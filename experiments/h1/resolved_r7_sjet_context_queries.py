"""Resolved qqbb W circuit, R7: do sjets query lep/nu or large jets to decide W?

This script separates attention presence from causal dependence.

Observational:
  - For leading truth-W sjet, subleading truth-W sjet, and leading non-W sjet
    (when present), report top query->category attention heads.

Causal:
  - Knock out direct attention edges from those sjet queries to lep/nu, H-ljet,
    all ljets, other sjets, and partner/W-sjet keys.
  - Report changes by baseline stratum (clean-exclusive, partial, channel-error).

This tests direct reads only. A null result does not rule out indirect context paths.

Usage:
  .venv/bin/python experiments/h1/resolved_r7_sjet_context_queries.py \
      --model thesis-ent1-bn1-d152 --skip 24 --batches 12
"""
import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

torch.set_num_threads(6)

from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from interp.activations import ActivationCache, hook_attention_heads
from models.registry import load_model


PAD, NU, LJ, SJ = 5, 2, 3, 4
W_CLASS = 2
NONE_CLASS = 0


def pt(p4):
    return torch.sqrt(torch.clamp(p4[..., 0] ** 2 + p4[..., 1] ** 2, min=0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="thesis-ent1-bn1-d152")
    ap.add_argument("--skip", type=int, default=24)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--ckpt-override", default=None)
    args = ap.parse_args()

    data = os.path.join(REPO, "tmp_data_20250321v1_signal/")
    stds = np.ones(7)
    stds[:4] = 1e5
    dl = ProportionalMemoryMappedDataset(
        N_Real_Vars_In_File=7,
        N_Real_Vars_To_Return=7,
        memmap_paths={d: f"{data}dsid_{d}.memmap" for d in range(510115, 510125)},
        max_objs_in_memmap=15,
        batch_size=args.batch_size,
        device="cpu",
        is_train=False,
        n_splits=2,
        validation_split_idx=0,
        n_targets=3,
        shuffle=False,
        shuffle_batch=False,
        means=None,
        stds=stds,
        objs_to_output=15,
        signal_only=True,
        has_eventNumbers=True,
    )
    for _ in range(args.skip):
        next(dl)

    model, _ = load_model(
        args.model,
        checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
        register_bottleneck_hook=False,
        checkpoint_override=args.ckpt_override,
    )
    model.eval()
    bn = model.bottleneck_attention
    nblk = model.num_attention_blocks
    nh = model.attention_blocks[0]["self_attention"].num_heads

    xs, ts = [], []
    for _ in range(args.batches):
        b = next(dl)
        t = b["types"]
        ok = ((t == NU).sum(1) == 1) & (((t == 0) | (t == 1)).sum(1) == 1)
        xs.append(b["x"][ok].clone())
        ts.append(t[ok].clone())
    X_all = torch.cat(xs)
    T_all = torch.cat(ts)
    TRU_all = torch.round(X_all[..., -1]).long()
    ar_all = torch.arange(len(X_all))
    lpos_all = ((T_all == 0) | (T_all == 1)).float().argmax(1)
    lvbb = TRU_all[ar_all, lpos_all] == 3
    boosted = (~lvbb) & (((T_all == LJ) & (TRU_all == 2)).sum(1) == 1)
    resolved = (~lvbb) & (((T_all == SJ) & (TRU_all == 2)).sum(1) == 2)
    idx = torch.where(resolved)[0]
    X = X_all[idx].clone()
    T = T_all[idx].clone()
    TRU = TRU_all[idx].clone()
    N = len(X)
    ar = torch.arange(N)
    lpos = ((T == 0) | (T == 1)).float().argmax(1)
    npos = (T == NU).float().argmax(1)
    jet_mask = (T == LJ) | (T == SJ)

    wrows = []
    other_rows = []
    other_valid = []
    for i in range(N):
        rows = torch.where((T[i] == SJ) & (TRU[i] == 2))[0]
        order = torch.argsort(pt(X[i, rows, :4]), descending=True)
        wrows.append(rows[order])
        others = torch.where((T[i] == SJ) & ~(TRU[i] == 2))[0]
        if len(others):
            oo = torch.argsort(pt(X[i, others, :4]), descending=True)
            other_rows.append(others[oo[0]])
            other_valid.append(True)
        else:
            other_rows.append(torch.tensor(0))
            other_valid.append(False)
    wrows = torch.stack(wrows)
    lead = wrows[:, 0]
    sub = wrows[:, 1]
    other1 = torch.stack(other_rows).long()
    other1_valid = torch.tensor(other_valid, dtype=torch.bool)

    # Standard hook pass for baseline and attention stats.
    cache = ActivationCache()
    lib_hooks = [
        m.register_forward_hook(fn, with_kwargs=True)
        for m, fn in hook_attention_heads(
            model,
            cache,
            detach=True,
            SINGLE_ATTENTION=False,
            bottleneck_attention_output=bn,
        )
    ]
    outputs = []
    attn_sum = defaultdict(float)
    attn_count = defaultdict(int)

    def add_attn_stats(sl, q_name, q_pos, q_valid, cats):
        bsz = sl.stop - sl.start
        bidx = torch.arange(bsz)
        qlocal = q_pos[sl]
        valid = q_valid[sl]
        if valid.sum() == 0:
            return
        for blk in range(nblk):
            aw_all = cache[f"block_{blk}_attention"]["attn_weights_per_head"]
            for h in range(nh):
                rows = aw_all[bidx, h, qlocal, :]
                for cname, cmask in cats.items():
                    vals = (rows * cmask).sum(1)
                    attn_sum[(q_name, blk, h, cname)] += vals[valid].sum().item()
                    attn_count[(q_name, blk, h, cname)] += int(valid.sum().item())

    with torch.no_grad():
        for c0 in range(0, N, args.batch_size):
            sl = slice(c0, min(c0 + args.batch_size, N))
            xx = X[sl]
            tt = T[sl]
            tru = TRU[sl]
            out = model(xx[..., :5], tt)
            outputs.append(out)
            wsj = (tt == SJ) & (tru == 2)
            hlj = (tt == LJ) & (tru == 1)
            all_lj = tt == LJ
            other_sj = (tt == SJ) & ~wsj
            lep = (tt == 0) | (tt == 1)
            nu = tt == NU
            lepnu = lep | nu
            cats_common = {
                "lepnu": lepnu,
                "lepton": lep,
                "nu": nu,
                "H_ljet": hlj,
                "all_ljets": all_lj,
                "other_sjets": other_sj,
                "W_sjets": wsj,
            }
            # Partner masks for lead/sub query, W-pair mask for other-sjet query.
            sub_key = torch.zeros_like(tt, dtype=torch.bool)
            sub_key[torch.arange(len(xx)), sub[sl]] = True
            lead_key = torch.zeros_like(tt, dtype=torch.bool)
            lead_key[torch.arange(len(xx)), lead[sl]] = True
            add_attn_stats(sl, "lead_Wsjet", lead, torch.ones(N, dtype=torch.bool), {**cats_common, "partner_Wsjet": sub_key})
            add_attn_stats(sl, "sub_Wsjet", sub, torch.ones(N, dtype=torch.bool), {**cats_common, "partner_Wsjet": lead_key})
            add_attn_stats(sl, "other_sjet1", other1, other1_valid, cats_common)
    base_out = torch.cat(outputs)
    for h in lib_hooks:
        h.remove()

    def readouts(out):
        pred = out.argmax(-1)
        margin = out[..., W_CLASS] - out[..., NONE_CLASS
        ]
        wsj_claims = (pred[ar.unsqueeze(1), wrows] == W_CLASS).sum(1)
        nonw_mask = jet_mask.clone()
        nonw_mask[ar, lead] = False
        nonw_mask[ar, sub] = False
        nonw_claim = ((pred == W_CLASS) & nonw_mask).any(1)
        other1_claim = np.full(N, np.nan)
        other1_margin = np.full(N, np.nan)
        if other1_valid.any():
            vv = other1_valid
            other1_claim[vv.numpy()] = (pred[ar[vv], other1[vv]] == W_CLASS).numpy().astype(float)
            other1_margin[vv.numpy()] = margin[ar[vv], other1[vv]].numpy()
        lep_w = pred[ar, lpos] == W_CLASS
        nu_w = pred[ar, npos] == W_CLASS
        channel_correct = (~lep_w) & (~nu_w)
        clean = channel_correct & (wsj_claims == 2) & (~nonw_claim)
        return {
            "lep_w": lep_w.numpy(),
            "nu_w": nu_w.numpy(),
            "channel_correct": channel_correct.numpy(),
            "clean": clean.numpy(),
            "both_wsj": (wsj_claims == 2).numpy(),
            "any_wsj": (wsj_claims >= 1).numpy(),
            "lead_w": (pred[ar, lead] == W_CLASS).numpy(),
            "sub_w": (pred[ar, sub] == W_CLASS).numpy(),
            "nonw_claim": nonw_claim.numpy(),
            "other1_w": other1_claim,
            "lead_margin": margin[ar, lead].numpy(),
            "sub_margin": margin[ar, sub].numpy(),
            "other1_margin": other1_margin,
            "lep_margin": margin[ar, lpos].numpy(),
        }

    base = readouts(base_out)
    clean = torch.as_tensor(base["clean"], dtype=torch.bool)
    channel_correct = torch.as_tensor(base["channel_correct"], dtype=torch.bool)
    partial = channel_correct & (~clean)
    channel_error = ~channel_correct

    print(f"model={args.model}  skip={args.skip}  batches={args.batches}")
    print(
        f"events={len(X_all)}  boosted={boosted.sum().item()}  resolved={N}; "
        f"clean={clean.sum().item()} partial={partial.sum().item()} error={channel_error.sum().item()} "
        f"other_sjet1_valid={other1_valid.sum().item()}"
    )

    print("\n=== Top Observational Sjet-Query Attention Heads ===")
    for q_name in ("lead_Wsjet", "sub_Wsjet", "other_sjet1"):
        print(f"{q_name}:")
        for cname in ("lepnu", "H_ljet", "all_ljets", "other_sjets", "W_sjets", "partner_Wsjet"):
            rows = []
            for blk in range(nblk):
                for h in range(nh):
                    n = attn_count.get((q_name, blk, h, cname), 0)
                    if n:
                        rows.append((attn_sum[(q_name, blk, h, cname)] / n, blk, h))
            if not rows:
                continue
            rows = sorted(rows, reverse=True)[:4]
            desc = "  ".join(f"b{blk}h{h}:{val:.3f}" for val, blk, h in rows)
            print(f"  {cname:14s} {desc}")

    # Surgical hook for edge KO.
    ko = {"active": False}

    def make_surgical(block_idx, block):
        bdown = block["bottleneck_down"] if bn is not None else None
        bup = block["bottleneck_up"] if bn is not None else None

        def hook_fn(module, inputs, kwargs, output):
            q0 = inputs[0]
            kpm = kwargs.get("key_padding_mask", None)
            kpm = F._canonical_mask(
                mask=kpm,
                mask_name="key_padding_mask",
                other_type=F._none_or_dtype(kpm),
                other_name="",
                target_type=q0.dtype,
            )
            q = k = v = q0.transpose(1, 0)
            tgt_len, bsz, embed_dim = q.shape
            q, k, v = F._in_projection_packed(q, k, v, module.in_proj_weight, module.in_proj_bias)
            head_dim = embed_dim // nh
            q = q.view(tgt_len, bsz * nh, head_dim).transpose(0, 1)
            k = k.view(tgt_len, bsz * nh, head_dim).transpose(0, 1)
            v = v.view(tgt_len, bsz * nh, head_dim).transpose(0, 1)
            q_scaled = q * math.sqrt(1.0 / float(head_dim))
            if kpm is not None:
                kpm_e = (
                    kpm.view(bsz, 1, 1, tgt_len)
                    .expand(-1, nh, -1, -1)
                    .reshape(bsz * nh, 1, tgt_len)
                )
                attn = torch.baddbmm(kpm_e, q_scaled, k.transpose(-2, -1))
            else:
                attn = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1).view(bsz, nh, tgt_len, tgt_len)
            if ko["active"] and block_idx in ko["blocks"]:
                car = torch.arange(bsz)
                for group in range(ko["q"].shape[0]):
                    qg = ko["q"][group]
                    kg = ko["kmask"][group]
                    valid = ko["valid"][group]
                    for head in ko["heads"]:
                        row = attn[car, head, qg, :]
                        row2 = row * (~kg).float()
                        row2 = row2 / row2.sum(-1, keepdim=True).clamp(min=1e-9)
                        row = torch.where(valid.unsqueeze(-1), row2, row)
                        attn[car, head, qg, :] = row
            out_by_head = torch.bmm(attn.view(bsz * nh, tgt_len, tgt_len), v)
            out_by_head = out_by_head.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            per_head = torch.empty(bsz, nh, tgt_len, module.out_proj.weight.shape[0])
            for head in range(nh):
                w_rows = module.out_proj.weight.transpose(0, 1)[head * head_dim : (head + 1) * head_dim]
                head_out = torch.matmul(out_by_head[:, head * head_dim : (head + 1) * head_dim], w_rows)
                per_head[:, head] = head_out.contiguous().view(tgt_len, bsz, -1).transpose(0, 1)
            if bn is not None:
                for head in range(nh):
                    scalar = torch.matmul(per_head[:, head].reshape(-1, embed_dim), bdown[head].weight.t())
                    per_head[:, head] = torch.matmul(scalar, bup[head].weight.t()).view(bsz, tgt_len, embed_dim)
            return per_head.sum(dim=1) + module.out_proj.bias, output[1]

        return hook_fn

    surgical_hooks = [
        block["self_attention"].register_forward_hook(make_surgical(i, block), with_kwargs=True)
        for i, block in enumerate(model.attention_blocks)
    ]

    def onehot(pos):
        m = torch.zeros(N, 15, dtype=torch.bool)
        m[ar, pos] = True
        return m

    h_ljet = (T == LJ) & (TRU == 1)
    all_ljets = T == LJ
    lepnu = (T == NU) | (T == 0) | (T == 1)
    other_sjets = (T == SJ) & ~(TRU == 2)
    w_sjets = (T == SJ) & (TRU == 2)
    lead_key = onehot(lead)
    sub_key = onehot(sub)

    q_specs = {
        "lead": (lead, torch.ones(N, dtype=torch.bool)),
        "sub": (sub, torch.ones(N, dtype=torch.bool)),
        "both_wsj": (None, torch.ones(N, dtype=torch.bool)),
        "other1": (other1, other1_valid),
    }
    key_specs = {
        "lepnu": lepnu,
        "H_ljet": h_ljet,
        "all_ljets": all_ljets,
        "other_sjets": other_sjets,
        "W_sjets": w_sjets,
        "partner": None,
    }

    def edge_groups(qname, kname, sl):
        if qname == "both_wsj":
            if kname == "partner":
                return [(lead[sl], sub_key[sl], torch.ones(sl.stop - sl.start, dtype=torch.bool)),
                        (sub[sl], lead_key[sl], torch.ones(sl.stop - sl.start, dtype=torch.bool))]
            km = key_specs[kname][sl]
            return [(lead[sl], km, torch.ones(sl.stop - sl.start, dtype=torch.bool)),
                    (sub[sl], km, torch.ones(sl.stop - sl.start, dtype=torch.bool))]
        q, valid = q_specs[qname]
        if kname == "partner":
            if qname == "lead":
                return [(lead[sl], sub_key[sl], torch.ones(sl.stop - sl.start, dtype=torch.bool))]
            if qname == "sub":
                return [(sub[sl], lead_key[sl], torch.ones(sl.stop - sl.start, dtype=torch.bool))]
            if qname == "other1":
                return [(other1[sl], w_sjets[sl], other1_valid[sl])]
            return []
        return [(q[sl], key_specs[kname][sl], valid[sl])]

    def fwd_ko(qname=None, kname=None, blocks=None, heads=None):
        outs = []
        with torch.no_grad():
            for c0 in range(0, N, args.batch_size):
                sl = slice(c0, min(c0 + args.batch_size, N))
                if qname is None:
                    ko["active"] = False
                else:
                    groups = edge_groups(qname, kname, sl)
                    ko.update(
                        active=True,
                        blocks=set(blocks),
                        heads=tuple(range(nh)) if heads is None else tuple(heads),
                        q=torch.stack([g[0] for g in groups]),
                        kmask=torch.stack([g[1] for g in groups]),
                        valid=torch.stack([g[2] for g in groups]),
                    )
                outs.append(model(X[sl, :, :5], T[sl]))
        ko["active"] = False
        return torch.cat(outs)

    surgical_base = fwd_ko()
    max_dev = (surgical_base - base_out).abs().max().item()
    if not torch.allclose(surgical_base, base_out, atol=1e-4):
        raise RuntimeError(f"surgical baseline mismatch {max_dev}")

    base_rd = base
    groups = [
        ("all", np.ones(N, dtype=bool)),
        ("clean", clean.numpy()),
        ("partial", partial.numpy()),
        ("error", channel_error.numpy()),
        ("other1_valid", other1_valid.numpy()),
    ]

    records = []

    def report_condition(qname, kname, blocks, label, heads=None):
        out = fwd_ko(qname, kname, blocks, heads=heads)
        post = readouts(out)
        for gname, mask in groups:
            if mask.sum() < 20:
                continue
            # For other1 KOs, non-valid events should not be included in all/clean/partial/error.
            mm = mask.copy()
            if qname == "other1":
                mm &= other1_valid.numpy()
            if mm.sum() < 20:
                continue
            rec = {
                "q": qname,
                "key": kname,
                "blocks": label,
                "group": gname,
                "n": int(mm.sum()),
                "delta_channel_correct": post["channel_correct"][mm].mean() - base_rd["channel_correct"][mm].mean(),
                "delta_clean": post["clean"][mm].mean() - base_rd["clean"][mm].mean(),
                "delta_lead_w": post["lead_w"][mm].mean() - base_rd["lead_w"][mm].mean(),
                "delta_sub_w": post["sub_w"][mm].mean() - base_rd["sub_w"][mm].mean(),
                "delta_nonw": post["nonw_claim"][mm].mean() - base_rd["nonw_claim"][mm].mean(),
                "delta_other1_w": np.nan,
                "med_delta_lead_margin": np.median(post["lead_margin"][mm] - base_rd["lead_margin"][mm]),
                "med_delta_sub_margin": np.median(post["sub_margin"][mm] - base_rd["sub_margin"][mm]),
                "med_delta_lep_margin": np.median(post["lep_margin"][mm] - base_rd["lep_margin"][mm]),
            }
            if qname == "other1":
                rec["delta_other1_w"] = np.nanmean(post["other1_w"][mm]) - np.nanmean(base_rd["other1_w"][mm])
            records.append(rec)

    # Compact causal pass: all-block/direct category KOs, plus block-local sweep for both W-sjets.
    for qname in ("lead", "sub", "both_wsj", "other1"):
        keys = ["lepnu", "H_ljet", "all_ljets", "other_sjets", "partner"]
        if qname == "other1":
            keys = ["lepnu", "H_ljet", "all_ljets", "W_sjets", "other_sjets"]
        for kname in keys:
            report_condition(qname, kname, set(range(nblk)), "all_blocks")
    for qname, kname in [("both_wsj", "lepnu"), ("both_wsj", "all_ljets"), ("both_wsj", "other_sjets"), ("both_wsj", "partner"), ("other1", "W_sjets"), ("other1", "all_ljets")]:
        for blk in range(nblk):
            report_condition(qname, kname, {blk}, f"b{blk}")
    for kname in ("lepnu", "all_ljets", "H_ljet"):
        for blk in range(nblk):
            for head in range(nh):
                report_condition("both_wsj", kname, {blk}, f"b{blk}h{head}", heads=(head,))

    print(f"\n=== Direct Sjet-Query Edge KO (surgical max dev {max_dev:.2e}) ===")
    print(
        f"{'q':10s} {'key':12s} {'blocks':10s} {'group':10s} {'n':>5s} "
        f"{'dChan':>7s} {'dClean':>7s} {'dLead':>7s} {'dSub':>7s} {'dNonW':>7s} {'dOth1':>7s} "
        f"{'med dLeadM':>11s} {'med dSubM':>10s}"
    )
    # Print the largest effects first, then key all-block rows.
    def effect_size(r):
        vals = [r["delta_channel_correct"], r["delta_clean"], r["delta_lead_w"], r["delta_sub_w"], r["delta_nonw"]]
        if np.isfinite(r["delta_other1_w"]):
            vals.append(r["delta_other1_w"])
        return max(abs(v) for v in vals)

    to_print = sorted(records, key=effect_size, reverse=True)[:70]
    for r in to_print:
        do = r["delta_other1_w"]
        do_s = "   nan" if not np.isfinite(do) else f"{do:+7.3f}"
        print(
            f"{r['q']:10s} {r['key']:12s} {r['blocks']:10s} {r['group']:10s} {r['n']:5d} "
            f"{r['delta_channel_correct']:+7.3f} {r['delta_clean']:+7.3f} {r['delta_lead_w']:+7.3f} "
            f"{r['delta_sub_w']:+7.3f} {r['delta_nonw']:+7.3f} {do_s} "
            f"{r['med_delta_lead_margin']:+11.3f} {r['med_delta_sub_margin']:+10.3f}"
        )

    out_dir = os.path.join(REPO, "tmp_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resolved_r7_sjet_context_queries_{args.model}_skip{args.skip}_b{args.batches}.csv")
    header = list(records[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in records:
            f.write(",".join(str(r[k]) for k in header) + "\n")
    print(f"\nsaved R7 records: {out_path}")

    for h in surgical_hooks:
        h.remove()


if __name__ == "__main__":
    main()

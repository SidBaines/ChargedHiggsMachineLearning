"""Resolved qqbb W circuit, R5: success-stratified probes and edge KO.

This script asks whether the pair-binding path found in R4 is specifically a
success-path circuit. It conditions resolved events on baseline behavior:

  - clean_exclusive: lep/nu none, both truth-W sjets claim W, no non-W jet claims W.
  - channel_correct_partial: lep/nu none, but not clean-exclusive.
  - channel_error: lep or nu claims W.

It reports R1-style feature/wire summaries and R4-style direct W-sjet<->W-sjet edge
knockouts within these baseline-defined strata.

Usage:
  .venv/bin/python experiments/h1/resolved_r5_success_strata.py \
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
WIRES = [(1, 3), (2, 2), (2, 0), (2, 3)]


def pt(p4):
    return torch.sqrt(torch.clamp(p4[..., 0] ** 2 + p4[..., 1] ** 2, min=0.0))


def mass(p4):
    return torch.sqrt(torch.clamp(p4[..., 3] ** 2 - (p4[..., :3] ** 2).sum(-1), min=0.0))


def qfmt(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "n/a"
    return f"med {np.median(x):+.3f}  p16/p84 {np.percentile(x,16):+.3f}/{np.percentile(x,84):+.3f}"


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
    for i in range(N):
        rows = torch.where((T[i] == SJ) & (TRU[i] == 2))[0]
        order = torch.argsort(pt(X[i, rows, :4]), descending=True)
        wrows.append(rows[order])
    wrows = torch.stack(wrows)
    lead = wrows[:, 0]
    sub = wrows[:, 1]

    attn_modules = [blk["self_attention"] for blk in model.attention_blocks]
    d_model = attn_modules[0].embed_dim
    d_head = d_model // nh

    # Baseline with standard hook, collecting wire decomposition.
    cache = ActivationCache()
    lib_handles = [
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
    wire_scalar = defaultdict(list)
    wire_value = defaultdict(list)
    wire_attn = defaultdict(list)

    def scalar_value_by_key(blk, h):
        x_in = cache[f"block_{blk}_attention"]["input"][0]
        attn = attn_modules[blk]
        wv = attn.in_proj_weight[2 * d_model : 3 * d_model]
        bv = attn.in_proj_bias[2 * d_model : 3 * d_model]
        v = x_in @ wv.t() + bv
        v_h = v[..., h * d_head : (h + 1) * d_head]
        wout_h = attn.out_proj.weight[:, h * d_head : (h + 1) * d_head]
        down_w = model.attention_blocks[blk]["bottleneck_down"][h].weight[0]
        return (v_h @ wout_h.t()) @ down_w

    with torch.no_grad():
        for c0 in range(0, N, args.batch_size):
            sl = slice(c0, min(c0 + args.batch_size, N))
            xx = X[sl]
            tt = T[sl]
            tru = TRU[sl]
            out = model(xx[..., :5], tt)
            outputs.append(out)
            bsz = len(xx)
            bidx = torch.arange(bsz)
            np_ = npos[sl]
            wsj = (tt == SJ) & (tru == 2)
            hlj = (tt == LJ) & (tru == 1)
            other_sj = (tt == SJ) & ~wsj
            other_lj = (tt == LJ) & ~hlj
            lep = (tt == 0) | (tt == 1)
            nu = tt == NU
            cats = {
                "W_sjets": wsj,
                "H_ljet": hlj,
                "other_sjets": other_sj,
                "other_ljets": other_lj,
                "lepton": lep,
                "nu_self": nu,
            }
            for key in WIRES:
                blk, h = key
                bna = cache[f"block_{blk}_attention"]["bottleneck_activation"]
                wire_scalar[key].append(bna[bidx, h, np_, 0])
                aw = cache[f"block_{blk}_attention"]["attn_weights_per_head"][:, h]
                a_nu = aw[bidx, np_]
                sv = scalar_value_by_key(blk, h)
                per_key = a_nu * sv
                for cname, cmask in cats.items():
                    wire_attn[(key, cname)].append((a_nu * cmask).sum(1))
                    wire_value[(key, cname)].append((per_key * cmask).sum(1))
    base_out = torch.cat(outputs)
    for h in lib_handles:
        h.remove()

    base_pred = base_out.argmax(-1)
    base_margin = base_out[..., W_CLASS] - base_out[..., NONE_CLASS]
    wsj_claims = (base_pred[ar.unsqueeze(1), wrows] == W_CLASS).sum(1)
    nonw_mask = jet_mask.clone()
    nonw_mask[ar, lead] = False
    nonw_mask[ar, sub] = False
    nonw_claim = ((base_pred == W_CLASS) & nonw_mask).any(1)
    lep_w = base_pred[ar, lpos] == W_CLASS
    nu_w = base_pred[ar, npos] == W_CLASS
    channel_correct = (~lep_w) & (~nu_w)
    clean = channel_correct & (wsj_claims == 2) & (~nonw_claim)
    correct_partial = channel_correct & (~clean)
    channel_error = ~channel_correct
    partial_one = correct_partial & (wsj_claims == 1)
    partial_zero = correct_partial & (wsj_claims == 0)
    partial_nonw = correct_partial & nonw_claim

    p4_pair = X[ar.unsqueeze(1), wrows, :4].sum(1)
    mjj = (mass(p4_pair) * 100.0).numpy()
    pair_pt = (pt(p4_pair) * 100.0).numpy()
    pair_sumpt = (pt(X[ar.unsqueeze(1), wrows, :4]) * 100.0).sum(1).numpy()
    ht = (pt(X[..., :4]) * 100.0 * jet_mask.float()).sum(1).numpy()
    pair_relpt = pair_pt / np.maximum(ht, 1e-6)
    wsj_sum_margin = base_margin[ar.unsqueeze(1), wrows].sum(1).numpy()
    wsj_min_margin = base_margin[ar.unsqueeze(1), wrows].min(1).values.numpy()
    lep_margin = base_margin[ar, lpos].numpy()

    groups = [
        ("all_resolved", torch.ones(N, dtype=torch.bool)),
        ("clean_exclusive", clean),
        ("channel_correct_partial", correct_partial),
        ("partial_one_wsj", partial_one),
        ("partial_zero_wsj", partial_zero),
        ("partial_nonw_claim", partial_nonw),
        ("channel_error", channel_error),
    ]

    print(f"model={args.model}  skip={args.skip}  batches={args.batches}")
    print(
        f"events={len(X_all)}  boosted={boosted.sum().item()}  resolved={N}; "
        f"channel_correct={channel_correct.sum().item()} clean={clean.sum().item()} "
        f"partial={correct_partial.sum().item()} error={channel_error.sum().item()}"
    )

    print("\n=== Baseline Success Strata ===")
    print(
        f"{'group':28s} {'n':>5s} {'PlepW':>7s} {'PanyW':>7s} {'Pboth':>7s} "
        f"{'PnonW':>7s} {'mjj med':>8s} {'relpt med':>9s} {'sumM med':>9s}"
    )
    for name, mask_t in groups:
        mask = mask_t.numpy().astype(bool)
        if mask.sum() == 0:
            continue
        print(
            f"{name:28s} {mask.sum():5d} {lep_w.numpy()[mask].mean():7.3f} "
            f"{(wsj_claims.numpy()[mask] >= 1).mean():7.3f} {(wsj_claims.numpy()[mask] == 2).mean():7.3f} "
            f"{nonw_claim.numpy()[mask].mean():7.3f} {np.median(mjj[mask]):8.1f} "
            f"{np.median(pair_relpt[mask]):9.3f} {np.median(wsj_sum_margin[mask]):9.2f}"
        )

    print("\n=== Wire Decomposition At Nu By Stratum ===")
    for key in WIRES:
        print(f"b{key[0]}h{key[1]}:")
        scalar = torch.cat(wire_scalar[key]).numpy()
        vals = {c: torch.cat(wire_value[(key, c)]).numpy() for c in ["W_sjets", "H_ljet", "other_sjets", "other_ljets", "lepton", "nu_self"]}
        atts = {c: torch.cat(wire_attn[(key, c)]).numpy() for c in ["W_sjets", "H_ljet", "other_sjets", "other_ljets", "lepton", "nu_self"]}
        for gname, mask_t in [("clean_exclusive", clean), ("channel_correct_partial", correct_partial), ("channel_error", channel_error)]:
            mask = mask_t.numpy().astype(bool)
            if mask.sum() == 0:
                continue
            jets_val = vals["W_sjets"][mask].mean() + vals["H_ljet"][mask].mean() + vals["other_sjets"][mask].mean() + vals["other_ljets"][mask].mean()
            lepnu_val = vals["lepton"][mask].mean() + vals["nu_self"][mask].mean()
            other_val = vals["H_ljet"][mask].mean() + vals["other_sjets"][mask].mean() + vals["other_ljets"][mask].mean()
            print(
                f"  {gname:24s} n={mask.sum():4d} scalar {scalar[mask].mean():+7.3f} "
                f"Wsj attn {atts['W_sjets'][mask].mean():.3f} Wsj val {vals['W_sjets'][mask].mean():+7.3f} "
                f"otherJet val {other_val:+7.3f} lepnu val {lepnu_val:+7.3f} allJet val {jets_val:+7.3f}"
            )

    # Surgical edge KO. We need a fresh by-hand hook after removing standard hooks.
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
                    for head in ko["heads"]:
                        row = attn[car, head, qg, :]
                        row = row * (~kg).float()
                        if ko["renorm"]:
                            row = row / row.sum(-1, keepdim=True).clamp(min=1e-9)
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

    lead_key = onehot(lead)
    sub_key = onehot(sub)
    edges_bidir = [(lead, sub_key), (sub, lead_key)]
    edges_sub_to_lead = [(sub, lead_key)]

    def fwd_ko(blocks=None, heads=None, edge_groups=None):
        outs = []
        with torch.no_grad():
            for c0 in range(0, N, args.batch_size):
                sl = slice(c0, min(c0 + args.batch_size, N))
                if blocks is None:
                    ko["active"] = False
                else:
                    ko.update(
                        active=True,
                        blocks=set(blocks),
                        heads=tuple(heads),
                        renorm=True,
                        q=torch.stack([g[0][sl] for g in edge_groups]),
                        kmask=torch.stack([g[1][sl] for g in edge_groups]),
                    )
                outs.append(model(X[sl, :, :5], T[sl]))
        ko["active"] = False
        return torch.cat(outs)

    # Validate inactive surgical path against baseline already collected.
    surgical_base = fwd_ko()
    max_dev = (surgical_base - base_out).abs().max().item()
    if not torch.allclose(surgical_base, base_out, atol=1e-4):
        raise RuntimeError(f"surgical baseline mismatch {max_dev}")

    def rd(out):
        pred = out.argmax(-1)
        margin = out[..., W_CLASS] - out[..., NONE_CLASS]
        c = (pred[ar.unsqueeze(1), wrows] == W_CLASS).sum(1)
        return {
            "lep_w": (pred[ar, lpos] == W_CLASS).numpy(),
            "nu_w": (pred[ar, npos] == W_CLASS).numpy(),
            "any_wsj": (c >= 1).numpy(),
            "both_wsj": (c == 2).numpy(),
            "lead_w": (pred[ar, lead] == W_CLASS).numpy(),
            "sub_w": (pred[ar, sub] == W_CLASS).numpy(),
            "wsj_sum_margin": margin[ar.unsqueeze(1), wrows].sum(1).numpy(),
            "lep_margin": margin[ar, lpos].numpy(),
        }

    base_rd = rd(base_out)
    conditions = [
        ("b2h3_sub_to_lead", {2}, (3,), edges_sub_to_lead),
        ("b2h3_bidir", {2}, (3,), edges_bidir),
        ("b2_all_heads_bidir", {2}, tuple(range(nh)), edges_bidir),
        ("b1h3_bidir", {1}, (3,), edges_bidir),
        ("all_blocks_all_heads_bidir", set(range(nblk)), tuple(range(nh)), edges_bidir),
    ]

    print(f"\n=== Edge KO By Baseline Stratum (surgical max dev {max_dev:.2e}) ===")
    print(
        f"{'condition':28s} {'group':24s} {'n':>5s} {'dPlep':>8s} {'dAny':>8s} "
        f"{'dBoth':>8s} {'dSub':>8s} {'med dSumM':>10s} {'lepFlip':>8s}"
    )
    records = []
    ko_groups = [
        ("clean_exclusive", clean),
        ("channel_correct_partial", correct_partial),
        ("channel_error", channel_error),
        ("partial_one_wsj", partial_one),
        ("partial_zero_wsj", partial_zero),
    ]
    for cname, blocks, heads, edges in conditions:
        out = fwd_ko(blocks, heads, edges)
        post = rd(out)
        for gname, mask_t in ko_groups:
            mask = mask_t.numpy().astype(bool)
            if mask.sum() < 20:
                continue
            rec = {
                "condition": cname,
                "group": gname,
                "n": int(mask.sum()),
                "delta_p_lep_w": post["lep_w"][mask].mean() - base_rd["lep_w"][mask].mean(),
                "delta_p_any_wsj": post["any_wsj"][mask].mean() - base_rd["any_wsj"][mask].mean(),
                "delta_p_both_wsj": post["both_wsj"][mask].mean() - base_rd["both_wsj"][mask].mean(),
                "delta_p_sub_w": post["sub_w"][mask].mean() - base_rd["sub_w"][mask].mean(),
                "med_delta_wsj_sum_margin": np.median(post["wsj_sum_margin"][mask] - base_rd["wsj_sum_margin"][mask]),
                "lep_flip": (post["lep_w"][mask] != base_rd["lep_w"][mask]).mean(),
            }
            records.append(rec)
            print(
                f"{cname:28s} {gname:24s} {rec['n']:5d} {rec['delta_p_lep_w']:+8.4f} "
                f"{rec['delta_p_any_wsj']:+8.4f} {rec['delta_p_both_wsj']:+8.4f} "
                f"{rec['delta_p_sub_w']:+8.4f} {rec['med_delta_wsj_sum_margin']:+10.3f} "
                f"{rec['lep_flip']:8.4f}"
            )

    out_dir = os.path.join(REPO, "tmp_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resolved_r5_success_strata_{args.model}_skip{args.skip}_b{args.batches}.csv")
    header = list(records[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for rec in records:
            f.write(",".join(str(rec[k]) for k in header) + "\n")
    print(f"\nsaved R5 edge-KO records: {out_path}")

    for h in surgical_hooks:
        h.remove()


if __name__ == "__main__":
    main()

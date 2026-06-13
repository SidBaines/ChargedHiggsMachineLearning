"""Resolved qqbb W circuit, R4: direct W-sjet<->W-sjet edge knockout.

R1 found the largest direct W-sjet partner attention in b2h3, b0h3, b1h3, b0h2.
This script causally tests whether those direct edges are necessary for natural
resolved-event W-sjet claiming.

Method: recompute attention by hand, zero selected post-softmax attention entries
from one truth-W sjet query to the partner truth-W sjet key, renormalize the row,
then apply the normal per-head out projection and bottleneck math. Inactive surgical
hook is validated against the standard bottleneck hook before interventions.

Readout: changes in W-sjet claims, lep/nu verdicts, and margins on fresh resolved
events. This only tests direct pair edges. A null result does not rule out indirect
pair/context effects.

Usage:
  .venv/bin/python experiments/h1/resolved_r4_sjet_edge_knockout.py \
      --model thesis-ent1-bn1-d152 --skip 24 --batches 12
"""
import argparse
import math
import os
import sys

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


def qfmt(x):
    x = np.asarray(x, dtype=np.float64)
    return f"med {np.median(x):+.3f}  mean {x.mean():+.3f}  p16/p84 {np.percentile(x,16):+.3f}/{np.percentile(x,84):+.3f}"


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

    def onehot(pos):
        m = torch.zeros(N, 15, dtype=torch.bool)
        m[ar, pos] = True
        return m

    lead_key = onehot(lead)
    sub_key = onehot(sub)
    edges_bidir = [(lead, sub_key), (sub, lead_key)]
    edges_lead_to_sub = [(lead, sub_key)]
    edges_sub_to_lead = [(sub, lead_key)]

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
                    for h in ko["heads"]:
                        row = attn[car, h, qg, :]
                        row = row * (~kg).float()
                        if ko["renorm"]:
                            row = row / row.sum(-1, keepdim=True).clamp(min=1e-9)
                        attn[car, h, qg, :] = row

            attn2 = attn.view(bsz * nh, tgt_len, tgt_len)
            out_by_head = torch.bmm(attn2, v)
            out_by_head = out_by_head.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            per_head = torch.empty(bsz, nh, tgt_len, module.out_proj.weight.shape[0])
            for h in range(nh):
                w_rows = module.out_proj.weight.transpose(0, 1)[h * head_dim : (h + 1) * head_dim]
                head_out = torch.matmul(out_by_head[:, h * head_dim : (h + 1) * head_dim], w_rows)
                per_head[:, h] = head_out.contiguous().view(tgt_len, bsz, -1).transpose(0, 1)
            if bn is not None:
                for h in range(nh):
                    scalar = torch.matmul(per_head[:, h].reshape(-1, embed_dim), bdown[h].weight.t())
                    per_head[:, h] = torch.matmul(scalar, bup[h].weight.t()).view(bsz, tgt_len, embed_dim)
            return per_head.sum(dim=1) + module.out_proj.bias, output[1]

        return hook_fn

    # Patch MHA to return weights, then validate inactive surgical hook against library hook.
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
    with torch.no_grad():
        ref = model(X[: min(256, N), :, :5], T[: min(256, N)])
    for h in lib_hooks:
        h.remove()
    surgical_hooks = [
        block["self_attention"].register_forward_hook(make_surgical(i, block), with_kwargs=True)
        for i, block in enumerate(model.attention_blocks)
    ]
    ko["active"] = False
    with torch.no_grad():
        got = model(X[: min(256, N), :, :5], T[: min(256, N)])
    max_dev = (ref - got).abs().max().item()
    if not torch.allclose(ref, got, atol=1e-4):
        raise RuntimeError(f"surgical hook mismatch: {max_dev}")

    def fwd_ko(blocks=None, heads=None, edge_groups=None, renorm=True):
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
                        renorm=renorm,
                        q=torch.stack([g[0][sl] for g in edge_groups]),
                        kmask=torch.stack([g[1][sl] for g in edge_groups]),
                    )
                outs.append(model(X[sl, :, :5], T[sl]))
        ko["active"] = False
        return torch.cat(outs)

    def readouts(out):
        pred = out.argmax(-1)
        margin = out[..., W_CLASS] - out[..., NONE_CLASS]
        wsj_claims = (pred[ar.unsqueeze(1), wrows] == W_CLASS).sum(1).numpy()
        return {
            "lep_w": (pred[ar, lpos] == W_CLASS).numpy(),
            "nu_w": (pred[ar, npos] == W_CLASS).numpy(),
            "both_wsj": wsj_claims == 2,
            "any_wsj": wsj_claims >= 1,
            "lead_w": (pred[ar, lead] == W_CLASS).numpy(),
            "sub_w": (pred[ar, sub] == W_CLASS).numpy(),
            "any_jet": (((pred == W_CLASS) & jet_mask).sum(1) > 0).numpy(),
            "wsj_sum_margin": margin[ar.unsqueeze(1), wrows].sum(1).numpy(),
            "wsj_min_margin": margin[ar.unsqueeze(1), wrows].min(1).values.numpy(),
            "lead_margin": margin[ar, lead].numpy(),
            "sub_margin": margin[ar, sub].numpy(),
            "lep_margin": margin[ar, lpos].numpy(),
        }

    base_out = fwd_ko()
    base = readouts(base_out)

    print(f"model={args.model}  skip={args.skip}  batches={args.batches}")
    print(
        f"events={len(X_all)}  boosted={boosted.sum().item()}  resolved={N}; "
        f"surgical max dev {max_dev:.2e}"
    )
    print("baseline resolved:")
    print(
        f"  P(lep=W)={base['lep_w'].mean():.4f}  P(any W-sjet)={base['any_wsj'].mean():.4f}  "
        f"P(both W-sjets)={base['both_wsj'].mean():.4f}  P(lead/sub W)={base['lead_w'].mean():.4f}/{base['sub_w'].mean():.4f}"
    )

    conditions = [
        ("all_blocks_all_heads_bidir", set(range(nblk)), tuple(range(nh)), edges_bidir, True),
        ("b0_all_heads_bidir", {0}, tuple(range(nh)), edges_bidir, True),
        ("b1_all_heads_bidir", {1}, tuple(range(nh)), edges_bidir, True),
        ("b2_all_heads_bidir", {2}, tuple(range(nh)), edges_bidir, True),
        ("b0h2_bidir", {0}, (2,), edges_bidir, True),
        ("b0h3_bidir", {0}, (3,), edges_bidir, True),
        ("b1h3_bidir", {1}, (3,), edges_bidir, True),
        ("b2h3_bidir", {2}, (3,), edges_bidir, True),
        ("candidate_heads_bidir", {0, 1, 2}, (2, 3), edges_bidir, True),
        ("b2h3_lead_to_sub", {2}, (3,), edges_lead_to_sub, True),
        ("b2h3_sub_to_lead", {2}, (3,), edges_sub_to_lead, True),
        ("all_blocks_all_heads_bidir_no_renorm", set(range(nblk)), tuple(range(nh)), edges_bidir, False),
    ]

    records = []
    print("\n=== Direct W-sjet<->W-sjet Edge KO ===")
    print(
        f"{'condition':36s} {'dPlep':>8s} {'dAny':>8s} {'dBoth':>8s} "
        f"{'dLead':>8s} {'dSub':>8s} {'med dSumM':>10s} {'med dLepM':>10s} {'flipLep':>8s}"
    )
    for name, blocks, heads, edges, renorm in conditions:
        out = fwd_ko(blocks, heads, edges, renorm=renorm)
        rd = readouts(out)
        rec = {
            "condition": name,
            "delta_p_lep_w": rd["lep_w"].mean() - base["lep_w"].mean(),
            "delta_p_any_wsj": rd["any_wsj"].mean() - base["any_wsj"].mean(),
            "delta_p_both_wsj": rd["both_wsj"].mean() - base["both_wsj"].mean(),
            "delta_p_lead_w": rd["lead_w"].mean() - base["lead_w"].mean(),
            "delta_p_sub_w": rd["sub_w"].mean() - base["sub_w"].mean(),
            "med_delta_wsj_sum_margin": np.median(rd["wsj_sum_margin"] - base["wsj_sum_margin"]),
            "med_delta_lep_margin": np.median(rd["lep_margin"] - base["lep_margin"]),
            "lep_flip": (rd["lep_w"] != base["lep_w"]).mean(),
        }
        records.append(rec)
        print(
            f"{name:36s} {rec['delta_p_lep_w']:+8.4f} {rec['delta_p_any_wsj']:+8.4f} "
            f"{rec['delta_p_both_wsj']:+8.4f} {rec['delta_p_lead_w']:+8.4f} {rec['delta_p_sub_w']:+8.4f} "
            f"{rec['med_delta_wsj_sum_margin']:+10.3f} {rec['med_delta_lep_margin']:+10.3f} {rec['lep_flip']:8.4f}"
        )

    out_dir = os.path.join(REPO, "tmp_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resolved_r4_sjet_edge_ko_{args.model}_skip{args.skip}_b{args.batches}.csv")
    header = list(records[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for rec in records:
            f.write(",".join(str(rec[k]) for k in header) + "\n")
    print(f"\nsaved edge-KO records: {out_path}")

    for h in surgical_hooks:
        h.remove()


if __name__ == "__main__":
    main()

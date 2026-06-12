"""Resolved qqbb W circuit, R9: causal replacement of the b2h3 partner message.

R8 found that the causal subleading-W-sjet -> leading-W-sjet partner path in b2h3
mostly carries relative-hardness / candidate-strength information. This script tests
whether that scalar message is causally sufficient to rescue failures, and whether a
simple analytical formula can replace it.

Intervention:
  In block 2 head 3, at the chosen query token, change the head's bottleneck scalar by

    target_partner_message - current_partner_message

  where partner_message is the contribution from the single partner key:

    attention(query, partner_key) * down_h(Wout_h V_h(partner_key))

This preserves all other heads, tokens, and non-partner key contributions. It is a
path-message scalar patch, not a full residual-stream patch.

Usage:
  .venv/bin/python experiments/h1/resolved_r9_partner_message_patch.py \
      --model thesis-ent1-bn1-d152 --skip 24 --batches 12
"""
import argparse
import csv
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


NU, LJ, SJ = 2, 3, 4
W_CLASS = 2
NONE_CLASS = 0
PATCH_BLOCK = 2
PATCH_HEAD = 3


def pt(p4):
    return torch.sqrt(torch.clamp(p4[..., 0] ** 2 + p4[..., 1] ** 2, min=0.0))


def mass(p4):
    return torch.sqrt(torch.clamp(p4[..., 3] ** 2 - (p4[..., :3] ** 2).sum(-1), min=0.0))


def fmt(x):
    if not np.isfinite(x):
        return "nan"
    return f"{x:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="thesis-ent1-bn1-d152")
    ap.add_argument("--skip", type=int, default=24)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--ckpt-override", default=None)
    ap.add_argument("--seed", type=int, default=0)
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
    cache = ActivationCache()
    handles = [
        m.register_forward_hook(fn, with_kwargs=True)
        for m, fn in hook_attention_heads(
            model,
            cache,
            detach=True,
            SINGLE_ATTENTION=False,
            bottleneck_attention_output=model.bottleneck_attention,
        )
    ]

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
    nblk = len(attn_modules)
    n_heads = attn_modules[0].num_heads
    d_model = attn_modules[0].embed_dim
    d_head = d_model // n_heads

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

    outputs = []
    sub_partner_msg = []
    lead_partner_msg = []
    sub_total_scalar = []
    lead_total_scalar = []
    max_msg_recon_err = 0.0
    with torch.no_grad():
        for c0 in range(0, N, args.batch_size):
            sl = slice(c0, min(c0 + args.batch_size, N))
            xx = X[sl]
            tt = T[sl]
            out = model(xx[..., :5], tt)
            outputs.append(out)
            bsz = len(xx)
            bidx = torch.arange(bsz)
            aw = cache[f"block_{PATCH_BLOCK}_attention"]["attn_weights_per_head"][:, PATCH_HEAD]
            bna = cache[f"block_{PATCH_BLOCK}_attention"]["bottleneck_activation"][:, PATCH_HEAD, :, 0]
            sv = scalar_value_by_key(PATCH_BLOCK, PATCH_HEAD)
            sub_q = sub[sl]
            lead_q = lead[sl]
            sub_msg = aw[bidx, sub_q, lead_q] * sv[bidx, lead_q]
            lead_msg = aw[bidx, lead_q, sub_q] * sv[bidx, sub_q]
            sub_partner_msg.append(sub_msg)
            lead_partner_msg.append(lead_msg)
            sub_total_scalar.append(bna[bidx, sub_q])
            lead_total_scalar.append(bna[bidx, lead_q])
            total_sub = (aw[bidx, sub_q, :] * sv).sum(1)
            total_lead = (aw[bidx, lead_q, :] * sv).sum(1)
            max_msg_recon_err = max(
                max_msg_recon_err,
                (total_sub - bna[bidx, sub_q]).abs().max().item(),
                (total_lead - bna[bidx, lead_q]).abs().max().item(),
            )

    base_out = torch.cat(outputs)
    for h in handles:
        h.remove()

    sub_partner_msg = torch.cat(sub_partner_msg).numpy()
    lead_partner_msg = torch.cat(lead_partner_msg).numpy()
    sub_total_scalar = torch.cat(sub_total_scalar).numpy()
    lead_total_scalar = torch.cat(lead_total_scalar).numpy()

    def readouts(out):
        pred = out.argmax(-1)
        margin = out[..., W_CLASS] - out[..., NONE_CLASS]
        wsj_claims = (pred[ar.unsqueeze(1), wrows] == W_CLASS).sum(1)
        nonw_mask = jet_mask.clone()
        nonw_mask[ar, lead] = False
        nonw_mask[ar, sub] = False
        nonw_claim = ((pred == W_CLASS) & nonw_mask).any(1)
        lep_w = pred[ar, lpos] == W_CLASS
        nu_w = pred[ar, npos] == W_CLASS
        channel_correct = (~lep_w) & (~nu_w)
        clean = channel_correct & (wsj_claims == 2) & (~nonw_claim)
        return {
            "channel_correct": channel_correct.numpy(),
            "clean": clean.numpy(),
            "any_wsj": (wsj_claims >= 1).numpy(),
            "both_wsj": (wsj_claims == 2).numpy(),
            "lead_w": (pred[ar, lead] == W_CLASS).numpy(),
            "sub_w": (pred[ar, sub] == W_CLASS).numpy(),
            "nonw_claim": nonw_claim.numpy(),
            "lead_margin": margin[ar, lead].numpy(),
            "sub_margin": margin[ar, sub].numpy(),
            "lep_margin": margin[ar, lpos].numpy(),
            "nu_margin": margin[ar, npos].numpy(),
            "wsj_sum_margin": (margin[ar, lead] + margin[ar, sub]).numpy(),
        }

    base = readouts(base_out)
    clean = torch.as_tensor(base["clean"], dtype=torch.bool)
    channel_correct = torch.as_tensor(base["channel_correct"], dtype=torch.bool)
    partial = channel_correct & (~clean)
    partial_zero = channel_correct & torch.as_tensor(~base["any_wsj"])
    partial_one = channel_correct & torch.as_tensor(base["any_wsj"]) & torch.as_tensor(~base["both_wsj"])
    partial_nonw = channel_correct & torch.as_tensor(base["nonw_claim"])
    error = ~channel_correct

    p4 = X[..., :4]
    pt_all = pt(p4) * 100.0
    ht_jets = (pt_all * jet_mask.float()).sum(1)
    pair_p4 = p4[ar, lead] + p4[ar, sub]
    pair_pt = pt(pair_p4) * 100.0
    pair_sumpt = pt_all[ar, lead] + pt_all[ar, sub]
    pair_relpt = (pair_pt / ht_jets.clamp(min=1e-6)).numpy()
    pair_relsumpt = (pair_sumpt / ht_jets.clamp(min=1e-6)).numpy()
    mjj = (mass(pair_p4) * 100.0).numpy()

    print(f"model={args.model}  skip={args.skip}  batches={args.batches}")
    print(
        f"events={len(X_all)}  boosted={boosted.sum().item()}  resolved={N}; "
        f"clean={clean.sum().item()} partial={partial.sum().item()} error={error.sum().item()} "
        f"partial_one={partial_one.sum().item()} partial_zero={partial_zero.sum().item()} "
        f"partial_nonw={partial_nonw.sum().item()}"
    )
    print(f"message/total scalar reconstruction max error = {max_msg_recon_err:.3e}")
    print(
        f"sub partner msg means: clean={sub_partner_msg[clean.numpy()].mean():+.3f} "
        f"partial_zero={sub_partner_msg[partial_zero.numpy()].mean():+.3f} "
        f"partial_one={sub_partner_msg[partial_one.numpy()].mean():+.3f} "
        f"error={sub_partner_msg[error.numpy()].mean():+.3f}"
    )

    # Surgical hook for bottleneck-scalar patching.
    patch = {"active": False}

    def make_surgical(block_idx, block):
        bdown = block["bottleneck_down"]
        bup = block["bottleneck_up"]

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
            head_dim = embed_dim // n_heads
            q = q.view(tgt_len, bsz * n_heads, head_dim).transpose(0, 1)
            k = k.view(tgt_len, bsz * n_heads, head_dim).transpose(0, 1)
            v = v.view(tgt_len, bsz * n_heads, head_dim).transpose(0, 1)
            q_scaled = q * math.sqrt(1.0 / float(head_dim))
            if kpm is not None:
                kpm_e = (
                    kpm.view(bsz, 1, 1, tgt_len)
                    .expand(-1, n_heads, -1, -1)
                    .reshape(bsz * n_heads, 1, tgt_len)
                )
                attn = torch.baddbmm(kpm_e, q_scaled, k.transpose(-2, -1))
            else:
                attn = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            out_by_head = torch.bmm(attn, v)
            out_by_head = out_by_head.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            per_head = torch.empty(bsz, n_heads, tgt_len, module.out_proj.weight.shape[0])
            for head in range(n_heads):
                w_rows = module.out_proj.weight.transpose(0, 1)[head * head_dim : (head + 1) * head_dim]
                head_out = torch.matmul(out_by_head[:, head * head_dim : (head + 1) * head_dim], w_rows)
                per_head[:, head] = head_out.contiguous().view(tgt_len, bsz, -1).transpose(0, 1)
            for head in range(n_heads):
                scalar = torch.matmul(per_head[:, head].reshape(-1, embed_dim), bdown[head].weight.t())
                scalar = scalar.view(bsz, tgt_len, -1)
                if patch["active"] and block_idx == PATCH_BLOCK and head == PATCH_HEAD:
                    car = torch.arange(bsz)
                    valid = patch["valid"]
                    if valid.any():
                        scalar[car[valid], patch["qpos"][valid], 0] += patch["delta"][valid].to(scalar.dtype)
                up = torch.matmul(scalar.reshape(-1, scalar.shape[-1]), bup[head].weight.t())
                per_head[:, head] = up.view(bsz, tgt_len, embed_dim)
            return per_head.sum(dim=1) + module.out_proj.bias, output[1]

        return hook_fn

    surgical_hooks = [
        block["self_attention"].register_forward_hook(make_surgical(i, block), with_kwargs=True)
        for i, block in enumerate(model.attention_blocks)
    ]

    def fwd_patch(qrole=None, target_msg=None, current_msg=None):
        outs = []
        with torch.no_grad():
            for c0 in range(0, N, args.batch_size):
                sl = slice(c0, min(c0 + args.batch_size, N))
                if qrole is None:
                    patch["active"] = False
                else:
                    qpos = sub[sl] if qrole == "sub" else lead[sl]
                    delta = torch.as_tensor(target_msg[sl] - current_msg[sl], dtype=torch.float32)
                    patch.update(
                        active=True,
                        qpos=qpos,
                        delta=delta,
                        valid=torch.ones(len(qpos), dtype=torch.bool),
                    )
                outs.append(model(X[sl, :, :5], T[sl]))
        patch["active"] = False
        return torch.cat(outs)

    surgical_base = fwd_patch()
    max_dev = (surgical_base - base_out).abs().max().item()
    if not torch.allclose(surgical_base, base_out, atol=1e-4):
        raise RuntimeError(f"surgical baseline mismatch {max_dev}")

    rng = np.random.default_rng(args.seed)

    def sample(mask, size=N):
        idxs = np.where(mask)[0]
        return sub_partner_msg[rng.choice(idxs, size=size, replace=True)]

    clean_m = clean.numpy()
    pzero_m = partial_zero.numpy()
    pone_m = partial_one.numpy()
    pnonw_m = partial_nonw.numpy()
    error_m = error.numpy()
    all_groups = {
        "clean": clean_m,
        "partial_one": pone_m,
        "partial_zero": pzero_m,
        "partial_nonw": pnonw_m,
        "error": error_m,
    }

    same_stratum = sub_partner_msg.copy()
    for mask in all_groups.values():
        idxs = np.where(mask)[0]
        if len(idxs) > 1:
            same_stratum[idxs] = sub_partner_msg[rng.permutation(idxs)]

    clean_mean = np.full(N, sub_partner_msg[clean_m].mean())
    clean_p75 = np.full(N, np.quantile(sub_partner_msg[clean_m], 0.75))
    clean_rand = sample(clean_m)
    pzero_rand = sample(pzero_m)
    pone_rand = sample(pone_m)
    error_rand = sample(error_m)
    zero = np.zeros(N)
    pysr_simple = (pair_relpt * 2.611763) - 0.32256523

    conditions = [
        ("sub_clean_random", "sub", clean_rand, sub_partner_msg),
        ("sub_clean_mean", "sub", clean_mean, sub_partner_msg),
        ("sub_clean_p75", "sub", clean_p75, sub_partner_msg),
        ("sub_pzero_random", "sub", pzero_rand, sub_partner_msg),
        ("sub_pone_random", "sub", pone_rand, sub_partner_msg),
        ("sub_error_random", "sub", error_rand, sub_partner_msg),
        ("sub_same_stratum_shuffle", "sub", same_stratum, sub_partner_msg),
        ("sub_zero", "sub", zero, sub_partner_msg),
        ("sub_pysr_simple", "sub", pysr_simple, sub_partner_msg),
        ("lead_clean_submsg_random", "lead", clean_rand, lead_partner_msg),
    ]

    records = []
    groups = {
        "all": np.ones(N, dtype=bool),
        "clean": clean_m,
        "partial": partial.numpy(),
        "partial_one": pone_m,
        "partial_zero": pzero_m,
        "partial_nonw": pnonw_m,
        "error": error_m,
    }

    def add_records(cond_name, qrole, target_msg, post):
        for gname, mask in groups.items():
            if mask.sum() < 20:
                continue
            rec = {
                "condition": cond_name,
                "qrole": qrole,
                "group": gname,
                "n": int(mask.sum()),
                "target_msg_mean": float(np.nanmean(target_msg[mask])),
                "delta_msg_mean": float(np.nanmean(target_msg[mask] - (sub_partner_msg if qrole == "sub" else lead_partner_msg)[mask])),
                "delta_channel_correct": float(post["channel_correct"][mask].mean() - base["channel_correct"][mask].mean()),
                "delta_clean": float(post["clean"][mask].mean() - base["clean"][mask].mean()),
                "delta_any_wsj": float(post["any_wsj"][mask].mean() - base["any_wsj"][mask].mean()),
                "delta_both_wsj": float(post["both_wsj"][mask].mean() - base["both_wsj"][mask].mean()),
                "delta_lead_w": float(post["lead_w"][mask].mean() - base["lead_w"][mask].mean()),
                "delta_sub_w": float(post["sub_w"][mask].mean() - base["sub_w"][mask].mean()),
                "delta_nonw": float(post["nonw_claim"][mask].mean() - base["nonw_claim"][mask].mean()),
                "med_delta_lead_margin": float(np.median(post["lead_margin"][mask] - base["lead_margin"][mask])),
                "med_delta_sub_margin": float(np.median(post["sub_margin"][mask] - base["sub_margin"][mask])),
                "med_delta_lep_margin": float(np.median(post["lep_margin"][mask] - base["lep_margin"][mask])),
                "med_delta_wsj_sum_margin": float(np.median(post["wsj_sum_margin"][mask] - base["wsj_sum_margin"][mask])),
            }
            records.append(rec)

    for cond_name, qrole, target_msg, current_msg in conditions:
        out = fwd_patch(qrole=qrole, target_msg=target_msg, current_msg=current_msg)
        post = readouts(out)
        add_records(cond_name, qrole, target_msg, post)

    print(f"\n=== Partner Message Replacement (surgical max dev {max_dev:.2e}) ===")
    print(
        f"{'condition':24s} {'group':13s} {'n':>5s} {'dMsg':>7s} "
        f"{'dChan':>7s} {'dClean':>7s} {'dAny':>7s} {'dBoth':>7s} "
        f"{'dLead':>7s} {'dSub':>7s} {'dNonW':>7s} {'med dSubM':>10s} {'med dSumM':>10s}"
    )
    key_groups = ("clean", "partial_zero", "partial_one", "partial_nonw", "error")
    for r in records:
        if r["group"] not in key_groups:
            continue
        print(
            f"{r['condition']:24s} {r['group']:13s} {r['n']:5d} "
            f"{fmt(r['delta_msg_mean']):>7s} {fmt(r['delta_channel_correct']):>7s} "
            f"{fmt(r['delta_clean']):>7s} {fmt(r['delta_any_wsj']):>7s} "
            f"{fmt(r['delta_both_wsj']):>7s} {fmt(r['delta_lead_w']):>7s} "
            f"{fmt(r['delta_sub_w']):>7s} {fmt(r['delta_nonw']):>7s} "
            f"{fmt(r['med_delta_sub_margin']):>10s} {fmt(r['med_delta_wsj_sum_margin']):>10s}"
        )

    out_dir = os.path.join(REPO, "tmp_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resolved_r9_partner_message_patch_{args.model}_skip{args.skip}_b{args.batches}.csv")
    header = list(records[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nsaved R9 records: {out_path}")

    msg_path = os.path.join(out_dir, f"resolved_r9_partner_messages_{args.model}_skip{args.skip}_b{args.batches}.csv")
    with open(msg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "clean",
                "partial_zero",
                "partial_one",
                "partial_nonw",
                "error",
                "sub_partner_msg",
                "lead_partner_msg",
                "sub_total_scalar",
                "lead_total_scalar",
                "pair_relpt",
                "pair_relsumpt",
                "mjj",
            ]
        )
        for i in range(N):
            writer.writerow(
                [
                    i,
                    int(clean_m[i]),
                    int(pzero_m[i]),
                    int(pone_m[i]),
                    int(pnonw_m[i]),
                    int(error_m[i]),
                    sub_partner_msg[i],
                    lead_partner_msg[i],
                    sub_total_scalar[i],
                    lead_total_scalar[i],
                    pair_relpt[i],
                    pair_relsumpt[i],
                    mjj[i],
                ]
            )
    print(f"saved R9 message table: {msg_path}")

    for h in surgical_hooks:
        h.remove()


if __name__ == "__main__":
    main()

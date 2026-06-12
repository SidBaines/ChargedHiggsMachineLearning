"""Resolved qqbb W circuit, R8: what information is passed on the sjet paths?

R7 showed that clean W-sjet labels causally depend on direct reads from lep/nu,
large jets, partner W-sjet, and other sjets. This script asks what those reads
carry, at the scalar bottleneck level.

For every resolved event it decomposes each attention head/query into:
  att_<query>_<category>_bXhY : attention mass from query to key category
  msg_<query>_<category>_bXhY : scalar contribution into that head/query bottleneck

The message is:
  sum_key attention(query,key) * down_h(Wout_h V_h(key))

This is still observational. It identifies information present in direct messages,
not by itself whether the downstream computation uses every correlated feature.

Usage:
  .venv/bin/python experiments/h1/resolved_r8_message_content.py \
      --model thesis-ent1-bn1-d152 --skip 24 --batches 12

Optional exploratory symbolic regression, after inspecting the named-feature probes:
  .venv/bin/python experiments/h1/resolved_r8_message_content.py \
      --run-pysr --pysr-target msg_both_lepnu_b0h0
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

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


def mass(p4):
    return torch.sqrt(torch.clamp(p4[..., 3] ** 2 - (p4[..., :3] ** 2).sum(-1), min=0.0))


def eta(p4):
    p = torch.sqrt(torch.clamp((p4[..., :3] ** 2).sum(-1), min=0.0))
    return 0.5 * torch.log((p + p4[..., 2]).clamp(min=1e-6) / (p - p4[..., 2]).clamp(min=1e-6))


def dphi(phi1, phi2):
    return torch.atan2(torch.sin(phi1 - phi2), torch.cos(phi1 - phi2))


def pearson_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    a = a[ok] - a[ok].mean()
    b = b[ok] - b[ok].mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-12:
        return np.nan
    return float((a * b).sum() / den)


def fmt(x):
    if not np.isfinite(x):
        return "nan"
    return f"{x:+.3f}"


def cv_r2(X, y):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 50 or np.nanstd(y[ok]) < 1e-9:
        return np.nan, np.nan
    n_splits = min(5, max(2, ok.sum() // 25))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    scores = cross_val_score(mdl, np.asarray(X)[ok], y[ok], cv=cv, scoring="r2")
    return float(scores.mean()), float(scores.std())


def cv_auc(X, y):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y).astype(int)
    counts = np.bincount(y)
    if len(counts) < 2 or counts.min() < 10:
        return np.nan, np.nan
    n_splits = min(5, counts.min())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    mdl = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    scores = cross_val_score(mdl, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def nanmean(x, mask):
    vals = np.asarray(x)[mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    return float(vals.mean())


def onehot_like(types, pos, valid=None):
    out = torch.zeros_like(types, dtype=torch.bool)
    ar = torch.arange(len(types))
    if valid is None:
        out[ar, pos] = True
    elif valid.any():
        out[ar[valid], pos[valid]] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="thesis-ent1-bn1-d152")
    ap.add_argument("--skip", type=int, default=24)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--ckpt-override", default=None)
    ap.add_argument("--run-pysr", action="store_true")
    ap.add_argument("--pysr-target", default="msg_both_lepnu_b0h0")
    ap.add_argument("--pysr-iterations", type=int, default=40)
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

    attn_modules = [blk["self_attention"] for blk in model.attention_blocks]
    nblk = len(attn_modules)
    n_heads = attn_modules[0].num_heads
    d_model = attn_modules[0].embed_dim
    d_head = d_model // n_heads

    outputs = []
    cols = defaultdict(list)
    max_scalar_recon_err = 0.0

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

    def add_col(name, vals):
        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
        cols[name].append(np.asarray(vals, dtype=np.float64))

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

            wsj = (tt == SJ) & (tru == 2)
            h_ljet = (tt == LJ) & (tru == 1)
            all_ljets = tt == LJ
            all_sjets = tt == SJ
            other_sjets = (tt == SJ) & ~wsj
            lepton = (tt == 0) | (tt == 1)
            nu = tt == NU
            lepnu = lepton | nu
            all_jets = (tt == LJ) | (tt == SJ)
            nonw_jets = all_jets & ~wsj

            role_pos = {
                "lead": lead[sl],
                "sub": sub[sl],
                "other1": other1[sl],
            }
            role_valid = {
                "lead": torch.ones(bsz, dtype=torch.bool),
                "sub": torch.ones(bsz, dtype=torch.bool),
                "other1": other1_valid[sl],
            }
            role_masks = {
                "lead": {
                    "self": onehot_like(tt, lead[sl]),
                    "partner": onehot_like(tt, sub[sl]),
                },
                "sub": {
                    "self": onehot_like(tt, sub[sl]),
                    "partner": onehot_like(tt, lead[sl]),
                },
                "other1": {
                    "self": onehot_like(tt, other1[sl], other1_valid[sl]),
                    "partner": wsj,
                },
            }
            base_masks = {
                "lepnu": lepnu,
                "lepton": lepton,
                "nu": nu,
                "H_ljet": h_ljet,
                "all_ljets": all_ljets,
                "all_sjets": all_sjets,
                "W_sjets": wsj,
                "other_sjets": other_sjets,
                "all_jets": all_jets,
                "nonW_jets": nonw_jets,
            }

            for blk in range(nblk):
                aw_all = cache[f"block_{blk}_attention"]["attn_weights_per_head"]
                bna = cache[f"block_{blk}_attention"]["bottleneck_activation"]
                for h in range(n_heads):
                    sv = scalar_value_by_key(blk, h)
                    aw = aw_all[:, h]
                    for role in ("lead", "sub", "other1"):
                        qpos = role_pos[role]
                        valid = role_valid[role]
                        row = aw[bidx, qpos, :]
                        total = (row * sv).sum(1)
                        ref = bna[bidx, h, qpos, 0]
                        if valid.any():
                            err = (total[valid] - ref[valid]).abs().max().item()
                            max_scalar_recon_err = max(max_scalar_recon_err, err)
                        masks = {**base_masks, **role_masks[role]}
                        for cname, cmask in masks.items():
                            att = (row * cmask).sum(1)
                            msg = (row * sv * cmask).sum(1)
                            if not valid.all():
                                att = att.clone()
                                msg = msg.clone()
                                att[~valid] = float("nan")
                                msg[~valid] = float("nan")
                            add_col(f"att_{role}_{cname}_b{blk}h{h}", att)
                            add_col(f"msg_{role}_{cname}_b{blk}h{h}", msg)

    out = torch.cat(outputs)
    for h in handles:
        h.remove()

    D = {k: np.concatenate(v) for k, v in cols.items()}
    for cname in (
        "lepnu",
        "lepton",
        "nu",
        "H_ljet",
        "all_ljets",
        "all_sjets",
        "W_sjets",
        "other_sjets",
        "all_jets",
        "nonW_jets",
        "partner",
    ):
        for blk in range(nblk):
            for h in range(n_heads):
                lk = f"msg_lead_{cname}_b{blk}h{h}"
                sk = f"msg_sub_{cname}_b{blk}h{h}"
                if lk in D and sk in D:
                    D[f"msg_both_{cname}_b{blk}h{h}"] = D[lk] + D[sk]
                    D[f"attmean_both_{cname}_b{blk}h{h}"] = 0.5 * (
                        D[f"att_lead_{cname}_b{blk}h{h}"] + D[f"att_sub_{cname}_b{blk}h{h}"]
                    )

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
    partial = channel_correct & (~clean)
    partial_zero = channel_correct & (wsj_claims == 0)
    partial_one = channel_correct & (wsj_claims == 1)
    partial_nonw = channel_correct & nonw_claim

    p4 = X[..., :4]
    pt_all = pt(p4) * 100.0
    m_all = mass(p4) * 100.0
    phi_all = torch.atan2(p4[..., 1], p4[..., 0])
    eta_all = eta(p4)
    ht_jets = (pt_all * jet_mask.float()).sum(1)
    pair_p4 = p4[ar, lead] + p4[ar, sub]
    pair_pt = pt(pair_p4) * 100.0
    pair_sumpt = pt_all[ar, lead] + pt_all[ar, sub]
    pair_m = mass(pair_p4) * 100.0
    abs_dphi = torch.abs(dphi(phi_all[ar, lead], phi_all[ar, sub]))
    abs_deta = torch.abs(eta_all[ar, lead] - eta_all[ar, sub])
    dr = torch.sqrt(abs_dphi**2 + abs_deta**2)
    lepnu_p4 = p4[ar, lpos] + p4[ar, npos]
    lepnu_pt = pt(lepnu_p4) * 100.0
    lepnu_mass = mass(lepnu_p4) * 100.0
    lepnu_sumpt = pt_all[ar, lpos] + pt_all[ar, npos]

    h_ljet_pt = np.full(N, np.nan)
    h_ljet_m = np.full(N, np.nan)
    h_ljet_tag = np.full(N, np.nan)
    h_ljet_margin = np.full(N, np.nan)
    ljet_pt_max = np.full(N, np.nan)
    ljet_pt_sum = np.zeros(N)
    other1_margin = np.full(N, np.nan)
    other1_w = np.full(N, np.nan)
    best_other_margin = np.full(N, np.nan)
    for i in range(N):
        hrows = torch.where((T[i] == LJ) & (TRU[i] == 1))[0]
        if len(hrows):
            r = hrows[0].item()
            h_ljet_pt[i] = float(pt_all[i, r])
            h_ljet_m[i] = float(m_all[i, r])
            h_ljet_tag[i] = float(X[i, r, 4])
            h_ljet_margin[i] = float(margin[i, r])
        lrows = torch.where(T[i] == LJ)[0]
        if len(lrows):
            lpts = pt_all[i, lrows]
            ljet_pt_max[i] = float(lpts.max())
            ljet_pt_sum[i] = float(lpts.sum())
        if other1_valid[i]:
            r = other1[i].item()
            other1_margin[i] = float(margin[i, r])
            other1_w[i] = float(pred[i, r] == W_CLASS)
        oj = nonw_mask[i]
        if oj.any():
            best_other_margin[i] = float(margin[i].masked_fill(~oj, -1e9).max())

    F = {
        "event_idx": np.arange(N, dtype=np.float64),
        "clean": clean.numpy().astype(float),
        "channel_correct": channel_correct.numpy().astype(float),
        "partial": partial.numpy().astype(float),
        "partial_zero": partial_zero.numpy().astype(float),
        "partial_one": partial_one.numpy().astype(float),
        "partial_nonw": partial_nonw.numpy().astype(float),
        "channel_error": (~channel_correct).numpy().astype(float),
        "both_wsj": (wsj_claims == 2).numpy().astype(float),
        "any_wsj": (wsj_claims >= 1).numpy().astype(float),
        "lead_w": (pred[ar, lead] == W_CLASS).numpy().astype(float),
        "sub_w": (pred[ar, sub] == W_CLASS).numpy().astype(float),
        "nonw_claim": nonw_claim.numpy().astype(float),
        "other1_w": other1_w,
        "lead_margin": margin[ar, lead].numpy(),
        "sub_margin": margin[ar, sub].numpy(),
        "wsj_sum_margin": (margin[ar, lead] + margin[ar, sub]).numpy(),
        "wsj_min_margin": torch.minimum(margin[ar, lead], margin[ar, sub]).numpy(),
        "lep_margin": margin[ar, lpos].numpy(),
        "nu_margin": margin[ar, npos].numpy(),
        "other1_margin": other1_margin,
        "best_other_margin": best_other_margin,
        "mjj": pair_m.numpy(),
        "pair_pt": pair_pt.numpy(),
        "pair_sumpt": pair_sumpt.numpy(),
        "pair_relpt": (pair_pt / ht_jets.clamp(min=1e-6)).numpy(),
        "pair_relsumpt": (pair_sumpt / ht_jets.clamp(min=1e-6)).numpy(),
        "dr": dr.numpy(),
        "abs_dphi": abs_dphi.numpy(),
        "abs_deta": abs_deta.numpy(),
        "lead_pt": pt_all[ar, lead].numpy(),
        "sub_pt": pt_all[ar, sub].numpy(),
        "pt_min": torch.minimum(pt_all[ar, lead], pt_all[ar, sub]).numpy(),
        "pt_max": torch.maximum(pt_all[ar, lead], pt_all[ar, sub]).numpy(),
        "pt_ratio": (torch.minimum(pt_all[ar, lead], pt_all[ar, sub]) / torch.maximum(pt_all[ar, lead], pt_all[ar, sub]).clamp(min=1e-6)).numpy(),
        "lead_m": m_all[ar, lead].numpy(),
        "sub_m": m_all[ar, sub].numpy(),
        "m_min": torch.minimum(m_all[ar, lead], m_all[ar, sub]).numpy(),
        "m_max": torch.maximum(m_all[ar, lead], m_all[ar, sub]).numpy(),
        "lead_tag": X[ar, lead, 4].numpy(),
        "sub_tag": X[ar, sub, 4].numpy(),
        "tag_min": torch.minimum(X[ar, lead, 4], X[ar, sub, 4]).numpy(),
        "tag_max": torch.maximum(X[ar, lead, 4], X[ar, sub, 4]).numpy(),
        "tag_mean": (0.5 * (X[ar, lead, 4] + X[ar, sub, 4])).numpy(),
        "ht_jets": ht_jets.numpy(),
        "n_sjets": (T == SJ).sum(1).numpy().astype(float),
        "n_ljets": (T == LJ).sum(1).numpy().astype(float),
        "lep_pt": pt_all[ar, lpos].numpy(),
        "nu_pt": pt_all[ar, npos].numpy(),
        "lepnu_pt": lepnu_pt.numpy(),
        "lepnu_sumpt": lepnu_sumpt.numpy(),
        "lepnu_relpt": (lepnu_pt / ht_jets.clamp(min=1e-6)).numpy(),
        "lepnu_mass": lepnu_mass.numpy(),
        "lepton_is_e": (T[ar, lpos] == 0).numpy().astype(float),
        "h_ljet_pt": h_ljet_pt,
        "h_ljet_m": h_ljet_m,
        "h_ljet_tag": h_ljet_tag,
        "h_ljet_margin": h_ljet_margin,
        "ljet_pt_max": ljet_pt_max,
        "ljet_pt_sum": ljet_pt_sum,
    }

    all_data = {**F, **D}
    feature_names = [
        "mjj",
        "pair_pt",
        "pair_sumpt",
        "pair_relpt",
        "pair_relsumpt",
        "dr",
        "abs_dphi",
        "abs_deta",
        "lead_pt",
        "sub_pt",
        "pt_min",
        "pt_max",
        "pt_ratio",
        "lead_m",
        "sub_m",
        "m_min",
        "m_max",
        "lead_tag",
        "sub_tag",
        "tag_min",
        "tag_max",
        "tag_mean",
        "ht_jets",
        "n_sjets",
        "n_ljets",
        "lep_pt",
        "nu_pt",
        "lepnu_pt",
        "lepnu_sumpt",
        "lepnu_relpt",
        "lepnu_mass",
        "lepton_is_e",
        "h_ljet_pt",
        "h_ljet_m",
        "h_ljet_tag",
        "ljet_pt_max",
        "ljet_pt_sum",
    ]
    outcome_names = [
        "clean",
        "channel_correct",
        "both_wsj",
        "any_wsj",
        "lead_w",
        "sub_w",
        "nonw_claim",
        "lead_margin",
        "sub_margin",
        "wsj_sum_margin",
        "lep_margin",
        "best_other_margin",
    ]
    feat_groups = {
        "pair_geom": ["mjj", "pair_pt", "pair_sumpt", "pair_relpt", "pair_relsumpt", "dr", "abs_dphi", "abs_deta"],
        "pair_members": ["lead_pt", "sub_pt", "pt_min", "pt_max", "pt_ratio", "lead_m", "sub_m", "lead_tag", "sub_tag", "tag_mean"],
        "lepnu": ["lep_pt", "nu_pt", "lepnu_pt", "lepnu_sumpt", "lepnu_relpt", "lepnu_mass", "lepton_is_e"],
        "largejet": ["n_ljets", "h_ljet_pt", "h_ljet_m", "h_ljet_tag", "ljet_pt_max", "ljet_pt_sum"],
        "global": ["ht_jets", "n_sjets", "n_ljets"],
        "all_phys": feature_names,
    }

    groups = {
        "all": np.ones(N, dtype=bool),
        "clean": clean.numpy(),
        "partial_one": partial_one.numpy(),
        "partial_zero": partial_zero.numpy(),
        "partial_nonw": partial_nonw.numpy(),
        "error": (~channel_correct).numpy(),
    }

    print(f"model={args.model}  skip={args.skip}  batches={args.batches}")
    print(
        f"events={len(X_all)}  boosted={boosted.sum().item()}  resolved={N}; "
        f"clean={clean.sum().item()} partial={partial.sum().item()} error={(~channel_correct).sum().item()} "
        f"partial_one={partial_one.sum().item()} partial_zero={partial_zero.sum().item()} "
        f"partial_nonw={partial_nonw.sum().item()} other1_valid={other1_valid.sum().item()}"
    )
    print(f"scalar decomposition max |sum_key msg - cached bottleneck| = {max_scalar_recon_err:.3e}")

    focus = [
        ("both lepnu b0h0", "msg_both_lepnu_b0h0"),
        ("both lepnu b1h3", "msg_both_lepnu_b1h3"),
        ("both lepnu b2h2", "msg_both_lepnu_b2h2"),
        ("lead lepnu b0h0", "msg_lead_lepnu_b0h0"),
        ("sub lepnu b0h0", "msg_sub_lepnu_b0h0"),
        ("both H_ljet b0h1", "msg_both_H_ljet_b0h1"),
        ("both H_ljet b1h1", "msg_both_H_ljet_b1h1"),
        ("both all_ljets b0h1", "msg_both_all_ljets_b0h1"),
        ("both all_ljets b1h1", "msg_both_all_ljets_b1h1"),
        ("sub partner b2h3", "msg_sub_partner_b2h3"),
        ("lead partner b2h3", "msg_lead_partner_b2h3"),
        ("other1 W_sjets b0h3", "msg_other1_W_sjets_b0h3"),
        ("other1 W_sjets b1h3", "msg_other1_W_sjets_b1h3"),
        ("other1 W_sjets b2h3", "msg_other1_W_sjets_b2h3"),
    ]
    focus = [(label, col) for label, col in focus if col in all_data]

    print("\n=== Focused Path Message Means ===")
    print(f"{'path':28s} {'all':>9s} {'clean':>9s} {'part1':>9s} {'part0':>9s} {'pNonW':>9s} {'error':>9s}")
    for label, col in focus:
        vals = all_data[col]
        print(
            f"{label:28s} {fmt(nanmean(vals, groups['all'])):>9s} "
            f"{fmt(nanmean(vals, groups['clean'])):>9s} {fmt(nanmean(vals, groups['partial_one'])):>9s} "
            f"{fmt(nanmean(vals, groups['partial_zero'])):>9s} {fmt(nanmean(vals, groups['partial_nonw'])):>9s} "
            f"{fmt(nanmean(vals, groups['error'])):>9s}"
        )

    print("\n=== Largest Clean-vs-Error Message Contrasts ===")
    msg_cols = [k for k in D if k.startswith("msg_")]
    contrasts = []
    for col in msg_cols:
        mc = nanmean(all_data[col], groups["clean"])
        me = nanmean(all_data[col], groups["error"])
        if np.isfinite(mc) and np.isfinite(me):
            contrasts.append((abs(mc - me), mc - me, col, mc, me))
    for _, delta, col, mc, me in sorted(contrasts, reverse=True)[:30]:
        print(f"  {col:34s} clean-error={delta:+.3f}  clean={mc:+.3f}  error={me:+.3f}")

    print("\n=== Top Physics Correlates Of Focused Messages ===")
    for label, col in focus:
        vals = all_data[col]
        scores = [(name, pearson_np(vals, all_data[name])) for name in feature_names]
        scores = sorted(scores, key=lambda x: -abs(x[1] if np.isfinite(x[1]) else 0.0))
        desc = "  ".join(f"{name}:{r:+.2f}" for name, r in scores[:6])
        print(f"  {label:28s} {desc}")

    print("\n=== Top Outcome Correlates Of Focused Messages ===")
    for label, col in focus:
        vals = all_data[col]
        scores = [(name, pearson_np(vals, all_data[name])) for name in outcome_names]
        scores = sorted(scores, key=lambda x: -abs(x[1] if np.isfinite(x[1]) else 0.0))
        desc = "  ".join(f"{name}:{r:+.2f}" for name, r in scores[:6])
        print(f"  {label:28s} {desc}")

    print("\n=== Named-Feature R2 For Focused Messages ===")
    print("Observational: high R2 means the message is decodable from that feature set, not that the model explicitly computes it.")
    X_groups = {
        g: np.stack([all_data[n] for n in names], axis=1)
        for g, names in feat_groups.items()
    }
    for label, col in focus:
        vals = all_data[col]
        rows = []
        for gname, Xg in X_groups.items():
            m, s = cv_r2(Xg, vals)
            rows.append((m if np.isfinite(m) else -np.inf, gname, m, s))
        rows = sorted(rows, reverse=True)
        best = rows[0]
        no_all = next(r for r in rows if r[1] != "all_phys")
        print(
            f"  {label:28s} best={best[1]} R2={best[2]:+.3f}+/-{best[3]:.3f}; "
            f"best_nonall={no_all[1]} R2={no_all[2]:+.3f}+/-{no_all[3]:.3f}"
        )

    print("\n=== Message-Only Outcome Probes ===")
    msg_groups = {
        "lepnu_support": [
            "msg_lead_lepnu_b0h0",
            "msg_sub_lepnu_b0h0",
            "msg_lead_lepnu_b1h3",
            "msg_sub_lepnu_b1h3",
            "msg_lead_lepnu_b2h2",
            "msg_sub_lepnu_b2h2",
        ],
        "largejet_support": [
            "msg_lead_H_ljet_b0h1",
            "msg_sub_H_ljet_b0h1",
            "msg_lead_all_ljets_b0h1",
            "msg_sub_all_ljets_b0h1",
            "msg_lead_H_ljet_b1h1",
            "msg_sub_H_ljet_b1h1",
            "msg_lead_all_ljets_b1h1",
            "msg_sub_all_ljets_b1h1",
        ],
        "partner_support": ["msg_lead_partner_b2h3", "msg_sub_partner_b2h3"],
        "focused_all": [col for _, col in focus],
    }
    for target in ("clean", "channel_correct", "both_wsj", "lead_w", "sub_w", "nonw_claim"):
        print(f"  target {target}:")
        for gname, names in msg_groups.items():
            names = [n for n in names if n in all_data]
            Xg = np.stack([all_data[n] for n in names], axis=1)
            m, s = cv_auc(Xg, all_data[target])
            print(f"    {gname:18s} AUC={m:.3f}+/-{s:.3f}")

    out_dir = os.path.join(REPO, "tmp_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resolved_r8_message_content_{args.model}_skip{args.skip}_b{args.batches}.csv")
    header = list(F.keys()) + sorted(D.keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(N):
            writer.writerow([all_data[k][i] for k in header])
    print(f"\nsaved R8 event/message table: {out_path}")

    if args.run_pysr:
        target = args.pysr_target
        if target not in all_data:
            print(f"\nPySR skipped: target column not found: {target}")
            return
        try:
            from pysr import PySRRegressor
        except Exception as exc:
            print(f"\nPySR skipped: import failed ({type(exc).__name__}: {exc})")
            return
        y = all_data[target]
        ok = np.isfinite(y)
        scores = [(name, abs(pearson_np(y, all_data[name]))) for name in feature_names]
        top_features = [name for name, _ in sorted(scores, key=lambda x: -x[1])[:8]]
        Xp = np.stack([all_data[n] for n in top_features], axis=1)
        ok &= np.isfinite(Xp).any(1)
        reg = PySRRegressor(
            niterations=args.pysr_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "log", "abs"],
            model_selection="best",
            maxsize=20,
            output_directory=os.path.join(REPO, "tmp_pysr", "resolved_r8_message_content"),
            verbosity=0,
        )
        print(f"\n=== PySR exploratory fit for {target} on {top_features} ===")
        reg.fit(Xp[ok], y[ok], variable_names=top_features)
        eq = reg.equations_
        cols = [c for c in ("complexity", "loss", "score", "equation") if c in eq.columns]
        print(eq[cols].to_string(index=False))


if __name__ == "__main__":
    main()

"""Resolved qqbb W circuit, R1: fresh-slice baseline + observational probes.

Questions:
  - In resolved qqbb, do the two truth-W sjets claim W together?
  - Do event/verdict margins track pairwise features such as m(jj), or mostly
    individual-sjet/context features?
  - Which attention/value paths visibly carry W-sjet-pair evidence?

This is observational only. It can suggest experiments, but cannot by itself prove
that the model computes a feature. Causal pair-mass surgery and real-event swaps
should follow any interesting signal found here.

Usage:
  .venv/bin/python experiments/h1/resolved_r1_baseline_probes.py \
      --model thesis-ent1-bn1-d152 --skip 24 --batches 12
"""
import argparse
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
WIRES = [(1, 3), (2, 2), (2, 0), (2, 3)]


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
    if ok.sum() < 3:
        return np.nan
    a = a[ok] - a[ok].mean()
    b = b[ok] - b[ok].mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cv_auc(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return np.nan, np.nan
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    mdl = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    scores = cross_val_score(mdl, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def cv_r2(X, y):
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, dtype=np.float64)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    scores = cross_val_score(mdl, X, y, cv=cv, scoring="r2")
    return float(scores.mean()), float(scores.std())


def fmt(mean, std):
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.3f} +/- {std:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="thesis-ent1-bn1-d152")
    ap.add_argument("--skip", type=int, default=24)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--ckpt-override", default=None)
    ap.add_argument("--batch-size", type=int, default=2048)
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
        types = b["types"]
        ok = ((types == NU).sum(1) == 1) & (((types == 0) | (types == 1)).sum(1) == 1)
        xs.append(b["x"][ok].clone())
        ts.append(types[ok].clone())
    X = torch.cat(xs)
    T = torch.cat(ts)
    N = len(X)
    TRU = torch.round(X[..., -1]).long()
    ar = torch.arange(N)
    npos = (T == NU).float().argmax(1)
    lpos = ((T == 0) | (T == 1)).float().argmax(1)
    lvbb = TRU[ar, lpos] == 3
    boosted = (~lvbb) & (((T == LJ) & (TRU == 2)).sum(1) == 1)
    resolved = (~lvbb) & (((T == SJ) & (TRU == 2)).sum(1) == 2)
    jet_mask = (T == LJ) | (T == SJ)
    res_idx = torch.where(resolved)[0]

    attn_modules = [blk["self_attention"] for blk in model.attention_blocks]
    d_model = attn_modules[0].embed_dim
    n_heads = attn_modules[0].num_heads
    d_head = d_model // n_heads

    outputs = []
    wire_scalars = defaultdict(list)
    nu_attn_cat = defaultdict(list)
    nu_value_cat = defaultdict(list)
    wsj_query_attn = defaultdict(list)

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
            lp_ = lpos[sl]
            res_local = resolved[sl]
            wsj = (tt == SJ) & (tru == 2)
            hlj = (tt == LJ) & (tru == 1)
            other_sj = (tt == SJ) & ~wsj
            other_lj = (tt == LJ) & ~hlj
            lep = (tt == 0) | (tt == 1)
            nu = tt == NU

            for key in WIRES:
                blk, h = key
                bna = cache[f"block_{blk}_attention"]["bottleneck_activation"]
                wire_scalars[key].append(bna[bidx, h, np_, 0])

                aw = cache[f"block_{blk}_attention"]["attn_weights_per_head"][:, h]
                a_nu = aw[bidx, np_]
                sv = scalar_value_by_key(blk, h)
                per_key = a_nu * sv
                cats = {
                    "W_sjets": wsj,
                    "H_ljet": hlj,
                    "other_sjets": other_sj,
                    "other_ljets": other_lj,
                    "lepton": lep,
                    "nu_self": nu,
                }
                for cname, cmask in cats.items():
                    nu_attn_cat[(key, cname)].append((a_nu * cmask).sum(1))
                    nu_value_cat[(key, cname)].append((per_key * cmask).sum(1))

            # Attention among the two truth-W sjets. This is forensic only, not causal.
            local_res_rows = torch.where(res_local)[0]
            for blk in range(len(attn_modules)):
                aw_all = cache[f"block_{blk}_attention"]["attn_weights_per_head"]
                for h in range(n_heads):
                    vals = []
                    aw = aw_all[:, h]
                    for ii in local_res_rows:
                        rows = torch.where(wsj[ii])[0]
                        if len(rows) != 2:
                            continue
                        r0, r1 = rows[0], rows[1]
                        q0 = aw[ii, r0]
                        q1 = aw[ii, r1]
                        pair_mask = torch.zeros_like(q0, dtype=torch.bool)
                        pair_mask[rows] = True
                        other_jets = ((tt[ii] == LJ) | (tt[ii] == SJ)) & ~pair_mask
                        lepnu = (tt[ii] == NU) | (tt[ii] == 0) | (tt[ii] == 1)
                        vals.append(
                            torch.stack(
                                [
                                    0.5 * (q0[r1] + q1[r0]),
                                    0.5 * (q0[r0] + q1[r1]),
                                    0.5 * (q0[other_jets].sum() + q1[other_jets].sum()),
                                    0.5 * (q0[lepnu].sum() + q1[lepnu].sum()),
                                ]
                            )
                        )
                    if vals:
                        wsj_query_attn[(blk, h)].append(torch.stack(vals))

    out = torch.cat(outputs)
    pred = out.argmax(-1)
    margin = out[..., W_CLASS] - out[..., NONE_CLASS]
    lep_margin = margin[ar, lpos]
    nu_margin = margin[ar, npos]
    lep_w = pred[ar, lpos] == W_CLASS
    nu_w = pred[ar, npos] == W_CLASS
    jet_claims = (pred == W_CLASS) & jet_mask
    n_jet_claims = jet_claims.sum(1)
    n_wsj_claims = (jet_claims & (T == SJ) & (TRU == 2)).sum(1)
    nonw_claims = jet_claims & ~((T == SJ) & (TRU == 2)) & resolved[:, None]

    p4 = X[..., :4]
    pt_all = pt(p4) * 100.0
    mass_all = mass(p4) * 100.0
    phi_all = torch.atan2(p4[..., 1], p4[..., 0])
    eta_all = eta(p4)
    ht_jets = (pt_all * jet_mask.float()).sum(1)

    rows = []
    for i in res_idx:
        wrows = torch.where((T[i] == SJ) & (TRU[i] == 2))[0]
        if len(wrows) != 2:
            continue
        r0, r1 = wrows[0].item(), wrows[1].item()
        p4_pair = p4[i, [r0, r1]].sum(0)
        pair_m = mass(p4_pair) * 100.0
        pair_pt = pt(p4_pair) * 100.0
        pair_sumpt = pt_all[i, [r0, r1]].sum()
        abs_dphi = torch.abs(dphi(phi_all[i, r0], phi_all[i, r1]))
        abs_deta = torch.abs(eta_all[i, r0] - eta_all[i, r1])
        dr = torch.sqrt(abs_dphi**2 + abs_deta**2)
        pts = pt_all[i, [r0, r1]]
        masses = mass_all[i, [r0, r1]]
        tags = X[i, [r0, r1], 4]
        w_margins = margin[i, [r0, r1]]
        hlj_rows = torch.where((T[i] == LJ) & (TRU[i] == 1))[0]
        if len(hlj_rows) > 0:
            hlj = hlj_rows[0].item()
            hlj_pt = pt_all[i, hlj]
            hlj_m = mass_all[i, hlj]
            hlj_tag = X[i, hlj, 4]
            hlj_margin = margin[i, hlj]
            hlj_claim = pred[i, hlj] == W_CLASS
        else:
            hlj_pt = hlj_m = hlj_tag = hlj_margin = torch.tensor(np.nan)
            hlj_claim = torch.tensor(False)
        other_jet_m = jet_mask[i].clone()
        other_jet_m[[r0, r1]] = False
        best_other_margin = margin[i].masked_fill(~other_jet_m, -1e9).max()
        rows.append(
            {
                "idx": int(i),
                "event_correct": float(not lep_w[i].item()),
                "lep_margin": float(lep_margin[i]),
                "nu_margin": float(nu_margin[i]),
                "n_jet_claims": int(n_jet_claims[i]),
                "n_wsj_claims": int(n_wsj_claims[i]),
                "any_nonw_claim": float(nonw_claims[i].any().item()),
                "h_ljet_claims_w": float(hlj_claim.item()),
                "wsj_sum_margin": float(w_margins.sum()),
                "wsj_min_margin": float(w_margins.min()),
                "wsj_max_margin": float(w_margins.max()),
                "mjj": float(pair_m),
                "pair_pt": float(pair_pt),
                "pair_sumpt": float(pair_sumpt),
                "pair_relpt": float(pair_pt / ht_jets[i].clamp(min=1e-6)),
                "pair_relsumpt": float(pair_sumpt / ht_jets[i].clamp(min=1e-6)),
                "dr": float(dr),
                "abs_dphi": float(abs_dphi),
                "abs_deta": float(abs_deta),
                "pt_min": float(pts.min()),
                "pt_max": float(pts.max()),
                "pt_ratio": float(pts.min() / pts.max().clamp(min=1e-6)),
                "m_min": float(masses.min()),
                "m_max": float(masses.max()),
                "tag_min": float(tags.min()),
                "tag_max": float(tags.max()),
                "tag_mean": float(tags.mean()),
                "ht_jets": float(ht_jets[i]),
                "n_sjets": int((T[i] == SJ).sum()),
                "n_ljets": int((T[i] == LJ).sum()),
                "h_ljet_pt": float(hlj_pt),
                "h_ljet_m": float(hlj_m),
                "h_ljet_tag": float(hlj_tag),
                "h_ljet_margin": float(hlj_margin),
                "best_other_margin": float(best_other_margin),
            }
        )

    feature_names = [
        "mjj",
        "pair_pt",
        "pair_sumpt",
        "pair_relpt",
        "pair_relsumpt",
        "dr",
        "abs_dphi",
        "abs_deta",
        "pt_min",
        "pt_max",
        "pt_ratio",
        "m_min",
        "m_max",
        "tag_min",
        "tag_max",
        "tag_mean",
        "ht_jets",
        "n_sjets",
        "n_ljets",
        "h_ljet_pt",
        "h_ljet_m",
        "h_ljet_tag",
    ]
    R = {k: np.array([row[k] for row in rows], dtype=np.float64) for k in rows[0]}
    feat_matrix = np.stack([R[k] for k in feature_names], axis=1)

    print(f"model={args.model}  skip={args.skip}  batches={args.batches}")
    print(
        f"events={N}  lvbb={lvbb.sum().item()}  boosted={boosted.sum().item()}  "
        f"resolved={resolved.sum().item()}  qq-neither={((~lvbb) & ~boosted & ~resolved).sum().item()}"
    )
    print("\n=== Resolved Baseline ===")
    print(f"resolved n={len(rows)}")
    print(f"  P(lep incorrectly W) = {lep_w[resolved].float().mean():.4f}")
    print(f"  P(nu=lep)            = {(nu_w[resolved] == lep_w[resolved]).float().mean():.4f}")
    print(f"  P(any jet claims W)  = {(n_jet_claims[resolved] > 0).float().mean():.4f}")
    print(f"  P(both W-sjets claim)= {(n_wsj_claims[resolved] == 2).float().mean():.4f}")
    print(f"  P(exactly 1 W-sjet)  = {(n_wsj_claims[resolved] == 1).float().mean():.4f}")
    print(f"  P(no W-sjet claims)  = {(n_wsj_claims[resolved] == 0).float().mean():.4f}")
    print(f"  P(non-W jet claims)  = {R['any_nonw_claim'].mean():.4f}")
    print(f"  P(H-ljet claims W)   = {R['h_ljet_claims_w'].mean():.4f}")
    for label, mask in [
        ("correct", R["event_correct"] == 1),
        ("error", R["event_correct"] == 0),
        ("both-wsj", R["n_wsj_claims"] == 2),
        ("one-wsj", R["n_wsj_claims"] == 1),
        ("zero-wsj", R["n_wsj_claims"] == 0),
    ]:
        if mask.sum() < 5:
            continue
        print(
            f"  {label:9s} n={mask.sum():4d} "
            f"mjj med {np.median(R['mjj'][mask]):7.1f} "
            f"pair_relpt med {np.median(R['pair_relpt'][mask]):.3f} "
            f"wsj_sum_margin med {np.median(R['wsj_sum_margin'][mask]):+.2f} "
            f"best_other_margin med {np.median(R['best_other_margin'][mask]):+.2f}"
        )

    print("\n=== Feature Correlations On Resolved Truth Pair ===")
    targets = {
        "wsj_sum_margin": R["wsj_sum_margin"],
        "wsj_min_margin": R["wsj_min_margin"],
        "hadronic_strength(-lep_margin)": -R["lep_margin"],
        "best_other_margin": R["best_other_margin"],
    }
    for tname, y in targets.items():
        scores = [(name, pearson_np(R[name], y)) for name in feature_names]
        scores = sorted(scores, key=lambda x: -abs(x[1] if np.isfinite(x[1]) else 0.0))
        print(f"  target {tname}:")
        for name, r in scores[:8]:
            print(f"    {name:14s} r={r:+.3f}")

    groups = {
        "mjj_only": ["mjj"],
        "pair_geom": ["mjj", "pair_pt", "pair_sumpt", "pair_relpt", "pair_relsumpt", "dr", "abs_dphi", "abs_deta"],
        "single_wsj": ["pt_min", "pt_max", "pt_ratio", "m_min", "m_max", "tag_min", "tag_max", "tag_mean"],
        "context": ["ht_jets", "n_sjets", "n_ljets", "h_ljet_pt", "h_ljet_m", "h_ljet_tag"],
        "single+context": [
            "pt_min",
            "pt_max",
            "pt_ratio",
            "m_min",
            "m_max",
            "tag_min",
            "tag_max",
            "tag_mean",
            "ht_jets",
            "n_sjets",
            "n_ljets",
            "h_ljet_pt",
            "h_ljet_m",
            "h_ljet_tag",
        ],
        "all_phys": feature_names,
    }
    print("\n=== Cross-Validated Predictive Probes (Resolved Only) ===")
    print("AUC targets are binary; R2 targets are continuous. Observational, not causal.")
    bin_targets = {
        "event_correct": R["event_correct"],
        "both_W_sjets_claim": (R["n_wsj_claims"] == 2).astype(float),
        "any_W_sjet_claim": (R["n_wsj_claims"] >= 1).astype(float),
    }
    cont_targets = {
        "wsj_sum_margin": R["wsj_sum_margin"],
        "wsj_min_margin": R["wsj_min_margin"],
        "hadronic_strength": -R["lep_margin"],
    }
    for yname, y in bin_targets.items():
        print(f"  binary target {yname}:")
        for gname, names in groups.items():
            cols = [feature_names.index(n) for n in names]
            mean, std = cv_auc(feat_matrix[:, cols], y)
            print(f"    {gname:15s} AUC {fmt(mean, std)}")
    for yname, y in cont_targets.items():
        print(f"  continuous target {yname}:")
        for gname, names in groups.items():
            cols = [feature_names.index(n) for n in names]
            mean, std = cv_r2(feat_matrix[:, cols], y)
            print(f"    {gname:15s} R2  {fmt(mean, std)}")

    print("\n=== Verdict Wire Forensics At Nu ===")
    for key in WIRES:
        vals = torch.cat(wire_scalars[key])[resolved].numpy()
        print(f"  b{key[0]}h{key[1]} scalar: mean {vals.mean():+.3f}, std {vals.std():.3f}")
        for cname in ["W_sjets", "H_ljet", "other_sjets", "other_ljets", "lepton", "nu_self"]:
            att = torch.cat(nu_attn_cat[(key, cname)])[resolved].numpy()
            val = torch.cat(nu_value_cat[(key, cname)])[resolved].numpy()
            print(f"    {cname:12s} attn {att.mean():.3f}  value-contrib {val.mean():+.3f}")

    print("\n=== Sjet-Sjet Attention Forensics ===")
    ss_rows = []
    nu_rows = []
    for blk in range(len(attn_modules)):
        for h in range(n_heads):
            vals = torch.cat(wsj_query_attn[(blk, h)]).numpy()
            ss_rows.append((blk, h, vals[:, 0].mean(), vals[:, 1].mean(), vals[:, 2].mean(), vals[:, 3].mean()))
            key = (blk, h)
            if ((key, "W_sjets") in nu_attn_cat):
                nu_w = torch.cat(nu_attn_cat[(key, "W_sjets")])[resolved].mean().item()
                nu_other = (
                    torch.cat(nu_attn_cat[(key, "H_ljet")])[resolved]
                    + torch.cat(nu_attn_cat[(key, "other_sjets")])[resolved]
                    + torch.cat(nu_attn_cat[(key, "other_ljets")])[resolved]
                ).mean().item()
                nu_rows.append((blk, h, nu_w, nu_other))
    print("  top W-sjet query -> partner heads:")
    for blk, h, partner, self_a, other, lepnu in sorted(ss_rows, key=lambda r: -r[2])[:8]:
        print(
            f"    b{blk}h{h}: partner {partner:.3f}  self {self_a:.3f}  "
            f"other-jets {other:.3f}  lep/nu {lepnu:.3f}"
        )
    print("  nu query -> W-sjet pair heads among cached wire heads:")
    for blk, h, wpair, other in sorted(nu_rows, key=lambda r: -r[2]):
        print(f"    b{blk}h{h}: W-pair {wpair:.3f}  other-jets {other:.3f}")

    out_dir = os.path.join(REPO, "tmp_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resolved_r1_{args.model}_skip{args.skip}_b{args.batches}.csv")
    header = list(rows[0].keys())
    np.savetxt(
        out_path,
        np.stack([R[k] for k in header], axis=1),
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    print(f"\nsaved event-level resolved features: {out_path}")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()

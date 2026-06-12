"""Resolved qqbb W circuit, R2: real-event W-sjet-pair swaps.

This is the first causal test for whether resolved handling is driven by pair mass
or by context-relative hardness/topology. It uses only real resolved W-sjet pairs as
donors, replacing the two truth-W sjet feature rows in a target resolved event while
keeping the target lepton, neutrino, H-ljet, other jets, and object types fixed.

Arms:
  - placebo_samebin: donor close in mjj and inserted pair-relative-pT.
  - mjj_low_to_high / mjj_high_to_low: large mjj change, matched inserted relpT.
  - relpt_low_to_high / relpt_high_to_low: large inserted relpT change, matched mjj.

Important caveat: real-pair transplantation is less OOD than synthetic four-vectors,
but it still breaks event-level recoil/coherence. Interpret effects only with the
match diagnostics and placebo arm.

Usage:
  .venv/bin/python experiments/h1/resolved_r2_pair_swaps.py \
      --model thesis-ent1-bn1-d152 --skip 24 --batches 12
"""
import argparse
import os
import sys
from dataclasses import dataclass

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


def summarize_bool(x):
    x = np.asarray(x).astype(bool)
    return float(x.mean()) if len(x) else np.nan


def qfmt(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "n/a"
    return f"med {np.median(x):+.3f}  mean {x.mean():+.3f}  p16/p84 {np.percentile(x,16):+.3f}/{np.percentile(x,84):+.3f}"


@dataclass
class Arm:
    name: str
    target_mask: np.ndarray
    donor_mask_fn: object
    cost_fn: object
    description: str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="thesis-ent1-bn1-d152")
    ap.add_argument("--skip", type=int, default=24)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--quantile", type=float, default=0.30)
    ap.add_argument("--max-pairs", type=int, default=500)
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
    res_idx = torch.where(resolved)[0]
    jet_mask = (T == LJ) | (T == SJ)

    def fwd(x, t):
        outs = []
        with torch.no_grad():
            for c0 in range(0, len(x), args.batch_size):
                outs.append(model(x[c0 : c0 + args.batch_size, :, :5], t[c0 : c0 + args.batch_size]))
        return torch.cat(outs)

    base_out = fwd(X, T)
    base_pred = base_out.argmax(-1)
    base_margin = base_out[..., W_CLASS] - base_out[..., NONE_CLASS]

    p4 = X[..., :4]
    pt_all = pt(p4) * 100.0
    mass_all = mass(p4) * 100.0
    ht_jets = (pt_all * jet_mask.float()).sum(1)

    rows = []
    for ev in res_idx.tolist():
        wrows = torch.where((T[ev] == SJ) & (TRU[ev] == 2))[0]
        if len(wrows) != 2:
            continue
        # Sort by descending pT so donor leading/subleading maps to target leading/subleading.
        order = torch.argsort(pt_all[ev, wrows], descending=True)
        wrows = wrows[order]
        pair_p4 = p4[ev, wrows].sum(0)
        pair_m = (mass(pair_p4) * 100.0).item()
        pair_pt = (pt(pair_p4) * 100.0).item()
        pair_sumpt = pt_all[ev, wrows].sum().item()
        ht_other = (ht_jets[ev] - pair_sumpt).item()
        relpt = pair_pt / max(ht_other + pair_sumpt, 1e-6)
        row = {
            "ev": ev,
            "r0": int(wrows[0]),
            "r1": int(wrows[1]),
            "mjj": pair_m,
            "pair_pt": pair_pt,
            "pair_sumpt": pair_sumpt,
            "ht_other": ht_other,
            "relpt": relpt,
            "pt0": pt_all[ev, wrows[0]].item(),
            "pt1": pt_all[ev, wrows[1]].item(),
            "m0": mass_all[ev, wrows[0]].item(),
            "m1": mass_all[ev, wrows[1]].item(),
            "tag0": X[ev, wrows[0], 4].item(),
            "tag1": X[ev, wrows[1], 4].item(),
            "base_lep_w": bool(base_pred[ev, lpos[ev]] == W_CLASS),
            "base_nu_w": bool(base_pred[ev, npos[ev]] == W_CLASS),
            "base_both_wsj": bool(((base_pred[ev, wrows] == W_CLASS).sum()) == 2),
            "base_any_wsj": bool(((base_pred[ev, wrows] == W_CLASS).sum()) >= 1),
            "base_any_jet": bool(((base_pred[ev] == W_CLASS) & jet_mask[ev]).sum() >= 1),
            "base_wsj_sum_margin": base_margin[ev, wrows].sum().item(),
            "base_wsj_min_margin": base_margin[ev, wrows].min().item(),
            "base_lep_margin": base_margin[ev, lpos[ev]].item(),
        }
        rows.append(row)

    M = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    n_res = len(rows)
    low_mjj = M["mjj"] <= np.quantile(M["mjj"], args.quantile)
    high_mjj = M["mjj"] >= np.quantile(M["mjj"], 1 - args.quantile)
    low_relpt = M["relpt"] <= np.quantile(M["relpt"], args.quantile)
    high_relpt = M["relpt"] >= np.quantile(M["relpt"], 1 - args.quantile)
    scale_mjj = max(np.subtract(*np.percentile(M["mjj"], [84, 16])), 1e-6)
    scale_relpt = max(np.subtract(*np.percentile(M["relpt"], [84, 16])), 1e-6)
    scale_sumpt = max(np.subtract(*np.percentile(M["pair_sumpt"], [84, 16])), 1e-6)

    def inserted_relpt(donor_idx, target_idx):
        return M["pair_pt"][donor_idx] / np.maximum(M["ht_other"][target_idx] + M["pair_sumpt"][donor_idx], 1e-6)

    def cost_match_both(d, i):
        ir = inserted_relpt(d, i)
        return (
            np.abs(M["mjj"][d] - M["mjj"][i]) / scale_mjj
            + np.abs(ir - M["relpt"][i]) / scale_relpt
            + 0.25 * np.abs(M["pair_sumpt"][d] - M["pair_sumpt"][i]) / scale_sumpt
        )

    def cost_match_relpt(d, i):
        ir = inserted_relpt(d, i)
        return (
            np.abs(ir - M["relpt"][i]) / scale_relpt
            + 0.25 * np.abs(M["pair_sumpt"][d] - M["pair_sumpt"][i]) / scale_sumpt
        )

    def cost_match_mjj(d, i):
        return np.abs(M["mjj"][d] - M["mjj"][i]) / scale_mjj

    def donor_static(mask):
        def fn(_i):
            return mask.copy()
        return fn

    arms = [
        Arm(
            "placebo_samebin",
            np.ones(n_res, dtype=bool),
            donor_static(np.ones(n_res, dtype=bool)),
            cost_match_both,
            "donor close in mjj and inserted relpt",
        ),
        Arm(
            "mjj_low_to_high",
            low_mjj,
            donor_static(high_mjj),
            cost_match_relpt,
            "low-mjj targets receive high-mjj donors, matched inserted relpt",
        ),
        Arm(
            "mjj_high_to_low",
            high_mjj,
            donor_static(low_mjj),
            cost_match_relpt,
            "high-mjj targets receive low-mjj donors, matched inserted relpt",
        ),
        Arm(
            "relpt_low_to_high",
            low_relpt,
            lambda i: inserted_relpt(np.arange(n_res), i) >= np.quantile(M["relpt"], 1 - args.quantile),
            cost_match_mjj,
            "low-relpt targets receive high-inserted-relpt donors, matched mjj",
        ),
        Arm(
            "relpt_high_to_low",
            high_relpt,
            lambda i: inserted_relpt(np.arange(n_res), i) <= np.quantile(M["relpt"], args.quantile),
            cost_match_mjj,
            "high-relpt targets receive low-inserted-relpt donors, matched mjj",
        ),
    ]

    rng = np.random.default_rng(0)

    def greedy_match(arm):
        targets = np.where(arm.target_mask)[0].copy()
        rng.shuffle(targets)
        used = np.zeros(n_res, dtype=bool)
        pairs = []
        for i in targets:
            dmask = arm.donor_mask_fn(i).astype(bool)
            dmask[i] = False
            dmask &= ~used
            donors = np.where(dmask)[0]
            if len(donors) == 0:
                continue
            costs = arm.cost_fn(donors, i)
            best = donors[int(np.argmin(costs))]
            used[best] = True
            pairs.append((i, best, float(np.min(costs))))
            if len(pairs) >= args.max_pairs:
                break
        return pairs

    def eval_arm(name, pairs, description):
        if len(pairs) == 0:
            print(f"\n=== {name}: no matches ===")
            return []
        target_local = np.array([p[0] for p in pairs], dtype=int)
        donor_local = np.array([p[1] for p in pairs], dtype=int)
        costs = np.array([p[2] for p in pairs], dtype=float)
        target_events = M["ev"][target_local].astype(int)
        donor_events = M["ev"][donor_local].astype(int)

        X2 = X[target_events].clone()
        T2 = T[target_events].clone()
        for j, (ti, di) in enumerate(zip(target_local, donor_local)):
            tgt_rows = [int(M["r0"][ti]), int(M["r1"][ti])]
            donor_rows = [int(M["r0"][di]), int(M["r1"][di])]
            X2[j, tgt_rows, :5] = X[int(M["ev"][di]), donor_rows, :5]
        out = fwd(X2, T2)
        pred = out.argmax(-1)
        margin = out[..., W_CLASS] - out[..., NONE_CLASS]

        local_ar = torch.arange(len(target_events))
        target_events_t = torch.as_tensor(target_events)
        r0 = torch.as_tensor(M["r0"][target_local].astype(int))
        r1 = torch.as_tensor(M["r1"][target_local].astype(int))
        wrows = torch.stack([r0, r1], dim=1)
        lep_pos = lpos[target_events_t]
        nu_pos = npos[target_events_t]

        post_wsj_claims = (pred[local_ar.unsqueeze(1), wrows] == W_CLASS).sum(1).numpy()
        post_lep_w = (pred[local_ar, lep_pos] == W_CLASS).numpy()
        post_nu_w = (pred[local_ar, nu_pos] == W_CLASS).numpy()
        post_wsj_sum_margin = margin[local_ar.unsqueeze(1), wrows].sum(1).numpy()
        post_wsj_min_margin = margin[local_ar.unsqueeze(1), wrows].min(1).values.numpy()
        post_lep_margin = margin[local_ar, lep_pos].numpy()
        post_any_jet = ((pred == W_CLASS) & jet_mask[target_events_t]).sum(1).numpy() > 0

        base_lep_w = M["base_lep_w"][target_local].astype(bool)
        base_nu_w = M["base_nu_w"][target_local].astype(bool)
        base_both = M["base_both_wsj"][target_local].astype(bool)
        base_any_wsj = M["base_any_wsj"][target_local].astype(bool)
        base_any_jet = M["base_any_jet"][target_local].astype(bool)
        base_wsj_sum = M["base_wsj_sum_margin"][target_local]
        base_wsj_min = M["base_wsj_min_margin"][target_local]
        base_lep_margin = M["base_lep_margin"][target_local]

        ins_relpt = inserted_relpt(donor_local, target_local)
        records = []
        for k, ti, di in zip(range(len(pairs)), target_local, donor_local):
            records.append(
                {
                    "arm": name,
                    "target_local": ti,
                    "donor_local": di,
                    "target_event": int(M["ev"][ti]),
                    "donor_event": int(M["ev"][di]),
                    "cost": costs[k],
                    "target_mjj": M["mjj"][ti],
                    "donor_mjj": M["mjj"][di],
                    "delta_mjj": M["mjj"][di] - M["mjj"][ti],
                    "target_relpt": M["relpt"][ti],
                    "inserted_relpt": ins_relpt[k],
                    "delta_relpt": ins_relpt[k] - M["relpt"][ti],
                    "target_sumpt": M["pair_sumpt"][ti],
                    "donor_sumpt": M["pair_sumpt"][di],
                    "delta_sumpt": M["pair_sumpt"][di] - M["pair_sumpt"][ti],
                    "base_lep_w": int(base_lep_w[k]),
                    "post_lep_w": int(post_lep_w[k]),
                    "base_nu_w": int(base_nu_w[k]),
                    "post_nu_w": int(post_nu_w[k]),
                    "base_both_wsj": int(base_both[k]),
                    "post_both_wsj": int(post_wsj_claims[k] == 2),
                    "base_any_wsj": int(base_any_wsj[k]),
                    "post_any_wsj": int(post_wsj_claims[k] >= 1),
                    "base_any_jet": int(base_any_jet[k]),
                    "post_any_jet": int(post_any_jet[k]),
                    "delta_wsj_sum_margin": post_wsj_sum_margin[k] - base_wsj_sum[k],
                    "delta_wsj_min_margin": post_wsj_min_margin[k] - base_wsj_min[k],
                    "delta_lep_margin": post_lep_margin[k] - base_lep_margin[k],
                }
            )

        print(f"\n=== {name} ===")
        print(f"{description}")
        print(f"n={len(pairs)}  unique donors={len(np.unique(donor_local))}  match cost: {qfmt(costs)}")
        print("match diagnostics:")
        print(f"  target mjj       {qfmt(M['mjj'][target_local])}")
        print(f"  donor mjj        {qfmt(M['mjj'][donor_local])}")
        print(f"  delta mjj        {qfmt(M['mjj'][donor_local] - M['mjj'][target_local])}")
        print(f"  target relpt     {qfmt(M['relpt'][target_local])}")
        print(f"  inserted relpt   {qfmt(ins_relpt)}")
        print(f"  delta relpt      {qfmt(ins_relpt - M['relpt'][target_local])}")
        print(f"  delta sumpt GeV  {qfmt(M['pair_sumpt'][donor_local] - M['pair_sumpt'][target_local])}")
        print("readouts:")
        print(f"  P(lep=W)          {base_lep_w.mean():.4f} -> {post_lep_w.mean():.4f}  delta {post_lep_w.mean()-base_lep_w.mean():+.4f}")
        print(f"  P(nu=W)           {base_nu_w.mean():.4f} -> {post_nu_w.mean():.4f}  delta {post_nu_w.mean()-base_nu_w.mean():+.4f}")
        print(f"  P(both W-sjets)   {base_both.mean():.4f} -> {(post_wsj_claims == 2).mean():.4f}  delta {(post_wsj_claims == 2).mean()-base_both.mean():+.4f}")
        print(f"  P(any W-sjet)     {base_any_wsj.mean():.4f} -> {(post_wsj_claims >= 1).mean():.4f}  delta {(post_wsj_claims >= 1).mean()-base_any_wsj.mean():+.4f}")
        print(f"  P(any jet claims) {base_any_jet.mean():.4f} -> {post_any_jet.mean():.4f}  delta {post_any_jet.mean()-base_any_jet.mean():+.4f}")
        print(f"  delta W-sjet sum margin: {qfmt(post_wsj_sum_margin - base_wsj_sum)}")
        print(f"  delta W-sjet min margin: {qfmt(post_wsj_min_margin - base_wsj_min)}")
        print(f"  delta lep margin:        {qfmt(post_lep_margin - base_lep_margin)}")
        print(f"  lep verdict flip rate:   {(post_lep_w != base_lep_w).mean():.4f}")

        if name.startswith("mjj_"):
            filt = np.abs(ins_relpt - M["relpt"][target_local]) <= 0.03
            label = "matched subset |delta relpt|<=0.03"
        elif name.startswith("relpt_"):
            filt = np.abs(M["mjj"][donor_local] - M["mjj"][target_local]) <= 10.0
            label = "matched subset |delta mjj|<=10 GeV"
        else:
            filt = (np.abs(ins_relpt - M["relpt"][target_local]) <= 0.03) & (
                np.abs(M["mjj"][donor_local] - M["mjj"][target_local]) <= 10.0
            )
            label = "matched subset |delta relpt|<=0.03 and |delta mjj|<=10"
        if filt.sum() >= 30:
            print(f"  {label}: n={int(filt.sum())}")
            print(
                f"    delta mjj med {np.median(M['mjj'][donor_local][filt] - M['mjj'][target_local][filt]):+.3f}; "
                f"delta relpt med {np.median((ins_relpt - M['relpt'][target_local])[filt]):+.4f}; "
                f"delta sumpt med {np.median((M['pair_sumpt'][donor_local] - M['pair_sumpt'][target_local])[filt]):+.1f} GeV"
            )
            print(
                f"    P(lep=W) delta {post_lep_w[filt].mean()-base_lep_w[filt].mean():+.4f}; "
                f"P(any W-sjet) delta {(post_wsj_claims[filt] >= 1).mean()-base_any_wsj[filt].mean():+.4f}; "
                f"P(both W-sjets) delta {(post_wsj_claims[filt] == 2).mean()-base_both[filt].mean():+.4f}; "
                f"med delta W-sjet sum margin {np.median((post_wsj_sum_margin - base_wsj_sum)[filt]):+.3f}; "
                f"lep flip {(post_lep_w[filt] != base_lep_w[filt]).mean():.4f}"
            )
        return records

    print(f"model={args.model}  skip={args.skip}  batches={args.batches}")
    print(
        f"events={N}  lvbb={lvbb.sum().item()}  boosted={boosted.sum().item()}  "
        f"resolved={n_res}  qq-neither={((~lvbb) & ~boosted & ~resolved).sum().item()}"
    )
    print(
        f"resolved quantiles q={args.quantile:.2f}: "
        f"mjj low<={np.quantile(M['mjj'], args.quantile):.1f}, "
        f"high>={np.quantile(M['mjj'], 1-args.quantile):.1f}; "
        f"relpt low<={np.quantile(M['relpt'], args.quantile):.3f}, "
        f"high>={np.quantile(M['relpt'], 1-args.quantile):.3f}"
    )
    print(
        "baseline resolved: "
        f"P(lep=W)={M['base_lep_w'].mean():.4f}, "
        f"P(both W-sjets)={M['base_both_wsj'].mean():.4f}, "
        f"P(any W-sjet)={M['base_any_wsj'].mean():.4f}"
    )

    all_records = []
    for arm in arms:
        pairs = greedy_match(arm)
        all_records.extend(eval_arm(arm.name, pairs, arm.description))

    out_dir = os.path.join(REPO, "tmp_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resolved_r2_pair_swaps_{args.model}_skip{args.skip}_b{args.batches}.csv")
    if all_records:
        header = list(all_records[0].keys())
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for rec in all_records:
                f.write(",".join(str(rec[k]) for k in header) + "\n")
        print(f"\nsaved swap records: {out_path}")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()

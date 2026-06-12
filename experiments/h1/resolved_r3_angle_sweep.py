"""Resolved qqbb W circuit, R3: synthetic pair-angle sweep.

This tests pair-geometry sensitivity while holding individual W-sjet features fixed.

Controls:
  - Co-rotate both truth-W sjets by the same angle: preserves pair mjj, pair pT,
    pair sum-pT, individual pT/pz/E/mass/tag, and scalar HT. Any effect is absolute
    direction/recoil sensitivity.
  - Relative-angle sweep: keep the leading truth-W sjet fixed and rotate the
    subleading truth-W sjet to target |delta_phi| values. This preserves each sjet's
    pT/pz/E/mass/tag and scalar HT, but changes pair mjj and pair vector pT together.

This is a synthetic intervention. It can show pair-geometry sensitivity, but it does
not isolate mjj from pair vector pT, and large angle changes may be OOD.

Usage:
  .venv/bin/python experiments/h1/resolved_r3_angle_sweep.py \
      --model thesis-ent1-bn1-d152 --skip 24 --batches 12
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

torch.set_num_threads(6)

from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from models.registry import load_model


PAD, NU, LJ, SJ = 5, 2, 3, 4
W_CLASS = 2
NONE_CLASS = 0


def pt(p4):
    return torch.sqrt(torch.clamp(p4[..., 0] ** 2 + p4[..., 1] ** 2, min=0.0))


def mass(p4):
    return torch.sqrt(torch.clamp(p4[..., 3] ** 2 - (p4[..., :3] ** 2).sum(-1), min=0.0))


def dphi(phi1, phi2):
    return torch.atan2(torch.sin(phi1 - phi2), torch.cos(phi1 - phi2))


def rotate_row_phi(x, rows, new_phi):
    p_t = torch.sqrt(torch.clamp(x[torch.arange(len(x)), rows, 0] ** 2 + x[torch.arange(len(x)), rows, 1] ** 2, min=0.0))
    x[torch.arange(len(x)), rows, 0] = p_t * torch.cos(new_phi)
    x[torch.arange(len(x)), rows, 1] = p_t * torch.sin(new_phi)


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
    ap.add_argument("--max-events", type=int, default=0, help="0 means all resolved events")
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

    model, handles = load_model(
        args.model,
        checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
        register_bottleneck_hook=True,
        checkpoint_override=args.ckpt_override,
    )
    model.eval()

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
    if args.max_events and len(idx) > args.max_events:
        idx = idx[: args.max_events]
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

    def fwd(x):
        outs = []
        with torch.no_grad():
            for c0 in range(0, len(x), args.batch_size):
                outs.append(model(x[c0 : c0 + args.batch_size, :, :5], T[c0 : c0 + args.batch_size]))
        return torch.cat(outs)

    def pair_features(x):
        p4_pair = x[ar.unsqueeze(1), wrows, :4].sum(1)
        p_m = mass(p4_pair) * 100.0
        p_pt = pt(p4_pair) * 100.0
        p_sumpt = (pt(x[ar.unsqueeze(1), wrows, :4]) * 100.0).sum(1)
        ht = (pt(x[..., :4]) * 100.0 * jet_mask.float()).sum(1)
        relpt = p_pt / ht.clamp(min=1e-6)
        phi0 = torch.atan2(x[ar, lead, 1], x[ar, lead, 0])
        phi1 = torch.atan2(x[ar, sub, 1], x[ar, sub, 0])
        abs_dp = torch.abs(dphi(phi1, phi0))
        return p_m.numpy(), p_pt.numpy(), p_sumpt.numpy(), relpt.numpy(), abs_dp.numpy()

    def readouts(out):
        pred = out.argmax(-1)
        margin = out[..., W_CLASS] - out[..., NONE_CLASS]
        wsj_claims = (pred[ar.unsqueeze(1), wrows] == W_CLASS).sum(1).numpy()
        return {
            "lep_w": (pred[ar, lpos] == W_CLASS).numpy(),
            "nu_w": (pred[ar, npos] == W_CLASS).numpy(),
            "both_wsj": wsj_claims == 2,
            "any_wsj": wsj_claims >= 1,
            "any_jet": (((pred == W_CLASS) & jet_mask).sum(1) > 0).numpy(),
            "wsj_sum_margin": margin[ar.unsqueeze(1), wrows].sum(1).numpy(),
            "wsj_min_margin": margin[ar.unsqueeze(1), wrows].min(1).values.numpy(),
            "lep_margin": margin[ar, lpos].numpy(),
        }

    base_out = fwd(X)
    base = readouts(base_out)
    base_mjj, base_pairpt, base_sumpt, base_relpt, base_absdp = pair_features(X)
    phi_lead = torch.atan2(X[ar, lead, 1], X[ar, lead, 0])
    phi_sub = torch.atan2(X[ar, sub, 1], X[ar, sub, 0])
    sign = torch.sign(dphi(phi_sub, phi_lead))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    print(f"model={args.model}  skip={args.skip}  batches={args.batches}")
    print(
        f"events={len(X_all)}  boosted={boosted.sum().item()}  resolved_used={N} "
        f"(available {resolved.sum().item()})"
    )
    print("baseline resolved:")
    print(f"  P(lep=W)={base['lep_w'].mean():.4f}  P(any W-sjet)={base['any_wsj'].mean():.4f}  P(both W-sjets)={base['both_wsj'].mean():.4f}")
    print(f"  mjj {qfmt(base_mjj)}")
    print(f"  pair relpt {qfmt(base_relpt)}")
    print(f"  |dphi| {qfmt(base_absdp)}")

    records = []

    def report(kind, label, x_mod):
        out = fwd(x_mod)
        rd = readouts(out)
        mjj, pairpt, sumpt, relpt, absdp = pair_features(x_mod)
        rec = {
            "kind": kind,
            "label": label,
            "mjj_med": np.median(mjj),
            "pairpt_med": np.median(pairpt),
            "relpt_med": np.median(relpt),
            "absdphi_med": np.median(absdp),
            "p_lep_w": rd["lep_w"].mean(),
            "p_any_wsj": rd["any_wsj"].mean(),
            "p_both_wsj": rd["both_wsj"].mean(),
            "p_any_jet": rd["any_jet"].mean(),
            "delta_p_lep_w": rd["lep_w"].mean() - base["lep_w"].mean(),
            "delta_p_any_wsj": rd["any_wsj"].mean() - base["any_wsj"].mean(),
            "delta_p_both_wsj": rd["both_wsj"].mean() - base["both_wsj"].mean(),
            "med_delta_wsj_sum_margin": np.median(rd["wsj_sum_margin"] - base["wsj_sum_margin"]),
            "med_delta_lep_margin": np.median(rd["lep_margin"] - base["lep_margin"]),
            "lep_flip": (rd["lep_w"] != base["lep_w"]).mean(),
        }
        records.append(rec)
        print(f"\n{kind:16s} {label}")
        print(f"  pair: mjj med {rec['mjj_med']:.1f}, relpt med {rec['relpt_med']:.3f}, |dphi| med {rec['absdphi_med']:.3f}")
        print(
            f"  P(lep=W) {base['lep_w'].mean():.4f}->{rec['p_lep_w']:.4f} ({rec['delta_p_lep_w']:+.4f}); "
            f"P(any W-sjet) {base['any_wsj'].mean():.4f}->{rec['p_any_wsj']:.4f} ({rec['delta_p_any_wsj']:+.4f}); "
            f"P(both) {base['both_wsj'].mean():.4f}->{rec['p_both_wsj']:.4f} ({rec['delta_p_both_wsj']:+.4f})"
        )
        print(
            f"  med delta W-sjet sum margin {rec['med_delta_wsj_sum_margin']:+.3f}; "
            f"med delta lep margin {rec['med_delta_lep_margin']:+.3f}; lep flip {rec['lep_flip']:.4f}"
        )

    # Co-rotation control.
    for delta in [0.5, 1.0, 1.57079632679, 3.14159265359]:
        xm = X.clone()
        rotate_row_phi(xm, lead, phi_lead + delta)
        rotate_row_phi(xm, sub, phi_sub + delta)
        report("co_rotate", f"delta={delta:.2f}", xm)

    # Relative-angle sweep. Preserve leading sjet; set subleading phi relative to it.
    for target in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.57079632679, 2.0, 2.5, 3.0, 3.14159265359]:
        xm = X.clone()
        rotate_row_phi(xm, sub, phi_lead + sign * target)
        report("relative_sweep", f"abs_dphi={target:.2f}", xm)

    out_dir = os.path.join(REPO, "tmp_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resolved_r3_angle_sweep_{args.model}_skip{args.skip}_b{args.batches}.csv")
    header = list(records[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for rec in records:
            f.write(",".join(str(rec[k]) for k in header) + "\n")
    print(f"\nsaved angle sweep records: {out_path}")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()

"""Audit-disjoint confirmation of four-angle probability averaging on signal."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from hd_interp import load_trained
from physics_rotation_average import ensemble
from physics_samples import sample_events
from utils.utils import check_category

MASSES_TEV = dict(zip(range(510115, 510125), [0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]))
RESIDUE = 4


def confirmation_sample(data_dir, count, seed, audit_count, audit_seed):
    """Reject rows from the earlier frozen single-rotation audit."""
    audit = sample_events(data_dir, RESIDUE, audit_count, 0, audit_seed)
    candidate = sample_events(data_dir, RESIDUE, count, 0, seed)
    keep = np.ones(len(candidate["row"]), dtype=bool)
    populations = {}
    metadata = {}
    for dsid in np.unique(candidate["dsid"]):
        in_dsid = candidate["dsid"] == dsid
        audit_rows = audit["row"][audit["dsid"] == dsid]
        keep[in_dsid] = ~np.isin(candidate["row"][in_dsid], audit_rows)
        shape = (data_dir / f"dsid_{dsid}.memmap.shape").read_text()
        n_full = int(shape.split(",")[0])
        n_residue = len(range(RESIDUE, n_full, 8))
        n_remaining = n_residue - len(audit_rows)
        n_kept = int((in_dsid & keep).sum())
        if not n_kept:
            raise ValueError(f"Empty confirmation stratum {dsid}")
        populations[int(dsid)] = n_remaining
        metadata[str(int(dsid))] = {
            "residue_population": n_residue, "audit_n": len(audit_rows),
            "candidate_n": int(in_dsid.sum()), "confirmation_n": n_kept,
            "overlap_removed": int((in_dsid & ~keep).sum()),
        }
        # Target the nonaudit finite population, scaled by each DSID's whole
        # dataset to residue size ratio (approximately eight).
        candidate["weight"][in_dsid] = (
            candidate["mc"][in_dsid] * (n_full / n_residue) * (n_remaining / n_kept)
        )
    test = {key: value[keep] for key, value in candidate.items()}
    assert (test["row"] % 8 == RESIDUE).all() and (test["label"] != 0).all()
    keys = set(zip(test["dsid"], test["row"]))
    assert len(keys) == len(test["row"])
    assert keys.isdisjoint(zip(audit["dsid"], audit["row"]))
    return test, populations, metadata


def measure(mask, dsids, populations, weights, base, avg):
    """Linearize the signed-weight paired ratio with stratified finite population variance."""
    denom = weights[mask].sum()
    baseline = float(weights[mask & base].sum() / denom)
    averaged = float(weights[mask & avg].sum() / denom)
    changed = avg.astype(np.int8) - base
    estimate = float(weights[mask] @ changed[mask] / denom)
    # Include sampled members outside the slice as zeros; the within-DSID
    # sample size was set before filtering to a topology or mass.
    influence = weights * mask * (changed - estimate)
    numerator_variance = 0.
    for dsid, population in populations.items():
        values = influence[dsids == dsid]
        if 1 < len(values) < population:
            numerator_variance += len(values) * (1 - len(values) / population) * values.var(ddof=1)
    se = float(np.sqrt(numerator_variance) / abs(denom))
    rescues = mask & ~base & avg
    regressions = mask & base & ~avg
    return {
        "n": int(mask.sum()), "baseline_accuracy": baseline, "mean_four_accuracy": averaged,
        "paired_difference": estimate, "paired_se": se,
        "paired_95_normal_ci": [estimate - 1.96 * se, estimate + 1.96 * se],
        "baseline_raw_perfect": int((mask & base).sum()),
        "mean_four_raw_perfect": int((mask & avg).sum()),
        "rescues_raw": int(rescues.sum()), "regressions_raw": int(regressions.sum()),
        "rescues_signed_weight": float(weights[rescues].sum()),
        "regressions_signed_weight": float(weights[regressions].sum()),
        "signed_denominator": float(denom),
        "negative_weight_events": int((mask & (weights < 0)).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("checkpoint", type=Path, help="Training output directory containing config.json")
    parser.add_argument("out", type=Path)
    parser.add_argument("--candidate-n", type=int, default=140000)
    parser.add_argument("--seed", type=int, default=8147)
    parser.add_argument("--audit-n", type=int, default=10000)
    parser.add_argument("--audit-seed", type=int, default=2701)
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    model, _, checkpoint = load_trained(args.checkpoint, "cpu")
    start = time.monotonic()
    test, populations, sample = confirmation_sample(
        args.data, args.candidate_n, args.seed, args.audit_n, args.audit_seed
    )
    print(f"sample n={len(test['row'])} after removing {sum(m['overlap_removed'] for m in sample.values())} audit rows; "
          f"{time.monotonic() - start:.1f}s", flush=True)
    angles = np.arange(4) * np.pi / 2
    native, _, _ = ensemble(model, test, [0.], args.batch)
    print(f"native predictions {time.monotonic() - start:.1f}s", flush=True)
    avg, _, _ = ensemble(model, test, angles, args.batch)
    print(f"averaged predictions {time.monotonic() - start:.1f}s", flush=True)
    truth = np.minimum(test["truth"], 2)
    padding = test["types"] == 5
    base_correct = ((native == truth) | padding).all(axis=1)
    avg_correct = ((avg == truth) | padding).all(axis=1)
    dsids = test["dsid"]
    cats = check_category(test["types"], test["truth"], 5)
    masks = {"overall": np.ones(len(dsids), dtype=bool)}
    masks.update({f"topology_{cat}": cats == cat for cat in range(6)})
    masks.update({f"mass_{mass:.1f}_TeV": dsids == dsid for dsid, mass in MASSES_TEV.items()})
    result = {
        "checkpoint": str(checkpoint), "data": str(args.data),
        "residue": RESIDUE, "exploratory_average_residue": 6,
        "angle_radians": angles.tolist(), "baseline_radians": [0.],
        "reduction": "arithmetic mean of per-object softmax probabilities before argmax",
        "sample_seed": args.seed, "requested_candidate_n": args.candidate_n,
        "earlier_audit_seed": args.audit_seed, "earlier_audit_requested_n": args.audit_n,
        "audit_overlap_in_analyzed_rows": 0,
        "variance_method": "paired signed-weight ratio influence, within-DSID sample variance "
                           "with finite-population correction for the nonaudit residue stratum",
        "sample_by_dsid": sample,
        "unclassified_topology_n": int((cats == -1).sum()),
        "assignment_changes_raw": int((((native != avg) & ~padding).any(axis=1)).sum()),
        "results": {key: measure(mask, dsids, populations, test["weight"], base_correct, avg_correct)
                    for key, mask in masks.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["results"]["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()

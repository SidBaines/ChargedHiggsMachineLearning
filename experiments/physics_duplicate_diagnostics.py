"""Report how exact source-row duplicates affect November score tails."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.physics_readouts import fit_readout
from experiments.physics_samples import operating_points, yield_summary


def load_json(path):
    return json.loads(path.read_text())


def group_ids(sample, membership):
    return np.array([(int(dsid), int(row)) in membership
                     for dsid, row in zip(sample["dsid"], sample["row"])])


def filter_rows(group, mask):
    return {key: arr[mask] for key, arr in group.items()}


def scope(sample, exclude):
    dup = group_ids(sample, exclude)
    return {"n": int(dup.sum()), "n_total": len(dup),
            "n_signal": int(((sample["label"] > 0) & dup).sum()),
            "weight": yield_summary(sample["weight"][dup]),
            "by_dsid": {str(dsid): int((sample["dsid"][dup] == dsid).sum())
                        for dsid in np.unique(sample["dsid"][dup])},
            "positive_weights": float(sample["weight"][dup & (sample["weight"] > 0)].sum()),
            "negative_weights": float(sample["weight"][dup & (sample["weight"] < 0)].sum())}


def fixed_cuts(events, predictions, cuts):
    out = {}
    for split, sample in events.items():
        data = predictions[split]
        for cls, label in [(1, "lvbb"), (2, "qqbb")]:
            score = data["score"].astype(float)
            in_channel = (data["channel"] == cls) & (sample["label"] == 0)
            passing = in_channel & (score >= cuts[label]["threshold"])
            below = score[in_channel & ~passing]
            out.setdefault(split, {})[label] = {
                "cut": cuts[label]["threshold"],
                "background": yield_summary(sample["weight"][passing]),
                "closest_duplicate_below_cut": float(below.max()) if len(below) else None,
                "passing_row_ids": np.c_[sample["dsid"][passing], sample["row"][passing]].tolist(),
            }
    return out


def students(directory, removed, direct_training):
    report = load_json(directory / "results.json")
    events = {}
    for residue in (2, 6):
        with np.load(directory / f"sample_r{residue}.npz") as raw:
            events[residue] = dict(raw)
    masks = {s: group_ids(e, removed) for s, e in events.items()}
    report_rows = {r: {"removed": scope(events[r], removed),
                       "direct_training_input_duplicate": scope(events[r], direct_training)}
                   for r in events}
    out = {"cohorts": report_rows, "models": {}}
    for name, original in report["models"].items():
        predictions = {}
        for r in (2, 6):
            with np.load(directory / f"{name}_r{r}.npz") as data:
                predictions[r] = dict(data)
        excluded = {r: filter_rows(events[r], masks[r]) for r in events}
        excluded_predictions = {r: filter_rows(predictions[r], masks[r]) for r in events}
        kept = {r: filter_rows(events[r], ~masks[r]) for r in events}
        kept_predictions = {r: filter_rows(predictions[r], ~masks[r]) for r in events}
        fixed = fixed_cuts(excluded, excluded_predictions, original["sb200"])
        fixed_after_removal = fixed_cuts(kept, kept_predictions, original["sb200"])
        yc, yt = kept_predictions[2], kept_predictions[6]
        working_points = operating_points(kept[2], kept[6], yc["score"], yt["score"],
                                          yc["channel"], yt["channel"])
        out["models"][name] = {"excluded_duplicate_passes": fixed,
                               "full_cohort_fixed_cut_after_removal": fixed_after_removal,
                               "recalibrated_after_exclusion": working_points,
                               "original_working_points": original["sb200"]}
    return out


def readouts(cache, reports, removed, source_pairs):
    splits = {"fit": (0, 20000, 55000), "calibration": (2, 16000, 355000),
              "evaluation": (4, 16000, 355000), "confirmation": (6, 16000, 355000)}
    groups, features = {}, {}
    fit_ids = None
    for name, (residue, ns, nb) in splits.items():
        pattern = f"readouts_*_{residue}_{ns}_{nb}_2701.npz"
        with np.load(next(cache.glob(pattern))) as data:
            keys = ("dsid", "row", "label", "weight", "train_weight")
            sample = {key: data[f"ev_{key}"] for key in keys}
            if name == "fit":
                fit_ids = set(zip(sample["dsid"], sample["row"]))
            masks = np.ones(len(sample["row"]), dtype=bool) if name == "fit" else group_ids(sample, removed)
            groups[name] = filter_rows(sample, masks)
            features[name] = {key: data[f"fr_{key}"][masks]
                              for key in ("native", "channel", "pooled", "physics")}
    fit_overlap = duplicates_of_fitted_inputs(source_pairs, fit_ids)
    cohorts = {}
    for name, (residue, ns, nb) in splits.items():
        with np.load(next(cache.glob(f"readouts_*_{residue}_{ns}_{nb}_2701.npz"))) as data:
            sample = {key: data[f"ev_{key}"] for key in ("dsid", "row", "label", "weight")}
            cohorts[name] = {"removed": scope(sample, removed),
                             "direct_fit_input_duplicate": scope(sample, fit_overlap)}
    settings = [load_json(p) for p in reports]
    scored = {s: {"score": features[s]["native"], "channel": features[s]["channel"]}
              for s in splits if s != "fit"}
    to_report = {}
    native = settings[0]["readouts"]["native"]
    cuts = native["evaluation"]["operating_points"]
    to_report["native"] = fixed_cuts({s: e for s, e in groups.items() if s != "fit"}, scored, cuts)
    for ref in settings:
        cfg = ref["config"]
        for name, entries in ref["readouts"].items():
            if name == "native":
                continue
            for seed, entry in entries.items():
                scores, head = fit_readout(groups["fit"], features, name, int(seed),
                                           cfg["epochs"], cfg["fit_batch"])
                loss_delta = float(np.max(np.abs(np.array(head["fit_loss_by_epoch"]) -
                                                entry["fit_loss_by_epoch"])))
                shifted = {split: {"score": raw, "channel": features[split]["channel"]}
                           for split, raw in scores.items()}
                original_cuts = entry["evaluation"]["operating_points"]
                to_report[f"{name}_{seed}"] = {
                    "fitted_loss_curve_max_error": loss_delta,
                    "excluded_duplicate_passes":
                        fixed_cuts({s: e for s, e in groups.items() if s != "fit"},
                                   shifted, original_cuts),
                }
                print("Readout duplicate tail", name, seed, "fit loss max error", loss_delta, flush=True)
    return {"cohorts": cohorts, "models": to_report}


def duplicates_of_fitted_inputs(pairs, fitted):
    result = set()
    for a, ar, b, br, _, _ in pairs:
        if (a, ar) in fitted and (b, br) not in fitted:
            result.add((b, br))
        if (b, br) in fitted and (a, ar) not in fitted:
            result.add((a, ar))
    return result


def main(args):
    torch.set_num_threads(args.threads)
    audit = load_json(args.audit)
    seen = set(map(tuple, audit["exclusions"]["teacher_seen_even"]))
    even = set(map(tuple, audit["exclusions"]["later_even_duplicate"]))
    excluded = seen | even
    with np.load(args.students_training_ids) as train:
        train_ids = set(zip(train["dsid"], train["row"]))
    training_overlap = duplicates_of_fitted_inputs(audit["pairs"], train_ids)
    result = {"exclusion_rule": "Union of even rows duplicate with an odd source row, and later residues with an identical earlier even row. Student and readout fits are kept fixed for diagnostic comparison.",
              "students": students(args.students, excluded, training_overlap),
              "intervention_arrays": {}}
    for name in ("jet_joint.npz", "geometry_toward_joint.npz"):
        with np.load(args.interventions / name) as sample:
            result["intervention_arrays"][name] = {"teacher_seen": scope(sample, seen),
                                                     "later_even": scope(sample, even)}
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print("Student and intervention duplicate tail checks complete", flush=True)
    if args.readouts:
        result["readouts"] = readouts(args.cache, args.readouts, excluded, audit["pairs"])
        args.out.write_text(json.dumps(result,indent=2)+"\n")
        print("Readout duplicate tail checks complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("audit","students","students-training-ids","interventions","cache","out"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    parser.add_argument("--readouts", type=Path, nargs="+")
    parser.add_argument("--threads", type=int, default=2)
    main(parser.parse_args())

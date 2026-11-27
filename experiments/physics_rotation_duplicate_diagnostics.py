"""Quantify exact duplicate membership at each existing model's fixed operating point."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from hd_interp import build_model, load_trained
from physics_rotation import predictions
from physics_samples import operating_points, sample_events, yield_summary


def selected(channel, scores, point, name):
    scores = np.asarray(scores, dtype=float)
    return (channel == (1 if name == "lvbb" else 2)) & (scores >= point[name]["threshold"])


def arm_scores(model, cal, test, operating_point):
    out = {}
    for name, sample in (("calibration", cal), ("confirmation", test)):
        records = {}
        angles = [0., np.pi/4, np.pi/2, np.pi, -np.pi/2] if name == "confirmation" else [0.]
        base_labels, base_channel, base_score = predictions(model, sample, 0., "cpu", 1024)
        for angle in angles:
            predictions_at_angle, channel, score = predictions(model, sample, angle, "cpu", 1024)
            changes = ((predictions_at_angle != base_labels) & (sample["types"] != 5)).any(1)
            error = abs(base_score - score)
            rows = {}
            for c, chan in ((1, 'lvbb'), (2, 'qqbb')):
                valid = channel == c
                threshold = operating_point[chan]["threshold"]
                accepted = selected(channel, score, operating_point, chan)
                original = selected(base_channel, base_score, operating_point, chan)
                rows[chan] = {
                    "threshold": threshold, "closest_eligible_score_below_cut": (
                        float(np.max(score[valid & ~accepted])) if (valid & ~accepted).any() else None),
                    "n_reco_valid": int(valid.sum()),
                    "passed": yield_summary(sample["weight"][accepted]),
                    "selected_ids": np.column_stack((sample["dsid"][accepted], sample["row"][accepted])).tolist(),
                    "n_selection_flip": int((accepted != original).sum()),
                }
            records[str(angle)] = {"per_channel": rows, "n_assignment_flip": int(changes.sum()),
                                   "total_abs_logodds_change": float(error.sum())}
        out[name] = {"all_duplicate_weights": yield_summary(sample["weight"]), "results": records}
    return out


def main(root, output, recheck_calibration):
    exclusions = json.loads((output / "row_exclusions.json").read_text())
    duplicates = {tuple(row) for name in exclusions for row in exclusions[name]}
    cohorts = {}
    calibration_full = None
    for name, residue, n_signal in (("calibration",2,0),("confirmation",6,30000)):
        all_events = sample_events(root / "tmp_data_20250321v1_all", residue, n_signal, 120000)
        subset = np.array([(int(d),int(r)) in duplicates for d,r in zip(all_events["dsid"],all_events["row"])])
        cohorts[name] = {k:v[subset] for k,v in all_events.items()}
        if recheck_calibration and name == "calibration":
            calibration_full = all_events, ~subset
    results = {}
    for filepath in sorted(output.glob("27*.json")) + [output/"teacher_reference.json"]:
        recorded = json.loads(filepath.read_text())
        if filepath.stem == "teacher_reference":
            model, _, _ = load_trained(root / "output/20260624-163646_TrainingOutput", "cpu")
        else:
            saved = torch.load(filepath.with_suffix(".pth"),map_location="cpu",weights_only=True)
            model = build_model(saved["config"], "cpu")
            model.load_state_dict(saved["model"])
            model.eval()
        result = arm_scores(model, cohorts["calibration"], cohorts["confirmation"],
                            recorded["operating_points"])
        if calibration_full is not None:
            cal, kept = calibration_full
            _, ch, score = predictions(model, cal, 0., "cpu", 1024)
            trimmed_cal = {key: value[kept] for key, value in cal.items()}
            nominal = operating_points(cal, cal, score, score, ch, ch)
            trimmed = operating_points(trimmed_cal, cal, score[kept], score, ch[kept], ch)
            result["calibration_recheck"] = {
                label: {"cpu_nominal": nominal[label]["threshold"],
                        "cpu_excluding_duplicates": trimmed[label]["threshold"],
                        "cpu_nominal_background": nominal[label]["calibration_background"],
                        "cpu_excluding_duplicates_background": trimmed[label]["calibration_background"]}
                for label in ("lvbb", "qqbb")
            }
        results[filepath.stem] = result
        print("Audited duplicate membership for", filepath.stem, flush=True)
    result = {"method":"Rescored only the exact duplicate events on CPU. Membership used each fitted arm's existing fixed cuts. The excluded rows are the union of teacher-seen even rows and exact duplicates of earlier even residues. This diagnostic keeps thresholds fixed.",
              "results":results}
    path = output / ("duplicate_membership_recalibrated.json" if recheck_calibration
                     else "duplicate_membership.json")
    path.write_text(json.dumps(result,indent=2) + "\n")
    print("Completed",list(results),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recheck-calibration", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    main(args.root, args.output, args.recheck_calibration)

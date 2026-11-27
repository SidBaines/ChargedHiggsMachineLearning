"""Evaluate the six fixed compact students on the larger matched readout sample."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from physics_samples import operating_points, sample_events, yield_summary
from physics_students import PAD, ROOT, save_json
from physics_students_formula import TOPOLOGIES, selection_model
from experiments.physics_readouts import cache_id
from utils.utils import check_category, check_valid

MASS_BY_DSID = dict(zip(range(510115, 510125), (800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000)))


@torch.inference_mode()
def student_predictions(model, sample, batch_size):
    n = len(sample["label"])
    scores, channel, perfect = (np.empty(n, dtype=dtype) for dtype in (np.float32, np.int8, bool))
    for start in range(0, n, batch_size):
        batch = slice(start, start + batch_size)
        x = torch.from_numpy(sample["x"][batch])
        types = torch.from_numpy(sample["types"][batch])
        truth = torch.from_numpy(sample["truth"][batch]).clamp(0, 2)
        logits = model(x, types)
        event = logits["event"]
        reco = logits["reco"]
        scores[batch] = torch.logsumexp(event[:, 1:], dim=-1) - event[:, 0]
        _, channel[batch] = check_valid(types, reco, PAD, categorical=True, returnTypes=True)
        perfect[batch] = ((reco.argmax(-1) == truth) | (types == PAD)).all(1)
        if start and start % (batch_size * 128) == 0:
            print(f"  processed {start:,}/{n:,}", flush=True)
    if not np.isfinite(scores).all():
        raise ValueError("Student produced a nonfinite score")
    return {"score": scores, "channel": channel, "perfect": perfect}


def teacher_predictions(cache_dir, cache_key, residue, sample):
    path = cache_dir / f"readouts_{cache_key}_{residue}_16000_355000_2701.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen readout cache for this checkpoint and dataset: {path}. "
                                "Run experiments.physics_readouts on CPU with inference batch 1024 first.")
    with np.load(path) as saved:
        for key in ("dsid", "row", "label", "weight"):
            np.testing.assert_array_equal(saved[f"ev_{key}"], sample[key])
        return {"score": saved["fr_native"], "channel": saved["fr_channel"], "perfect": saved["fr_perfect"]}


def cached_student_predictions(path, checkpoint, cache_key, residue, batch_size, sample):
    with checkpoint.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    fingerprint = f"{cache_key}:{residue}:{batch_size}:{digest}"
    if path.exists():
        with np.load(path) as saved:
            if "fingerprint" in saved and saved["fingerprint"].item() == fingerprint:
                return {key: saved[key] for key in ("score", "channel", "perfect")}
    model = selection_model(checkpoint.parent, checkpoint.stem, "cpu")
    output = student_predictions(model, sample, batch_size)
    save_predictions(path, output, fingerprint=fingerprint)
    return output


def save_predictions(path, output, **metadata):
    with NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **output, **metadata)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize(cal, test, prediction):
    cal_pred, test_pred = prediction[2], prediction[6]
    signal = test["label"] != 0
    score = np.asarray(test_pred["score"], dtype=float)
    selected = test_pred["channel"] != 0
    correct = test_pred["perfect"]
    truth_category = test["category"]
    weight = test["weight"]
    points = operating_points(cal, test, cal_pred["score"], score, cal_pred["channel"], test_pred["channel"])
    for channel, label in enumerate(("lvbb", "qqbb"), 1):
        point = points[label]
        sb = point["signal"]
        point["signal_by_mass_gev"] = {str(MASS_BY_DSID.get(int(dsid), dsid)): values for dsid, values in sb.items()}
        passes = (test_pred["channel"] == channel) & (test["label"] == channel) & (score >= point["threshold"])
        point["signal_all_masses"] = yield_summary(weight[passes])
        for summary in (point["calibration_background"], point["test_background"],
                        point["signal_all_masses"], *sb.values()):
            summary["sqrt_sumw2"] = float(np.sqrt(summary["sumw2"]))
    return {
        "auc_abs_yield_weighted": float(roc_auc_score(signal, score, sample_weight=np.abs(weight))),
        "auc_abs_yield_weighted_valid_only": float(roc_auc_score(signal[selected], score[selected],
                                                                sample_weight=np.abs(weight[selected]))),
        "valid_reco_count_fraction": float(selected.mean()),
        "reco_perfect_signed": float(np.average(correct[signal], weights=weight[signal])),
        "signal_perfect_by_topology": {
            name: {
                "all": yield_summary(weight[signal & (truth_category == i)]),
                "perfect": yield_summary(weight[signal & (truth_category == i) & correct]),
            } for i, name in enumerate(TOPOLOGIES)},
        "sb200": points,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "tmp_data_20250321v1_all")
    parser.add_argument("--students", type=Path, default=Path("output/physics_interp/students"))
    parser.add_argument("--teacher-cache", type=Path, default=Path("tmp_readouts_cache"))
    parser.add_argument("--out", type=Path, default=Path("output/physics_interp/students_sb200"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    training = json.loads((args.students / "training.json").read_text())
    teacher_cache_key = cache_id(SimpleNamespace(ckpt=Path(training["teacher_checkpoint"]),
                                                data_dir=args.data, device="cpu", inference_batch=1024))
    keys = ["teacher", *training["models"]]
    report = {
        "data_dir": str(args.data), "teacher_checkpoint": training["teacher_checkpoint"],
        "teacher_cache_dir": str(args.teacher_cache), "teacher_cache_key": teacher_cache_key,
        "sample_seed": 2701,
        "requested_signal_background": [16000, 355000], "calibration": 2, "evaluation": 6,
        "scores": "logsumexp of signal event logits minus background logit",
        "validity_and_channel": "utils.utils.check_valid on model object logits",
        "models": {},
    }
    predictions = {key: {} for key in keys}
    samples = {}
    started = time.monotonic()
    for residue in (2, 6):
        sample = sample_events(args.data, residue, 16000, 355000, seed=2701)
        samples[residue] = {key: sample[key] for key in ("row", "dsid", "label", "weight", "mc")}
        samples[residue]["category"] = check_category(sample["types"], sample["truth"], PAD)
        print(f"r{residue}: loaded {len(sample['label']):,} shared events in {time.monotonic() - started:.1f}s", flush=True)
        ids = args.out / f"sample_r{residue}.npz"
        if ids.exists():
            with np.load(ids) as saved:
                for key in ("row", "dsid", "label", "weight", "mc"):
                    np.testing.assert_array_equal(saved[key], samples[residue][key])
        else:
            np.savez_compressed(ids, **samples[residue])
        for key in keys:
            before = time.monotonic()
            path = args.out / f"{key}_r{residue}.npz"
            if key == "teacher":
                output = teacher_predictions(args.teacher_cache, teacher_cache_key, residue, sample)
                save_predictions(path, output)
            else:
                output = cached_student_predictions(path, args.students / f"{key}.pt",
                                                    teacher_cache_key, residue, args.batch_size, sample)
            if len(output["score"]) != len(sample["label"]):
                raise ValueError(f"Mismatched score count for {key}, residue {residue}")
            predictions[key][residue] = output
            if residue == 6:
                report["models"][key] = summarize(samples[2], samples[6], predictions[key])
                save_json(args.out / "results.json", report)
            print(f"r{residue}: {key} finished in {time.monotonic() - before:.1f}s", flush=True)
        del sample


if __name__ == "__main__":
    main()

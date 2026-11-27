"""Stratified, read-only sampling and working points for the November physics studies."""
from pathlib import Path

import numpy as np


def sample_events(data_dir, residue, n_signal, n_background, seed=2701):
    """Uniformly sample each DSID's row % 8 == residue stratum. Odd rows trained the teacher.

    Expand raw MC weights by the sampling fraction to estimate the full dataset yield.
    Return stable (DSID,row) identifiers for verifying disjointness and pairing edits.
    """
    if residue not in range(8) or min(n_signal, n_background) < 0:
        raise ValueError("Invalid row residue or requested event count")
    files = []
    for path in sorted(Path(data_dir).glob("dsid_*.memmap")):
        dsid = int(path.stem.split("_")[1])
        n, absolute, signed = Path(str(path) + ".shape").read_text().split(",")
        if int(n):
            files.append((dsid, path, int(n), float(absolute), float(signed)))
    if not files:
        raise FileNotFoundError(data_dir)
    signals = {d for d, *_ in files if 500000 < d < 600000}
    n_by_class = {is_signal: sum(n for d, _, n, _, _ in files if (d in signals) == is_signal)
                  for is_signal in (False, True)}
    total_background = sum(w for d, _, _, _, w in files if d not in signals)
    chunks = {}
    for dsid, path, n, absolute, signed in files:
        signal = dsid in signals
        budget = n_signal if signal else n_background
        available = len(range(residue, n, 8))
        if not budget or not available:
            continue
        count = min(available, max(1, round(budget * n / n_by_class[signal])))
        rng = np.random.default_rng([seed, residue, dsid])
        row = 8 * np.sort(rng.choice(available, size=count, replace=False)) + residue
        raw = np.memmap(path, dtype=np.float32, mode="r", shape=(n, 17, 8))[row]
        label = raw[:, 0, 0].astype(np.int64)
        if not np.all((label != 0) == signal) or not np.all(raw[:, 1, 1] == dsid):
            raise ValueError(f"DSID/label mismatch in {path}")
        types = raw[:, 2:, 0].astype(np.int64)
        x = raw[:, 2:, 1:6].copy()
        x[:, :, :4] /= 1e5
        pad = types == 5
        x[pad] = [1e-3, 1e-3, 0, 2e-3, 0]
        truth = np.where(pad, 0, raw[:, 2:, -1]).astype(np.int64)
        mc = raw[:, 1, 0].astype(float)
        weight = mc * (n / count)
        train_factor = (total_background / len(signals) if signal else signed) / absolute
        for name, value in dict(x=x, types=types, truth=truth, label=label, row=row,
                                dsid=np.full(count, dsid), weight=weight, mc=mc,
                                train_weight=np.abs(weight * train_factor)).items():
            chunks.setdefault(name, []).append(value)
    data = {key: np.concatenate(parts) for key, parts in chunks.items()}
    permutation = np.random.default_rng([seed, residue]).permutation(len(data["label"]))
    return {key: value[permutation] for key, value in data.items()}


def yield_summary(weights):
    return {"yield": float(weights.sum()), "sumw2": float(weights @ weights), "n": len(weights)}


def operating_points(cal, test, cal_score, test_score, cal_channel, test_channel, target=200):
    """Select one inclusive score cut per reconstructed channel on calibration background.

    Descend by complete ties to the first crossing of the signed target, choosing the
    nearer of the two attainable yields. Keep this threshold fixed on independent data.
    Signal yield requires correct true channel as in the thesis metric.
    """
    test_score = np.asarray(test_score, dtype=float)
    if not np.isfinite(test_score).all() or not np.isfinite(target) or target <= 0:
        raise ValueError("The yield target must be finite and positive, and assessment scores finite")
    result = {}
    for channel, name in [(1, "lvbb"), (2, "qqbb")]:
        bkg = (cal["label"] == 0) & (cal_channel == channel)
        vals = np.asarray(cal_score[bkg], dtype=float)
        w = cal["weight"][bkg]
        if not len(vals) or not np.isfinite(vals).all():
            raise ValueError(f"No finite calibration background in {name}")
        order = np.argsort(-vals, kind="stable")
        scores = vals[order]
        last = np.r_[scores[1:] != scores[:-1], True]
        levels = np.r_[np.nextafter(scores[0], np.inf), scores[last]]
        cumulative = np.r_[0., np.cumsum(w[order])[last]]
        crossed = np.flatnonzero(cumulative >= target)
        if len(crossed):
            j = crossed[0]
            idx = j if abs(cumulative[j] - target) < abs(cumulative[j-1] - target) else j-1
        else:
            idx = int(np.argmax(cumulative))
        threshold = float(levels[idx])
        selected = (test_score >= threshold) & (test_channel == channel)
        result[name] = {
            "threshold": threshold, "target_bracketed": bool(len(crossed)),
            "calibration_background": yield_summary(w[vals >= threshold]),
            "test_background": yield_summary(test["weight"][selected & (test["label"] == 0)]),
            "signal": {str(int(d)): yield_summary(test["weight"][selected & (test["label"] == channel)
                                                                & (test["dsid"] == d)])
                       for d in np.unique(test["dsid"][test["label"] != 0])}
        }
    return result

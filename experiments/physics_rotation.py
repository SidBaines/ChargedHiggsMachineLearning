"""Audit invariance to a global rotation about the beam, including the MET object."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hd_interp import load_trained
from physics_samples import operating_points, sample_events
from utils.utils import check_category, check_valid


def rotate(x, angle):
    if isinstance(angle, (int, float)) and angle == 0:
        return x
    a = torch.as_tensor(angle, device=x.device, dtype=x.dtype)
    c, s = a.cos(), a.sin()
    px, py = x[..., 0], x[..., 1]
    return torch.cat(((px*c - py*s).unsqueeze(-1), (px*s + py*c).unsqueeze(-1),
                      x[..., 2:]), dim=-1)


@torch.inference_mode()
def predictions(model, data, angle, device, batch=512):
    labels, channels, scores = [], [], []
    for i in range(0, len(data["x"]), batch):
        x = torch.as_tensor(data["x"][i:i+batch], device=device)
        t = torch.as_tensor(data["types"][i:i+batch], device=device)
        out = model(rotate(x, angle), t)
        z = out["event"]
        scores.append((torch.logsumexp(z[:,1:], dim=1)-z[:,0]).cpu().numpy())
        reco = out["reco"].cpu()
        labels.append(reco.argmax(-1).numpy())
        channels.append(check_valid(t.cpu(), reco, 5, True, True)[1].numpy())
    return tuple(np.concatenate(values) for values in (labels, channels, scores))


def reco_accuracy(data, predicted):
    signal = data["label"] > 0
    truth = np.minimum(data["truth"], 2)
    exact = ((predicted == truth) | (data["types"] == 5)).all(axis=1)
    cats = check_category(data["types"], data["truth"], 5)
    result = {}
    for k in ["all", *range(6)]:
        mask = signal if k == "all" else signal & (cats == k)
        w = data["weight"][mask]
        result[str(k)] = {"n": int(mask.sum()),
                          "accuracy": float(np.average(exact[mask], weights=w)) if len(w) else None}
    return result


def evaluate(model, cal, test, angles, device):
    _, cc, cs = predictions(model, cal, 0, device)
    baseline, ch, scores = predictions(model, test, 0, device)
    base_wp = operating_points(cal, test, cs, scores, cc, ch)
    def passes(channel, score):
        score = np.asarray(score, dtype=float)
        return ((channel == 1) & (score >= base_wp["lvbb"]["threshold"])) | (
                (channel == 2) & (score >= base_wp["qqbb"]["threshold"]))
    keep = passes(ch, scores)
    result = {}
    for angle in angles:
        pred, chan, z = predictions(model, test, angle, device)
        changed = (pred != baseline) & (test["types"] != 5)
        diff = abs(z-scores)
        row = {"reco_accuracy": reco_accuracy(test, pred),
               "operating_points": operating_points(cal, test, cs, z, cc, chan)}
        for label, mask in [("signal", test["label"] > 0), ("background", test["label"] == 0)]:
            if not mask.any():
                continue
            row[label] = {
                "n": int(mask.sum()), "assignment_flip": float(changed[mask].any(1).mean()),
                "object_flip": float(changed[mask].sum()/(test["types"][mask] != 5).sum()),
                "selection_flip": float((keep[mask] != passes(chan,z)[mask]).mean()),
                "n_selection_flip": int((keep[mask] != passes(chan,z)[mask]).sum()),
                "channel_flip": float((chan[mask] != ch[mask]).mean()),
                "mean_abs_logodds_change": float(diff[mask].mean()),
                "p95_abs_logodds_change": float(np.quantile(diff[mask],.95))}
        result[str(angle)] = row
        print("angle", round(angle,3), "assignment flip", row["signal"]["assignment_flip"], flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--signal", type=int, default=12000)
    p.add_argument("--background", type=int, default=48000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(3)
    net, _, ck = load_trained(args.checkpoint, args.device)
    cal = sample_events(args.data, 2, 0, args.background)
    test = sample_events(args.data, 4, args.signal, args.background)
    study = evaluate(net, cal, test, [0., np.pi/4, np.pi/2, np.pi, -np.pi/2], args.device)
    args.out.write_text(json.dumps({"checkpoint":str(ck), "residues":[2,4], "results":study},indent=2))

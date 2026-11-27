"""Post-hoc frozen symmetry control: average probabilities over four azimuths."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from hd_interp import load_trained
from physics_rotation import reco_accuracy, rotate
from physics_samples import operating_points, sample_events
from utils.utils import check_valid


@torch.inference_mode()
def ensemble(model, data, angles, batch):
    scores, channels, predictions = [], [], []
    for i in range(0, len(data["x"]), batch):
        x = torch.from_numpy(data["x"][i:i+batch])
        types = torch.from_numpy(data["types"][i:i+batch])
        objects, events = [], []
        for angle in angles:
            out = model(rotate(x, angle),types)
            objects.append(out["reco"].double().softmax(-1))
            events.append(out["event"].double().softmax(-1))
        reco = torch.stack(objects).mean(0)
        event = torch.stack(events).mean(0)
        scores.append((event[:,1:].sum(1).log()-event[:,0].log()).numpy())
        channels.append(check_valid(types,reco,5,True,True)[1].numpy())
        predictions.append(reco.argmax(-1).numpy())
    return tuple(np.concatenate(values) for values in (predictions,channels,scores))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data",type=Path)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("out",type=Path)
    args = p.parse_args()
    torch.set_num_threads(3)
    net, _, checkpoint = load_trained(args.checkpoint,"cpu")
    cal = sample_events(args.data,2,0,120000)
    test = sample_events(args.data,6,30000,120000)
    sig = test["label"] != 0
    angles = np.arange(4)*np.pi/2
    result = {"checkpoint":str(checkpoint), "angles":angles.tolist(), "calibration_residue":2,
              "test_residue":6, "post_hoc":True}
    for arm, group in [("native",np.array([0.])),("mean_four",angles)]:
        _, cc, cs = ensemble(net,cal,group,512)
        pred, chan, score = ensemble(net,test,group,512)
        wp = operating_points(cal,test,cs,score,cc,chan)
        metrics = {"reco":reco_accuracy(test,pred), "operating_points":wp,
                   "auc":float(roc_auc_score(sig,score,sample_weight=abs(test["weight"])))}
        # A fixed subset has the same rows across arms and probes. Verify both a
        # member of the four-fold group and an arbitrary off-grid orientation.
        idx = slice(0,16000)
        subset = {key:v[idx] for key,v in test.items()}
        metrics["rotation"] = {}
        for theta in [np.pi/2,0.37]:
            rp, _, rs = ensemble(net,subset,group+theta,512)
            changed = ((rp != pred[idx]) & (subset["types"] != 5)).any(1)
            metrics["rotation"][str(theta)] = {name:float(changed[mask].mean())
                                               for name, mask in [("signal",sig[idx]),("background",~sig[idx])]}
            metrics["rotation"][str(theta)]["mean_abs_logodds_change"] = float(abs(rs-score[idx]).mean())
        result[arm] = metrics
        print(arm,metrics["reco"]["all"],metrics["rotation"],flush=True)
        args.out.write_text(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()

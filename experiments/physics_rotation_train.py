"""Paired fine-tunes of one trained d152 checkpoint with alternate azimuth treatments."""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from hd_interp import build_model, load_trained
from physics_rotation import predictions, reco_accuracy, rotate
from physics_samples import operating_points, sample_events


def start_model(checkpoint, arm, device):
    original, config, _ = load_trained(checkpoint, device)
    if arm != "relative":
        return original, config
    config = dict(config, feature_set=["eta", "pt", "m", "tag", "sin_dphi_lep", "cos_dphi_lep"])
    model = build_model(config, device)
    weights = original.state_dict()
    old = weights["object_net.0.weight"]  # m2, pt, eta, phi, tag, type embedding
    weights["object_net.0.weight"] = torch.cat([old[:, :4], torch.zeros_like(old[:, :1]), old[:, 4:]], 1)
    model.load_state_dict(weights)
    return model, config


def fit(args, model, arm, seed, train, batch_size, steps, verbose=True):
    device = args.device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    angles = np.random.default_rng([seed, 110])
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    tensors = {key: torch.as_tensor(train[key], device=device) for key in ("x","types","truth","label")}
    wt = train["train_weight"]
    tensors["weight"] = torch.from_numpy((wt/wt.mean()).astype("float32")).to(device)
    model.train()
    loss_values = []
    t0 = time.monotonic()
    for i in range(steps):
        idx = torch.as_tensor(rng.integers(len(wt),size=batch_size),device=device)
        x, types, truth, label, w = (tensors[k][idx] for k in ("x","types","truth","label","weight"))
        if arm == "augmented":
            theta = torch.from_numpy(angles.uniform(-np.pi,np.pi,(batch_size,1)).astype("float32")).to(device)
            x = rotate(x, theta)
        out = model(x, types)
        per_obj = F.cross_entropy(out["reco"].transpose(1,2), truth.clamp(max=2), reduction="none")
        nonpad = types != 5
        rw = w * (label > 0)
        reco = (((per_obj*nonpad).sum(1)/nonpad.sum(1).clamp(min=1))*rw).sum()/rw.sum().clamp(min=1e-8)
        event = (F.cross_entropy(out["event"], label, reduction="none")*w).sum()/w.sum()
        loss = reco + event
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        lr_factor = min(1., (i+1)/30) * (0.02+0.98*(1+math.cos(math.pi*i/steps))/2)
        optim.param_groups[0]["lr"] = args.lr*lr_factor
        optim.step()
        if i in {0, steps//4, steps//2, 3*steps//4, steps-1}:
            loss_values.append([i+1,float(loss.detach().cpu())])
            if verbose:
                print(seed,arm,"step",i+1,"loss",loss_values[-1][1],"seconds",round(time.monotonic()-t0,1),flush=True)
    model.eval()
    return loss_values


def score(model, cal, test, device):
    _, cc, zc = predictions(model,cal,0.,device,1024)
    r, ch, z = predictions(model,test,0.,device,1024)
    op = operating_points(cal,test,zc,z,cc,ch)
    def keep(channel, score):
        score = score.astype(float)
        return ((channel == 1) & (score >= op["lvbb"]["threshold"])) | ((channel == 2) & (score >= op["qqbb"]["threshold"]))
    base_keep = keep(ch,z)
    signal = test["label"] > 0
    result = {"reconstruction":reco_accuracy(test,r), "auc_abs_mc":roc_auc_score(signal,z,sample_weight=abs(test["weight"])),
              "operating_points":op, "rotation":{}}
    for angle in [np.pi/4,np.pi/2,np.pi,-np.pi/2]:
        rr, rc, rz = predictions(model,test,angle,device,1024)
        changed = ((rr != r) & (test["types"] != 5)).any(1)
        result["rotation"][str(angle)] = {name:{"assignment_flip":float(changed[m].mean()),
                                                "selection_flip":float((keep(rc,rz)[m] != base_keep[m]).mean()),
                                                "mean_abs_logodds_change":float(abs(z[m]-rz[m]).mean())}
                                           for name,m in [("signal",signal),("background",~signal)]}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("--device",default="cpu")
    p.add_argument("--steps",type=int,default=1600)
    p.add_argument("--batch",type=int,default=512)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--benchmark",action="store_true")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,nargs="+",default=[2701,2702,2703])
    p.add_argument("--arms",nargs="+",choices=["ordinary","augmented","relative"],default=["ordinary","augmented","relative"])
    args=p.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True)
    train=sample_events(args.data,1,48000,96000)
    if args.benchmark:
        for batch_size in (128,512,1024):
            model, _ = start_model(args.checkpoint,"augmented",args.device)
            t0=time.monotonic()
            fit(args,model,"augmented",2701,train,batch_size,25,False)
            print("benchmark",args.device,batch_size,round(batch_size*25/(time.monotonic()-t0)),"events/s",flush=True)
        return
    cal=sample_events(args.data,2,0,120000)
    test=sample_events(args.data,6,30000,120000)
    for seed in args.seed:
        for arm in args.arms:
            if (args.out/"STOP").exists():
                return
            output=args.out/f"{seed}_{arm}.json"
            if output.exists():
                continue
            model,cfg=start_model(args.checkpoint,arm,args.device)
            ck=args.out/f"{seed}_{arm}.pth"
            if ck.exists():
                saved=torch.load(ck, map_location=args.device,weights_only=True)
                model.load_state_dict(saved["model"])
                learning=saved["curve"]
            else:
                learning=fit(args,model,arm,seed,train,args.batch,args.steps)
                torch.save({"model":model.state_dict(),"curve":learning,"config":cfg},ck.with_suffix(".partial"))
                os.replace(ck.with_suffix(".partial"),ck)
            model.eval()
            results=score(model,cal,test,args.device)
            results.update(seed=seed,arm=arm,steps=args.steps,batch=args.batch,lr=args.lr,
                           n_train=len(train["label"]),train_residue=1,cal_residue=2,test_residue=6,
                           checkpoint=str(args.checkpoint),curve=learning)
            output.with_suffix(".partial").write_text(json.dumps(results,indent=2))
            os.replace(output.with_suffix(".partial"),output)
            print("completed",seed,arm,results["reconstruction"]["all"],flush=True)


if __name__=="__main__":
    main()

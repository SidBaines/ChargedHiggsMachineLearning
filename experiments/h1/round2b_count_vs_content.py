"""H1 round 2b: sjet COUNT vs CONTENT.

Round 2 showed the lep/nu channel decision is driven primarily by the sjet system
(masking sjets flips 57-82% of qqbb events; jet transplant flips both directions).
Disambiguate: does the model count sjets, or read their kinematics?

  D1 qqbb: own ljets + same NUMBER of sjets, but sampled from lvbb donor events
           (count preserved, content swapped) -> high flips = content matters
  D2 qqbb: same but donors are other qqbb events (control for the sampling itself)
  D3 lvbb: own ljets + own-count sjets from qqbb donors
  D4 lvbb: own jets + EXTRA lvbb-donor sjets appended to reach qqbb-median count
           (content innocuous, count raised) -> high flips = count matters

Usage: .venv/bin/python tmp_h1_round2b.py [--model ...] [--batches 6]
"""
import os, sys, argparse
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
torch.set_num_threads(4)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

p = argparse.ArgumentParser()
p.add_argument("--model", default="thesis-ent1-bn1-d152")
p.add_argument("--batches", type=int, default=6)
ARGS = p.parse_args()

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
DSIDS = list(range(510115, 510125))
PAD, NU, SJ, LJ = 5, 2, 4, 3  # types: 3=LJET, 4=SJET — first run of this script had them swapped (its "sjet" pools were really LJET pools)
W_CLASS = 2
stds = np.ones(7); stds[:4] = 1e5
torch.manual_seed(0)

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)

BN = 1 if "bn1" in ARGS.model else None
model, _ = load_model(ARGS.model, checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False)
model.eval()
cache = ActivationCache()
handles = [m.register_forward_hook(fn, with_kwargs=True)
           for m, fn in hook_attention_heads(model, cache, detach=True,
                 SINGLE_ATTENTION=False, bottleneck_attention_output=BN)]

def fwd(x, types):
    with torch.no_grad():
        return model(x[..., :5], types)

EV = []
for _ in range(ARGS.batches):
    b = next(dl)
    x, types = b["x"], b["types"]
    pred = fwd(x, types).argmax(-1)
    tru = torch.round(x[..., -1]).long()
    nu_m = types == NU; lep_m = (types == 0) | (types == 1)
    ok = (nu_m.sum(1) == 1) & (lep_m.sum(1) == 1)
    for i in torch.arange(len(types))[ok]:
        i = i.item()
        np_ = nu_m[i].float().argmax().item(); lp_ = lep_m[i].float().argmax().item()
        EV.append(dict(x=x[i].clone(), types=types[i].clone(),
                       lvbb=(tru[i, lp_].item() == 3),
                       p_nu=pred[i, np_].item(), p_lep=pred[i, lp_].item(),
                       n_sj=(types[i] == SJ).sum().item()))

LV = [e for e in EV if e["lvbb"]]; QQ = [e for e in EV if not e["lvbb"]]
nsj_lv = np.array([e["n_sj"] for e in LV]); nsj_qq = np.array([e["n_sj"] for e in QQ])
print(f"model={ARGS.model}  events={len(EV)} (lvbb {len(LV)} / qqbb {len(QQ)})")
print(f"sjet count: lvbb mean {nsj_lv.mean():.2f} (median {int(np.median(nsj_lv))})  "
      f"qqbb mean {nsj_qq.mean():.2f} (median {int(np.median(nsj_qq))})")
for n in range(8):
    print(f"  n_sjets={n}: lvbb {np.mean(nsj_lv==n):.3f}  qqbb {np.mean(nsj_qq==n):.3f}")

# pools of individual sjet feature rows per channel
def sjet_pool(events):
    rows = []
    for e in events:
        for r in range(15):
            if e["types"][r].item() == SJ:
                rows.append(e["x"][r])
    return torch.stack(rows)
POOL = {"lv": sjet_pool(LV), "qq": sjet_pool(QQ)}

def rebuild(e, sjets):
    """Own lep/nu/ljets + provided sjet rows."""
    x = torch.zeros(15, 7); t = torch.full((15,), PAD, dtype=e["types"].dtype)
    k = 0
    for r in range(15):
        if e["types"][r].item() in (0, 1, NU, LJ):
            x[k] = e["x"][r]; t[k] = e["types"][r]; k += 1
    for srow in sjets:
        if k >= 15: break
        x[k] = srow; t[k] = SJ; k += 1
    return x, t

def run(events, label, donor=None, target_count=None, n_max=1500):
    rng = np.random.default_rng(0)
    xs, ts = [], []
    n = min(n_max, len(events))
    for k in range(n):
        e = events[k]
        cnt = e["n_sj"] if target_count is None else max(e["n_sj"], target_count)
        if donor is None:   # own sjets, possibly padded with own duplicates? not used
            sj = [e["x"][r] for r in range(15) if e["types"][r].item() == SJ]
        else:
            pool = POOL[donor]
            if target_count is None:
                sj = [pool[j] for j in rng.integers(len(pool), size=cnt)]
            else:           # keep own sjets, append donor sjets up to target
                sj = [e["x"][r] for r in range(15) if e["types"][r].item() == SJ]
                sj += [pool[j] for j in rng.integers(len(pool), size=cnt - len(sj))]
        x, t = rebuild(e, sj)
        xs.append(x); ts.append(t)
    X = torch.stack(xs); T = torch.stack(ts)
    preds = []
    for c in range(0, n, 1024):
        preds.append(fwd(X[c:c + 1024], T[c:c + 1024]).argmax(-1))
    pred = torch.cat(preds)
    nu_m = T == NU; lep_m = (T == 0) | (T == 1)
    ar = torch.arange(n)
    p_nu = pred[ar, nu_m.float().argmax(1)]; p_lep = pred[ar, lep_m.float().argmax(1)]
    print(f"  {label:52s} P(nu=W)={(p_nu==W_CLASS).float().mean():.4f}  "
          f"P(lep=W)={(p_lep==W_CLASS).float().mean():.4f}  "
          f"P(nu=lep)={(p_nu==p_lep).float().mean():.4f}  (n={n})")

print(f"\n=== D. sjet count vs content (own lep/nu/ljets kept throughout) ===")
print(f"  {'qqbb as-is':52s} P(nu=W)={np.mean([e['p_nu']==W_CLASS for e in QQ]):.4f}")
run(QQ, "qqbb: rebuild w/ OWN sjets (control)")
run(QQ, "D2 qqbb: own-count sjets from OTHER-qqbb pool", donor="qq")
run(QQ, "D1 qqbb: own-count sjets from LVBB pool  <-- KEY", donor="lv")
print(f"  {'lvbb as-is':52s} P(nu=W)={np.mean([e['p_nu']==W_CLASS for e in LV]):.4f}")
run(LV, "lvbb: rebuild w/ OWN sjets (control)")
run(LV, "lvbb: own-count sjets from OTHER-lvbb pool", donor="lv")
run(LV, "D3 lvbb: own-count sjets from QQBB pool  <-- KEY", donor="qq")
tgt = int(np.median(nsj_qq))
run(LV, f"D4 lvbb: own sjets + lvbb-pool extras to n={tgt} <-- KEY", donor="lv", target_count=tgt)

for h in handles: h.remove()
print("\ndone.")

"""RT4: is W-candidacy gated on ABSOLUTE pT or pT RELATIVE to the event scale?
(thesis model, slice C boosted qqbb)

The dataset mixes H+ masses 0.8-3 TeV, so true-W pT spans a huge range; "pT dominates
candidacy" (A1-pT, E2) could mean either. Discriminating interventions:
  (a) W-ljet x0.5 (fixed mass)            -- reproduces A1-pT
  (b) WHOLE EVENT x0.5 (all real objects, per-object masses fixed)
        absolute-pT account: claims drop like (a); relative: claims survive
  (c) everything EXCEPT the W x2          -- relative: claims drop; absolute: no change
  (d) everything EXCEPT the W x0.5        -- relative: claims rise; absolute: no change
Plus observational: baseline claim rate & soft-miss threshold by DSID (mass point).
"""
import os, sys
import numpy as np
import torch
import argparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="thesis-ent1-bn1-d152")
ARGS = ap.parse_args()
torch.set_num_threads(6)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
PAD, NU, LJ, SJ = 5, 2, 3, 4
W = 2
stds = np.ones(7); stds[:4] = 1e5
torch.manual_seed(0)

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in range(510115, 510125)},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)
for _ in range(12):
    next(dl)

BN = 1 if "bn1" in ARGS.model else None
model, _ = load_model(ARGS.model, checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False)
model.eval()
cache = ActivationCache()
handles = [m.register_forward_hook(fn, with_kwargs=True)
           for m, fn in hook_attention_heads(model, cache, detach=True,
                 SINGLE_ATTENTION=False, bottleneck_attention_output=BN)]

def fwd(x, types):
    outs = []
    with torch.no_grad():
        for c in range(0, len(x), 2048):
            outs.append(model(x[c:c + 2048, :, :5], types[c:c + 2048]))
    return torch.cat(outs)

Xs, Ts, Ds = [], [], []
for _ in range(6):
    b = next(dl)
    types = b["types"]
    ok = ((types == NU).sum(1) == 1) & (((types == 0) | (types == 1)).sum(1) == 1)
    Xs.append(b["x"][ok].clone()); Ts.append(types[ok].clone()); Ds.append(b["dsids"][ok].clone())
X = torch.cat(Xs); T = torch.cat(Ts); DS = torch.cat(Ds).long(); N = len(X)
TRU = torch.round(X[..., -1]).long()
ar = torch.arange(N)
npos = (T == NU).float().argmax(1); lpos = ((T == 0) | (T == 1)).float().argmax(1)
lvbb = TRU[ar, lpos] == 3
boosted = (~lvbb) & (((T == LJ) & (TRU == 2)).sum(1) == 1)

bsel = torch.where(boosted)[0]
wp = ((T[bsel] == LJ) & (TRU[bsel] == 2)).float().argmax(1)
bar = torch.arange(len(bsel))
dsid_b = DS[bsel]
real = (T[bsel] != PAD)

m2_all = torch.clamp(X[bsel][..., 3] ** 2 - (X[bsel][..., :3] ** 2).sum(-1), min=0)

def scale_objs(scale_w, scale_rest):
    X2 = X[bsel].clone()
    sc = torch.full(X2.shape[:2], 1.0)
    sc[real] = scale_rest
    sc[bar, wp] = scale_w
    X2[..., :3] = X2[..., :3] * sc.unsqueeze(-1)
    X2[..., 3] = torch.sqrt((X2[..., :3] ** 2).sum(-1) + m2_all)
    X2[~real] = X[bsel][~real]   # keep padding rows exactly as they were
    return X2

ARMS = [
    ("baseline", 1.0, 1.0),
    ("(a) W x0.5, rest x1", 0.5, 1.0),
    ("(b) whole event x0.5", 0.5, 0.5),
    ("(c) W x1, rest x2", 1.0, 2.0),
    ("(d) W x1, rest x0.5", 1.0, 0.5),
    ("(e) whole event x2", 2.0, 2.0),
]
print(f"boosted n={len(bsel)}")
print(f"\n{'arm':24s} {'P(W claims)':>12s} {'P(lep=W)':>9s}")
res = {}
for name, sw, sr in ARMS:
    X2 = scale_objs(sw, sr)
    o = fwd(X2, T[bsel]); pr = o.argmax(-1)
    cw = (pr[bar, wp] == W); lw = (pr[bar, lpos[bsel]] == W)
    res[name] = cw
    print(f"{name:24s} {cw.float().mean():12.4f} {lw.float().mean():9.4f}")

print("\n=== observational: baseline claim rate + W-pT by DSID (mass point) ===")
pt_w = torch.sqrt(X[bsel][bar, wp, 0] ** 2 + X[bsel][bar, wp, 1] ** 2) * 100
print(f"  {'dsid':>7s} {'n':>5s} {'med W-pT GeV':>13s} {'P(claim)':>9s} {'P(claim| pT in 300-500)':>24s}")
band = (pt_w > 300) & (pt_w < 500)
for d in range(510115, 510125):
    m = dsid_b == d
    if m.sum() < 30: continue
    mb = m & band
    pb = res["baseline"][mb].float().mean().item() if mb.sum() >= 20 else float("nan")
    print(f"  {d:7d} {m.sum().item():5d} {pt_w[m].median():13.0f} {res['baseline'][m].float().mean():9.4f} "
          f"{pb:24.4f}  (n_band={mb.sum().item()})")

for h in handles: h.remove()
print("\ndone.")

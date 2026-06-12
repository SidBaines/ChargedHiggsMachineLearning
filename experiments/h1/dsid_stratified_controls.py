"""DSID-stratified controls for two high-level H1 variable claims.

Checks whether:
  1. LW1's b1h3 ~ pT(lepW)/HT(jets) association survives within DSID/mass point.
  2. RT4's relative-hardness intervention (rest x2 at fixed W) suppresses W claims
     within each DSID, not only after pooling mass points.

This is a confound check, not a new circuit-localization experiment.
Usage:
  .venv/bin/python experiments/h1/dsid_stratified_controls.py \
    [--model thesis-ent1-bn1-d152] [--skip 18] [--batches 6]
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="thesis-ent1-bn1-d152")
ap.add_argument("--skip", type=int, default=18)
ap.add_argument("--batches", type=int, default=6)
ap.add_argument("--ckpt-override", default=None)
ARGS = ap.parse_args()
torch.set_num_threads(6)

from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from interp.activations import ActivationCache, hook_attention_heads
from models.registry import load_model

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
PAD, NU, LJ, SJ = 5, 2, 3, 4
W = 2
stds = np.ones(7)
stds[:4] = 1e5

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7,
    N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in range(510115, 510125)},
    max_objs_in_memmap=15,
    batch_size=2048,
    device="cpu",
    is_train=False,
    n_splits=2,
    validation_split_idx=0,
    n_targets=3,
    shuffle=False,
    shuffle_batch=False,
    means=None,
    stds=stds,
    objs_to_output=15,
    signal_only=True,
    has_eventNumbers=True,
)
for _ in range(ARGS.skip):
    next(dl)

model, _ = load_model(
    ARGS.model,
    checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
    register_bottleneck_hook=False,
    checkpoint_override=ARGS.ckpt_override,
)
model.eval()
bn = 1 if "bn1" in ARGS.model else None
cache = ActivationCache()
handles = [
    m.register_forward_hook(fn, with_kwargs=True)
    for m, fn in hook_attention_heads(
        model, cache, detach=True, SINGLE_ATTENTION=False, bottleneck_attention_output=bn
    )
]


def fwd(x, types):
    outs = []
    with torch.no_grad():
        for c in range(0, len(x), 2048):
            outs.append(model(x[c : c + 2048, :, :5], types[c : c + 2048]))
    return torch.cat(outs)


Xs, Ts, Ds = [], [], []
for _ in range(ARGS.batches):
    b = next(dl)
    types = b["types"]
    ok = ((types == NU).sum(1) == 1) & (((types == 0) | (types == 1)).sum(1) == 1)
    Xs.append(b["x"][ok].clone())
    Ts.append(types[ok].clone())
    Ds.append(b["dsids"][ok].clone().long())

X = torch.cat(Xs)
T = torch.cat(Ts)
DS = torch.cat(Ds)
N = len(X)
TRU = torch.round(X[..., -1]).long()
ar = torch.arange(N)
npos = (T == NU).float().argmax(1)
lpos = ((T == 0) | (T == 1)).float().argmax(1)
lvbb = TRU[ar, lpos] == 3
boosted = (~lvbb) & (((T == LJ) & (TRU == 2)).sum(1) == 1)
print(f"slice batches {ARGS.skip + 1}-{ARGS.skip + ARGS.batches}: events {N}")


def pt_of(rows):
    return torch.sqrt((rows[..., :2] ** 2).sum(-1))


# ---- LW1 ratio claim, stratified by DSID ----
lsel = torch.where(lvbb)[0]
lar = torch.arange(len(lsel))
lp, np_ = lpos[lsel], npos[lsel]
plep = X[lsel][lar, lp]
pnu = X[lsel][lar, np_]
pt_lepW = torch.sqrt(((plep[:, :2] + pnu[:, :2]) ** 2).sum(-1))
pt_all_l = pt_of(X[lsel])
ht_jets = (pt_all_l * ((T[lsel] == LJ) | (T[lsel] == SJ)).float()).sum(1)
ratio = pt_lepW / ht_jets.clamp(min=0.01)

s13, s22 = [], []
with torch.no_grad():
    for c in range(0, len(lsel), 2048):
        sl = slice(c, min(c + 2048, len(lsel)))
        model(X[lsel][sl][..., :5], T[lsel][sl])
        car = torch.arange(sl.stop - sl.start)
        s13.append(cache["block_1_attention"]["bottleneck_activation"][car, 3, np_[sl], 0])
        if bn is not None and model.num_attention_blocks > 2:
            s22.append(cache["block_2_attention"]["bottleneck_activation"][car, 2, np_[sl], 0])
s13 = torch.cat(s13)

print("\n=== LW1 ratio correlation within lvbb ===")
print(f"{'scope':>10s} {'n':>5s} {'rho ratio,b1h3':>15s} {'rho pTlepW,b1h3':>17s} {'rho HT,b1h3':>13s}")


def rho(a, b):
    if len(a) < 50:
        return float("nan")
    return spearmanr(a.numpy(), b.numpy()).statistic


print(
    f"{'all':>10s} {len(lsel):5d} {rho(ratio, s13):15.3f} "
    f"{rho(pt_lepW, s13):17.3f} {rho(ht_jets, s13):13.3f}"
)
for d in range(510115, 510125):
    m = DS[lsel] == d
    if int(m.sum()) < 100:
        continue
    print(
        f"{d:10d} {int(m.sum()):5d} {rho(ratio[m], s13[m]):15.3f} "
        f"{rho(pt_lepW[m], s13[m]):17.3f} {rho(ht_jets[m], s13[m]):13.3f}"
    )

# ---- RT4 relative-hardness intervention, stratified by DSID ----
bsel = torch.where(boosted)[0]
bar = torch.arange(len(bsel))
wp = ((T[bsel] == LJ) & (TRU[bsel] == 2)).float().argmax(1)
real = T[bsel] != PAD
m2_all = torch.clamp(X[bsel][..., 3] ** 2 - (X[bsel][..., :3] ** 2).sum(-1), min=0)

def rest_scaled(scale_rest):
    X2 = X[bsel].clone()
    sc = torch.full(X2.shape[:2], 1.0)
    sc[real] = scale_rest
    sc[bar, wp] = 1.0
    X2[..., :3] = X2[..., :3] * sc.unsqueeze(-1)
    X2[..., 3] = torch.sqrt((X2[..., :3] ** 2).sum(-1) + m2_all)
    X2[~real] = X[bsel][~real]
    return X2

base_pr = fwd(X[bsel], T[bsel]).argmax(-1)
rest2_pr = fwd(rest_scaled(2.0), T[bsel]).argmax(-1)
base_claim = base_pr[bar, wp] == W
rest2_claim = rest2_pr[bar, wp] == W
pt_w = pt_of(X[bsel][bar, wp]) * 100
band = (pt_w > 300) & (pt_w < 500)

print("\n=== RT4 rest x2 suppression within boosted ===")
print(f"{'scope':>10s} {'n':>5s} {'base':>8s} {'restx2':>8s} {'drop':>8s} {'base|300-500':>13s}")
def row(name, mask):
    if int(mask.sum()) == 0:
        return
    mb = mask & band
    band_val = base_claim[mb].float().mean().item() if int(mb.sum()) >= 20 else float("nan")
    base = base_claim[mask].float().mean().item()
    rest = rest2_claim[mask].float().mean().item()
    print(f"{name:>10s} {int(mask.sum()):5d} {base:8.3f} {rest:8.3f} {base-rest:8.3f} {band_val:13.3f}")

row("all", torch.ones(len(bsel), dtype=torch.bool))
for d in range(510115, 510125):
    row(str(d), DS[bsel] == d)

for h in handles:
    h.remove()
print("\ndone.")

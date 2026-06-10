"""H1 round 2c: stratify the jet-masking flips by the TRUE hadronic-W topology.

Sid's hypothesis (2026-06-10): the lep/nu verdict = NOT(some jet is the H+'s W).
Jets compute their own W-candidacy; lep/nu read the aggregate answer.

Predictions tested here:
  A. qqbb strata by truth-2 location: resolved (sjets), boosted (W-ljet), partial/mixed.
     mask-sjets should flip RESOLVED events specifically; mask-ljets the BOOSTED ones.
  B. Surgical: masking ONLY the truth-2 jets should flip ~everything; masking the same
     NUMBER of non-W jets (sham) should flip ~nothing.
  C. Prediction-level XOR: P(lep=W) should anticorrelate with "some jet predicted W",
     event-by-event, in baseline AND under corruption (mediation readout).

Usage: .venv/bin/python tmp_h1_round2c.py [--model ...] [--batches 6]
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
PAD, NU, SJ, LJ = 5, 2, 4, 3  # types: 3=LJET, 4=SJET (RunLowLevelInterp.py:77-80) — was swapped in the first run of this script!
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
    out = []
    with torch.no_grad():
        for c in range(0, len(x), 2048):
            out.append(model(x[c:c + 2048, :, :5], types[c:c + 2048]))
    return torch.cat(out)

# ---------- flat event tensors ----------
Xs, Ts = [], []
for _ in range(ARGS.batches):
    b = next(dl)
    types = b["types"]
    nu_m = types == NU; lep_m = (types == 0) | (types == 1)
    ok = (nu_m.sum(1) == 1) & (lep_m.sum(1) == 1)
    Xs.append(b["x"][ok].clone()); Ts.append(types[ok].clone())
X = torch.cat(Xs); T = torch.cat(Ts); N = len(X)
TRU = torch.round(X[..., -1]).long()
ar = torch.arange(N)
npos = (T == NU).float().argmax(1); lpos = ((T == 0) | (T == 1)).float().argmax(1)
lvbb = TRU[ar, lpos] == 3
jet_m = (T == SJ) | (T == LJ)
sjW = ((T == SJ) & (TRU == 2)).sum(1); ljW = ((T == LJ) & (TRU == 2)).sum(1)

# strata (meaningful for qqbb; lvbb should have no truth-2 jets)
strata = {
    "resolved (>=2 sjW, 0 ljW)": (~lvbb) & (sjW >= 2) & (ljW == 0),
    "partial (1 sjW, 0 ljW)":    (~lvbb) & (sjW == 1) & (ljW == 0),
    "boosted (>=1 ljW, 0 sjW)":  (~lvbb) & (ljW >= 1) & (sjW == 0),
    "mixed (sjW>0 & ljW>0)":     (~lvbb) & (sjW > 0) & (ljW > 0),
    "no W-labeled jet":          (~lvbb) & (sjW == 0) & (ljW == 0),
}
print(f"model={ARGS.model}  events={N} (lvbb {lvbb.sum()} / qqbb {(~lvbb).sum()})")
print(f"lvbb events with any truth-2 jet: {((lvbb)&((sjW+ljW)>0)).sum().item()} (sanity, expect 0)")
print("qqbb strata:")
for k, m in strata.items():
    print(f"  {k:28s} {m.sum().item():5d}  ({m.sum().item()/(~lvbb).sum().item()*100:.1f}% of qqbb)")

base_out = fwd(X, T)
base_pred = base_out.argmax(-1)
b_lep = base_pred[ar, lpos]; b_nu = base_pred[ar, npos]

def flip_table(name, X2, T2):
    pred = fwd(X2, T2).argmax(-1)
    f = pred[ar, npos] != b_nu
    print(f"\n  {name}")
    print(f"    {'lvbb':28s} flip {f[lvbb].float().mean():.4f}  (n={lvbb.sum()})")
    for k, m in strata.items():
        if m.sum() > 30:
            print(f"    {k:28s} flip {f[m].float().mean():.4f}  (n={m.sum()})")
    return pred, f

def mask_where(m):
    X2, T2 = X.clone(), T.clone()
    X2[m] = 0.0; T2[m] = PAD
    return X2, T2

print("\n=== B. flips by stratum (nu pred flip vs baseline) ===")
pred_msj, f_msj = flip_table("mask ALL sjets", *mask_where(T == SJ))
pred_mlj, f_mlj = flip_table("mask ALL ljets", *mask_where(T == LJ))
pred_w, f_w = flip_table("SURGICAL: mask only truth-2 jets", *mask_where(jet_m & (TRU == 2)))

# sham: per event mask as many random NON-W jets as that event has W jets
rng = np.random.default_rng(0)
m_sham = torch.zeros_like(T, dtype=torch.bool)
nW = (jet_m & (TRU == 2)).sum(1)
for i in range(N):
    k = nW[i].item()
    if k == 0: continue
    cand = torch.where(jet_m[i] & (TRU[i] != 2))[0].numpy()
    if len(cand) == 0: continue
    pick = rng.choice(cand, size=min(k, len(cand)), replace=False)
    m_sham[i, pick] = True
X2, T2 = mask_where(m_sham)
pred_s = fwd(X2, T2).argmax(-1)
f_s = pred_s[ar, npos] != b_nu
print(f"\n  SHAM: mask same # of non-W jets (only events with >=1 W jet & >=1 non-W jet)")
has_sham = m_sham.any(1)
for k, m in strata.items():
    mm = m & has_sham
    if mm.sum() > 30:
        print(f"    {k:28s} flip {f_s[mm].float().mean():.4f}  (n={mm.sum()})")
print(f"    {'lvbb (no W jets -> no sham)':28s} n/a")

# ---------- C. prediction-level XOR / mediation ----------
def any_jet_W(pred, T2):
    return ((pred == W_CLASS) & ((T2 == SJ) | (T2 == LJ))).any(1)

print("\n=== C. event-level XOR: 'some jet predicted W' vs 'lep predicted W' ===")
ajw = any_jet_W(base_pred, T)
for lab, m in (("lvbb", lvbb), ("qqbb", ~lvbb)):
    a = ajw[m]; l = (b_lep == W_CLASS)[m]
    both = (a & l).float().mean(); neither = (~a & ~l).float().mean()
    print(f"  {lab}: P(jetW)={a.float().mean():.4f} P(lepW)={l.float().mean():.4f} "
          f"P(both)={both:.4f} P(neither)={neither:.4f} P(exactly one)={1-both-neither:.4f}")

print("\n  mediation under 'mask ALL sjets' (qqbb only):")
X2, T2 = mask_where(T == SJ)
ajw2 = any_jet_W(pred_msj, T2)
qq = ~lvbb
ct = np.zeros((2, 2), dtype=int)
for a in (0, 1):
    for f in (0, 1):
        ct[a, f] = ((ajw2 == bool(a)) & (f_msj == bool(f)) & qq).sum().item()
print(f"    rows: remaining-jet-predicted-W after mask (no/yes); cols: nu flipped (no/yes)")
print(f"    {ct[0]}   flip rate | no jetW:  {ct[0,1]/max(ct[0].sum(),1):.4f}")
print(f"    {ct[1]}   flip rate | yes jetW: {ct[1,1]/max(ct[1].sum(),1):.4f}")

print("\n  mediation under SURGICAL truth-2 mask (qqbb only):")
X2, T2 = mask_where(jet_m & (TRU == 2))
ajw3 = any_jet_W(pred_w, T2)
ct = np.zeros((2, 2), dtype=int)
for a in (0, 1):
    for f in (0, 1):
        ct[a, f] = ((ajw3 == bool(a)) & (f_w == bool(f)) & qq).sum().item()
print(f"    {ct[0]}   flip rate | no jetW:  {ct[0,1]/max(ct[0].sum(),1):.4f}")
print(f"    {ct[1]}   flip rate | yes jetW: {ct[1,1]/max(ct[1].sum(),1):.4f}")

for h in handles: h.remove()
print("\ndone.")

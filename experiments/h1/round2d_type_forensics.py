"""Data forensics: verify the type mapping (3=ljet, 4=sjet) from kinematics alone,
and recompute the qqbb truth-W strata with correct labels.

No model needed - pure data. Masses in GeV (memmap units are MeV/1e5).
"""
import os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
DSIDS = list(range(510115, 510125))
PAD, NU = 5, 2
T3, T4 = 3, 4   # to be identified
stds = np.ones(7); stds[:4] = 1e5

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)

Xs, Ts = [], []
for _ in range(6):
    b = next(dl)
    types = b["types"]
    ok = ((types == NU).sum(1) == 1) & (((types == 0) | (types == 1)).sum(1) == 1)
    Xs.append(b["x"][ok].clone()); Ts.append(types[ok].clone())
X = torch.cat(Xs); T = torch.cat(Ts); N = len(X)
TRU = torch.round(X[..., -1]).long()
ar = torch.arange(N)
lpos = ((T == 0) | (T == 1)).float().argmax(1)
lvbb = TRU[ar, lpos] == 3

# invariant mass per object, in GeV (units: MeV/1e5 -> *1e5/1e3)
p2 = (X[..., :3] ** 2).sum(-1)
m = torch.sqrt(torch.clamp(X[..., 3] ** 2 - p2, min=0)) * 100.0
pt = torch.sqrt(X[..., 0] ** 2 + X[..., 1] ** 2) * 100.0

print(f"events: {N} (lvbb {lvbb.sum()}, qqbb {(~lvbb).sum()})")
print("\n=== per-type kinematics (which is the large-R jet?) ===")
for tt in (0, 1, NU, T3, T4):
    mm = T == tt
    print(f"  type {tt}: n/evt={mm.sum().item()/N:.2f}  "
          f"mass GeV: median {m[mm].median():7.1f}  mean {m[mm].mean():7.1f}  "
          f"pT GeV: median {pt[mm].median():7.1f}")

print("\n=== truth-2 (W-hadronic) objects: mass by type ===")
for tt in (T3, T4):
    mm = (T == tt) & (TRU == 2)
    if mm.sum() > 0:
        print(f"  type {tt} & truth-2: n={mm.sum().item()}  "
              f"mass GeV: median {m[mm].median():7.1f}  16-84%: "
              f"{np.percentile(m[mm].numpy(), 16):.1f}-{np.percentile(m[mm].numpy(), 84):.1f}")
print("=== truth-1 (Higgs) objects: mass by type (expect ~125 on the ljet type) ===")
for tt in (T3, T4):
    mm = (T == tt) & (TRU == 1)
    if mm.sum() > 0:
        print(f"  type {tt} & truth-1: n={mm.sum().item()}  "
              f"mass GeV: median {m[mm].median():7.1f}")

print("\n=== qqbb strata, CORRECT labels (3=ljet, 4=sjet) ===")
ljW = ((T == 3) & (TRU == 2)).sum(1)   # type 3 = LJET
sjW = ((T == 4) & (TRU == 2)).sum(1)   # type 4 = SJET
qq = ~lvbb
strata = {
    "boosted: W = 1 ljet":        qq & (ljW == 1) & (sjW == 0),
    "boosted: W = 2+ ljets (?)":  qq & (ljW >= 2) & (sjW == 0),
    "resolved: W = 2 sjets":      qq & (sjW == 2) & (ljW == 0),
    "semi: W = 1 sjet":           qq & (sjW == 1) & (ljW == 0),
    "W = 3+ sjets (?)":           qq & (sjW >= 3) & (ljW == 0),
    "mixed ljet+sjet":            qq & (sjW > 0) & (ljW > 0),
    "no truth-2 jet":             qq & (sjW == 0) & (ljW == 0),
}
for k, mm in strata.items():
    print(f"  {k:26s} {mm.sum().item():5d}  ({mm.sum().item()/qq.sum().item()*100:.1f}%)")

# resolved events: dijet mass of the two truth-2 sjets (expect ~mW)
res = strata["resolved: W = 2 sjets"]
if res.sum() > 10:
    mjj = []
    for i in torch.where(res)[0]:
        rows = torch.where((T[i] == 4) & (TRU[i] == 2))[0]
        p4 = X[i, rows, :4].sum(0)
        mjj.append((torch.sqrt(torch.clamp(p4[3]**2 - (p4[:3]**2).sum(), min=0)) * 100).item())
    mjj = np.array(mjj)
    print(f"\n  resolved W=2sjets: m(jj) median {np.median(mjj):.1f} GeV "
          f"(16-84%: {np.percentile(mjj,16):.1f}-{np.percentile(mjj,84):.1f}) [expect ~80]")
# semi events: mass of the single truth-2 sjet
semi = strata["semi: W = 1 sjet"]
if semi.sum() > 10:
    mm = torch.zeros(0)
    rows = (T == 4) & (TRU == 2) & semi.unsqueeze(1)
    print(f"  semi W=1sjet: sjet mass median {m[rows].median():.1f} GeV "
          f"(16-84%: {np.percentile(m[rows].numpy(),16):.1f}-{np.percentile(m[rows].numpy(),84):.1f})")
# boosted: mass of the truth-2 ljet
boo = strata["boosted: W = 1 ljet"]
rows = (T == 3) & (TRU == 2) & boo.unsqueeze(1)
print(f"  boosted W-ljet mass: median {m[rows].median():.1f} GeV "
      f"(16-84%: {np.percentile(m[rows].numpy(),16):.1f}-{np.percentile(m[rows].numpy(),84):.1f}) [expect ~80]")
rows = (T == 3) & (TRU == 1)
print(f"  H-ljet (truth-1) mass: median {m[rows].median():.1f} GeV [expect ~125]")

# per-mass-point composition (dsids 510115..510124 = 0.8..3.0 TeV)
print("\n=== qqbb strata fraction vs mass point ===")
# recover dsid per event by re-iterating loader? use weights? simpler: reload with dsids
dl2 = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)
DS = []
for _ in range(6):
    b = next(dl2)
    types = b["types"]
    ok = ((types == NU).sum(1) == 1) & (((types == 0) | (types == 1)).sum(1) == 1)
    DS.append(b["dsids"][ok].clone())
DS = torch.cat(DS)
masses = {510115: 0.8, 510116: 0.9, 510117: 1.0, 510118: 1.2, 510119: 1.4,
          510120: 1.6, 510121: 1.8, 510122: 2.0, 510123: 2.5, 510124: 3.0}
print(f"  {'mass':>5s} {'n_qq':>6s} {'W=ljet':>7s} {'W=2sj':>6s} {'W=1sj':>6s} {'mixed':>6s}")
for d, mt in masses.items():
    sel = qq & (DS == d)
    if sel.sum() == 0: continue
    print(f"  {mt:5.1f} {sel.sum().item():6d} "
          f"{(strata['boosted: W = 1 ljet'] & sel).sum().item()/sel.sum().item():7.2f} "
          f"{(strata['resolved: W = 2 sjets'] & sel).sum().item():6d} "
          f"{(strata['semi: W = 1 sjet'] & sel).sum().item()/sel.sum().item():6.2f} "
          f"{(strata['mixed ljet+sjet'] & sel).sum().item():6d}")
print("\ndone.")

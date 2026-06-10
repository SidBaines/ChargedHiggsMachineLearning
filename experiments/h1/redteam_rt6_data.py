"""RT6: data-hygiene checks for the H1 wave analyses (red-team session 2026-06-11).

1. eventNumber parity split: float32 storage sanity, even/odd fractions per DSID.
2. DSID + mass-point composition of analysis slices A (batches 1-6), B (7-12),
   C (13-18) vs the full val population; slices disjointness double-check.
3. Strata exhaustiveness: does boosted|resolved partition qqbb?
4. Kinematic re-verification of the types_dict (the round-2d guard).
"""
import os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
DSIDS = list(range(510115, 510125))
PAD, NU, LJ, SJ = 5, 2, 3, 4

print("=== 1. eventNumber parity / float32 sanity (full memmaps) ===")
print(f"  {'dsid':>7s} {'N':>8s} {'frac even':>10s} {'frac int':>9s} {'max evtNum':>12s} {'mH':>7s}")
for d in DSIDS:
    path = f"{DATA}dsid_{d}.memmap"
    with open(path + ".shape") as f:
        n = int(f.read().split(",")[0])
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(n, 17, 8))
    ev = np.asarray(mm[:, 1, 2], dtype=np.float64)
    mh = np.asarray(mm[:, 0, 1], dtype=np.float64)
    frac_even = (ev % 2 == 0).mean()
    frac_int = np.isclose(ev, np.round(ev)).mean()
    print(f"  {d:7d} {n:8d} {frac_even:10.4f} {frac_int:9.4f} {ev.max():12.0f} {np.median(mh):7.0f}")

print("\n=== 2./3. slice composition & strata coverage (loader as used by the waves) ===")
stds = np.ones(7); stds[:4] = 1e5
dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)

slices = {"A(1-6)": [], "B(7-12)": [], "C(13-18)": []}
for sname in slices:
    for _ in range(6):
        b = next(dl)
        slices[sname].append(b)

val_total = {d: ((np.memmap(f"{DATA}dsid_{d}.memmap", dtype=np.float32, mode="r",
                            shape=(int(open(f"{DATA}dsid_{d}.memmap.shape").read().split(',')[0]), 17, 8))[:, 1, 2] % 2) == 0).sum()
             for d in DSIDS}
tot = sum(val_total.values())
print(f"  val population: {tot} events; per-dsid fractions: "
      + " ".join(f"{d % 100:02d}:{val_total[d]/tot:.3f}" for d in DSIDS))
for sname, bs in slices.items():
    ds = torch.cat([b["dsids"] for b in bs]).numpy().astype(int)
    comp = " ".join(f"{d % 100:02d}:{(ds == d).mean():.3f}" for d in DSIDS)
    print(f"  slice {sname:8s} n={len(ds)}  {comp}")

print("\n  strata coverage + filter pass-rate per slice (after the waves' ok-filter):")
for sname, bs in slices.items():
    X = torch.cat([b["x"] for b in bs]); T = torch.cat([b["types"] for b in bs])
    ok = ((T == NU).sum(1) == 1) & (((T == 0) | (T == 1)).sum(1) == 1)
    Xo, To = X[ok], T[ok]
    TRU = torch.round(Xo[..., -1]).long()
    ar = torch.arange(len(Xo))
    lpos = ((To == 0) | (To == 1)).float().argmax(1)
    lvbb = TRU[ar, lpos] == 3
    boosted = (~lvbb) & (((To == LJ) & (TRU == 2)).sum(1) == 1)
    resolved = (~lvbb) & (((To == SJ) & (TRU == 2)).sum(1) == 2)
    neither = (~lvbb) & ~boosted & ~resolved
    print(f"  slice {sname:8s} ok={ok.float().mean():.4f}  lvbb={lvbb.float().mean():.4f} "
          f"boosted={boosted.float().mean():.4f} resolved={resolved.float().mean():.4f} "
          f"qqbb-NEITHER={neither.float().mean():.5f} (n={neither.sum().item()})")

print("\n=== 4. types_dict kinematic guard (slice A) ===")
X = torch.cat([b["x"] for b in slices["A(1-6)"]]); T = torch.cat([b["types"] for b in slices["A(1-6)"]])
m_obj = torch.sqrt(torch.clamp(X[..., 3] ** 2 - (X[..., :3] ** 2).sum(-1), min=0)) * 100
pt_obj = torch.sqrt(X[..., 0] ** 2 + X[..., 1] ** 2) * 100
for t, nm in ((0, "ele"), (1, "mu"), (2, "nu"), (3, "LJET?"), (4, "SJET?"), (5, "pad")):
    mm_ = T == t
    if mm_.sum() == 0:
        print(f"  type {t} ({nm}): none"); continue
    print(f"  type {t} ({nm:5s}): n/evt={mm_.float().sum(1).mean():.2f}  "
          f"med mass={m_obj[mm_].median():6.1f} GeV  med pT={pt_obj[mm_].median():6.1f} GeV")
print("\ndone.")

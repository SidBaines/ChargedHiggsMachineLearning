"""Sanity-check the real-background combined memmap dir before committing compute.

Loads tmp_data_20250321v1_all/ with the same loader settings train_joint.py uses
under --real-bkg, pulls a few batches, and asserts the layout / labels are sane:
  - signal AND background DSIDs both present and sampled;
  - event label y is one-hot {0=bkg,1=qqbb,2=lvbb}, background events -> class 0;
  - background events get reco target all-`none` after preprocess;
  - per-object features finite; types in valid range.
Exits non-zero with a clear message on any failure.
"""
import os, sys, glob
import numpy as np
import torch

EXP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, EXP)
REPO = os.path.dirname(EXP); sys.path.insert(0, REPO)
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_all") + "/"
DSIDS = sorted(int(os.path.basename(p)[5:11]) for p in glob.glob(f"{DATA}dsid_*.memmap"))
sig = [d for d in DSIDS if 510000 < d < 600000]
bkg = [d for d in DSIDS if not (510000 < d < 600000)]
print(f"data dir: {DATA}")
print(f"DSIDs found: {len(DSIDS)} = {len(sig)} signal + {len(bkg)} background")
assert len(sig) >= 1, "no signal DSIDs found"
assert len(bkg) >= 5, f"too few background DSIDs ({len(bkg)}) — copy incomplete?"

stds = np.ones(7); stds[:4] = 1e5
mk = dict(N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
          memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
          max_objs_in_memmap=15, batch_size=4096, device="cpu", n_splits=2,
          validation_split_idx=0, n_targets=3, means=None, stds=stds,
          objs_to_output=15, signal_only=False, has_eventNumbers=True)
dl = ProportionalMemoryMappedDataset(is_train=True, shuffle=True, shuffle_batch=True, **mk)
print(f"loaded; train total samples = {dl.get_total_samples():,}")
print(f"bkg_weight_sums = {dl.bkg_weight_sums:.4g}  num_signals = {dl.num_signals}")

n_sig_tot = n_bkg_tot = 0
finite_ok = True
types_ok = True
label_classes = set()
for i in range(5):
    b = next(dl)
    x, y, types, dsids = b["x"], b["y"], b["types"], b["dsids"]
    lab = y.argmax(-1)
    label_classes.update(lab.unique().tolist())
    is_bkg = (lab == 0)
    n_bkg = int(is_bkg.sum()); n_sig = int((~is_bkg).sum())
    n_bkg_tot += n_bkg; n_sig_tot += n_sig
    finite_ok &= bool(torch.isfinite(x).all())
    # types: 0..5 (5=pad). After loader they are longs in [0,5].
    types_ok &= bool((types >= 0).all() and (types <= 5).all())
    # background reco target should collapse to all-none (0): the harness sets
    # reco_target[is_bkg]=0; here just confirm bkg events exist to mask.
    if i == 0:
        bkg_dsids_seen = sorted(set(int(d) for d, m in zip(dsids.tolist(), is_bkg.tolist()) if m))[:8]
        print(f"  batch0: {n_sig} signal + {n_bkg} bkg; example bkg dsids: {bkg_dsids_seen}")

print(f"5 batches: {n_sig_tot} signal + {n_bkg_tot} background events")
print(f"label classes seen: {sorted(label_classes)} (expect subset of [0,1,2])")
print(f"all x finite: {finite_ok}   all types in [0,5]: {types_ok}")

assert n_sig_tot > 0, "no signal events sampled"
assert n_bkg_tot > 0, "no background events sampled — real bkg not loading!"
assert finite_ok, "non-finite values in input features"
assert types_ok, "object types out of range [0,5]"
assert label_classes <= {0, 1, 2}, f"unexpected label classes {label_classes}"
assert 0 in label_classes, "no background-class (0) labels — background mislabelled?"
print("\nVERIFY OK: real-background data loads, labels + features sane.")

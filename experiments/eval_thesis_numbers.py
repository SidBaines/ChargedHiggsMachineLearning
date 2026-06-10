"""Phase 0 sanity gate: run the recovered thesis model on its ACTUAL validation data
(20250321v1 signal memmaps) and compare val PerfectRecoPct per mass against the wandb
targets archived in docs/run_configs/20250512-093728_DSSARVTSBN3_YesEnt1_YesBn.json.

Conventions mirrored from RunLowLevelInterp.py + TrainLowLevelReconstruction.py:
- normalization: SCALE_DATA (4-momenta / 1e5, no mean subtraction, no mean/std files)
- 15 objects, N_Real_Vars_InFile=7, model input x[...,:5], truth x[...,-1]
- split: n_splits=2, validation_split_idx=0, has_eventNumbers=True (the TRAINING split)
"""
import os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from models.registry import load_model
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from metrics.lowlevelrecometrics import HEPMetrics

_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_data_20250321v1_signal/")
_DRIVE = ("/Volumes/Seagate/heppc_recovered/data/"
          "20250321v1_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5_NotPhiRotated_"
          "WithRecoMasses_15_MetCut_RemovedUncertainTruth_WithTagInfo_"
          "KeepAllOldSelIncludingNegative_RemovedEventsWhereTruthIsCutByMaxObjs/")
DATA = _LOCAL if os.path.exists(os.path.join(_LOCAL, "dsid_510115.memmap")) else _DRIVE
DSIDS = list(range(510115, 510125))
N_CTX, MAX_OBJS, NV_FILE = 6, 15, 7
BATCH = 3000

WANDB_VAL_TARGETS = {  # val/PerfectRecoPct_<mass> at epoch 29 (run 20250512-093728)
    "0.8": 0.6975, "0.9": 0.7485, "1.0": 0.7842, "1.2": 0.8277, "1.4": 0.8560,
    "1.6": 0.8760, "1.8": 0.8904, "2.0": 0.9023, "2.5": 0.9204, "3.0": 0.9316,
}

missing = [d for d in DSIDS if not os.path.exists(f"{DATA}dsid_{d}.memmap")]
if missing:
    sys.exit(f"NOT READY: missing signal memmaps for {missing} — run tmp_fetch_signal_norms.sh")

stds = np.ones(NV_FILE); stds[:4] = 1e5  # SCALE_DATA convention
dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=NV_FILE, N_Real_Vars_To_Return=NV_FILE,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
    max_objs_in_memmap=MAX_OBJS, batch_size=BATCH, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=MAX_OBJS, signal_only=True, has_eventNumbers=True,
)
print(f"val samples: {dl.get_total_samples()}")

model, _ = load_model("thesis-ent1-bn1-d152", checkpoint_root=os.path.join(REPO, "tmp_checkpoints"))

metrics = HEPMetrics(
    N_CTX - 1, MAX_OBJS, is_categorical=True, num_categories=3,
    max_bkg_levels=[100, 200], max_buffer_len=int(dl.get_total_samples()),
    total_weights_per_dsid=dl.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000],
)
metrics.reset()
dl._reset_indices()

n_batches = len(dl) - 1
for i in range(n_batches):
    batch = next(dl)
    x, types, dsids, MCWts = batch["x"], batch["types"], batch["dsids"], batch["MC_Wts"]
    with torch.no_grad():
        out = model(x[..., :5], types)
    metrics.update(out.squeeze(), x[..., -1], MCWts, dsids, types)
    if i % 10 == 9:
        print(f"  batch {i+1}/{n_batches}")

rv = metrics.compute_and_log(1, "val", 0, 3, False, None, calc_all=True)

print("\n=== val PerfectRecoPct: recovered model vs wandb (epoch 29) ===")
print(f"{'mass':>6s} {'now':>8s} {'wandb':>8s} {'diff':>8s}")
for mass, target in WANDB_VAL_TARGETS.items():
    key = next((k for k in rv if k.endswith(f"PerfectRecoPct_{mass}")), None)
    now = rv.get(key, float("nan")) if key else float("nan")
    print(f"{mass:>6s} {now:8.4f} {target:8.4f} {now-target:+8.4f}")
print("\nper-category reco:")
for k in sorted(rv):
    if "tRecoPct_all_cat" in k:
        print(f"  {k}: {rv[k]:.4f}")

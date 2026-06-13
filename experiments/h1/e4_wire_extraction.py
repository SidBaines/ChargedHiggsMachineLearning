"""E4 (part 1): symbolic extraction of the verdict wires — fit the b2h2 ("W found")
and b1h3 (hadronic-vs-leptonic comparator) bottleneck scalars at the nu position as
formulas over NAMED physics features, on the thesis model.

Feature set is informed by the whole H1 programme (candidacy = relative hardness x
m2-window x anti-tag; leptonic evidence = magnitudes, lepton-weighted, ratio to jet
HT — LW1/RT4): per-event scalars only, no raw 4-vectors. Round-1 PySR underwhelmed
by searching blind; this is the informed retry. Linear ridge on the same features
is reported as the baseline PySR must beat.

By default, targets are fit on val batches 19-22 and scored on 23-24. Override
--skip/--batches/--train-batches to move the extraction to a fresh slice; this is
important because the default feature set was discovered during the H1 programme.
Part 2 (e4_wire_replacement.py) splices the fitted formulas back into the network
via the override hooks and measures verdict agreement + task retention.

Usage: .venv/bin/python experiments/h1/e4_wire_extraction.py [--niterations 40]
       [--skip 18] [--batches 6] [--train-batches 4]
Writes tmp_pysr/<target>_equations.csv + a summary.
"""
import os, sys, argparse
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--niterations", type=int, default=40)
ap.add_argument("--maxsize", type=int, default=22)
ap.add_argument("--procs", type=int, default=4)
ap.add_argument("--skip", type=int, default=18,
                help="number of validation-loader batches to skip before collecting data")
ap.add_argument("--batches", type=int, default=6,
                help="number of validation-loader batches to collect")
ap.add_argument("--train-batches", type=int, default=4,
                help="first k collected batches used for fitting; remainder is held out")
ARGS = ap.parse_args()
assert 0 < ARGS.train_batches < ARGS.batches, "--train-batches must be in (0, --batches)"
torch.set_num_threads(4)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
PAD, NU, LJ, SJ = 5, 2, 3, 4
stds = np.ones(7); stds[:4] = 1e5
torch.manual_seed(0)

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in range(510115, 510125)},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)
for _ in range(ARGS.skip):
    next(dl)

model, _ = load_model("thesis-ent1-bn1-d152", checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False)
model.eval()
cache = ActivationCache()
handles = [m.register_forward_hook(fn, with_kwargs=True)
           for m, fn in hook_attention_heads(model, cache, detach=True,
                 SINGLE_ATTENTION=False, bottleneck_attention_output=1)]

Xs, Ts = [], []
for _ in range(ARGS.batches):
    b = next(dl)
    t = b["types"]
    ok = ((t == NU).sum(1) == 1) & (((t == 0) | (t == 1)).sum(1) == 1)
    Xs.append(b["x"][ok].clone()); Ts.append(t[ok].clone())
X = torch.cat(Xs); T = torch.cat(Ts); N = len(X)
ar = torch.arange(N)
npos = (T == NU).float().argmax(1); lpos = ((T == 0) | (T == 1)).float().argmax(1)

# ---- targets: wire scalars at nu ----
s22, s13 = [], []
with torch.no_grad():
    for c in range(0, N, 2048):
        sl = slice(c, min(c + 2048, N))
        model(X[sl][..., :5], T[sl])
        car = torch.arange(sl.stop - sl.start)
        s13.append(cache["block_1_attention"]["bottleneck_activation"][car, 3, npos[sl], 0])
        s22.append(cache["block_2_attention"]["bottleneck_activation"][car, 2, npos[sl], 0])
y_b1h3 = torch.cat(s13).numpy(); y_b2h2 = torch.cat(s22).numpy()

# ---- named physics features (units of 100 GeV; m2 in (100 GeV)^2 — what the model sees) ----
def pt_of(rows): return torch.sqrt((rows[..., :2] ** 2).sum(-1))
plep = X[ar, lpos]; pnu = X[ar, npos]
pt_lep = pt_of(plep); pt_nu = pt_of(pnu)
pt_lepW = torch.sqrt(((plep[:, :2] + pnu[:, :2]) ** 2).sum(-1))
isl = (T == LJ); iss = (T == SJ)
pt_all = pt_of(X)
m2_all = torch.clamp(X[..., 3] ** 2 - (X[..., :3] ** 2).sum(-1), min=0)
ht_j = (pt_all * (isl | iss).float()).sum(1)
# leading and subleading ljet (by pT)
ptl = torch.where(isl, pt_all, torch.zeros_like(pt_all))
v1, i1 = ptl.max(1)
ptl2 = ptl.scatter(1, i1.unsqueeze(1), 0.0)
v2, i2 = ptl2.max(1)
m2_1 = m2_all[ar, i1]; tag_1 = X[ar, i1, 4]
m2_2 = m2_all[ar, i2]; tag_2 = X[ar, i2, 4]
# best W-window ljet: pT among ljets with m in [60,100] GeV (m2 in [0.36,1.0])
inwin = isl & (m2_all > 0.36) & (m2_all < 1.00)
ptw = torch.where(inwin, pt_all, torch.zeros_like(pt_all))
vW, iW = ptw.max(1)
tag_W = torch.where(vW > 0, X[ar, iW, 4], torch.zeros_like(vW))
n_lj = isl.sum(1).float()

FEATS = {
    "pt_lep": pt_lep, "pt_nu": pt_nu, "pt_lepW": pt_lepW,
    "ht_jets": ht_j, "n_ljets": n_lj,
    "pt_lj1": v1, "m2_lj1": m2_1, "tag_lj1": tag_1,
    "pt_lj2": v2, "m2_lj2": m2_2, "tag_lj2": tag_2,
    "pt_Wwin": vW, "tag_Wwin": tag_W,
    "r_lepW_ht": pt_lepW / ht_j.clamp(min=0.01),
    "r_lj1_ht": v1 / ht_j.clamp(min=0.01),
}
Xf = np.stack([v.numpy() for v in FEATS.values()], axis=1).astype(np.float64)
names = list(FEATS.keys())
print(f"events {N}, features {len(names)}: {names}")
print(f"slice protocol: skipped {ARGS.skip}, collected {ARGS.batches} batches, "
      f"fit first {ARGS.train_batches}, test last {ARGS.batches - ARGS.train_batches}")

ntr = int(N * ARGS.train_batches / ARGS.batches)
idx = np.arange(N)                            # loader order == batch order
tr, te = idx[:ntr], idx[ntr:]

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
os.makedirs(os.path.join(REPO, "tmp_pysr"), exist_ok=True)
from pysr import PySRRegressor

for tgt_name, y in (("b2h2_nu", y_b2h2), ("b1h3_nu", y_b1h3)):
    ridge = Ridge(alpha=1.0).fit(Xf[tr], y[tr])
    r2_lin = r2_score(y[te], ridge.predict(Xf[te]))
    print(f"\n=== target {tgt_name}: ridge baseline R2(test) = {r2_lin:.4f} ===")
    mdl = PySRRegressor(
        niterations=ARGS.niterations, maxsize=ARGS.maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "sqrt", "tanh"],
        elementwise_loss="loss(y, yp) = (y - yp)^2",
        procs=ARGS.procs, populations=24, deterministic=False,
        verbosity=0, progress=False,
        output_directory=os.path.join(REPO, "tmp_pysr"), run_id=tgt_name,
    )
    mdl.fit(Xf[tr], y[tr], variable_names=names)
    yp = mdl.predict(Xf[te])
    r2_sym = r2_score(y[te], yp)
    print(f"PySR best (R2 test {r2_sym:.4f}, vs ridge {r2_lin:.4f}):")
    print(f"  {mdl.get_best()['equation']}")
    eqs = mdl.equations_[["complexity", "loss", "equation"]]
    print("pareto front (top 8 by complexity):")
    for _, row in eqs.iloc[:8].iterrows():
        print(f"  c={row['complexity']:2.0f} loss={row['loss']:.4f}  {row['equation']}")
    eqs.to_csv(os.path.join(REPO, "tmp_pysr", f"{tgt_name}_equations.csv"), index=False)

for h in handles: h.remove()
print("\ndone — equations in tmp_pysr/")

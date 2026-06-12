"""E4 (part 2): the faithfulness test — splice the PySR-extracted wire formulas
back INTO the network and measure what survives.

The verdict wires (b2h2, b1h3) at the nu AND lepton positions are overwritten with
the symbolic formulas fitted in e4_wire_extraction.py (evaluated from named physics
features of the event), using the exact override-hook math of wave2_wires.py
(add (target - s_cur) * w_up at the position; the hook chain applies bottleneck
first, override second). Everything else in the network stays, so this is a
readout-level behavioral-compression test, not a full symbolic replacement of the
model.

Conditions: intact | replace b2h2 | replace b1h3 | replace BOTH | mean-control
(wires set to their train-set means — destroys wire info; the floor).

Metrics (default held-out batches 23-24): nu/lep verdict agreement with the intact
model, channel accuracy vs truth, lockstep P(nu=lep). Override
--skip/--batches/--train-batches to evaluate on a fresh slice after refitting
or reusing compatible formulas.

Usage: .venv/bin/python experiments/h1/e4_wire_replacement.py
       [--pick-b2h2 best] [--pick-b1h3 best]   (or an integer complexity)
       [--skip 18] [--batches 6] [--train-batches 4]
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import sympy

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--pick-b2h2", default="best")
ap.add_argument("--pick-b1h3", default="best")
ap.add_argument("--skip", type=int, default=18,
                help="number of validation-loader batches to skip before collecting data")
ap.add_argument("--batches", type=int, default=6,
                help="number of validation-loader batches to collect")
ap.add_argument("--train-batches", type=int, default=4,
                help="first k collected batches define train-set means; remainder is held out")
ap.add_argument("--nan-policy", choices=["mean", "zero", "keep"], default="mean",
                help="how to handle non-finite formula outputs before splicing")
ARGS = ap.parse_args()
assert 0 < ARGS.train_batches < ARGS.batches, "--train-batches must be in (0, --batches)"
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
for _ in range(ARGS.skip):
    next(dl)

model, _ = load_model("thesis-ent1-bn1-d152", checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False)
model.eval()
cache = ActivationCache()
lib_handles = [m.register_forward_hook(fn, with_kwargs=True)
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
TRU = torch.round(X[..., -1]).long()
lvbb = TRU[ar, lpos] == 3

# ---- features (identical to e4_wire_extraction.py) ----
def pt_of(rows): return torch.sqrt((rows[..., :2] ** 2).sum(-1))
plep = X[ar, lpos]; pnu = X[ar, npos]
pt_lep = pt_of(plep); pt_nu = pt_of(pnu)
pt_lepW = torch.sqrt(((plep[:, :2] + pnu[:, :2]) ** 2).sum(-1))
isl = (T == LJ); iss = (T == SJ)
pt_all = pt_of(X)
m2_all = torch.clamp(X[..., 3] ** 2 - (X[..., :3] ** 2).sum(-1), min=0)
ht_j = (pt_all * (isl | iss).float()).sum(1)
ptl = torch.where(isl, pt_all, torch.zeros_like(pt_all))
v1, i1 = ptl.max(1)
ptl2 = ptl.scatter(1, i1.unsqueeze(1), 0.0)
v2, i2 = ptl2.max(1)
m2_1 = m2_all[ar, i1]; tag_1 = X[ar, i1, 4]
m2_2 = m2_all[ar, i2]; tag_2 = X[ar, i2, 4]
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
names = list(FEATS.keys())
Xf = {k: v.numpy().astype(np.float64) for k, v in FEATS.items()}
ntr = int(N * ARGS.train_batches / ARGS.batches)
te = np.arange(ntr, N)
print(f"slice protocol: skipped {ARGS.skip}, collected {ARGS.batches} batches, "
      f"train-mean first {ARGS.train_batches}, test last {ARGS.batches - ARGS.train_batches}")

# ---- true train-set wire means for the information-destroying control ----
true22, true13 = [], []
with torch.no_grad():
    for c in range(0, N, 2048):
        sl = slice(c, min(c + 2048, N))
        model(X[sl][..., :5], T[sl])
        car = torch.arange(sl.stop - sl.start)
        true13.append(cache["block_1_attention"]["bottleneck_activation"][car, 3, npos[sl], 0])
        true22.append(cache["block_2_attention"]["bottleneck_activation"][car, 2, npos[sl], 0])
true_b1h3 = torch.cat(true13)
true_b2h2 = torch.cat(true22)

# ---- load + evaluate the chosen formulas ----
def load_formula(tgt, pick):
    df = pd.read_csv(os.path.join(REPO, "tmp_pysr", f"{tgt}_equations.csv"))
    if pick == "best":
        row = df.iloc[df["loss"].idxmin()]
    else:
        cand = df[df["complexity"] <= int(pick)]
        row = cand.iloc[cand["loss"].idxmin()]
    eq = row["equation"]
    syms = {n: sympy.Symbol(n) for n in names}
    expr = sympy.sympify(eq, locals={**syms, "square": lambda x: x ** 2})
    fn = sympy.lambdify([syms[n] for n in names], expr, modules=["numpy"])
    vals = fn(*[Xf[n] for n in names]).astype(np.float32)
    print(f"{tgt} (pick={pick}, complexity {row['complexity']:.0f}, loss {row['loss']:.4f}):\n  {eq}")
    return torch.from_numpy(np.broadcast_to(vals, (N,)).copy())

f_b2h2 = load_formula("b2h2_nu", ARGS.pick_b2h2)
f_b1h3 = load_formula("b1h3_nu", ARGS.pick_b1h3)

# ---- override machinery (wave2_wires conventions; bottleneck hook runs FIRST) ----
attn_modules = [blk["self_attention"] for blk in model.attention_blocks]
OV = {"active": False, "cfg": {}, "positions": None}   # cfg: (blk,h)->target[n] (chunk-sliced)
def make_override(blk):
    def hook(module, args, kwargs, output):
        if not OV["active"]:
            return output
        heads = [(b, h) for (b, h) in OV["cfg"] if b == blk]
        if not heads:
            return output
        out, w = output
        out = out.clone()
        bidx = torch.arange(out.shape[0])
        for pos in OV["positions"]:
            for (b, h) in heads:
                tgt = OV["cfg"][(b, h)]
                s_cur = cache[f"block_{blk}_attention"]["bottleneck_activation"][bidx, h, pos, 0]
                w_up = model.attention_blocks[blk]["bottleneck_up"][h].weight[:, 0]
                out[bidx, pos] = out[bidx, pos] + (tgt - s_cur).unsqueeze(-1) * w_up
        return (out, w)
    return hook
ov_handles = [m.register_forward_hook(make_override(bi), with_kwargs=True)
              for bi, m in enumerate(attn_modules)]

def fwd_repl(cfg_full=None):
    """cfg_full: dict (blk,h) -> target tensor [N], applied at nu AND lep."""
    outs = []
    with torch.no_grad():
        for c in range(0, N, 2048):
            sl = slice(c, min(c + 2048, N))
            if cfg_full is None:
                OV["active"] = False
            else:
                OV.update(active=True,
                          cfg={k: v[sl] for k, v in cfg_full.items()},
                          positions=[npos[sl], lpos[sl]])
            outs.append(model(X[sl][..., :5], T[sl]))
    OV["active"] = False
    return torch.cat(outs)

# ---- conditions ----
out0 = fwd_repl(None)
pr0 = out0.argmax(-1)
nu0, lep0 = pr0[ar, npos], pr0[ar, lpos]
truth_nu = torch.where(lvbb, torch.full_like(nu0, W), torch.zeros_like(nu0))

mean22 = torch.full((N,), true_b2h2[:ntr].mean().item())
mean13 = torch.full((N,), true_b1h3[:ntr].mean().item())
def finite_mean(x):
    m = torch.isfinite(x)
    return x[m].mean().item() if m.any() else float("nan")

def sanitize_formula(name, vals, fill):
    bad = ~torch.isfinite(vals)
    n_bad = int(bad.sum())
    if n_bad:
        te_bad = int(bad[te].sum())
        print(f"WARNING: {name} formula produced {n_bad}/{len(vals)} non-finite values "
              f"({te_bad}/{len(te)} held-out); nan-policy={ARGS.nan_policy}")
        vals = vals.clone()
        if ARGS.nan_policy == "mean":
            vals[bad] = fill
        elif ARGS.nan_policy == "zero":
            vals[bad] = 0.0
        elif ARGS.nan_policy == "keep":
            pass
    return vals

print(f"true train-set wire means: b2h2={mean22[0].item():.4f}, b1h3={mean13[0].item():.4f} "
      f"(finite formula means were {finite_mean(f_b2h2[:ntr]):.4f}, {finite_mean(f_b1h3[:ntr]):.4f})")
f_b2h2 = sanitize_formula("b2h2", f_b2h2, mean22[0])
f_b1h3 = sanitize_formula("b1h3", f_b1h3, mean13[0])
CONDS = [
    ("intact", None),
    ("replace b2h2", {(2, 2): f_b2h2}),
    ("replace b1h3", {(1, 3): f_b1h3}),
    ("replace BOTH", {(2, 2): f_b2h2, (1, 3): f_b1h3}),
    ("mean-control BOTH", {(2, 2): mean22, (1, 3): mean13}),
]
print(f"\nheld-out events: {len(te)} (lvbb {lvbb[te].float().mean():.2f})")
print(f"{'condition':>20s} {'nu==intact':>11s} {'lep==intact':>12s} {'nu==truth':>10s} {'P(nu=lep)':>10s}")
for nm, cfg in CONDS:
    o = out0 if cfg is None else fwd_repl(cfg)
    pr = o.argmax(-1)
    nu_, lep_ = pr[ar, npos], pr[ar, lpos]
    print(f"{nm:>20s} {(nu_[te]==nu0[te]).float().mean():11.4f} "
          f"{(lep_[te]==lep0[te]).float().mean():12.4f} "
          f"{(nu_[te]==truth_nu[te]).float().mean():10.4f} "
          f"{(nu_[te]==lep_[te]).float().mean():10.4f}")
print(f"{'(intact vs truth)':>20s} {'':>11s} {'':>12s} {(nu0[te]==truth_nu[te]).float().mean():10.4f}")

for h in lib_handles + ov_handles: h.remove()
print("\ndone.")

"""F1: circuit-formation timeline — when does each element of the channel-decision
circuit form during training?

Runs a compact probe battery on EVERY checkpoint of an organism training run and
plots each circuit property vs epoch:

  task        P(true W-jet claims W) in boosted (candidacy quality)
  lockstep    P(pred_nu = pred_lep) (the shared-wiring signature)
  mask-flip   P(lep flips ->W) when the true W is masked (ask-the-jets broadcast)
  rel-hard    claim suppression under rest x2 (context-relative scoring, RT4)
  tie-both    P(both claim) at an exact tie (emergent exclusivity / comparator)
  tie-cost    winner's mean margin cost vs solo twin (SB6)
  x-system    P(ins claims) at pT = 1.0 / 1.4 / 2.0 x pT(lepW) in lvbb (A4b)

Questions: does candidacy precede the broadcast? does the comparator (tie-both
dropping from ~independence to the ranked level) form abruptly? does the
cross-system threshold drift?

Usage: .venv/bin/python experiments/h1/f1_formation_dynamics.py
       [--run output/20260610-203258_TrainingOutput] [--model ent1-d20-2blk]
       [--batches 6] [--skip 18] [--every 1]
Writes tmp_plots/f1_formation_<runstamp>.png + .csv
"""
import os, sys, glob, re, argparse, csv
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="output/20260610-203258_TrainingOutput")
ap.add_argument("--model", default="ent1-d20-2blk")
ap.add_argument("--batches", type=int, default=6)
ap.add_argument("--skip", type=int, default=18)
ap.add_argument("--every", type=int, default=1, help="probe every k-th checkpoint")
ap.add_argument("--step-range", default=None,
                help="only probe checkpoints with global step in [lo,hi], e.g. '0,1100'")
ARGS = ap.parse_args()
torch.set_num_threads(6)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
PAD, NU, LJ, SJ = 5, 2, 3, 4
W = 2
WTAG = -2.6
stds = np.ones(7); stds[:4] = 1e5
torch.manual_seed(0)

ckpts = sorted(glob.glob(os.path.join(REPO, ARGS.run, "models", "*", "chkpt*.pth")),
               key=lambda p: int(re.search(r"_(\d+)\.pth", p).group(1)))
if ARGS.step_range:
    lo_, hi_ = (int(v) for v in ARGS.step_range.split(","))
    ckpts = [c for c in ckpts if lo_ <= int(re.search(r"_(\d+)\.pth", c).group(1)) <= hi_]
ckpts = ckpts[::ARGS.every]
assert ckpts, f"no checkpoints under {ARGS.run}"
print(f"{len(ckpts)} checkpoints from {ARGS.run}")

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in range(510115, 510125)},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)
for _ in range(ARGS.skip):
    next(dl)

Xs, Ts = [], []
for _ in range(ARGS.batches):
    b = next(dl)
    types = b["types"]
    ok = ((types == NU).sum(1) == 1) & (((types == 0) | (types == 1)).sum(1) == 1)
    Xs.append(b["x"][ok].clone()); Ts.append(types[ok].clone())
X = torch.cat(Xs); T = torch.cat(Ts); N = len(X)
TRU = torch.round(X[..., -1]).long()
ar = torch.arange(N)
npos = (T == NU).float().argmax(1); lpos = ((T == 0) | (T == 1)).float().argmax(1)
lvbb = TRU[ar, lpos] == 3
boosted = (~lvbb) & (((T == LJ) & (TRU == 2)).sum(1) == 1)

# --- boosted working set (rel-hard, mask, tie) ---
bsel = torch.where(boosted & ((T == PAD).sum(1) >= 1))[0]
NB = len(bsel)
wp = ((T[bsel] == LJ) & (TRU[bsel] == 2)).float().argmax(1)
bar = torch.arange(NB)
slot = (T[bsel] == PAD).float().argmax(1)
real = T[bsel] != PAD
m2_all = torch.clamp(X[bsel][..., 3] ** 2 - (X[bsel][..., :3] ** 2).sum(-1), min=0)
rng = np.random.default_rng(0)
donor_idx = rng.integers(NB, size=NB)
ddir = X[bsel[donor_idx], wp[donor_idx], :3].clone()
donor_t = T[bsel[donor_idx], wp[donor_idx]]
donor_y = TRU[bsel[donor_idx], wp[donor_idx]]
print(f"boosted donor directions: true-W-ljet fraction "
      f"{(((donor_t == LJ) & (donor_y == 2)).float().mean()):.4f}; "
      f"type counts {[int((donor_t == k).sum()) for k in range(6)]}")

# --- lvbb working set (cross-system) ---
lsel = torch.where(lvbb & ((T == PAD).sum(1) >= 1))[0]
NL = len(lsel)
lar = torch.arange(NL)
slot_l = (T[lsel] == PAD).float().argmax(1)
lp, np_l = lpos[lsel], npos[lsel]
pvec = X[lsel][lar, lp, :2] + X[lsel][lar, np_l, :2]
pt_lepW = torch.sqrt((pvec ** 2).sum(-1)).clamp(min=0.01)
ddir_l = ddir[rng.integers(NB, size=NL) % NB][:NL] if NB else None
print(f"boosted+slot n={NB}, lvbb+slot n={NL}")

def set_kin(rows3, pt_t, m_gev):
    pt_now = torch.sqrt(rows3[:, 0] ** 2 + rows3[:, 1] ** 2).clamp(min=1e-6)
    p3 = rows3 * (pt_t / pt_now).unsqueeze(-1)
    E = torch.sqrt((p3 ** 2).sum(-1) + (m_gev / 100.0) ** 2)
    return p3, E

# prebuilt intervention inputs (model-independent)
XT = {}
XT["base_all"] = (X, T)
Xm = X.clone(); Tm = T.clone()
bsel_all = torch.where(boosted)[0]
wp_all = ((T[bsel_all] == LJ) & (TRU[bsel_all] == 2)).float().argmax(1)
Xm[bsel_all, wp_all] = 0.0; Tm[bsel_all, wp_all] = PAD
XT["mask"] = (Xm, Tm)

X2 = X[bsel].clone()                                   # rest x2 (masses fixed)
sc = torch.full(X2.shape[:2], 1.0); sc[real] = 2.0; sc[bar, wp] = 1.0
X2[..., :3] = X2[..., :3] * sc.unsqueeze(-1)
X2[..., 3] = torch.sqrt((X2[..., :3] ** 2).sum(-1) + m2_all)
X2[~real] = X[bsel][~real]
XT["restx2"] = (X2, T[bsel].clone())

Xb = X[bsel].clone()                                   # solo600 baseline
p3, E = set_kin(Xb[bar, wp, :3], 6.0, 80)
Xb[bar, wp, :3] = p3; Xb[bar, wp, 3] = E; Xb[bar, wp, 4] = WTAG
XT["solo600"] = (Xb, T[bsel].clone())
Xt = Xb.clone(); Tt = T[bsel].clone()                  # tie
rows = torch.zeros(NB, 7)
p3, E = set_kin(ddir, 6.0, 80)
rows[:, :3] = p3; rows[:, 3] = E; rows[:, 4] = WTAG
Xt[bar, slot] = rows; Tt[bar, slot] = LJ
XT["tie"] = (Xt, Tt)

for rr in (1.0, 1.4, 2.0):                             # cross-system doses
    Xc = X[lsel].clone(); Tc = T[lsel].clone()
    rows = torch.zeros(NL, 7)
    p3, E = set_kin(ddir_l, rr * pt_lepW, 80)
    rows[:, :3] = p3; rows[:, 3] = E; rows[:, 4] = WTAG
    Xc[lar, slot_l] = rows; Tc[lar, slot_l] = LJ
    XT[f"xsys{rr}"] = (Xc, Tc)

def probe(model):
    BN = 1 if "bn1" in ARGS.model else None
    cache = ActivationCache()
    hs = [m.register_forward_hook(fn, with_kwargs=True)
          for m, fn in hook_attention_heads(model, cache, detach=True,
                SINGLE_ATTENTION=False, bottleneck_attention_output=BN)]
    def fwd(x, t):
        outs = []
        with torch.no_grad():
            for c in range(0, len(x), 4096):
                outs.append(model(x[c:c + 4096, :, :5], t[c:c + 4096]))
        return torch.cat(outs)
    r = {}
    out0 = fwd(*XT["base_all"]); pr0 = out0.argmax(-1)
    r["lockstep"] = (pr0[ar, npos] == pr0[ar, lpos]).float().mean().item()
    prm = fwd(*XT["mask"]).argmax(-1)
    r["mask_flip"] = ((prm[bsel_all, lpos[bsel_all]] == W)
                      & (pr0[bsel_all, lpos[bsel_all]] != W)).float().mean().item()
    out_s = fwd(*XT["solo600"]); ms = out_s[bar, wp]; ms = ms[..., W] - ms[..., 0]
    cl_s = out_s.argmax(-1)[bar, wp] == W
    r["task_solo600"] = cl_s.float().mean().item()
    cl_r = fwd(*XT["restx2"]).argmax(-1)[bar, wp] == W
    base_b = pr0[bsel][bar, wp] == W if False else None  # (kept simple: use solo)
    r["relhard_supp"] = ((cl_s & ~cl_r).float().sum() / cl_s.float().sum()).item() if cl_s.sum() else float("nan")
    out_t = fwd(*XT["tie"]); prt = out_t.argmax(-1)
    cw, ci = prt[bar, wp] == W, prt[bar, slot] == W
    r["tie_both"] = (cw & ci).float().mean().item()
    mt = out_t[bar, wp]; mt = mt[..., W] - mt[..., 0]
    r["tie_margin_cost"] = (mt - ms).mean().item()
    for rr in (1.0, 1.4, 2.0):
        pr = fwd(*XT[f"xsys{rr}"]).argmax(-1)
        r[f"xsys{rr}"] = (pr[lar, slot_l] == W).float().mean().item()
    for h in hs: h.remove()
    return r

rows_out = []
for ck in ckpts:
    ep = int(re.search(r"_(\d+)\.pth", ck).group(1))  # global step (monotone across epochs)
    model, _ = load_model(ARGS.model, checkpoint_root="unused",
                          register_bottleneck_hook=False, checkpoint_override=ck)
    model.eval()
    r = probe(model)
    r["epoch"] = ep
    rows_out.append(r)
    print(f"ep {ep:2d}: task {r['task_solo600']:.3f} lockstep {r['lockstep']:.3f} "
          f"mask {r['mask_flip']:.3f} relhard {r['relhard_supp']:.3f} "
          f"tie-both {r['tie_both']:.3f} cost {r['tie_margin_cost']:+.2f} "
          f"xsys(1/1.4/2) {r['xsys1.0']:.2f}/{r['xsys1.4']:.2f}/{r['xsys2.0']:.2f}", flush=True)

stamp = os.path.basename(ARGS.run.rstrip("/")).split("_")[0]
if ARGS.step_range:
    stamp += f"_steps{ARGS.step_range.replace(',', '-')}"
os.makedirs(os.path.join(REPO, "tmp_plots"), exist_ok=True)
csv_path = os.path.join(REPO, "tmp_plots", f"f1_formation_{stamp}.csv")
with open(csv_path, "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    wcsv.writeheader(); wcsv.writerows(rows_out)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
eps = [r["epoch"] for r in rows_out]
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
a = axes[0, 0]
a.plot(eps, [r["task_solo600"] for r in rows_out], "o-", label="P(W-jet claims) solo600")
a.plot(eps, [r["lockstep"] for r in rows_out], "s-", label="P(nu=lep) lockstep")
a.plot(eps, [r["mask_flip"] for r in rows_out], "^-", label="mask->lep flip (broadcast)")
a.legend(fontsize=8); a.set_title("task + broadcast")
a = axes[0, 1]
a.plot(eps, [r["relhard_supp"] for r in rows_out], "o-", label="rest x2 suppression")
a.plot(eps, [r["tie_both"] for r in rows_out], "s-", label="tie: P(both claim)")
a.legend(fontsize=8); a.set_title("context-relativity + comparator")
a = axes[1, 0]
a.plot(eps, [r["tie_margin_cost"] for r in rows_out], "o-")
a.set_title("SB6 winner margin cost (tie - solo)"); a.set_xlabel("global step")
a = axes[1, 1]
for rr, mk in ((1.0, "o-"), (1.4, "s-"), (2.0, "^-")):
    a.plot(eps, [r[f"xsys{rr}"] for r in rows_out], mk, label=f"r={rr}")
a.legend(fontsize=8); a.set_title("cross-system P(ins claims) at r x pT(lepW)")
a.set_xlabel("global step")
plt.tight_layout()
png_path = os.path.join(REPO, "tmp_plots", f"f1_formation_{stamp}.png")
plt.savefig(png_path, dpi=120)
print(f"\nwrote {csv_path}\n      {png_path}")

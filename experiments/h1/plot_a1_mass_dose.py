"""A1 figure: W-ljet mass dose-response for the thesis model vs both d20 organisms.

Sets the truth-W ljet's mass to a grid of values in boosted qqbb events and plots
(top) how often that jet claims the W and (bottom) how often the lepton is called W.
Headline: the 677k thesis model has a genuine W-mass window peaked at m_W; both
independently-trained 20k organisms are monotonic ("lighter => more W-like") --
the window is a capacity/scale effect. See the 2026-06-10 session log, waves 1 & 3.

Output: tmp_plots/a1_mass_dose_response.png
"""
import os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
torch.set_num_threads(4)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
PAD, NU, LJ, SJ = 5, 2, 3, 4
W = 2
stds = np.ones(7); stds[:4] = 1e5
GRID = [5, 20, 40, 60, 70, 80, 90, 100, 110, 125, 150, 175, 200, 250, 300]

MODELS = [
    ("thesis-ent1-bn1-d152", None, "thesis model (677k, ent+bn1, 3 blocks)", "tab:blue"),
    ("ent1-d20-2blk", None, "organism 2025 (20k, ent, 2 blocks)", "tab:orange"),
    ("ent1-d20-2blk",
     os.path.join(REPO, "output/20260610-203258_TrainingOutput/models/Nplits2_ValIdx0/chkpt29_10290.pth"),
     "organism 2026 (20k, retrained tonight)", "tab:green"),
]

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in range(510115, 510125)},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)
Xs, Ts = [], []
for _ in range(6):
    b = next(dl)
    types = b["types"]
    ok = ((types == NU).sum(1) == 1) & (((types == 0) | (types == 1)).sum(1) == 1)
    Xs.append(b["x"][ok].clone()); Ts.append(types[ok].clone())
X = torch.cat(Xs); T = torch.cat(Ts)
TRU = torch.round(X[..., -1]).long()
ar = torch.arange(len(X))
lpos = ((T == 0) | (T == 1)).float().argmax(1)
lvbb = TRU[ar, lpos] == 3
boosted = (~lvbb) & (((T == LJ) & (TRU == 2)).sum(1) == 1)
sel = torch.where(boosted)[0]
wp = ((T[sel] == LJ) & (TRU[sel] == 2)).float().argmax(1)
sar = torch.arange(len(sel))
n = len(sel)
print(f"boosted qqbb events: {n}")

results = {}
for name, ckpt, label, color in MODELS:
    bn = 1 if "bn1" in name else None
    model, _ = load_model(name, checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                          register_bottleneck_hook=False, checkpoint_override=ckpt)
    model.eval()
    cache = ActivationCache()
    handles = [m.register_forward_hook(fn, with_kwargs=True)
               for m, fn in hook_attention_heads(model, cache, detach=True,
                     SINGLE_ATTENTION=False, bottleneck_attention_output=bn)]
    claims, leps = [], []
    for m_gev in GRID:
        X2 = X[sel].clone()
        p2 = (X2[sar, wp, :3] ** 2).sum(-1)
        X2[sar, wp, 3] = torch.sqrt(p2 + (m_gev / 100.0) ** 2)
        prs = []
        with torch.no_grad():
            for c in range(0, n, 2048):
                prs.append(model(X2[c:c + 2048, :, :5], T[sel][c:c + 2048]).argmax(-1))
        pr = torch.cat(prs)
        claims.append((pr[sar, wp] == W).float().mean().item())
        leps.append((pr[sar, lpos[sel]] == W).float().mean().item())
    results[label] = (claims, leps, color)
    for h in handles: h.remove()
    print(f"  {label}: done")

fig, axes = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True)
for label, (claims, leps, color) in results.items():
    err = [np.sqrt(p * (1 - p) / n) for p in claims]
    axes[0].errorbar(GRID, claims, yerr=err, marker="o", ms=4, color=color, label=label)
    err = [np.sqrt(p * (1 - p) / n) for p in leps]
    axes[1].errorbar(GRID, leps, yerr=err, marker="o", ms=4, color=color, label=label)
for ax in axes:
    ax.axvline(80.4, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axvline(125, color="gray", ls=":", lw=1, alpha=0.7)
    ax.grid(alpha=0.25)
axes[0].text(80.4, 0.30, " $m_W$", color="gray")
axes[0].text(125, 0.30, " $m_h$", color="gray")
axes[0].set_ylabel("P(W-ljet claims W)")
axes[0].set_title("A1: set the true-W ljet's mass, watch who claims the W\n"
                  f"(boosted qqbb val events, n={n}; lep/ν untouched)")
axes[0].legend(loc="lower right", fontsize=9)
axes[1].set_ylabel("P(lepton called W)  —  the verdict flip")
axes[1].set_xlabel("W-ljet mass set to [GeV]")
fig.tight_layout()
out = f"{REPO}/tmp_plots/a1_mass_dose_response.png"
os.makedirs(f"{REPO}/tmp_plots", exist_ok=True)
fig.savefig(out, dpi=140)
print(f"saved {out}")

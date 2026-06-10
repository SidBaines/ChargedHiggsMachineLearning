"""Re-analysis of the round-1 b2h3->nu bottleneck-scalar histogram (Sid's question):
the 'true W->lv' distribution was bimodal with one hump on the 'other' peak — are
those the MISreconstructed lvbb events?

Preregistration: if the message encodes the model's VERDICT (not the truth), then
(i) the overlapping hump = true-lvbb events predicted 'none'; (ii) AUC of the scalar
vs PREDICTION >> AUC vs truth (0.83 from round 1); (iii) conditioning on prediction
makes the truth-split collapse. Also: hunt the third mode within true-lvbb-correct
(candidates: e vs mu, mass point, n_ljets).

Thesis model only (organism has no bottleneck).
"""
import os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
torch.set_num_threads(4)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
DSIDS = list(range(510115, 510125))
PAD, NU, LJ, SJ = 5, 2, 3, 4
W = 2
stds = np.ones(7); stds[:4] = 1e5
N_BATCHES = 12

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)

model, _ = load_model("thesis-ent1-bn1-d152", checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False)
model.eval()
cache = ActivationCache()
handles = [m.register_forward_hook(fn, with_kwargs=True)
           for m, fn in hook_attention_heads(model, cache, detach=True,
                 SINGLE_ATTENTION=False, bottleneck_attention_output=1)]

msgs = {(b, h): [] for b in range(3) for h in range(4)}   # scalar into nu, all heads
truth_lv, pred_nu_W, lep_is_e, dsid_l, n_lj = [], [], [], [], []
for _ in range(N_BATCHES):
    b = next(dl)
    x, types = b["x"], b["types"]
    with torch.no_grad():
        out = model(x[..., :5], types)
    tru = torch.round(x[..., -1]).long()
    pred = out.argmax(-1)
    ok = ((types == NU).sum(1) == 1) & (((types == 0) | (types == 1)).sum(1) == 1)
    ii = torch.arange(len(types))[ok]
    npos = (types[ok] == NU).float().argmax(1)
    lpos = ((types[ok] == 0) | (types[ok] == 1)).float().argmax(1)
    truth_lv.append((tru[ii, lpos] == 3).numpy())
    pred_nu_W.append((pred[ii, npos] == W).numpy())
    lep_is_e.append((types[ii, lpos] == 0).numpy())
    dsid_l.append(b["dsids"][ok].numpy())
    n_lj.append((types[ok] == LJ).sum(1).numpy())
    for blk in range(3):
        bna = cache[f"block_{blk}_attention"]["bottleneck_activation"]  # [B,H,N,1]
        for h in range(4):
            msgs[(blk, h)].append(bna[ii, h, npos, 0].numpy())

truth_lv = np.concatenate(truth_lv); pred_nu_W = np.concatenate(pred_nu_W)
lep_is_e = np.concatenate(lep_is_e); dsid_l = np.concatenate(dsid_l)
n_lj = np.concatenate(n_lj)
M = {k: np.concatenate(v) for k, v in msgs.items()}
m23 = M[(2, 3)]
N = len(m23)
print(f"events: {N}")

print("\n=== AUC of each head's scalar-into-nu: vs TRUTH vs PREDICTION ===")
print(f"  {'head':>6s} {'AUC truth':>10s} {'AUC pred':>10s}")
for blk in range(3):
    for h in range(4):
        a_t = roc_auc_score(truth_lv, M[(blk, h)])
        a_p = roc_auc_score(pred_nu_W, M[(blk, h)])
        print(f"  b{blk}h{h:1d}  {max(a_t,1-a_t):10.4f} {max(a_p,1-a_p):10.4f}")

print("\n=== b2h3 by truth x prediction (Sid's split) ===")
groups = [
    ("true lvbb, pred W (correct)",  truth_lv & pred_nu_W),
    ("true lvbb, pred none (MISS)",  truth_lv & ~pred_nu_W),
    ("other,     pred none (correct)", ~truth_lv & ~pred_nu_W),
    ("other,     pred W (false pos)", ~truth_lv & pred_nu_W),
]
for lab, g in groups:
    print(f"  {lab:32s} n={g.sum():6d}  msg mean {m23[g].mean():+.3f}  median {np.median(m23[g]):+.3f}")
auc_t = roc_auc_score(truth_lv, m23); auc_p = roc_auc_score(pred_nu_W, m23)
print(f"  b2h3: AUC vs truth {max(auc_t,1-auc_t):.4f}   vs prediction {max(auc_p,1-auc_p):.4f}")
# conditioning on prediction: does truth still separate?
for plab, pm in (("pred W", pred_nu_W), ("pred none", ~pred_nu_W)):
    if (truth_lv[pm].mean() not in (0.0, 1.0)):
        a = roc_auc_score(truth_lv[pm], m23[pm])
        print(f"  within {plab}: AUC vs truth {max(a,1-a):.4f}  (n={pm.sum()})")

# third-mode hunt within true-lvbb-correct
g = truth_lv & pred_nu_W
print("\n=== third-mode hunt within 'true lvbb, pred W' ===")
for lab, sub in (("electron", lep_is_e[g]), ("muon", ~lep_is_e[g])):
    print(f"  {lab:9s} n={sub.sum():6d}  msg mean {m23[g][sub].mean():+.3f}  median {np.median(m23[g][sub]):+.3f}")
a_e = roc_auc_score(lep_is_e[g], m23[g])
print(f"  AUC(e vs mu | correct lvbb) = {max(a_e,1-a_e):.4f}")
lo = dsid_l[g] <= 510119
print(f"  low-mass (<=1.4TeV) mean {m23[g][lo].mean():+.3f}  high-mass {m23[g][~lo].mean():+.3f}  "
      f"AUC {max(roc_auc_score(lo, m23[g]),1-roc_auc_score(lo, m23[g])):.4f}")
one_lj = n_lj[g] == 1
print(f"  n_ljet==1 mean {m23[g][one_lj].mean():+.3f}  n_ljet>=2 {m23[g][~one_lj].mean():+.3f}  "
      f"AUC {max(roc_auc_score(one_lj, m23[g]),1-roc_auc_score(one_lj, m23[g])):.4f}")

print("\n=== the strong verdict carriers (b1h3, b2h2) by truth x prediction ===")
for key in ((1, 3), (2, 2)):
    mm = M[key]
    print(f"  b{key[0]}h{key[1]}:")
    for lab, g in groups:
        print(f"    {lab:32s} n={g.sum():6d}  msg mean {mm[g].mean():+.3f}  median {np.median(mm[g]):+.3f}")
    for plab, pm in (("pred W", pred_nu_W), ("pred none", ~pred_nu_W)):
        a = roc_auc_score(truth_lv[pm], mm[pm])
        print(f"    within {plab}: AUC vs truth {max(a,1-a):.4f}")

# plots
fig, axes = plt.subplots(3, 1, figsize=(8, 11))
ax = axes[0]
ax.hist(m23[truth_lv], bins=80, alpha=0.55, density=True, label="true W→lν", color="tab:blue")
ax.hist(m23[~truth_lv], bins=80, alpha=0.55, density=True, label="other", color="tab:orange")
ax.legend(); ax.set_title("round-1 view: split by TRUTH only")
ax = axes[1]
colors = ("tab:blue", "tab:cyan", "tab:orange", "tab:red")
for (lab, g), c in zip(groups, colors):
    if g.sum() > 20:
        ax.hist(m23[g], bins=80, alpha=0.5, density=True, label=f"{lab} (n={g.sum()})", color=c)
ax.legend(fontsize=8); ax.set_title("b2h3: split by truth × prediction")
ax.set_xlabel("b2h3 bottleneck scalar at the neutrino position")
ax = axes[2]
m22 = M[(2, 2)]
for (lab, g), c in zip(groups, colors):
    if g.sum() > 20:
        ax.hist(m22[g], bins=80, alpha=0.5, density=True, label=f"{lab} (n={g.sum()})", color=c)
ax.legend(fontsize=8); ax.set_title("b2h2 — the actual verdict carrier (AUC vs pred 0.996)")
ax.set_xlabel("b2h2 bottleneck scalar at the neutrino position")
fig.tight_layout(); fig.savefig(f"{REPO}/tmp_plots/h1_b2h3_message_hist_split.png", dpi=130)
print("\nsaved tmp_plots/h1_b2h3_message_hist_split.png")
for h in handles: h.remove()
print("done.")

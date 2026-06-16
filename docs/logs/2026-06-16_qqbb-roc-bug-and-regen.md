# 2026-06-16 — qqbb ROC-AUC bug fix + ch.7 figure regeneration

## What we did

Sid's thesis had two ch.7 figures (`TransformerVsDenseNNComparison`,
`TransformerReconstructionAndClassificationPerformance`) with the **qqbb ROC-AUC subplot
removed** and a `\note{Had to remove… there was a bug…}`. Task: find the bug, fix it,
regenerate, and restore the figures.

1. **Traced the plot** to `CompareAllModelsQqbb.py` (`jNNpNNlowLevel` block) → metrics from
   `metrics/lowlevelmetrics.py::HEPMetrics`.
2. **Diagnosed the bug** (see below). Confirmed from the committed `roc_aucs_qqbb.pkl`
   (low-level qqbb ROC fell 0.65→0.086, vs lvbb 0.92–0.999 and high-level qqbb 0.63→0.90
   both fine) — exactly matching the thesis note's suspicion.
3. **Fixed** `compute_auc` (rank on `all_probs[:, target_class]`, not hardcoded col 1).
4. **Regenerated** on heppc402 via `/data/baines/piienv` (CPU torch, data over NFS). Used a
   truncate-the-script + override-`model_params` driver pattern (`make_regen_drivers.py`,
   `make_regen_treco.py`) → `*_FIXED.pkl`; plotted locally with usetex + y-axis capped at
   1.0 (`plot_all_roc_ylim1.py`).
5. **Updated the thesis**: restored qqbb ROC panels, relaid both figures 4×1→2×2, removed
   notes, swapped in 4 new PDFs; full `latexmk` build clean.

## The bug

`compute_auc` loops `channel in ['lvbb','qqbb']` with `target_class = 1` (lvbb) / `2`
(qqbb), but gathered the ranking score as `self.all_probs[…, 1]` (**hardcoded column 1**)
for both. So qqbb AUC ranked qqbb-signal-vs-bkg on the **lvbb** probability → meaningless
(and for the pre-split qqbb model, col 1 is forced to ≈0 by `outputs[:,1]=-100`, so it was
ranking on ~noise → AUC well below 0.5). Fix: `self.all_probs[…, target_class]` at the two
sites (lines ~239, ~265 pre-fix). lvbb unchanged (target_class=1==old col 1); only low-level
3-class metrics affected (high-level binary uses col 1 == signal, already correct).

## Results (corrected qqbb ROC-AUC, 0.8→3.0 TeV)

- Low-level pre-split: 0.830→0.992  (was 0.655→0.086)
- Low-level combined:  0.898→0.994  (was 0.694→0.497)
- Low-level transformer-reco: 0.922→0.997 (was 0.642→0.189)
- High-level Joint / Parametrised: **identical** to before (unaffected) — 0.635→0.900 / 0.693→0.965

Validation: high-level qqbb AUC and **all** `sig_rems`/`S_b200` reproduced the committed
pkl values exactly (combined `sig_rems` to ~0.1%, float nondeterminism) → confirms same
model + same data + same code path; only the intended metric moved.

## Commits

- Fix: `79d3312` on `heppc-work` (this repo) → pushed to GitHub.
- Thesis: `6425031` on `sid-opus-final-pass` (`SidBaines/PhDThesis`) → pushed.
- Cluster repo: fix applied but **uncommitted** (`metrics/lowlevelmetrics.py.prefix_bak` is
  the original).

## Learnings

- **Most of the new, durable infra knowledge → [`../CLUSTER_AND_DATA_GUIDE.md`](../CLUSTER_AND_DATA_GUIDE.md)** (env, data
  locations, run recipe, gotchas). Read that before any future cluster run.
- The committed `*_{channel}.pkl` at repo root ARE the thesis-plot values, but hold only
  aggregated arrays (no per-event probs) → must re-run models to recompute AUC.
- `CompareAllModels*` load-pkl cell is `if 1:` and silently clobbers freshly computed values
  — disable when regenerating.
- NFS reads dominate runtime; don't run two big jobs concurrently; `/data/baines` was full.
- Regen scaffolding left untracked in repo root: `make_regen_drivers.py`,
  `make_regen_treco.py`, `plot_all_roc_ylim1.py`, `compare_and_plot_fixed.py`.

# heppc402 + data + how-to-run guide (living doc)

> **Purpose.** A practical, verified reference for future agents (and Sid): how to reach
> the QMUL cluster `heppc402`, where the datasets and model checkpoints live, which dataset
> is for which task, and how to actually run this repo's code there. Append your own
> gotchas/hints at the bottom (**Running log**) as you learn them.
>
> Confidence: everything in §1–§5 was directly verified on 2026-06-16 unless tagged
> *(inferred)* or *(from <other doc>)*. Trust the code/cluster over this doc if they disagree,
> and fix the doc.
>
> Related docs: [`SESSION_FINDINGS_2026-06-08.md`](SESSION_FINDINGS_2026-06-08.md) (repo
> layout, branches, data provenance chain, local-Mac scratch data); the thesis-repo
> auto-memory `qmul-plot-regeneration.md` (ROOT-based appendix plots, a *different* env) and
> `qqbb-roc-auc-bug.md` (the bug fixed 2026-06-16).

---

## 1. Accessing heppc402

- **Host / user:** `baines@heppc402.ph.qmul.ac.uk`. SSH **key works non-interactively**
  (`ssh -o BatchMode=yes baines@heppc402.ph.qmul.ac.uk '<cmd>'` runs without a prompt).
- OS is **CentOS 7** — too old for the VS Code remote server (this is why Sid can't open it
  in VS Code), but plain `ssh`/`scp`/`rsync` are fine.
- Every login prints a QMUL banner; filter it out of command output with e.g.
  `... | grep -vi "warning\|queen mary\|school of\|available to"`.
- 32 cores; shared machine — check `uptime` before launching anything heavy.
- **Long jobs:** launch with `nohup … > some.log 2>&1 &` so they survive the SSH session,
  then poll the log from separate `ssh` calls. Holding an interactive SSH open for a
  multi-hour job is fragile.

## 2. Python environments on heppc402

Two **different** envs for two different worlds — don't mix them:

- **PyTorch / this repo:** `/data/baines/piienv/bin/python` (Python 3.9, `torch 2.4.1+cu121`
  but **CPU-only** — `torch.cuda.is_available()` is `False`). Verified to import everything
  the training/eval scripts need: `numpy, jaxtyping, einops, matplotlib, transformer_lens,
  wandb, sklearn`, plus the repo's `dataloaders`, `models.models.TestNetwork`,
  `metrics.lowlevelmetrics`, `utils.utils`. **Use this for `Train*`/`CompareAllModels*`/
  interp.**
- **ROOT (for the thesis appendix ROOT plots, NOT this repo):**
  `source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos7-gcc11-opt/setup.sh`
  (ROOT 6.28/04 + uproot/numpy). *(from `qmul-plot-regeneration.md`.)* Conda envs
  (`myroot` etc.) are broken/incomplete.

There is **no GPU** in practice (CPU torch). Heavy training → consider RunPod (see
RESEARCH_PLAN). Evaluation/inference for the comparison plots runs fine on CPU, just slow.

## 3. The code

- **Repo on cluster:** `/users/baines/Code/ChargedHiggs_ExperimentalML` (this same repo;
  `/users` is the home filesystem, fast-ish). Model checkpoints live under its
  `output/<TIMESTAMP>_TrainingOutput/models/<channel>_Nplits2_ValIdx0/chkpt*.pth`
  (`output/` is gitignored — checkpoints are NOT in git, only on the cluster).
- Entry points: `TrainLowLevelReconstruction.py`, `TrainLowLevelClassifier.py`,
  `TrainHighLevelClassifier.py`, `RunLowLevelInterp.py`; comparison/plotting:
  `CompareAllModelsQqbb.py`, `CompareAllModelsLvbb.py`. All are Jupyter `# %%` cell-tagged
  but run fine top-to-bottom as plain scripts.
- **`CompareAllModels{Qqbb,Lvbb}.py` structure** (these make the thesis ch.7 classification
  figures):
  - `model_params` dict (top) selects which models to evaluate; most entries are commented
    out. Each entry has a `type` (`'hl'`, `'Low-level pre-split NN'`, `'Low-level combined
    NN'`, `'Low-level NN (transformer-reco inputs)'`) which picks an eval branch, a
    checkpoint `filepath`, and arch params.
  - One big loop evaluates every model into `vms_MCWts[name]`, then fills `sig_rems`
    (= `S_b200`), `roc_aucs`, etc.
  - Results are pickled to `*_{channel}.pkl` (save cell guarded by `if 0:`) and reloaded
    (load cell guarded by `if 1:`). **The committed `*_qqbb.pkl`/`*_lvbb.pkl` in the repo
    root are the values that fed the thesis plots** — but they store only the final
    aggregated arrays, **not** per-event probabilities, so you cannot recompute AUC from
    them; you must re-run the model.
  - `plotSaveDir` + `models_to_plot` (further down) select which subset of models go into
    which figure. The two ch.7 classification figures share one set of pkls:
    - `jNNpNNlowLevel` → 4 models: pre-split, combined, HL Joint, HL Parametrised.
    - `jNNpNNlowLevelTransformerReco` → those 4 **plus** `Low-level NN (transformer-reco
      inputs)` (the 4 are identical curves; only the extra olive line is added).
  - Plotting uses `plt.rcParams['text.usetex']=True`, which needs a LaTeX install. The
    cluster path of least resistance is to **compute on the cluster → pickle → scp pkls to a
    machine with LaTeX → plot there** (that's what was done 2026-06-16).

## 4. Where the data lives (and which is for what)

All under **`/data/atlas/baines/`**, which is **NFS** (`hepraid11:/srv/nfs/data-atlas`,
~80 TB, ~74% full). Reads are **memory-mapped over NFS → slow random access** (cold reads
of a few-GB val set take ~tens of minutes; once page-cache-warm, batches fly). Datasets are
per-DSID `dsid_<DSID>[_<channel>].memmap` + `.memmap.shape` files, plus `mean.npy`/`std.npy`.

| Dataset dir (under `/data/atlas/baines/`) | Size | What it is / use for |
|---|---|---|
| `20250619v1_Split_WithEventNumbers_…_WithRecoMasses_30_MetCut_OldTruth_RemovedUncertainTruth_WithTagInfo` | ~0.7 GB/channel | **Channel-split** low-level classifier data (`dsid_*_qqbb.memmap`, `dsid_*_lvbb.memmap`), signal+background. Use for the **pre-split** low-level transformer classifiers. |
| `20250619v2_WithEventNumbers_…_WithRecoMasses_30_MetCut_OldTruth_RemovedUncertainTruth_WithTagInfo` | ~4.4 GB | **Combined** (not channel-split) low-level data. Use for the **combined 3-class** classifier. |
| `20250321v2_AppliedRecoNNSplit_WithEventNumbers_…_MetCut_RemovedUncertainTruth_WithTagInfo_KeepAllOldSelIncludingNegative` | ~6.8 GB | Events **reconstructed+split by the transformer reco network** then re-classified. Use for `Low-level NN (transformer-reco inputs)` (the figure-2 / `jNNpNNlowLevelTransformerReco` model). Big & slow. |
| `20250322v4_highLevel_MetCut_OldTruth_RemovedUncertainTruth` | ~0.2 GB | **High-level** reconstructed-event variables (per-channel `{channel}_mean.npy`/`std.npy`). Use for the baseline high-level **DNN** classifiers (`type:'hl'`). |

The exact dir name is built at runtime by concatenating flags in the script
(`DATA_PATH = base + '_MetCut'*MET_CUT_ON + …`), so it depends on the config block of the
branch you're running — read the branch, don't guess.

**Checkpoints** (cluster only, gitignored): `/users/baines/Code/ChargedHiggs_ExperimentalML/
output/<TIMESTAMP>_TrainingOutput/models/<qqbb|lvbb|combined>_Nplits2_ValIdx0/chkpt*.pth`.
The specific ones feeding the ch.7 classification figures are listed in each
`CompareAllModels*.py` `model_params` (mostly commented out — grep for `'filepath'`).

**DSID → mass (signal):** `510115..510124` = `0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5,
3.0` TeV. Backgrounds: `410470` (ttbar, dominant), `363xxx` (V+jets/diboson), `700xxx`.

**`/data/baines/` is a *different* NFS mount** (`hepraid8`) and was **100% full** (~880 MB
free) on 2026-06-16 — fine for the `piienv` and small outputs, **cannot** stage multi-GB
data there. Provenance of the memmaps (raw ntuples → C++ → `preprocessLowLevel.py`): see
`SESSION_FINDINGS_2026-06-08.md` §"Cluster data provenance".

**Mac note:** the right datasets for these classifier plots are **not** on the Mac — local
`*.memmap` dirs (`20250311_MechInterpTmp/data`, `tmp_data_20250321v1_signal`,
`HplusWh/20250201_ML_Testing/data`) are signal-only / un-split / partial-background and a
debug-subset; not usable for the classifier comparison.

## 5. Recipe: regenerate the ch.7 classification figures (worked example, 2026-06-16)

Goal was to recompute corrected qqbb (and re-plot lvbb) ROC-AUC after a metrics bugfix.
Pattern that worked (scripts left untracked in repo root):

1. **Edit `metrics/lowlevelmetrics.py`** locally, `scp` to cluster (back up first:
   `cp metrics/lowlevelmetrics.py metrics/lowlevelmetrics.py.prefix_bak`).
2. **Build a slim driver** from the big Compare script: `make_regen_drivers.py` /
   `make_regen_treco.py` take the original, **truncate it right before the "Save the
   calculated data" cell** (drops the buggy-pkl reload + the LaTeX plotting), **override
   `model_params`** with an explicit small dict of just the models you want, and append a
   save block writing `*_{channel}_FIXED.pkl`. This reuses the script's exact eval/config
   code paths (faithful) without manual cell-uncommenting.
3. `scp` drivers up; run with `nohup MPLBACKEND=Agg /data/baines/piienv/bin/python -u
   CompareAllModels…_REGEN.py > regen.log 2>&1 &`; poll the log for `REGEN DONE`.
4. `scp` the `*_FIXED.pkl` back; plot **locally** with usetex (`plot_all_roc_ylim1.py`).
   lvbb needed no re-run — the committed `roc_aucs_lvbb.pkl` is already correct.

Validation trick: the **unchanged** metric (`sig_rems`/`S_b200`) and the high-level AUCs
**reproduce the committed pkl values exactly** → confirms you're running the right model on
the right data; only the intended metric should move.

## 6. Gotchas (read before you burn an hour)

- **NFS is the bottleneck**, not CPU. A multi-GB val set's first pass is slow (state `Dl`,
  `wait_on_page_bit_killable`); CPU% looks idle because it's I/O-waiting. It IS progressing
  (watch `read_bytes` in `/proc/<pid>/io`). Don't kill it thinking it hung.
- **Don't run two big-dataset jobs at once** — they contend for the same NFS server and
  ~halve each other's throughput.
- `CompareAllModels*.py` **load-pkl cell is `if 1:`** → it will silently overwrite your
  freshly computed values with the old pkls. Disable it (or use the truncate-driver recipe)
  when regenerating.
- `text.usetex=True` needs LaTeX (+dvipng). Not the path of least resistance on the cluster
  → compute there, plot where LaTeX lives.
- **pkls don't store per-event probabilities** → you cannot recompute ROC/AUC from them;
  re-run the model.
- `/data/baines` can be 100% full — check `df` before writing.
- Permutation-invariant nets give tiny float-level run-to-run differences (object shuffling)
  → `sig_rems` for the combined model differed by ~0.1% between runs; expected, harmless.
- To render a thesis PDF page for a visual check on the Mac (no poppler): `pip install
  pymupdf` into a venv and use `fitz`; match **full-caption** text (the short caption also
  appears in the List of Figures), and note the PDF page index ≠ printed page (twoside
  blanks + roman front matter offset, ~+17 in the thesis as of 2026-06-16).

## 7. What was achieved 2026-06-16

- **Found & fixed a bug** in `metrics/lowlevelmetrics.py::compute_auc`: it hardcoded the
  ranking probability to column 1 (the lvbb score) for **both** channels, so the **qqbb**
  ROC-AUC was computed by ranking qqbb-signal-vs-background on the lvbb score → garbage
  (AUC fell below 0.5 and decreased with mass). Fixed to rank on `all_probs[:, target_class]`.
  Only low-level 3-class metrics were affected (high-level binary metrics, where column 1 ==
  signal, were already correct; `sig_qqbb_expected`/`S_b200` uses a separate, correct
  function and was fine). This explains why only the qqbb **ROC** panels (not the qqbb
  `S_b200` panels) had been removed from the thesis.
- **Regenerated** all four ch.7 ROC-AUC plots on heppc402 (qqbb recomputed with the fix;
  lvbb re-used the already-correct committed pkls), with the y-axis capped at 1.0. Corrected
  qqbb ROC now rises toward ~0.99 (pre-split 0.83→0.99, combined 0.90→0.99, transformer-reco
  0.92→0.997).
- **Commits:** fix = `79d3312` on `heppc-work` (this repo, pushed to GitHub
  `SidBaines/ChargedHiggsMachineLearning`). Thesis = `6425031` on `sid-opus-final-pass`
  (`SidBaines/PhDThesis`): qqbb ROC panels restored, both figures relaid 4×1→2×2, notes
  removed, 4 PDFs updated. Compiles clean.
- The cluster repo has the fix **applied but uncommitted** (`metrics/lowlevelmetrics.py`,
  original at `…prefix_bak`); the canonical fix is on GitHub.

---

## Running log (append below — newest first)

- **2026-06-16** — Created this guide. Did the qqbb ROC-AUC bugfix + ch.7 figure regen
  (see §7). Session log: [`logs/2026-06-16_qqbb-roc-bug-and-regen.md`](logs/2026-06-16_qqbb-roc-bug-and-regen.md).
<!-- Future agents: add a dated bullet here for cluster/data/run things you learn or change. -->

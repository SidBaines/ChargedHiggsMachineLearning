# Picking the work back up — investigation findings (2026-06-08)

> Written by Claude during a session where Sid returned to this PhD work after a long
> gap and wanted to (a) find which repo/branch held his transformer-interpretability
> work, (b) check whether the data still exists, and (c) work out how to restart.
> This doc captures **everything found**, including context that lives *outside* this
> repo. Paths to things outside this repo are given as absolute paths.
>
> **Confidence note:** facts about git state, file presence, sizes, data paths and
> dependencies were verified directly. Some internal code details (exact line numbers,
> precise model hyperparameters) came from a broad code read and are marked *(inferred)*
> where not re-verified. Trust the code over this doc where they disagree.

---

## TL;DR

- **This repo (`ChargedHiggsMachineLearning`) is the canonical, consolidated codebase.** It is
  the clean version of work that was prototyped in scratch folders elsewhere on the Mac
  (see "Related work outside this repo"). It's also the only interpretability codebase
  that is **backed up on GitHub**.
- Sid had been working **on another machine ("HEP PC" / HEPPC)** and pushed to the
  **`heppc-work`** branch. That branch ≈ `main` plus committed *result artifacts*.
- **The blocker for running anything is data + model weights**, not code. The training
  scripts read from `/data/atlas/baines/…` on the HEP PC (not present on this Mac), and
  model checkpoints live in a gitignored `output/` dir (not in the repo). Sid was
  searching an external hard drive for these.
- **Fastest way to run *something* on the Mac today** is actually a self-contained scratch
  folder (`/Users/sidbaines/Documents/PhD/Work/20250311_MechInterpTmp/`) which still has
  both data and a trained `model.pth`. See below.

---

## 1. What this repo is

`SidBaines/ChargedHiggsMachineLearning` (public on GitHub; formerly `ChargedHiggsML2024Jan`).

Per the README: *"Code for training, interpreting & applying transformer-like neural
networks to low-level reconstructed objects in particle physics events, for event
reconstruction/classification."* Physics context: ATLAS search for a charged Higgs
H⁺ → W h (the `HplusWh` analysis).

### Layout
- `models/models.py` — model definitions: `ConfigurableNN`, `TestNetwork`,
  `LorentzInvariantFeatures`, multihead-attention blocks. Both **custom PyTorch
  transformers** and **TransformerLens** (`transformer_lens`) models are used across the
  repo.
- `dataloaders/` — `lowleveldataloader.py`, `highleveldataloader.py` (read preprocessed
  memmap data).
- `interp/` — **the mechanistic-interpretability package**:
  - `activations.py` — `ModelActivationExtractor`, `ActivationCache`, `get_residual_stream`, `hook_attention_heads`
  - `attention.py` — `AttentionAnalyzer`
  - `causal_analysis.py` — `DirectLogitContributionAnalyzer`, `DirectLogitAttributor`, `ActivationPatcher`
  - `residual_analysis.py` — linear probes (`fit_linear_probe`, `probe_accuracy`), `plot_pca`
  - `ablation.py` — `ablate_attention_head`
  - `symbolic_regression.py` — symbolic regression (uses **PySR**) to test whether learned
    features correspond to meaningful physics variables
  - `physics_filters.py`, `condition_factories.py`, `data_filtering.py`, `mechinterputils.py`
- `metrics/` — `lowlevelmetrics.py`, `highlevelmetrics.py`, `lowlevelrecometrics.py`
- `preprocessing-scripts/` — `preprocessLowLevel.py`, `preprocessHighLevel.py`,
  `preprocessLowLevelApplyRecoSplitChannels.py`, `preprocessLowToHighApplyReco.py`,
  `preprocessLowLevelSplit.py` (raw ROOT → preprocessed memmaps)
- `postprocessing-scripts/ApplyRecoAndClassifiersToRoot.py` — apply trained nets, write back to ROOT
- `cpp_code/` — ROOT preprocessing / decoration with old classifications & published NN scores
- `utils/` — `utils.py`, `metrics_storage.py`
- Committed result artifacts (on `heppc-work`): `roc_aucs_*.pkl`, `sig_rems_*.pkl`,
  `tot_wts_per_dsid_*.pkl` (for `lvbb` / `qqbb` / `central` channels).

### Main entry points (from README)
- `python TrainLowLevelReconstruction.py` — train a reconstruction net (signal only) on low-level objects
- `python TrainLowLevelClassifier.py` — classifier (signal1 [vs signal2] vs background) on low-level objects
- `python TrainHighLevelClassifier.py` — classifier (signal vs background) on high-level reconstructed event vars
- `python RunLowLevelInterp.py` — the interpretability playground (loads a trained model, runs the `interp/` tools)

All run files are split into Jupyter cell tags (`# %%`), especially the interp script.

### Research angle worth remembering
`RunLowLevelInterp.py` loads model variants trained with **attention bottlenecks**
(`bottleneck_attention`) and **attention-entropy penalties** ("encourage attention to a
single / two particle(s)") — i.e. training models to be *more interpretable*, then
analysing them. This is the substantive interpretability direction, beyond vanilla probing.

---

## 2. Branches — which one was being worked on

All three branches were last touched 2025-10-29. Diverged from a common base at
`2320310` (2025-06-18).

| Branch | Last commit | vs `main` | Notes |
|---|---|---|---|
| `main` | 2025-10-29 19:47 "Added examples & updated README.md" | — | canonical / cleaned reference |
| **`heppc-work`** | 2025-10-29 16:51 "Added all artifacts from HEPPC" | **2 ahead, 1 behind** | the work-machine branch; **uniquely holds the run-result `.pkl` artifacts** |
| `cleaner` | 2025-05-14 | far behind | old tidy-up branch, stale |

- `heppc-work`'s 2 unique commits: "Added all artifacts from HEPPC" and "Local metric
  storage; some other runs" → experimental result artifacts produced on the HEP PC.
- `main`'s 1 unique commit: a later README/examples update.
- **They are nearly identical in code.** Recommendation: base future work off `main`, pull
  in `heppc-work`'s artifacts (or simply work on `heppc-work`; the difference is tiny).

This local clone is checked out on **`heppc-work`** (all branches fetched).

---

## 3. Data & model weights (the actual blocker)

- Training scripts hard-code an absolute data path on the HEP PC, e.g. (from
  `TrainLowLevelReconstruction.py`):
  `DATA_PATH = /data/atlas/baines/20250321v1_WithEventNumbers_…` and a later
  `/data/atlas/baines/20250429v1_…` variant. **These paths do not exist on this Mac.**
- Data format: **preprocessed per-DSID memmap files + `mean.npy` / `std.npy`**, shape
  `[batch, object, variable]`. Generated by the `preprocessing-scripts/` from raw ATLAS
  ROOT ntuples.
- **Model checkpoints are NOT in the repo.** `.gitignore` excludes `output/`, `wandb/`,
  `tmp*`, `*.png`, `*.pdf`. Trained weights live under `output/` on the HEP PC.
- **To run this repo you therefore need**, from the HEP PC or Sid's external hard drive:
  1. the preprocessed memmap dataset(s) under `/data/atlas/baines/…` (or repoint `DATA_PATH`), and
  2. a trained model checkpoint (or retrain via the `Train*` scripts).

### When the hard drive turns up, look for
- Any `/data/atlas/baines/…` directories (the memmap datasets — names contain `20250321v1`
  / `20250429v1`, `WithEventNumbers`, `WithRecoMasses`, etc.).
- `output/` directories with model checkpoints (`*.pth`) and `wandb/` run logs.

### Can we use LOCAL data instead of the hard drive? (checked 2026-06-08)

**Yes — for the reconstruction task + interp, no hard drive needed.** The local folder
`/Users/sidbaines/Documents/PhD/Work/20250311_MechInterpTmp/data/` holds per-DSID
`.memmap` files (signal DSIDs 510115–510124) in the **same format** this repo's
`dataloaders/lowleveldataloader.py` expects — in fact that dataloader is a near-identical
refactor of the local `mynewdataloader.py`. Verified compatibility:

- Memmap layout `(n_samples, max_objs+2, N_vars+1)` float32, `.shape` file =
  `total_samples,sum_abs_weights,sum_weights` — identical convention.
- Local dims (from file sizes): `112 floats/sample = 14×8` → **`max_objs_in_memmap=12`,
  `N_Real_Vars_In_File=7`** (px,py,pz,E,tag,recoInclusion,trueInclusion; last col is the
  per-object truth label for the reco net). Repo reconstruction config uses `N_Real_Vars=7`
  with the same column meaning → **feature schema matches**.
- `mean.npy`/`std.npy` have 8 values, sliced `[1:]`→7, as the repo expects.

To point the repo at the local data, set the older-schema values:
`max_n_objs_in_file/max_objs_in_memmap=12` (repo default is 15), `has_eventNumbers=False`
(local files predate eventNumbers; splits train/val by index parity), and
`DATA_PATH` → the local `…/20250311_MechInterpTmp/data/` folder. The model
(`TestNetwork`, DeepSets + self-attention) is permutation-equivariant over objects, so
12-vs-15 objects is fine. `TrainLowLevelReconstruction.py` is already `signal_only=True`,
which matches the signal-only local data exactly.

**Unverified (do once an env exists, ~5 min):** that the 7 columns are in the *same order*
(inferred from sizes+code, high confidence). Also, the local `model.pth` was trained by the
predecessor code — it may not load into the repo `TestNetwork` without matching arch config;
easier to retrain a small recon model with repo code on local data, or run the
self-contained local `20250311` code as-is.

**Still needs the hard drive:** the classifier tasks (`TrainLowLevelClassifier.py`,
`TrainHighLevelClassifier.py`) need **background** DSIDs (local folder is signal-only);
pretrained repo checkpoints (gitignored `output/`); and exact reproduction of the
`20250321v1`/`20250429v1` `WithEventNumbers` runs.

#### EMPIRICALLY VERIFIED 2026-06-08 (venv built + data loaded)

A venv was built at `.venv/` (python 3.12; torch 2.12, numpy 2.4, einops, sklearn, uproot,
awkward, vector, wandb, etc. — note these are NEWER than the original pinned versions, so
watch for API drift in the training scripts). Ran `tmp_verify_local_data.py` (gitignored;
at repo root). Findings:

- **Schema/column order CONFIRMED.** Object row = `[type, px, py, pz, E, tagInfo,
  recoInclusion, trueInclusion]`; `type ∈ {0..5}`, `trueInclusion ∈ {0,1,2,3}` (matches the
  `preprocessLowLevel.py` encoding); `mean.npy`/`std.npy` have 8 values → `[1:]` = 7. The
  repo's `lowleveldataloader` loaded it cleanly and returned `x=[B,12,7]`, `types=[B,12]`,
  `y=[B,3]`. Use `max_objs_in_memmap=12, N_Real_Vars_In_File=7, has_eventNumbers=False,
  signal_only=True`.
- **⚠️ The local data is a DEBUG SUBSET, not the full dataset.** Each of the 10 signal
  files (`dsid_510115..510124`) holds **exactly 2000 real events** (contiguous at the
  front), zero-padded up to the original allocation; the `.shape` files still report the
  full original counts (8k–87k). Total ≈ **20k signal events, signal-only**. Loading as-is
  iterates over 75–98% empty (`dsid==0`, all-zero) events — **filter `dsid==0` or slice
  `[:2000]` per file, and set the `.shape` count to 2000**, else empty events pollute as
  garbage class-0 samples.
- **Implication:** fine for a pipeline/interp **smoke test** on this Mac with no hard drive;
  NOT enough to train a performant model or do the classifier (needs background + full
  stats from the hard drive / HEP PC).

#### Local `model.pth` does NOT load into this repo's `TestNetwork` (checked 2026-06-08)

Tried (`tmp_try_load_model.py`, gitignored). `20250311_MechInterpTmp/model.pth` is a
state_dict for the **predecessor architecture** (the scratch `DeepSetsWithResidualSelf
AttentionTriple` class), which differs structurally from the repo's `TestNetwork`:
only **7 of 37** tensors match (the `type_embedding` + `object_net` front-end). The
transformer body differs in naming AND structure — checkpoint has flat
`self_attention`/`2`/`3` + explicit `layer_norm`s + single-Linear `post_attention` +
3-layer classifier; repo has `attention_blocks.{i}.*` with GELU-MLP post-attention + a
1-layer classifier. `load_state_dict` fails (even non-strict, on `classifier.0` shape).
=> That checkpoint is only usable with the local `20250311` scratch code. To use the repo's
clean `interp/` toolkit you must (a) **retrain** a model with the repo's `Train*` scripts,
or (b) obtain a repo-trained checkpoint from the HEP PC / hard drive.

---

## 4. Environment

- **No pinned requirements file in the repo.** Dependencies, from the imports:
  `torch, torchvision, transformer_lens, einops, jaxtyping, uproot, awkward, vector,
  scikit-learn, scipy, seaborn, matplotlib, numpy, pandas, pysr, wandb`.
- **No usable Python environment exists on this Mac yet.** No conda, no pyenv; system /
  Homebrew pythons do **not** have torch or transformer_lens. A fresh venv must be built
  before anything runs. (Python 3.12 is available via Homebrew.)
- `pysr` pulls in a Julia backend on first use — worth knowing it's heavier than a pure-pip dep.
- Devices: scripts handle CUDA / CPU; some scratch versions used Apple `mps`. The HEP PC
  presumably had a GPU.

---

## 5. Related work outside this repo (on this Mac)

These are **scratch / precursor** versions of the same line of work. They live under
`/Users/sidbaines/Documents/PhD/Work/`. **None of the interp scratch folders are
git-backed** (only-on-this-Mac → at risk of loss):

- **`/Users/sidbaines/Documents/PhD/Work/20250311_MechInterpTmp/`** — scratch version of
  this repo's *low-level reco-truth-matching + mech interp* (object-level tagging of which
  objects are W / Higgs decay products). Architecture
  `DEEPSETS_SELFATTENTION_RESIDUAL_X3` with bespoke hooks in `attachModelHooks.py`.
  **Self-contained: has intact data memmaps (signal DSIDs 510115–510124) AND a trained
  `model.pth` (6.1 MB).** → This is the **fastest way to run something on the Mac with no
  hard drive** (needs only a Python env). Recent `output/` run-dirs are empty; last file
  is `tmp/clustering1.ipynb` (2025-03-18). Has some likely small code bugs *(inferred)* —
  mid-debugging when set down.

- **`/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/`** — earlier
  TransformerLens (`HookedTransformer`, ~d_model 64 / 2 layers / ~5 heads) + a custom small
  SAE on the residual stream; *event-level* classification (signal lvbb / qqbb / bkg).
  **Data is MISSING** (`tmpdata/` empty, `data` symlink broken). Empty `.venv`. Latest
  activity = Jan-2025 toy adversarial / feature-decorrelation experiments. Note: a clone of
  `ai-safety-foundation/sparse_autoencoder` sits in this folder but the latest scripts use
  a custom SAE, not that library.

- **`/Users/sidbaines/Documents/PhD/Work/HplusWh/20251212_MassSculpting/`** — Dec-2025
  physics study of background mass-sculpting (the decorrelation theme). Git-tracked but
  **local-only (no remote)**; has ~1 GB of background ROOT files and its own `env/` venv.
  Not mech-interp, but the same intellectual thread.

- `/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel/` — the original physics ML repo
  (GitHub `SidBaines/HPlusWhBoostedMachineLearningEventSelection`, only `main` on the
  remote). 2022–2023 NN/event-selection notebooks; predates the interp work.

---

## 6. Suggested next steps (nothing here has been done yet)

1. **Quick local sanity run (no hard drive needed):** stand up a Python env and run the
   self-contained `20250311_MechInterpTmp/` (it has data + `model.pth`). Optionally check
   whether its data + checkpoint are loadable by *this* repo's `lowleveldataloader.py` /
   `models.py` — if so, the clean `interp/` toolkit can be exercised on local data
   immediately. **(Compatibility is unverified.)**
2. **Recover the real data + weights** from the HEP PC / external hard drive (see §3),
   then either repoint `DATA_PATH` or place files under `/data/atlas/baines/…`.
3. **Build the env** for this repo (deps in §4). Consider pinning a `requirements.txt` once
   it works, since the repo lacks one.
4. **Back up the at-risk scratch folders** (§5) — they are not in any git repo.
5. Decide whether to consolidate `heppc-work`'s artifacts onto `main`.

---

## 7. Open questions for Sid (as of 2026-06-08)

- Which task is the priority — *object-level* decay-product tagging (the reconstruction /
  truth-matching line, = `20250311_MechInterpTmp` scratch) or *event-level* channel
  classification? Both are in this repo.
- What does the external hard drive actually contain (datasets? checkpoints?)?
- Work off `main` + cherry-pick artifacts, or just continue on `heppc-work`?

---

## 8. Resume checklist (paused 2026-06-08, waiting on the external hard drive)

State at pause: repo cloned & on `heppc-work`; a working venv exists at `.venv/`
(git-excluded); local data + checkpoint compatibility fully investigated. The user chose to
pause and resume once the hard drive (with the full datasets + repo-trained checkpoints) is
found.

**Already done — don't redo:**
- `.venv/` built (python 3.12; torch 2.12, numpy 2.4, einops, sklearn, uproot, awkward,
  vector, wandb, matplotlib, seaborn, scipy, jaxtyping, pandas). NOT installed:
  `transformer_lens`, `pysr` (Julia), `torchvision` — add these only for the high-level
  classifier / symbolic-regression / comparison scripts.
- Confirmed local data schema is compatible (see §3) but is only a 20k-event signal-only
  smoke subset, and the local `model.pth` is NOT loadable by the repo (predecessor arch).
- Scratch verification scripts live at repo root (gitignored `tmp*`):
  `tmp_verify_local_data.py`, `tmp_try_load_model.py`.

### ✅ STATUS UPDATE 2026-06-09 (late evening): cluster reachable, transfer IN PROGRESS

- **The HEP PC is alive and reachable**: `baines@heppc402.ph.qmul.ac.uk` (QMUL). The
  "recover from hard drive" framing is obsolete — data can be rsynced straight off it.
- An rsync of `/data/atlas/HplusWh/20250313_…DeltaR0.5/` → `/Volumes/Seagate/` was running
  (started ~23:27, under `caffeinate`). At ~23:45 it had 2.8 GB / 113 files, mid-407343;
  alphabetical order means ttbar 410xxx and **signal 510115–124 come near the end** — don't
  judge completeness until it exits. Verify afterwards with a file-count/size diff vs cluster.
- Seagate has **874 GiB free** → easily room to ALSO pull the derived memmaps
  (`/data/atlas/baines/2025…`) and the cluster repo's gitignored `output/` checkpoints +
  `wandb/`, which this transfer does NOT include. Strongly recommended while access lasts.
- No `.pth` / `.memmap` / `mean.npy` anywhere else on the Seagate (deep search done).

### Cluster data provenance (for copying source data off the cluster — added 2026-06-09)

The memmaps are derived. The data chain on the cluster:
- `/data/atlas/HplusWh/20241021_RawNtuples/` — INPUT to the C++ stage. Itself a local
  cluster copy (~2024-10-21) of CERN EOS "L1 ntuples": `/eos/user/l/lubaines/ATLAS_SHARE/
  HpWh_L1ntuples/` (lubaines = Sid) + `/eos/user/t/tqiu/H+Wh_ntuples/` (older: blumen,
  adsalvad). Those are standard ATLAS flat ntuples produced on the grid from DAOD_TOPQ1
  derivations of mc16a/d/e MC (`user.rhulsken.mc16_13TeV.<DSID>…TOPQ1…Nominal_v0_1l_out`).
  Need this + a local ROOT toolchain only if varying physics-level choices. (Runner skips
  `*CORRUPTED*` dirs — cluster copy may be partial.) Below this = EOS / grid re-derivation.
- C++ stage detail: `cpp_code/mainCodeRunner.sh` runs `bin/roo MCbase.cpp <sample> <out>`
  over `20241021_RawNtuples/`. `MCbase.cpp` is the CONFIG (parsed by `configparser.h`):
  `path=…20241021_RawNtuples/`, `low_level_delta_R_ljet_sjet_cut=Enable` (→ the DeltaR0.5
  removal), MET 30 GeV, b-tag WP 77p, categories 0–10, ≥1 large-R jet; `EventLoop.C` builds
  reco + truth-inclusion labels. xsecs from PMG DB (`main/xsec.h`). NOTE: runner is left
  mid-experiment — writes to `20250610.tmp2/` and filters to ONLY dsid 510120, so that dir
  is partial; `20250313_…DeltaR0.5/` is the COMPLETE processed set. `bin/roo` is a committed
  LINUX binary — would need recompiling against ROOT on the Mac.
- **`/data/atlas/HplusWh/20250313_WithTrueInclusion_FixedOverlapWHsjet_SmallJetCloseToLargeJetRemovalDeltaR0.5/`**
  — the "top-level data we process": INPUT to `preprocessing-scripts/preprocessLowLevel.py`
  (hard-coded `DATA_PATH`, line ~383). Contains `user.*.root` per DSID (signal 510115–124 +
  all backgrounds). **Copy this whole dir to regenerate memmaps with any *preprocessing*
  settings (object count, phi-rotation, MET cut, tag info, normalization…) and retrain
  locally — Python only.** (Check `ls -lt /data/atlas/HplusWh/` for a newer set, e.g.
  `20250610.tmp2` appears in `cpp_code/mainCodeRunner.sh`.)
- `/data/atlas/baines/2025…WithRecoMasses_15…` — the derived memmaps actually trained on;
  `output/` (gitignored on cluster) holds repo-trained checkpoints. Grab these too only to
  reproduce exact prior results.

Transfer: `rsync -avh --progress USER@CLUSTER:/data/atlas/HplusWh/20250313_…DeltaR0.5/ /Volumes/<drive>/HplusWh_20250313/`.

### What the C++ stage removes vs keeps — event selection (added 2026-06-09)

The `20250313_…` files ARE event-filtered (not pure object selection), but only lightly.
Because the config sets `write_all_events = Enable`, the **only** hard event drop is in
`LowLevel_Loop()` — `main.C:168-169` (`LowLevelPass = LowLevel_Loop(); if (!LowLevelPass) continue;`).
The category gate on `main.C:172` (`if (WriteAllEvents || category∈{0,3,8,9,10})`) is
**neutralised** by `WriteAllEvents=true`, so all categories are written; and the reco
`Loop()`'s internal cuts (MET, lepton-pT, mass windows, category) compute variables but do
NOT gate the `Fill()`.

Events REMOVED:
- **0 large-R ("fat") jets → dropped.** Hard-coded in `LowLevel_Loop` (`EventLoop.C:~1306`:
  `if (ljetCandidates.size()==0) return false;`). The optional large-R pT/mass/ΔR-to-lepton
  cuts are all DISABLED here, so any large-R jet counts. No config flag for this — keeping
  0-ljet events would need a code edit.
- **Exactly 1 lepton** — inherited from the upstream `_1l_out` ntuples; `LowLevel_Loop` takes
  the leading electron (else leading muon). (The stricter `Leptons.size()!=1` cut at
  `EventLoop.C:3036` is in the reco `Loop()`, which doesn't gate writing here.)

Events KEPT (NOT cut here, despite the directory name): no MET cut; no W/Higgs mass window;
no lepton-pT / W-pT / angle cuts (all `Disable`d); no category selection (all categories incl.
1,2,4–7,11 and unreconstructable `-1` are written); no b-tag/Xbb requirement (Xbb only sets a
jet's *type*).

Object-level pruning (event kept, objects dropped): small-R jets within ΔR<0.5 of any large-R
jet removed (`EventLoop.C:1333`, `LowLevelDeltaRLjetSjetCut=Enable` → the "DeltaR0.5" in the
name). A neutrino is built from MET+lepton; per-object `recoInclusion`/`trueInclusion` labels
and best reco Wh masses (`mH`, `mWh_qqbb`, `mWh_lvbb`) are added.

⇒ Copying `20250313_…` keeps the full **1-lepton, ≥1-large-R-jet** population with all reco
categories + truth labels. The ONLY permanent loss vs `20241021_RawNtuples/` is **0-large-R-jet
(purely resolved) events**. The event-removals you'd most likely want to *vary* (MET cut,
Xbb-required, mH-selection, drop-uncertain-truth, drop-events-where-truth-cut-by-max-objs) all
happen LATER in the Python `preprocessLowLevel.py` → fully controllable by re-running the Python.

### ⚠️ TO INVESTIGATE: object-type / `N_CTX` encoding (3 vs 5) may differ between old local data and current code

Current `LowLevel_Loop` (`EventLoop.C:~1295-1340`) assigns object types:
`0=electron, 1=muon, 2=neutrino, 3=non-Xbb large-R jet, 4=small-R jet, 5=Xbb-tagged large-R jet`.
BUT the old local `20250311_MechInterpTmp/data/` memmaps appeared to use **`5 = padding`** with
all large-R jets as a single type `3` (no separate Xbb type) — see the verification dump where
the empty trailing object rows had `type=5`.

**This is NOT confirmed, and must be investigated before feeding the old memmaps to the current
model** — crucially, **the old local data files may have been generated BEFORE this code change**,
so the apparent mismatch could be real, or could just reflect an older encoding. Steps:
(a) inspect what type values + counts the old local memmaps actually contain (incl. the padding
token); (b) determine what the *current* `preprocessLowLevel.py` + `LowLevel_Loop` write (incl.
padding token and `N_CTX` — `TrainLowLevelReconstruction.py` uses `N_CTX=7`); (c) reconcile.
A mismatch would silently break things: the model's `key_padding_mask` uses
`object_types == num_particle_types-1`, and the type-embedding size = `num_particle_types`.

**When the hard drive arrives:**
1. Locate `/data/atlas/baines/…` memmap datasets (full stats, incl. background DSIDs) and
   any `output/` repo-trained checkpoints (`*.pth` matching `models.models.TestNetwork`).
2. Point `DATA_PATH` in `RunLowLevelInterp.py` / `Train*` at the recovered data (these use
   `max_n_objs_in_file=15`, `has_eventNumbers=True` — the NEWER schema, not the local 12/no-
   eventNumber subset).
3. If a repo-trained checkpoint exists → load it into `TestNetwork` and run
   `RunLowLevelInterp.py` (the clean `interp/` toolkit). If not → retrain via
   `TrainLowLevelReconstruction.py` first.
4. Watch for API drift: the venv has newer library versions than the original pins
   (`MLEventSel_OnlyImportant/myenv.requirements.txt` had torch 2.5.1 / numpy 2.1.2 etc.).
   If something breaks, pin down to those.

**No-hard-drive fallback (smoke test only):** retrain a quick `TestNetwork` on the local
20k subset (filter `dsid==0` / slice `[:2000]`, set `max_objs=12, N_vars=7,
has_eventNumbers=False, signal_only=True`), then run the interp toolkit — model will be weak
but the pipeline is exercised. Or run the self-contained `20250311_MechInterpTmp` scratch
code (its `model.pth` + `attachModelHooks.py` already match).

# 2026-06-09/10 — Planning session + data recovery kickoff

## What we did

- **Discovered the data recovery is live, not stalled**: a `caffeinate`d rsync from
  `baines@heppc402.ph.qmul.ac.uk:/data/atlas/HplusWh/20250313_…DeltaR0.5/` →
  `/Volumes/Seagate/` has been running since ~23:27 on 06-09 (≈4.8 GB / 106 ROOT files by
  mid-session; copies alphabetically so signal DSIDs 510115–124 land last). Confirmed the
  Seagate has 874 GiB free and nothing else physics-related on it.
- **Surveyed the whole codebase** (3 parallel deep-dives: training/models, interp toolkit,
  metrics/results). Headlines: solid MC-weighted metrics + physics-aware architecture;
  interp library (`interp/`) is decent but `mechinterputils.py` (~3k lines) duplicates it
  and contains a syntax error (`wefwef`, ~line 856); 9 model variants live as `if 0:`
  blocks in `RunLowLevelInterp.py:174-361` (line ~320 marks the thesis model:
  entropy-penalty + bottleneck-1); no test split (50/50 parity only); hard-coded 0.75
  training fraction in `TrainLowLevelClassifier.py:321`; configs not serialized with
  checkpoints; 5× `PlotAllmWhDists` variants unconsolidated.
- **Read thesis ch. 7 + App. ZD** (`~/Documents/PhD/PhDThesis/Chapters/`). Performance
  story is complete (transformer ≫ DNN, better expected limits). Interp section is the
  frontier: attention bottleneck + entropy penalty (claimed novel in HEP) with quantified
  performance cost; attention-pattern figures; head-ablation tables (redundancy in cats
  4–5); robustness studies (lepton angles critical, neutrino angles irrelevant ⇒
  neutrino class inherited from lepton). Probes/DLA/PySR tried, "no insights yet".
  Several `\note{}` correction gaps incl. a lost qqbb ROC plot ("code on hepcluster which
  I can no longer access" — now recoverable!).
- **Wrote the research plan**: `docs/RESEARCH_PLAN_2026-06.md` (phases: foundations →
  enablers → model organisms + tradeoff curve → circuit hunting H1–H4 → presentation).
- Gave Sid probe commands for heppc (find `output/` checkpoints, wandb dirs, memmaps);
  he's running them now.

## Decisions

- **Primary goal = interpretability research for its own sake** (thesis submitted, viva
  passed, only corrections remain). Secondary: cleaner plots → corrections; eventually
  interactive demos for the public repo.
- Plan approach agreed: model-organisms-first (minimal models on easy categories, then
  transfer findings up), H1 (neutrino-pairing circuit) as opening target. No pushback yet;
  Sid will flag any as setup proceeds.
- Adopted this research-log convention (now in `CLAUDE.md`).

## Learnings

- **wandb creds in `~/.netrc` are stale** ("relogin required") — needs
  `wandb login --relogin` with a fresh key before we can mine old run configs.
- **The thesis-era work was done on a THIRD machine** (not this Mac, not necessarily
  heppc) — so heppc may or may not hold the actual thesis-model checkpoints. Sid's probe
  results will tell; if machine #3 holds them, ask whether it's still accessible.
- Claude's sandbox cannot ssh to heppc (permission-gated) — give Sid commands to run, or
  he can use `! <cmd>` in-session.
- The rsync flags that work for the exFAT Seagate: `-rlt --partial --info=progress2
  --modify-window=1` (no `-a`; perms/owners unsupported). The other drive
  ("Seagate Expansion Drive") is NTFS = read-only on macOS — don't target it.
- Compute: new MacBook M5 Pro / 48 GB arrives ~2026-06-11; work moves there. RunPod
  available as fallback (classifier training on 2.7M bkg events may want it; signal-only
  reco training should be fine on MPS).

## heppc inventory (from Sid's probes, late 2026-06-10)

- `~/Code/` repo clones: **`ChargedHiggs_CodeForThesis`** (1.5G, last touched 2025-06-18;
  `output/` = PlotsForThesis + yields only — plotting, not training) and
  **`ChargedHiggs_ExperimentalML`** (2.3G, last touched 2025-10-29 = the `heppc-work`
  clone; `output/` full of TrainingOutput dirs incl. a 2025-07-08..11 burst, + `wandb/`).
  Also `ChargedHiggs_ProcessingForIntNote` (4.6G, contains a `particle_transformer/` —
  brief ParT benchmarking), `ChargedHiggsML2025Jan` (548K), `20250609/` (28M).
- `/data/atlas/baines/`: the full memmap-dataset zoo, names = preprocessing flags. Thesis-
  relevant: `20250321v1_…15_MetCut…` (reco), `20250321v2_AppliedRecoNNSplit_…30…`
  (classifier), `20250429v1_…OnlyLjetBosonTruth`, high-level dirs, 2025-06-18/19 sets.
  **`TmpCommonModelResults/`** (12 subdirs) = the model store the `CompareAllModels*`
  scripts read. Mystery **`20250711_ChargedHiggsCode/`** code copy (matches the July
  TrainingOutput burst).
- **Bonus**: Nov-2024 `tmp_shuffled_*.bin` + `models/` (97 subdirs, Oct-2024) are almost
  certainly the missing data+models of the old TransformerLens/SAE event-classifier work
  (`MLEventSel_OnlyImportant` on the Mac, broken `data` symlink) — that thread is
  recoverable too.
- Sid confirms heppc most likely IS the thesis-training machine ("third machine" mystery
  resolved); access retained for a while, so urgency reduced.
- heppc git is ancient (no `git -C`); write probe commands accordingly.
- Recovery queued: 5 repo dirs (~8.5 GB) → `/Volumes/Seagate/heppc_recovered/Code/`,
  chained after the ROOT rsync via ssh ControlMaster (auth up front) + pgrep-wait.
- Decisions: model-organisms approach + H1 opener confirmed by Sid, with the explicit
  precondition that the lepton→neutrino association must be re-verified in each organism.

## Late-night probe round 2 + overnight queue (2026-06-10, ~00:30–01:00)

- **`ChargedHiggs_CodeForThesis` is NOT a git repo** (plain copy) — the thesis-plot code
  state exists only there → wholesale copy is the most valuable recovery item.
- `ChargedHiggs_ExperimentalML` = our `heppc-work` @ `9fbe40e`, clean (only `__pycache__`
  untracked) → no unique code; value = `output/` (TrainingOutputs 2025-02 → 2025-07) +
  `wandb/`. Remote is `git@github.com:SidBaines/ChargedHiggsML2025Jan.git`.
- Dataset sizes: `20250321v1` reco 12G; `20250321v2` classifier 17G; `20250429v1` 1.3G +
  high-level 761M/22M; June-2025 sets 4.1–4.4G ×4 (one empty: `20250618v1_…RemovedWrong
  TruthForTraining…` = 0); `TmpCommonModelResults` 3.2G (run dirs 2025-02-11→03-10);
  `20250711_ChargedHiggsCode` 1.6M (code snapshot). Round-3 total ≈ 51G — take all.
- Overnight queue verified on the Mac: ROOT rsync running; ssh ControlMaster alive;
  watcher chain healthy BUT Sid's `caffeinate` prefix was lost → fixed by attaching
  `caffeinate -i -w <watcher-pid>` (nohup'd). Learning: **verify the process tree after
  launching chained jobs** — a dropped wrapper is silent until 3am.
- Round 3 (datasets, ~51G) command handed to Sid for the morning; optional round 4 =
  Oct/Nov-2024 `models/` + `tmp_shuffled_*.bin` (the old MLEventSel/SAE-era artifacts).

## Morning of 06-10: overnight results + the transfer debugging saga

**Outcome: ROOT transfer ✅ complete** (31G, 678 `.root` files, signal 5101xx ×102 +
V+jets 700xxx ×323 present, zero leftover rsync partials). Repo+dataset transfer
restarted cleanly at ~09:45 via `tmp_recover_heppc.sh` (gitignored, repo root) after a
chain of failures, each with a lesson:

1. **The queued repo rsync hung all night at a password prompt.** The overnight ssh
   ControlMaster died (idle gateway kill); ssh hit `Broken pipe` on the dead socket,
   fell back to a fresh connection, and sat at `password:` from 05:29. Diagnosis trail:
   0.18s CPU over 3h, zero network, nothing written, then the prompt visible in the tty.
2. **Root cause discovered: heppc402's final hop was password-auth** (Sid's `id_ed25519`
   wasn't in its `authorized_keys`; only the ITS-gateway hop used a key). **Fixed
   permanently** with `ssh-copy-id` → passwordless verified. Queued/unattended transfers
   are now actually safe.
3. **Modern-rsync gotcha**: rsync ≥3.2.4 (local 3.4.1) no longer lets the remote shell
   word-split a quoted `"path1 path2"` source list → `chdir "…path1 path2…"` failure.
   Use one source per arg: `host:path1 :path2 :path3`. Also: terminal copy-paste had
   injected literal newlines into the quoted list (use script files, not pasted
   multiline commands → hence `tmp_recover_heppc.sh`).
4. **The Seagate "two drives" were one drive all along**: a single NTFS disk. Writable
   mount = ntfs-3g/macFUSE at `/Volumes/Seagate`; after any unplug, macOS auto-mounts it
   READ-ONLY as `/Volumes/Seagate Expansion Drive` (fskit). **Remount recipe** (from zsh
   history):
   ```
   diskutil unmount /dev/disk4s1
   sudo mkdir -p /Volumes/Seagate
   sudo /opt/homebrew/bin/ntfs-3g /dev/disk4s1 /Volumes/Seagate \
     -o local,allow_other,auto_xattr,windows_names,uid=$(id -u),gid=$(id -g)
   ```
   (If "dirty volume": `sudo ntfsfix -d /dev/disk4s1` first. Device id can change —
   check `diskutil list external`.)
5. Laptop moved mid-morning → internet + drive disconnected; harmless (no active
   writes), but triggered the read-only remount above.

**State at session pause**: `tmp_recover_heppc.sh` running under caffeinate — stage 1
repos (~8.5G, ~45k files) then stage 2 datasets (~51G) → `/Volumes/Seagate/heppc_recovered/`.
Reconvene when done.

## Midday 06-10: drive corruption episode + THESIS MODEL RECOVERED & LOADED

**Second yank (mid-write this time) corrupted NTFS dirs** under `CodeForThesis/lwtnn/build/`
(EIO on stat/mkdir; `ntfsfix` clears the dirty flag but NOT index corruption; rsync was
silently skipping corrupt dirs = hole risk). Fix: killed transfer, `rm -rf` partial copy
(ghost dirs that survive rm parked at `Code/.CORRUPT_GHOSTS_ignore/` — only a Windows
chkdsk can reclaim), added `--exclude=lwtnn/build` (disposable Boost/Eigen tree, ~60k
files) to `tmp_recover_heppc.sh`, re-ran. **Graceful-pause ritual: Ctrl+C → `diskutil
unmount /Volumes/Seagate` → unplug.**

**Recovery status**: ROOT 32G ✅ (verified readable); repos 5.1G ✅; datasets ~51G ⏳.

**Verification wins (while datasets transfer):**
- `ExperimentalML/output/`: **876 training runs, 2250 checkpoints** recovered. All
  model-variant runs referenced in `RunLowLevelInterp.py:174-361` present EXCEPT
  `20250328-113053` (missing on heppc too, presumably deleted).
- **THE THESIS MODEL LOADS**: `20250512-093728…/chkpt29_164490.pth` → strict
  `load_state_dict` into repo `TestNetwork` (config from `RunLowLevelInterp.py:319-340`:
  d=152, 3 blocks, 4 heads, MLP 400, bottleneck=1, entropy-trained) ✅ + clean forward
  pass. 676,587 params. Test script: `tmp_load_thesis_model.py` (gitignored).
- **Type-encoding question RESOLVED for the thesis era**: checkpoint `type_embedding` is
  `[6,6]` → **N_CTX=6 = 5 object types + padding=5** (OLD encoding, same as the local
  20250311 scratch memmaps; no separate Xbb type). `RunLowLevelInterp.py:79` (`N_CTX=6`)
  is the active branch. ⇒ When regenerating memmaps for thesis-model comparisons, must
  reproduce 5-types+padding=5 (check how `preprocessLowLevel.py` maps the C++ type 5=Xbb
  → presumably collapses to 3; verify when regenerating).
- **Sid's last pre-pause run (20250709-095246) is a proto-model-organism**: d=20,
  2 blocks, 4 heads, entropy penalty, NO bottleneck → loads fine, **20,559 params**.
  Phase 2 has a head start.
- July 2025 high-level runs saved `*_config.json` next to outputs — config serialization
  was already started for the high-level task; extend the pattern to low-level (Phase 1.2).

## Afternoon 06-10: model registry built+validated; first interp pass of the new era

- **`models/registry.py` created (Phase 1.3 done, uncommitted)**: the nine `if 0:` blocks
  of `RunLowLevelInterp.py:174-361` are now `VARIANTS` (dataclass entries with kwargs +
  checkpoint relpaths + provenance) and `load_model(name)`, which **auto-registers the
  bottleneck forward hook** — critical, because `TestNetwork.forward()` does NOT apply
  the bottleneck (docstring "kinda hacky"); validation showed max|Δlogit| up to ~79
  between hooked/unhooked. Validation (`tmp_validate_registry.py`): **all 8 recovered
  variants strict-load and run** (`plain-d152-ln` checkpoint lost). Key entries:
  `thesis-ent1-bn1-d152` (677k params), `plain-d152` (unconstrained twin baseline),
  `ent1-d20-2blk` (20k-param proto-organism).
- **Thesis model run over local 20k signal events** (`tmp_thesis_model_local_pass.py`,
  truncated /tmp copies, LOCAL norms → qualitative only):
  - **Attention entropy confirms single-particle training worked**: block 2 heads at
    0.025–0.24 nats (block 0: 0.45–1.7).
  - **Type→type attention heatmaps reproduced** (`tmp_plots_attention/*.png`), rich
    structure: b0h2 = "collect sjets" head; b0h3 = leptons read neutrino; **b2h3 =
    neutrino reads leptons (H1's predicted direction!)**; b1h1/b2h1 = everything reads
    ljet; b2h0 = everyone reads leptons.
  - ⚠️ **Quantitative reco accuracy on local data is BAD** (~0.47 per-object, 0 perfect
    events, predictions collapse to class 0; excluding truth==3 only →0.54). NOT
    surprising-in-hindsight: local 20250311 derivation differs from training data
    (truth scheme "OldTruth"? + non-matching normalization constants). **Do NOT chase
    this**; redo quantitatively on `20250321v1` (its first files have already landed,
    incl. `mean.npy`/`std.npy`) — that's the real thesis-numbers sanity check.
- wandb still "relogin required" (CLI's "already logged in" only checks .netrc exists).
  Sid to `wandb login --relogin` with a fresh key.
- Sid's direction: registry ✓, thesis-model pass ✓, but expect to TRAIN OUR OWN organisms
  (better frontier expected with current codebase+context) rather than only reuse old ones.

## wandb mined (after Sid's relogin, afternoon 06-10)

- Entity `luke-sid-baines-blank`; reco runs live in **`HEP-Transformers-TruthMatchingReco`**
  (203 runs; names = output-dir timestamps + variant code, e.g.
  `_20250512-093728_LowLevel_DSSARVTSBN3_YesEnt1_YesBn` = thesis model); classifier/
  high-level runs in `HEP-Transformers` (341 runs).
- **Hyperparameters were NOT logged to wandb** (config = `{magic: enable}` only) → the
  registry + script blocks remain the only config source. But **final summaries are
  rich**: archived to `docs/run_configs/*.json` for all 9 registry runs (incl. the
  lost-checkpoint `20250328-113053`).
- **Thesis-model quantitative targets now on file** (val PerfectRecoPct by mass):
  0.8 TeV → 0.698, 1.0 → 0.784, 1.4 → 0.856, 2.0 → 0.902, 3.0 → 0.932. These are the
  numbers the 20250321v1 rerun must reproduce (Phase 0.5 gate).
- **Cross-check win**: wandb-logged final per-head attention entropies (block 2:
  0.024/0.023/0.17/0.11) match what we measured locally on different data
  (0.025/0.048/0.14/0.24 in the same ranking) → our hook/eval pipeline reproduces
  training-time behavior.

## Evening 06-10: ✅ PHASE 0 GATE PASSED — thesis numbers reproduced to 4 s.f.

- **Normalization discovery**: thesis pipeline used `SCALE_DATA` (4-momenta / 1e5, no
  mean-subtraction); `mean.npy`/`std.npy` were NEVER used (`NORMALISE_DATA=False`,
  `RunLowLevelInterp.py:46-47,120-130`). This was the cause of the morning's bad local
  numbers: with correct scaling, local per-object acc 0.47→0.79, and the confusion matrix
  (saved `tmp_plots/confusion_local_thesis_model.png|.npy`) shows the residual gap is the
  local subset's OLD 4-class truth scheme (truth-3→pred-2 at 95%).
- Signal memmaps + norms of 20250321v1 side-fetched to SSD
  (`tmp_data_20250321v1_signal/`, 1.4G) — bandwidth tip: suspend the main rsync
  (Ctrl+Z/fg) while small fetches run.
- **`tmp_eval_thesis_20250321v1.py`** (full val split, has_eventNumbers=True =
  TRAINING split; the interp script itself had quietly used the index-parity default):
  recovered thesis model val per-category PerfectRecoPct = **0.3097 / 0.3709 / 0.2917 /
  0.4959 / 0.9587 / 0.9271** vs thesis Table `RecoTransformerModifications`
  (bottleneck+entropy row) **0.3101 / 0.3709 / 0.2917 / 0.4957 / 0.9587 / 0.9272** —
  match to ~4 s.f. Per-mass PerfectRecoPct 0.738(0.8 TeV)→0.941(3.0 TeV).
- Per-mass values sit +1–4% above the wandb-logged epoch-29 numbers — those were logged
  during training (different buffering); the dedicated-eval thesis table is the standard
  and is matched exactly. ⇒ **environment + data + model + metrics pipeline fully
  verified end-to-end on this Mac.** Phase 0.5 done; remaining Phase 0 item = memmap
  regeneration from ROOT (type-encoding check) which is now non-blocking for interp.

## Late afternoon 06-10: H1 investigation round 1 (thesis model, 20250321v1 val signal)

Scripts: `tmp_h1_step12.py`, `tmp_h1_step3_ablate.py` (+ inline experiments). ~50k events.

**Established facts:**
1. **Truth-label encoding (file)**: {0:none, 1:H, 2:W-hadronic, 3:W-leptonic} → training
   collapses to model classes {0,1,2,2}. Leptons/ν only ever {0,3}; jets {0,1,2}.
   (Also fully explains the morning's local-data confusion matrix.)
2. **Behavioral coupling**: P(pred_ν = pred_lep) = 0.9953; truth(ν)=truth(lep) always.
3. **Channel-conditional attention routing**: ν→lep mean attention in true-Wlν vs other:
   b1h3 0.49/0.08, b2h0 0.49/0.01, b2h3 0.40/0.00 (b0h0 always-on 0.54/0.40). The
   late-head attention pattern itself encodes the (already-computed) channel decision.
4. **The b2h3 bottleneck scalar into ν separates true channel at AUC 0.83**
   (`tmp_plots/h1_b2h3_message_hist.png`).
5. **Ablations (per-head exact subtraction, verified Σ per-head ≈ block output, rel ~5e-3
   — residual = shared out_proj bias):** REFUTE the naive "b2h3 is THE carrier" picture.
   b2h3/b2h0/b1h3 individually or jointly: ν-acc essentially unchanged (0.96). Largest
   single-head hit to lep/ν = **b0h0** (0.96→0.89); b0h1 is the jet-relevant head
   (jet 0.95→0.90, evt-perfect 0.85→0.64). Even ablating ALL of block 0: ν-acc 0.82,
   **P(ν=lep) ≥ 0.97 under every ablation tried** — coupling is hyper-redundant.
6. **Input corruptions**: randomizing lepton φ/pz → only 2.8%/3.1% of ν/lep predictions
   flip; ν corruption even less (1.2%); ljet φ-rotation 1.4%, **ljet mass ×0.5 only
   4.5%**, ljet tag-zero 2.3%. P(ν=lep) stays ≥0.99 in all cases.

**Interpretation / status of H1:** the naive message-passing circuit (lepton →[one head]→
neutrino) is WRONG. The ν/lep "W vs none" assignment is computed globally and
redundantly: no single head, block-0 ensemble, or single-object kinematic corruption
breaks it. Possible reframings for next round: (a) decision is event-global (residual
streams progressively merge: cos(ν,lep) = 0.65→0.79→0.88 entering blocks 0/1/2; all
tokens may converge on a shared channel representation); (b) flips concentrate in
low-margin events — measure flip rate vs logit margin; (c) feature-sweep/saliency to
find which inputs DO drive the decision (untested candidates: lepton/ν pT, MET-related
magnitudes, sjet system, multi-object combinations); (d) compare with the 20k-param
`ent1-d20-2blk` organism — less capacity for redundancy, maybe a crisper circuit.

**Methodology learnings:**
- For bottleneck models the per-head cache decomposition enables clean ablations:
  register `hook_attention_heads` first, then a subtracting hook (chained hooks receive
  the previous hook's modified output).
- `hook_attention_heads` returns BOTH attention modules and MLP Sequentials — filter by
  module type before indexing by block.
- Inline heredoc python via Bash kept corrupting long f-strings (stray tokens) — write
  scripts to files (`tmp_*.py`) instead of long heredocs.

## Evening 06-10: first locally-trained model organism LAUNCHED (MPS)

- **Training works on this Mac**: `tmp_train_organism.py` mirrors the
  `TrainLowLevelReconstruction.py` entropy-penalty recipe (Adam, basic_lr_scheduler
  3e-4→5e-7 log-decay, warmup 100, wd 1e-6, batch 4096, entropy_weight 1e-2 target 0,
  30 epochs) on the SSD signal data (1.40M train / 1.40M val, eventNumber split).
  Differences from original: uses `interp.activations.hook_attention_heads(detach=False)`
  for the grad-attached attention cache (mechinterputils needs pysr, not installed), and
  **serializes full config to wandb AND `output/<ts>/config.json`** (Phase 1.2 pattern).
- Smoke test (30 steps) ✓ on MPS; full run launched ~16:17: run
  `_20260610-161723_LowLevel_ORG2026_d20b2_YesEnt1_NoBn` (wandb ag6vm16o), d=20,
  2 blocks, 4 heads, 20,559 params — architecture-identical to old `ent1-d20-2blk`.
  ETA ~2-2.5h (dataloader-bound, ~1.3 it/s).
- **Comparison target** (old organism, 10 epochs, from archived wandb summary):
  val PerfectRecoPct_all 0.8055 (0.652@0.8 TeV → 0.907@3.0 TeV).
- Gotchas hit: `HEPLossWithEntropy` returns `(loss, dict)` tuple; `mechinterputils`
  import fails on missing pysr (avoid importing it); `compute_and_log` prints the whole
  metric dict (noisy but harmless).
- **Attempt 1 NaN'd late in epoch 0 (MPS)**: healthy to step 300 (loss 0.27), then NaN →
  all-zero val metrics → "Only self-attention is supported" error (NaN≠NaN breaks the
  q==k identity check in `activations.py:144` — that error is a SYMPTOM of NaN weights,
  not a hook bug). Root cause (likely): padding objects are all-zero ⇒
  `pt=sqrt(0)` has infinite gradient and `eta=asinh(0/0)` NaNs in backward
  (`models.py:66-67`; `nan_to_num` only fixes forward). CPU/CUDA survived this for the
  2025 runs; **MPS doesn't**. Fix in trainer: `sanitize_padding()` (benign nonzero
  4-momenta for padding rows — they're masked everywhere) + grad-norm clip 1.0 +
  non-finite-loss batch skipping. Attempt 2 launched ~16:25
  (`_20260610-1625*_LowLevel_ORG2026_d20b2_YesEnt1_NoBn`). If training models on MPS
  again (the M5!), keep these guards.

## Night 06-10: organism attempt 2 crashed the MACHINE; relaunched on CPU

- **Attempt 2 (gg9bt2mj) was healthy but took the whole Mac down at 16:48** (hard reset:
  `ResetCounter` diag 16:48:51; wandb internal log stops mid-stream 16:46:59, no
  traceback anywhere — the process tree died with the OS). Sid confirms the machine
  crashed. Strong suspicion: Metal/MPS kernel panic under the sustained training load
  (the NaN guards themselves worked: 1 non-finite grad-norm skipped, training continued).
- **Before dying it validated the recipe**: epoch 0 val PerfectRecoPct_all = **0.7295**
  (lvbb 0.827 / qqbb 0.533) after ONE epoch vs the old organism's 0.8055 after 10 —
  comfortably on track. `chkpt0_343.pth` + `config.json` saved.
- **CPU smoke test passed** (30 steps in ~36 s ≈ 1 s/step — comparable to MPS, which was
  dataloader-bound anyway, and immune to Metal panics). Added `--device` flag to
  `tmp_train_organism.py`.
- **Attempt 3 launched on CPU** ~20:33 (`OMP_NUM_THREADS=8`, nohup + caffeinate -i,
  log `tmp_organism_train_cpu.log`). ETA ~3.5–4 h.
- **Learnings**: (i) on this Mac, treat sustained MPS training as a machine-crash risk —
  for small organisms CPU is just as fast (tiny model = dataloader-bound); on the M5,
  smoke-test MPS with a short run before trusting it overnight. (ii) The 16:48 reboot
  also killed the stage-2 dataset rsync; Seagate was unplugged without the graceful-pause
  ritual → run `ntfsfix` before the ntfs-3g remount next plug-in, then rerun
  `tmp_recover_heppc.sh` (idempotent).
- Phase 1.1 done: `requirements.txt` frozen (55 pkgs) from the venv that reproduced the
  thesis numbers — torch 2.12.0, numpy 2.4.6, wandb 0.27.2 (no pysr yet — not installed).

## Night 06-10: H1 round 2 — CLEAR SIGNAL: the channel decision reads the JET system

> ⚠️ **CORRECTION (round 2d below): every "sjet"/"ljet" in rounds 2-2c is SWAPPED**
> (scripts used SJ=3/LJ=4; truth is 3=ljet, 4=sjet). Numbers are valid; names are not.

Scripts: `tmp_h1_round2.py` (margins, jet-system swap, corruption grid),
`tmp_h1_round2b.py` (sjet count-vs-content), both `--model`-parameterized; run on
**thesis-ent1-bn1-d152 AND ent1-d20-2blk** (12,288 val events: 8,174 lvbb / 4,114 qqbb).
Checkpoints side-fetched from heppc to `tmp_checkpoints/` (Seagate unplugged) — pass
`checkpoint_root` to `load_model`.

**Findings (consistent across BOTH models — 33× parameter gap):**
1. **Margins explain round 1's "corruptions don't flip"**: under lepton-φ randomization,
   flips live entirely in the lowest |ν margin| decile (thesis: 15.5% flip in decile 0,
   ~0 above decile 3; AUC(|margin|→no-flip)=0.96; organism even sharper, 0.04% overall).
   The decision has a wide margin; single-feature corruption only flips borderline events.
2. **Jet-system swap (decisive)**: transplant ALL jets from an opposite-channel event
   (lep/ν untouched; repack control clean at ±0.3%): lvbb leps + qqbb jets →
   P(lep=W) 0.97→**0.61**; qqbb leps + lvbb jets → 0.06→**0.66** (organism 0.63/0.56).
   Same-channel-donor controls ~0.93/0.09 ⇒ the flip is channel content, not
   foreign-jet incoherence. **P(ν=lep) ≥ 0.95 under EVERY intervention** — the lep/ν
   coupling itself is never broken; only the joint verdict moves.
3. **Sjets are the main carrier**: masking all sjets flips **82%** of qqbb events to
   lep=W (thesis; organism 57%) vs 4-5% of lvbb. Masking ljets: only 21%/14%.
   Tag-zeroing: ≤7%. ⇒ "no hadronic-W candidate among the sjets ⇒ the lepton must be
   the W" is nearly deterministic; presence of one only partially overrides a good
   leptonic side (asymmetry).
4. **Not count, mostly coherence**: n_sjets only weakly separates channels (lvbb median
   1, qqbb 2, big overlap). D4: padding lvbb events to qqbb-median count with innocuous
   donor sjets ≈ no effect (0.94→0.93). D2: replacing qqbb sjets with same-count
   random OTHER-qqbb sjets already flips 16% (relational structure broken); lvbb-pool
   donors add ~13pt more (D1, 0.34). Coherent-system transplant ≫ pooled-sjet
   transplant ⇒ the model reads **within-event relational structure of the sjet system**
   (W→qq̄-compatible pairing w.r.t. the rest of the event), not sjet marginals.
5. **Lepton/MET magnitudes are the secondary input** (angles ~irrelevant): lep+ν ×2
   flips 22%/27% of qqbb (thesis/organism); ×0.5 flips 14%/17% of lvbb. A continuous
   "how W-like is the leptonic side" score.

**H1 reformulated (v2):** the model computes a global event-channel score
(lvbb vs qqbb) ≈ [coherent hadronic-W structure in the sjet system] vs [hardness of
the lepton+MET system], broadcasts it (residual-stream convergence, round 1), and lep+ν
inherit the verdict in lockstep. The organism shows the identical mechanism ⇒ use it
for the circuit-level dig.

**Next experiments queued:** (i) find WHERE the channel score is computed in the
organism (per-block patching of the score between channel-opposite event pairs);
(ii) H2 tie-in: which heads read sjet-pair structure — fit attention logits / bottleneck
messages against pairwise invariants (dijet mass, ΔR) on the organism; (iii) probe for
"dijet-mass-≈-mW" in the residual stream (targeted probe, H3-style).

**Methodology:** event REBUILD (repack into fresh 15-slot tensors) is a clean, verified
intervention primitive (controls at ±0.3%); object order doesn't matter (permutation-
equivariant model, shuffle-trained). Donor-pool sampling vs whole-system transplant
cleanly separates marginal-content vs relational-structure hypotheses.

## Night 06-10 (round 2c): Sid's "ask the jets" hypothesis — first three tests PASS

> ⚠️ **CORRECTION (round 2d below)**: "partial (1 sjW)" is really **boosted (W = 1 ljet)**
> and "boosted (≥1 ljW)" is really **resolved (W = 2 sjets)**. Numbers valid, names swapped.

Sid's reframe: each jet computes "am I the H+'s W?"; lep/ν read the aggregate answer
(NOT(some jet is the W) ⇒ we are). `tmp_h1_round2c.py`, both models.

- **Truth-topology stratification (Sid's question: was 82/21 just prevalence? YES):**
  qqbb = **76.9% "partial"** (exactly ONE sjet labeled truth-2 — NO 2-sjet resolved
  events exist in this truth scheme; check `preprocessLowLevel.py` ΔR-unique matching)
  + **23.1% "boosted"** (W = ljet). Round-2's "mask ljets flips 21%" ≈ the boosted
  fraction exactly. Stratified: mask-ljets flips 80% of boosted vs 2.9% of partial
  (thesis); mask-sjets flips 89.5% of partial.
- **Surgical vs sham**: masking ONLY the truth-2 jet flips 77%/66% (thesis/organism,
  partial stratum); masking the same NUMBER of non-W jets flips 19.6%/**4.5%**.
  lvbb surgical = 0.0000 flips (no truth-2 jets exist → perfect negative control).
- **Prediction-level XOR**: P(exactly one of {some-jet-pred-W, lep-pred-W}) =
  0.99/0.96 (lvbb/qqbb, thesis) — the model behaves as if "exactly one W per event"
  were a hard constraint, on errors too.
- **Mediation (key)**: after masking the truth-W jet, flips track whether the model
  RE-FINDS a W among remaining jets: thesis 87.8% flip when no jet claims W vs 11.8%
  when one does; organism 71.8% vs **5.7%**. The lep/ν verdict reads "did any jet
  claim the W", not the jets' raw content.
- **Wrinkles**: (i) thesis-model boosted stratum is generically fragile (sham flips 54%
  there vs organism 17%); (ii) under mask-ALL-sjets (heavier OOD) the thesis model's
  exclusivity breaks (flips ~82% regardless of remaining jetW) while the organism's
  mediation survives — another reason to dig on the organism first.
- **Queued next (H1 v3 mechanistic)**: activation-level mediation (patch jet-token
  residual streams, freeze lep/ν streams); logit-lens timing (do jets "know" before
  lep/ν?); attention W-ljet-vs-H-ljet contrast in boosted events; bottleneck-message
  patching (thesis model); coherent-W INSERTION into lvbb (sufficiency).

## Late night 06-10 (round 2d): TYPE-LABEL BUG FOUND — Sid's "1 sjet?!" instinct was right

Sid flagged "W = exactly one sjet, zero 2-sjet events" as unphysical → investigation
(`tmp_h1_round2d_forensics.py` + code archaeology) found the cause was OUR bug, not the
truth scheme: **rounds 2/2b/2c used SJ=3, LJ=4 but the real mapping is 3=LJET, 4=SJET**
(`RunLowLevelInterp.py:77-80`, `preprocessing-scripts/preprocessLowLevel.py:377`;
confirmed from kinematics: type 3 median mass 114.5 GeV / pT 535 GeV at 1.6/event;
type 4: 8.9 GeV / 62 GeV at 3.9/event).

**Corrected truth picture — perfectly physical:** qqbb = **76.9% boosted (W = exactly
one LJET, mass median 85.0 GeV ≈ m_W)** + **23.1% resolved (W = exactly two SJETS)**;
no other strata exist. Boosted fraction rises 0.62→0.88 across 0.8→3.0 TeV ✓.
H-ljet (truth-1) mass 119.8 GeV ✓. The per-object label is `trueInclusion` = membership
in the truth-matched reconstruction (`preprocessLowLevel.py:64`); x[...,5] is
`recoInclusion` (the cut-based algo's choice — useful for H4 later!).

**Corrected round-2 narrative (numbers unchanged, objects renamed):**
- "mask sjets flips 82% of qqbb" → **mask LJETS flips 82%** (removes the boosted
  W-ljet — and the H-ljet — in the 77% stratum).
- "mask ljets flips 21%" → **mask SJETS flips 21% ≈ exactly the resolved fraction**.
- Stratified (2c, corrected): mask-ljets flips 89.5% of boosted / 58.3% of resolved;
  mask-sjets flips 80.0% of resolved / 2.9% of boosted. Surgical truth-W removal:
  77% boosted / 56% resolved (organism 66%/31%); sham: 20% boosted / 54% resolved
  (organism 4.5%/17%) ⇒ the boosted-W mechanism is crisp; the resolved stratum is
  generically more fragile (thesis sham≈surgical there) — model handles it worse.
- 2b reinterpretation: the "sjet pools" were LJET pools; "count vs content" tested
  ljet count (lvbb 1 = H; qqbb 2 = H+W) — adding a second H-like ljet to lvbb does
  NOT fake a W (D4, ~no effect) ⇒ the model checks W-likeness of the ljet, not ljet
  multiplicity. Round-2's jet-system swap + surgical/sham/XOR/mediation conclusions
  are type-agnostic and stand unchanged.
- **H1 v2 restated**: channel verdict ≈ "is there a hadronic-W candidate — usually a
  single W-mass-window LJET (77%), else an sjet pair (23%)?" vs leptonic-side hardness.
- **Tension to resolve next**: round-1 logged "ljet mass ×0.5 → only 4.5% flips" — if
  that really was type 3 (ljet), it conflicts with a W-ljet mass-window detector;
  re-verify with correct ids, then scale the truth-W-ljet mass in boosted events
  specifically. (Round-1 inline corruption namings now all suspect.)
- **Curiosity**: resolved-stratum m(jj) of the two truth-W sjets: median 167 GeV,
  W-peak at the low edge (16% ≈ 78 GeV) + long tail to ~400 — the ΔR-based truth
  matching is loose here; check `app:TruthMatching:Type2` + matching code when
  regenerating memmaps. May explain why the model is less crisp on resolved events.

**Learning (promote to convention): before any interp analysis, hard-code the
types_dict from the source-of-truth script and SANITY-CHECK it against kinematics
(masses/pT/multiplicities) — a swapped label survives every downstream test silently.**
Scripts fixed: 2c/2b constants now SJ=4/LJ=3; 2.py grid labels corrected.

## Late night 06-10: preregistered H1-v3 test battery written (no experiments run)

Sid asked for a full brainstorm BEFORE running anything →
[`docs/H1_ASK_THE_JETS_TEST_PLAN.md`](../H1_ASK_THE_JETS_TEST_PLAN.md): 4 competing
hypotheses (H-A "ask the jets" / H-B "parallel global" / H-C "lepton-primary" /
H-D "output bookkeeping"), ~30 experiments in 6 families (input dose-response &
insertion; output-level laws; attention analyses; activation patching/probing/steering/
scrubbing; geometry/PySR; training-dynamics on the new organism's 30 checkpoints),
each with preregistered predictions + outcome interpretations, ordered in 5 waves with
pivot criteria. Key discriminator: direction/timing of information flow (D1 patching ×
D2 logit-lens — organism has no LayerNorm so the lens is exact).

## Late night 06-10: WAVE 1 RUN (D2, A1, B1-B3, A6) — H-A's structure confirmed,
## but the candidacy FEATURE diverges across scale

Scripts: `tmp_h1_wave1_behav.py`, `tmp_h1_wave1_lens.py` (both `--model`/`--batches`/
`--ckpt-override`; lens gotcha: snapshot captured streams before re-calling
`classifier()` — its pre-hook overwrites the final depth). Old organism + thesis model,
12,288 val events. New organism still training (rerun on it when done).

**vs preregistrations (docs/H1_ASK_THE_JETS_TEST_PLAN.md):**
- **B1 ✓** XOR(jet-system claims, lep=W) holds: 0.985/0.96/0.87 organism,
  0.991/0.977/0.912 thesis (lvbb/boosted/resolved). #claiming jets: boosted → exactly 1
  in 99.3/99.6%; resolved → 2 in only 47%/63% (1 in 33%/24%) — **pair binding is weak**,
  esp. in the organism. XOR drops to 0.80/0.85 among error events.
- **B2 ✓✓** lep margin vs best-jet margin: r = −0.96/−0.97 overall and **−0.82…−0.92
  WITHIN each stratum** — a shared scalar read with opposite signs, event-by-event.
  Plots: `tmp_plots/wave1_B2_*.png`.
- **B3 ✓ (with a twist)** errors are detector misses (P(no claim|error)=0.90-0.98 vs
  0.02-0.16|correct) — but missed boosted W-ljets are **SOFT (pT med 444-454 vs 691-694
  GeV), NOT off-mass (89.6-89.9 vs 84.8)**. pT, not mass, drives misses. Resolved:
  correct events have HIGHER m(jj) (169-173 vs 121) — anti-window, confounded w/ hardness.
- **A1 ⚡ THE HEADLINE — models diverge:**
  - **Thesis model: textbook mass bump peaking at m_W**: claim 0.34 @5 GeV → **0.96 @80**
    → 0.81 @125 → plateau ~0.83 @150-200 → 0.72 @300; P(lep=W) mirrors exactly
    (0.42 → 0.018 → 0.12). A real W-mass-window feature.
  - **Organism: monotonic decreasing** (0.96 @5 GeV → 0.71 @300, no bump): its
    "W-candidacy" is more like "the lighter / not-H ljet" — a relative discriminant
    that works in-distribution but is NOT a W detector. (OOD caveat at m≤20 GeV.)
  - ⇒ **Behavior transfers across scale; the FEATURE does not.** Organism findings
    about WHAT is computed must be re-verified on the big model (mechanism-level
    organisms-first still fine: structure matched everywhere else).
- **A6 ✓** exact permutation invariance (0 pred mismatches, |Δlogit| ≤ 2e-5). Cite forever.
- **D2 ✓ (H-A timing)** logit-lens staircase, both models: **jet claims crystallize one
  block before the lep/ν verdict**; the verdict + broadcast land in the final block
  (organism: W-claim 0→0.45→0.90; AUC ch|lep 0.85→0.85→0.99; thesis: 0.04→0.55→0.89→0.93
  and 0.87→0.95→0.98→0.99 with ν lagging lep early). Even the H-jet token ends at
  AUC 0.87-0.92 (broadcast — matches round-1 residual convergence).
  - **Bonus finding**: the lep token's OWN embedding already separates the channel at
    AUC 0.85-0.87 (lepton hardness differs by channel) but this never improves until
    the final-block read — *informative but overridden* by the jet verdict. Consistent
    with "leptonic side = secondary input".

**Hypothesis scoreboard after wave 1:** H-C (lepton-primary) dead. H-D (output
bookkeeping) disfavored (XOR = anti-correlated internal margins, not output artifact).
H-A structure (jets compute first; lep/ν read late; exclusion via shared score)
supported on both models. H-B not dead: the final-block read could be of an
*event-level summary* rather than the W-jet token specifically — that's exactly what
D1 patching (wave 3) separates. New wave-2 priorities: A2 (H-ljet dose-response —
competition/elimination test), pT dose-response (B3 says pT drives misses), and the
lep-prior override (patch lep embeddings).

## Next session

- Ingest Sid's heppc probe results → prioritize + run recovery rsyncs (checkpoints first).
- wandb relogin → archive thesis-model hyperparameters.
- When ROOT transfer completes: verify counts/sizes vs cluster, then Phase 0.4
  (regenerate memmaps; resolve the 3-vs-5 type-encoding question).

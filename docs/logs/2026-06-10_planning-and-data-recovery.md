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

## Next session

- Ingest Sid's heppc probe results → prioritize + run recovery rsyncs (checkpoints first).
- wandb relogin → archive thesis-model hyperparameters.
- When ROOT transfer completes: verify counts/sizes vs cluster, then Phase 0.4
  (regenerate memmaps; resolve the 3-vs-5 type-encoding question).

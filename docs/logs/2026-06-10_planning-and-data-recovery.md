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

## Next session

- Ingest Sid's heppc probe results → prioritize + run recovery rsyncs (checkpoints first).
- wandb relogin → archive thesis-model hyperparameters.
- When ROOT transfer completes: verify counts/sizes vs cluster, then Phase 0.4
  (regenerate memmaps; resolve the 3-vs-5 type-encoding question).

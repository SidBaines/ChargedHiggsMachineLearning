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

## Next session

- Ingest Sid's heppc probe results → prioritize + run recovery rsyncs (checkpoints first).
- wandb relogin → archive thesis-model hyperparameters.
- When ROOT transfer completes: verify counts/sizes vs cluster, then Phase 0.4
  (regenerate memmaps; resolve the 3-vs-5 type-encoding question).

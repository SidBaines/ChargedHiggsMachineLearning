# 2026-06-11 — New MacBook Pro (M5 Pro, 48 GB) setup

Sid migrated to the new M5 Pro MacBook, copying over **only** this repo folder and the
thesis folder (`~/Documents/PhD/PhDThesis/`). This session: audit what survived the
copy, rebuild the toolchain, and verify the research pipeline end-to-end.

## What survived the copy (audited)

The Finder copy brought the *whole working directory*, including the gitignored
research assets — much better than feared:

| asset | status |
|---|---|
| repo @ `heppc-work`, all branches, remote intact | ✓ |
| `tmp_data_20250321v1_signal/` (1.4 GB signal memmaps, DSIDs 510115–124) | ✓ |
| `tmp_checkpoints/` (thesis `20250512-093728` chkpt29 + `20250709-095246` chkpt9) | ✓ |
| `output/` incl. the new organism `20260610-203258_TrainingOutput` (2.7 MB, 30 chkpts) | ✓ |
| `experiments/` + all h1 scripts, `wandb/` local runs, `requirements.txt` (frozen 2026-06-10) | ✓ |
| `.venv/` | ✗ (not copied — rebuilt this session) |
| symlink `experiments/tmp_data_20250321v1_signal → ../tmp_data_20250321v1_signal` | ✗ (symlink lost in copy — recreated) |

**⇒ Stage B (and all thesis-model/organism work) is fully unblocked on this machine
without the Seagate drive.** All h1 scripts load via
`checkpoint_root=REPO/tmp_checkpoints` and `REPO/tmp_data_20250321v1_signal/`.

## What did NOT come over (lives only on the old Mac / Seagate / heppc)

1. **`~/Documents/PhD/Work/` scratch folders** — absent on this Mac. Notably:
   - `Work/20250311_MechInterpTmp/` — predecessor scratch code + its `model.pth`
     (not git-backed anywhere; flagged at-risk in `SESSION_FINDINGS_2026-06-08.md` §5).
     Mostly superseded now, but it's the only copy of the predecessor architecture run.
   - `Work/HplusWh/20251212_MassSculpting/` — git repo with **no remote** (~1 GB).
   - `Work/HplusWh/MLEventSel_OnlyImportant/` — TransformerLens + SAE work, no git.
2. **Seagate drive** (not attached): `heppc_recovered/` — full memmap datasets
   (incl. background DSIDs → needed for classifier work), all 8 registry-variant
   checkpoints (`models/registry.py` `DEFAULT_CHECKPOINT_ROOT` points at it;
   `experiments/validate_registry.py` needs it), recovered heppc repos, 20250313 ROOT
   files (needed to regenerate memmaps / vary preprocessing — Phase 0.4).
3. **Credentials**: GitHub SSH key DID come over (`~/.ssh/github-SidBaines`), but the
   remote is https (first push will want auth — `gh auth login` or switch remote to
   ssh). No wandb login (`~/.netrc` absent → `wandb login` before any wandb pull,
   Phase 0.3). No heppc ssh key found — if the old Mac had passwordless
   `baines@heppc402.ph.qmul.ac.uk`, copy that key too (or use password).

## Toolchain built this session

- Pre-existing: Xcode CLT, Apple git 2.50.1 (user.name/email already configured),
  system Python 3.9 only. **No Homebrew, no uv, no node.**
- Installed **uv 0.11.20** via the official standalone installer (`~/.local/bin`;
  self-updates with `uv self update`; no sudo needed — Homebrew install requires
  Sid's password so was left to him; after that: `brew install gh` etc.).
- `uv python install 3.12` → CPython 3.12.13; `uv venv .venv`;
  `uv pip install -r requirements.txt` (the 2026-06-10 freeze incl. torch 2.12.0)
  — clean install, ~1 min.
- Fixed git noise: the copy flipped all file modes 644→755, showing 148 phantom
  modified files. `git config core.fileMode false` (local) → tree clean again.

## Verification — ✅ Phase-0 gate re-passed on the M5

`experiments/eval_thesis_numbers.py` (full val split, CPU, ~3.5 min): per-category
PerfectRecoPct = **0.3097 / 0.3709 / 0.2917 / 0.4959 / 0.9587 / 0.9271** — *identical*
to the verified 2026-06-10 run on the old Mac (which matched the thesis Table to
~4 s.f.). Per-mass 0.738 (0.8 TeV) → 0.941 (3.0 TeV), same as before; the +1–4%
offset vs the wandb epoch-29 numbers is the known training-time-buffering artifact
(see 2026-06-10 log, "PHASE 0 GATE PASSED" section). ⇒ env + data + checkpoints +
metrics all verified end-to-end on the new machine; Stage B ready to run.

## Learnings

- **Finder copy keeps gitignored files but drops symlinks' validity is fine — what it
  actually loses is the venv (abs paths) and flips exec bits.** On any future machine
  migration: expect to (a) rebuild `.venv` from `requirements.txt`, (b) recreate the
  `experiments/tmp_data_20250321v1_signal` symlink, (c) `git config core.fileMode
  false` or re-chmod.
- `eval_thesis_numbers.py`'s `_LOCAL` resolves relative to `experiments/`, not repo
  root — it silently relied on the (uncommitted) symlink. The h1 scripts use
  `os.path.join(REPO, …)` and are robust.
- MPS is still untrusted for training on this hardware family (2026-06-10 crash was
  on the old Mac; the M5 is untested — worth a cautious retry someday, but CPU
  training of the organism was fine at ~5 min/epoch on the old machine and will be
  faster here).

## Remaining setup to-dos (user-side)

- **Homebrew** (needs admin password): `/bin/bash -c "$(curl -fsSL
  https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` — then
  optionally `brew install gh` (GitHub auth) and whatever else laptop-wide.
- **wandb**: `wandb login` (no `~/.netrc` came over) — needed for Phase 0.3.
- **GitHub push auth**: remote is https; either `gh auth login` (after brew) or
  switch remote to ssh (the `~/.ssh/github-SidBaines` key DID come over).
- **heppc ssh**: no cluster key found in `~/.ssh` — copy from old Mac if
  passwordless access existed, else password works.
- **PATH**: uv installer added `~/.local/bin` via `~/.zshrc` — new shells fine.

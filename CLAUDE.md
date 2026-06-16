# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repo.

## What this repo is

`ChargedHiggsMachineLearning` — code for training, interpreting & applying
transformer-like neural networks to low-level reconstructed objects in ATLAS
particle-physics events (the H⁺ → W h / `HplusWh` analysis), for event
reconstruction/classification. See `README.md` for run commands and `README_metrics.md`.

This is the **canonical, consolidated** version of work that was prototyped in scratch
folders elsewhere on this machine.

## START HERE if you're picking this work back up

📄 **Read [`docs/HANDOFF_2026-06-12.md`](docs/HANDOFF_2026-06-12.md) first** — current
state, paused jobs, next-steps queue, and the gotchas. Then `experiments/README.md`
(conventions + script inventory) and the narrative notebook
`experiments/h1_narrative.ipynb` (the science, claims + evidence + confidence).

📄 **For the expected-limits plot (`fig:TransformerVsOriginalExpectedLimits`) — how it's
produced, lxplus access, and reco-vs-classifier model provenance — read
[`docs/LIMITS_PIPELINE_AND_MODEL_PROVENANCE.md`](docs/LIMITS_PIPELINE_AND_MODEL_PROVENANCE.md)**
(2026-06-16; numbers reproduced end-to-end). Key: the **reconstruction** model (this repo's,
being retrained) and the **classifier** models (out of this repo's scope) are distinct
dependencies — both feed the plot.

(Historical context from the original 2026-06-08 pickup — repo layout, data
provenance, the old-Mac/heppc/Seagate situation — is in
[`docs/SESSION_FINDINGS_2026-06-08.md`](docs/SESSION_FINDINGS_2026-06-08.md).)
It is a full investigation write-up (2026-06-08) covering: what this repo is and how it's
laid out, the branch situation (`main` vs `heppc-work` vs `cleaner`), **where the data and
model weights actually live** (`/data/atlas/baines/…` on the "HEP PC", not on this Mac),
the environment situation (no Python env exists yet — must be built), related scratch/
precursor folders elsewhere on the machine, and suggested next steps.

That doc has context a fresh session cannot otherwise discover (e.g. that the runnable
data + a trained `model.pth` still exist in a scratch folder at
`/Users/sidbaines/Documents/PhD/Work/20250311_MechInterpTmp/`).

## Quick facts

- **Entry points:** `TrainLowLevelReconstruction.py`, `TrainLowLevelClassifier.py`,
  `TrainHighLevelClassifier.py`, `RunLowLevelInterp.py` (interp playground). All use
  Jupyter `# %%` cell tags.
- **Interp toolkit:** the `interp/` package (activation extraction, attention analysis,
  direct logit attribution, activation patching, linear probes, ablation, PySR symbolic
  regression).
- **Branch:** this clone is on `heppc-work` (≈ `main` + committed result artifacts).
- **Blocker to running anything:** data + model checkpoints are not on this Mac and not in
  the repo (`output/`, `wandb/` are gitignored). See the findings doc §3.
- **Deps (no pinned requirements file):** `torch, torchvision, transformer_lens, einops,
  jaxtyping, uproot, awkward, vector, scikit-learn, scipy, seaborn, matplotlib, numpy,
  pandas, pysr, wandb`.

## Research log convention (agreed 2026-06-10)

Every working session gets a research log, tracked in git:

1. **Write a log** at `docs/logs/YYYY-MM-DD_<short-topic>.md` covering: what we did, key
   decisions (and why), results/numbers produced, and a **"Learnings"** section — gotchas,
   dead ends, and facts future sessions (Claude or Sid) should know. Update it as the
   session goes; don't reconstruct from memory at the end.
2. **Add a one-line entry** to the "Session log" section at the bottom of
   [`docs/RESEARCH_PLAN_2026-06.md`](docs/RESEARCH_PLAN_2026-06.md) (the living plan doc).
3. **Update the plan doc itself** if we pivot or finish/add phases — it should always
   reflect current intent, with the logs holding the history.
4. If a learning is really a durable convention (a rule for how we work), promote it into
   this `CLAUDE.md` rather than leaving it buried in a log.
5. Commit the docs (`CLAUDE.md`, `docs/`) at session end — docs-only commits are
   pre-authorized; everything else still needs explicit say-so (below).

## Conventions / cautions

- Don't commit or push without the user's explicit say-so (exception: docs-only commits
  per the research-log convention above).
- `output/`, `wandb/`, `tmp*`, `*.png`, `*.pdf` are gitignored — don't force-add them.
- This `CLAUDE.md` and `docs/SESSION_FINDINGS_2026-06-08.md` are session notes added on
  2026-06-08; they are untracked until the user decides to commit them.

# 2026-06-12 — Codex analysis-fix pass

Running log for the `codex-analysis` worktree. Purpose: address scientific-validity
issues found in the branch review, preserving headline results at the top as the
work evolves.

## Headline results

- **SB1 donor-direction bug fixed and re-run.** The bug was real: the old sampling
  could decouple donor event and donor W-position index. The corrected guardrail now
  reports 100% true-W-ljet donor directions. Thesis SB1 headline numbers are robust:
  symmetric crossing remains at r≈1.00 and asymmetric crossing remains around
  r≈1.05-1.07.
- **Still in progress:** F1 re-run, E4 mean-control issue, over-strong wording
  around E4/B-i/suite, and missing DSID-stratified controls.
- **F1 epoch-level re-run also robust so far.** Corrected seed-0 organism timeline
  preserves the old qualitative result: circuit signatures are present at the first
  checkpoint, cross-system threshold calibrates early, and competition cost deepens
  smoothly to about `-1.62`.
- **F1 step-resolved re-run robust.** Corrected winner-recipe step trace preserves
  the "claim first, compare second" result and the two-phase competition-pressure
  trajectory: early overshoot to about `-0.89`, relaxation to about `-0.16..-0.20`,
  then late re-deepening to about `-0.89` by step 1100.
- **E4 control bugs fixed; headline numerically stable.** Replacement now uses true
  train-set bottleneck means for the mean-control floor and reports non-finite
  formula outputs. The simple c=6+c=8 formula pair keeps `95.53%` of held-out ν
  decisions after a one-event sqrt-domain fill; the mean-control floor is `69.51%`.

## Running log

### Setup

- Created separate worktree:
  `/Users/sidbaines/Documents/PhD/20260608_ChargedHiggsMachineLearning_codex-analysis`
  on branch `codex-analysis`, based on `sid-fable-experiments` at `3ab38b7`.
- Original review worktree left untouched.
- Added local symlinks in the worktree to reuse gitignored runtime artifacts from
  the original checkout: `.venv`, `tmp_data_20250321v1_signal`, `tmp_checkpoints`,
  and `output`.

### Donor-direction indexing fix

- Fixed independent donor event/W-position sampling in:
  `experiments/h1/stageb_sb1_dose_response.py`,
  `experiments/h1/f1_formation_dynamics.py`, and
  `experiments/h1_narrative.py`.
- Added donor-composition guardrails to SB1 and F1. This catches the exact failure
  mode from the review: synthetic "W directions" must be sampled from true W-ljets.

### Corrected SB1 re-run

Command:

```bash
.venv/bin/python experiments/h1/stageb_sb1_dose_response.py
```

Key output:

```text
donor_dir1: donor true-W-ljet fraction 1.0000; type counts [0, 0, 0, 3085, 0, 0]
donor_dir2: donor true-W-ljet fraction 1.0000; type counts [0, 0, 0, 3085, 0, 0]
```

- Asym design at r=1.00: `P(c2 wins|one)=0.1314`; at r=1.05:
  `0.4099`; at r=1.15: `0.9007`. Crossing remains around r≈1.05-1.07.
- Sym design at r=1.00: `P(c2 wins|one)=0.5039`. Crossing remains exactly at
  the symmetric point within sampling noise.
- Both-claim peak remains broad around the tie: asym `both=0.7439` at r=1.00,
  sym `both=0.7008` at r=1.00.

Verdict: the donor bug was real but did **not** materially change the thesis SB1
claim about an unbiased symmetric comparator and a small in-event coherence
handicap.

### Corrected F1 epoch-level re-run

Command:

```bash
.venv/bin/python experiments/h1/f1_formation_dynamics.py \
  --run output/20260610-203258_TrainingOutput
```

Guardrail:

```text
boosted donor directions: true-W-ljet fraction 1.0000; type counts [0, 0, 0, 3087, 0, 0]
```

Key corrected trajectory:

- First checkpoint (`step=343`): `task=0.904`, `lockstep=0.962`,
  `mask_flip=0.400`, `relhard=0.623`, `tie_both=0.839`,
  `tie_cost=-0.61`, `xsys(1/1.4/2)=0.61/0.87/0.95`.
- By `step=686`: `lockstep=0.991`, `xsys r=1.0` drops to `0.19`, consistent
  with rapid threshold calibration.
- Final checkpoint (`step=10290`): `task=0.928`, `lockstep=0.994`,
  `mask_flip=0.457`, `relhard=0.553`, `tie_both=0.880`,
  `tie_cost=-1.62`, `xsys(1/1.4/2)=0.07/0.69/0.95`.

Verdict: the donor bug does **not** change the seed-0 epoch-level F1 conclusion.
The old wording "everything qualitative exists at checkpoint 0; training is
calibration" remains supported for this run, with the caveat that "checkpoint 0"
means after 343 optimizer steps.

### Corrected F1 step-resolved re-run

Command:

```bash
.venv/bin/python experiments/h1/f1_formation_dynamics.py \
  --run output/20260612-092102_TrainingOutput --step-range 0,1100
```

Guardrail:

```text
boosted donor directions: true-W-ljet fraction 1.0000; type counts [0, 0, 0, 3087, 0, 0]
```

Key corrected trajectory:

- Step 10-30: early task/selectivity collapse (`task=0.649 → 0.231`,
  `tie_both=0.747 → 0.149`) with no broadcast yet (`mask_flip=0.000`).
- Step 50-70: transient indiscriminate W-claiming (`task≈1.0`, `tie_both≈1.0`)
  while lockstep collapses (`0.365 → 0.242 → 0.291`).
- Step 90-170: relational consolidation starts (`mask_flip=0.119 → 0.405`,
  `relhard=0.558 → 0.606`) and competition cost deepens to `-0.89`.
- Step 230-450: high task/lockstep return while competition cost relaxes
  (`-0.41 → -0.16..-0.20`).
- Step 700-1100: competition cost re-deepens (`-0.52 → -0.89`), with high
  lockstep (`~0.99`) and high task (`~0.93`).

Verdict: the corrected run preserves the original F1 headline: the donor bug does
not explain the selectivity crisis, birth-order result, or two-phase competition
pressure. The safest wording is: **under the winner recipe, early training moves
through a transient selectivity/lockstep crisis, then relational consolidation;
competition pressure overshoots, relaxes, and later re-deepens.**

### E4 replacement controls

Code changes:

- `e4_wire_extraction.py` and `e4_wire_replacement.py` now expose
  `--skip`, `--batches`, and `--train-batches`, so the E4 protocol can be moved
  onto a genuinely fresh validation slice without code edits.
- `e4_wire_replacement.py` now states in its docstring that this is a
  **readout-level behavioral-compression** test, not a full symbolic model
  replacement.
- The mean-control condition now uses the **true train-set bottleneck activation
  means**, not the formula-prediction means.
- Formula outputs are checked for non-finite values before splicing. Default
  `--nan-policy mean` fills undefined formula outputs with the true train-set wire
  mean and reports the count.

Default best-complexity run:

```bash
.venv/bin/python experiments/h1/e4_wire_replacement.py
```

- True train-set means: `b2h2=0.6518`, `b1h3=-0.4212`.
- Formula means were close: `0.6404`, `-0.4255`.
- `replace BOTH`: `nu==intact 0.9636`, `nu==truth 0.9402`.
- `mean-control BOTH`: `nu==intact 0.6951`, `nu==truth 0.6941`.

Simple headline formula run:

```bash
.venv/bin/python experiments/h1/e4_wire_replacement.py \
  --pick-b1h3 6 --pick-b2h2 8
```

- The c=8 b2h2 formula has a sqrt-domain issue in normal NumPy evaluation:
  `1/12288` events non-finite, including `1/4096` held-out events.
- With default mean-fill, `replace BOTH`: `nu==intact 0.9553`,
  `nu==truth 0.9309`, `P(nu=lep)=0.9880`.
- `mean-control BOTH`: `nu==intact 0.6951`, `nu==truth 0.6941`.

Verdict: the E4 numerical headline is stable, but the corrected language should be
strictly behavioral: **two simple readout-level formulas reproduce most final
verdict decisions when spliced into two dominant wires; this is not a full
mechanistic symbolic extraction of the jet-side algorithm.**

### Wording/scope corrections

Updated the main docs and narrative source to keep claims aligned with the evidence:

- `docs/logs/2026-06-11_stage-b-competition.md`: replaced "weak B-i is dead" with
  "no separate directed winner→loser inhibition edge was found; B-ii is strongly
  supported and localized in this model." This preserves the edge-KO result without
  claiming every possible multi-hop/token-state description is ruled out.
- `docs/logs/2026-06-12_faithfulness-e4.md`: reframed E4 as readout-level verdict
  compression, updated corrected metrics, added the post-hoc feature-discovery
  caveat, and noted the formula-domain guard.
- `docs/logs/2026-06-12_suite-d20-half.md`: softened "interpretability nearly free"
  to "small overall cost in this partial d20 suite" and marked variance reduction
  as a 3-seed observation.
- `docs/RESEARCH_PLAN_2026-06.md` and
  `docs/H1_ASK_THE_JETS_TEST_PLAN.md`: updated the session summaries/addendum to
  the same scoped language and recorded that the SB1 donor bug was fixed/re-run
  without moving headline numbers.
- `experiments/README.md`: scoped Stage-B and A4b insertion language; synthetic
  insertion results are now explicitly model-counterfactual.
- `experiments/h1_narrative.py`: adjusted scoreboard/comments so regenerated
  notebooks inherit the scoped claims.

### DSID-stratified controls

Added `experiments/h1/dsid_stratified_controls.py` to check whether two key
high-level variable claims are mass-point artifacts.

Command:

```bash
.venv/bin/python experiments/h1/dsid_stratified_controls.py
```

Results on the default thesis slice (batches 19-24):

1. **LW1 ratio correlation survives within every DSID.**

   Overall in lvbb: `rho(pT(lepW)/HT(jets), b1h3@nu) = -0.795`.
   Within DSID, the same correlation remains strong:
   `-0.814, -0.813, -0.795, -0.808, -0.809, -0.797, -0.774, -0.740,
   -0.701, -0.666` for DSIDs 510115-510124. This makes a pure pooled
   mass-point confound unlikely. The weakening at high DSID is worth tracking,
   but the sign and large magnitude are stable.

2. **RT4 rest×2 suppression survives within every boosted DSID.**

   Overall boosted W-claim rate drops `0.914 -> 0.586` when the rest of the event
   is doubled at fixed W. Per-DSID drops are all substantial: `27-37 pp`. This
   supports relative event-hardness as a within-mass-point effect, not only a
   pooled mass-spectrum artifact.

Verdict: the DSID control strengthens both the LW1 ratio claim and RT4 relative
hardness claim. It does not replace generator-level validation, but it addresses
the immediate mass-point pooling confound.

### Verification

Commands run:

```bash
.venv/bin/python -m py_compile \
  experiments/h1/stageb_sb1_dose_response.py \
  experiments/h1/f1_formation_dynamics.py \
  experiments/h1/e4_wire_extraction.py \
  experiments/h1/e4_wire_replacement.py \
  experiments/h1/dsid_stratified_controls.py \
  experiments/h1_narrative.py

git diff --check
```

Both passed.

Worktree status after excluding local runtime symlinks/caches:

- Modified tracked files: docs/test-plan/research-plan/log wording, README,
  SB1/F1/E4 scripts, narrative source.
- New files: this running log and `experiments/h1/dsid_stratified_controls.py`.

## Final status of this pass

- Donor-direction bug fixed and checked: affected SB1/F1 headline results are robust.
- E4 replacement controls fixed: true-mean control and non-finite formula guard
  added; numerical headline stable but now correctly framed as readout-level
  behavioral compression.
- DSID controls added: immediate mass-point pooling confounds for LW1/RT4 are
  substantially reduced.
- Documentation language softened where prior wording exceeded the evidence.

## Recommended next runs

These are ordered by expected impact on the scientific story, not by runtime.

1. **Prospective E4 validation.** Refit/evaluate the E4 formulas on a genuinely
   fresh validation slice, or use a nested protocol:
   old slices for feature discovery, fresh slice A for fitting, fresh slice B for
   replacement evaluation. This is the cleanest upgrade from post-hoc compression
   to stronger faithfulness evidence.
2. **Finish the d152 Phase-2 suite.** The d20 result is useful but not decisive.
   The important question is whether the small interpretability cost under the sane
   recipe survives at thesis scale. Run d152 `{none, ent, bn1, both}` across seeds,
   plus the legacy comparability runs, and compare category-level costs, especially
   cats 0-3.
3. **Spot-check Stage-B/F1 on the best 0.845 d20 model.** The organism analysed
   earlier was undertrained relative to the improved recipe. Re-run a compact SB1,
   SB4, F1-final, and A4b crossing check on the 0.845 model to see whether structural
   universality survives better training.
4. **Jet-side E4: extract the candidacy formula.** Current E4 only compresses the
   verdict readout. Next target should be the per-jet W-claim score/logit/margin,
   with features including relative pT/hardness, m²-window, tag, event HT, and
   lead/sublead context. Validate with intervention/replacement, not just R².
5. **Background or mixed signal/background robustness.** Most H1 evidence is
   signal-only. Before broad physics claims, check whether b2h2/b1h3, relative
   hardness, and the candidacy/verdict split survive on background or mixed samples.
6. **H-side competition battery.** Run the Stage-B analogue for H candidates:
   two H-like candidates, tag/mass/pT factorials, edge KO, and comparison to the
   W-side b2h2 machinery. This tells us whether candidate competition is per-class
   or shared.

## Other circuits / experiments worth studying

1. **Higgs-candidate circuit.** Search for Xbb-tag use, H mass-window use,
   large-R vs small-R preference, H-pair competition, and whether H evidence
   suppresses W candidacy or only enters downstream.
2. **Resolved qqbb circuit.** The current story says resolved detection is weak and
   unspecific. Map whether the model computes pair mass, binds two sjets into one W,
   or fails because pairwise binding is absent/weak.
3. **Transformer-vs-cut-based reco comparison.** Where the transformer beats the old
   algorithm, identify the extra variable/path; where it fails, test whether the
   same circuit is miscalibrated or a shortcut circuit is active.
4. **Error circuits.** Stratify failures rather than only correct behavior:
   W missed despite good mass, false W claims on H-like jets, lvbb/qqbb channel
   flips, high-mass vs low-mass failures. Then test whether errors are circuit
   miscalibration or distinct mechanisms.
5. **Training-origin comparison.** Use random-init/early checkpoints, label shuffles,
   channel-balanced subsets, entropy/bottleneck ablations, and seed comparisons to
   separate architecture bias from learned circuit structure.
6. **Lepton/MET circuit.** LW1 suggests magnitudes/vector-sum hardness, not true W
   transverse mass. Map where lepton/MET magnitude enters, whether `pT(lep+nu)` is
   explicitly computed, and why the lepton appears weighted more than MET.
7. **Mass-generalization / DSID-conditioning circuit.** Test whether DSID or mass
   point is decodable from streams, whether thresholds shift by DSID, and whether
   relative-hardness truly normalizes mass scale or leaves residual mass-specific
   subcircuits.

Priority recommendation: prospective E4, finish d152 suite, H-side competition,
resolved-channel circuit, then background robustness. Those are most likely to
change the science rather than just add supporting detail.

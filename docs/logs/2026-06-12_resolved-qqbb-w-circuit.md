# 2026-06-12 — Resolved qqbb W circuit

## Aim

Study how the thesis reconstruction model handles resolved `qqbb` W cases, where
the hadronic W is represented by two truth-2 small-R jets (`sjet`, type 4), not one
large-R W jet (`ljet`, type 3).

Main questions:

- How does the model combine two sjets into one W, if it does?
- Does it compute or approximate pair mass?
- Are failures due to missing pairwise binding, weak signal, or readout conflict with
  the boosted topology?
- Which attention/value paths carry sjet-sjet or sjet-pair evidence?
- Can pairwise-mass probes or causal interventions predictably control the resolved
  verdict?

## Initial Context Read

Read:

- `CLAUDE.md`
- `docs/HANDOFF_2026-06-12.md`
- `experiments/README.md`
- `docs/RESEARCH_PLAN_2026-06.md`
- `docs/H1_ASK_THE_JETS_TEST_PLAN.md`
- `experiments/h1/wave3_wires_positions_resolved.py`

Installed `ripgrep` via Homebrew for faster repo-wide searches.

## Starting Evidence State

Established from prior logs/scripts:

- Correct object mapping is `3=ljet`, `4=sjet`.
- `qqbb` splits into about 77% boosted (one truth-W ljet) and 23% resolved
  (two truth-W sjets), and boosted/resolved exhaustively partition `qqbb` on checked
  slices.
- The broader H1 story is well supported for the channel verdict: candidate jets score
  "am I the W?" against event context; the verdict is broadcast to lepton/neutrino
  mainly through bottleneck wires `b1h3` and `b2h2`, with secondary `b2h0/b2h3`.
- Boosted W-ljet handling is much crisper than resolved handling.

Resolved-specific evidence currently looks suggestive, not complete:

- Prior stratified masking/surgical removal found resolved truth-W removal weaker and
  sham fragility higher than boosted. This is consistent with weak pair binding, but
  not decisive.
- Wave 1 claim bookkeeping reported resolved events often have poor pair-level
  bookkeeping: two W-sjet claims only about half the time in the thesis model, with many
  one-claim cases.
- A3-lite (`wave3_wires_positions_resolved.py`) found neutrino verdict-head attention
  in resolved events spread across the W-sjet pair and other jets, unlike boosted
  events where `b2h2` focuses strongly on the W-ljet. It also found the resolved
  qqbb-side value contribution is carried by W-sjets, but weakly and unspecifically.

Open caveat:

- None of this yet proves whether the model computes pair mass, computes another
  pairwise relation, uses independent single-sjet evidence, or simply relies on weak
  correlated topology cues.

## Candidate First Experiments

1. Reproduce and extend the resolved baseline on a fresh slice (`--skip 24`):
   per-stratum accuracy, W-claim counts, pair-mass distributions, and wire/value
   decompositions.
2. Observational probes:
   regress resolved W-sjet claim margins and event verdict margins against
   `m(jj)`, pair pT, ΔR, Δφ, individual pT/mass/tag, HT-normalized features, and
   boosted-topology readout scalars.
3. Pair-mass causal surgery:
   modify one W-sjet four-vector to sweep `m(jj)` while trying to keep individual
   sjet features as controlled as possible; compare to controls that perturb
   individual sjet pT/mass without targeted pair-mass motion.
4. Path tests:
   compare sjet-sjet, sjet-to-lepton/neutrino, and lepton/neutrino-to-sjet attention
   and value contributions, then use edge knockouts/overwrites on candidate paths.
5. Readout-conflict tests:
   intervene on resolved W-sjet evidence while clamping or replacing the known verdict
   wires to distinguish upstream weak detection from downstream broadcast conflict.

## Learnings

- Treat "attention to the W-sjet pair" as weak evidence only. Attention mass is not
  causal weight, and the prior H1 work already found cases where high attention did
  not align with causal effect.
- Pair-mass tests need careful controls because changing a four-vector can also change
  single-sjet pT, η/φ, HT, and event-level relative hardness.

## Decisions

- Use fresh validation batches by default (`--skip 24`) unless sample size becomes too
  small. Main downside is wider intervals for resolved-only estimates; report `n`
  throughout.
- Start thesis-model-only. Add organisms later if the thesis-model signal is clear or
  if a comparison would adjudicate a mechanism.
- Synthetic pair-mass interventions are allowed, but only alongside real-event swaps.
  Benefit: synthetic sweeps give factorial control over intended variables; risk:
  off-manifold artifacts. Treat synthetic-only evidence as provisional unless OOD
  diagnostics and real-swap controls agree.

## R1 Baseline + Observational Probes

Script:

```bash
.venv/bin/python experiments/h1/resolved_r1_baseline_probes.py \
  --model thesis-ent1-bn1-d152 --skip 24 --batches 12
```

Fresh slice: batches 25-36, `N=24576`; `lvbb=16330`, boosted `qqbb=6267`,
resolved `qqbb=1979`, `qq-neither=0`.

Key resolved baseline:

- `P(lep incorrectly W)=0.1369`; `P(nu=lep)=0.9859`.
- `P(any jet claims W)=0.7938`.
- `P(both W-sjets claim)=0.4512`; `P(exactly 1 W-sjet)=0.2602`;
  `P(no W-sjet claims)=0.2885`.
- `P(non-W jet claims)=0.2794`, but `P(H-ljet claims W)=0.0056`. The non-W claims are
  mostly not the H-ljet taking W.

Resolved truth-pair summaries:

- Correct events: `n=1708`, median `mjj=167.7 GeV`, median pair-relative-pT `0.225`,
  median W-sjet sum margin `+2.41`.
- Errors: `n=271`, median `mjj=98.8 GeV`, median pair-relative-pT `0.175`,
  median W-sjet sum margin `-4.21`.
- Both W-sjets claim: median `mjj=165.5`, pair-relative-pT `0.335`, W-sjet sum margin
  `+5.06`.
- Zero W-sjets claim: median `mjj=119.7`, pair-relative-pT `0.152`, W-sjet sum margin
  `-3.55`.

Observational predictive probes, 5-fold CV on resolved events:

- Event correctness: `mjj` alone AUC `0.639 +/- 0.028`; pair geometry
  `0.686 +/- 0.015`; single-W-sjet features `0.677 +/- 0.040`; context
  `0.711 +/- 0.029`; all physical features `0.724 +/- 0.023`.
- Both W-sjets claim: `mjj` alone AUC `0.527 +/- 0.014`; pair geometry
  `0.651 +/- 0.014`; single-W-sjet features `0.673 +/- 0.017`; context
  `0.752 +/- 0.020`; all physical features `0.785 +/- 0.018`.
- W-sjet sum margin: `mjj` alone R2 `0.017 +/- 0.011`; pair geometry
  `0.158 +/- 0.028`; single-W-sjet features `0.179 +/- 0.022`; context
  `0.315 +/- 0.043`; all physical features `0.405 +/- 0.043`.
- Hadronic verdict strength (`-lep_margin`): all physical features only R2
  `0.229 +/- 0.023`; `mjj` alone R2 `0.022 +/- 0.010`.

Wire/path forensics at the neutrino:

- `b2h2` scalar mean `-0.480`; W-sjets attention `0.460`, value contribution `-0.246`;
  other jets attention `~0.532`, value contribution `~-0.243`. This replicates the
  weak/unspecific resolved readout: W-sjet pair and other jets contribute comparable
  "found" signal.
- `b1h3` reads broader hadronic vs leptonic evidence: W-sjets value `+0.572`,
  other sjets `+0.316`, other ljets `+0.121`, lepton/nu `-0.281` combined.
- `b2h3` has W-sjet and other-ljet contributions with opposite signs (`+0.285` vs
  `-0.276`), so it may be cancelling topology/conflict information rather than acting
  as a simple W-found wire.
- Sjet-sjet attention: top partner-attention heads are `b2h3` (`partner=0.300`,
  `self=0.304`, `other-jets=0.271`), `b0h3` (`partner=0.266`, `self=0.325`,
  `other-jets=0.398`), then `b1h3` (`partner=0.150`). This is a path hint only.

Current interpretation:

- Evidence is **against a simple pair-mass-window detector** being the main resolved
  mechanism. `mjj` alone is weak for claims/margins, and correct/both-claim events have
  high median `mjj`, not an 80 GeV peak.
- Evidence is **more consistent with context-relative hardness / event topology**
  dominating the resolved signal. Pair features add some information, but context and
  single-W-sjet hardness features are stronger observational predictors.
- This does **not** rule out pairwise computation. It may be computing pairwise
  quantities that correlate with hardness/topology, using pair binding only as a weak
  auxiliary, or computing them in a way not captured by these linear probes.

Next causal tests suggested by R1:

1. Real-event resolved swaps: replace the truth W-sjet pair with resolved donor pairs
   matched on pair-relative-pT but different `mjj`, and vice versa. This tests whether
   `mjj` has residual causal control when hardness/context are approximately matched.
2. Synthetic controlled sweep: modify one W-sjet to sweep `mjj` while monitoring
   induced changes in pair pT, sum pT, individual pT/mass/tag, HT, and OOD distance.
3. Path tests for `b2h3`, `b0h3`, and `b1h3` sjet-sjet edges, with edge knockouts
   rather than attention-only claims.

## R2 Design — Real-Event Resolved Pair Swaps

Next experiment: use only real resolved W-sjet pairs as donors. Replace the two truth-W
sjet feature rows in a target resolved event with the two truth-W sjet rows from a
different resolved event, keeping target lepton/nu/H-ljet/rest event fixed. This is not
fully on-manifold because the donor pair may not recoil coherently against the target
event, but it avoids synthetic four-vectors and gives a direct causal perturbation.

Arms:

- **same-bin donor placebo**: replace with a donor close in `mjj` and pair-relative-pT.
  This estimates the generic cost of donor-pair transplantation.
- **low→high `mjj` at matched relative-pT**: low-`mjj` target gets high-`mjj` donor,
  matched on pair-relative-pT. If pair mass causally drives the resolved W claim, this
  should move W-sjet claims and lep/nu verdicts in the high-`mjj` direction.
- **high→low `mjj` at matched relative-pT**: reverse contrast.
- **low→high pair-relative-pT at matched `mjj`** and reverse. This is the positive
  control suggested by R1: relative hardness/context should have stronger causal effect
  than `mjj` if the observational probes were pointing at the mechanism.

Readouts:

- Target pair W-sjet claim counts, any jet claim, lep/nu verdict, W-sjet margins.
- Donor-match diagnostics: achieved changes in `mjj`, pair-relative-pT, pair sum-pT,
  individual pT/mass/tag, and baseline prediction composition of target/donor pools.
- Interpret pair swaps conservatively: a large effect can reflect donor-pair coherence,
  direction/recoil, or interaction with target context unless the matched diagnostics
  support the intended variable isolation.

## R2 Results — Real-Event Pair Swaps

Script:

```bash
.venv/bin/python experiments/h1/resolved_r2_pair_swaps.py \
  --model thesis-ent1-bn1-d152 --skip 24 --batches 12
```

Same fresh slice as R1: `N=24576`, resolved `n=1979`.

Baseline resolved: `P(lep=W)=0.1369`, `P(both W-sjets)=0.4512`,
`P(any W-sjet)=0.7115`.

Important control:

- **same-bin donor placebo** (`n=500`, close in `mjj` and inserted relative-pT):
  `P(lep=W)` increased `+0.048`, `P(any W-sjet)` decreased `-0.084`,
  `P(both W-sjets)` decreased `-0.086`, median W-sjet-sum-margin delta `-0.677`.
- Matched subset (`|delta relpt|<=0.03`, `|delta mjj|<=10`, `n=424`) was similar:
  `P(lep=W)` `+0.047`, `P(any W-sjet)` `-0.094`, median W-sjet-sum-margin
  `-0.687`.
- Interpretation: real-pair transplant itself disrupts resolved detection. Any targeted
  feature effect should be compared against this disruption floor.

`mjj` contrast with relative-pT matched:

- **low→high `mjj`**, high-quality subset `|delta relpt|<=0.03`, `n=301`:
  median `delta mjj=+302 GeV`, median `delta relpt≈0`, median `delta sumpt=+45.9 GeV`.
  Readouts: `P(lep=W)` `+0.010`, `P(any W-sjet)` `-0.063`,
  `P(both W-sjets)` `-0.276`, median W-sjet-sum-margin `-1.408`.
- **high→low `mjj`**, high-quality subset `|delta relpt|<=0.03`, `n=242`:
  median `delta mjj=-338 GeV`, median `delta relpt≈0`, median `delta sumpt=-71.4 GeV`.
  Readouts: `P(lep=W)` `+0.083`, `P(any W-sjet)` `-0.157`,
  `P(both W-sjets)` `-0.033`, median W-sjet-sum-margin `-0.959`.
- Interpretation: changing `mjj` while approximately matching inserted relative-pT does
  **not** produce a clean "higher mjj means stronger resolved W" effect. Both directions
  reduce W-sjet margins, and low→high especially damages two-sjet joint claiming. This
  is more consistent with transplant/coherence effects and/or individual donor features
  than with a robust pair-mass score.

Relative-pT contrast with `mjj` matched:

- **low→high inserted relative-pT**, matched subset `|delta mjj|<=10 GeV`, `n=256`:
  median `delta mjj≈0`, median `delta relpt=+0.331`, median `delta sumpt=+482 GeV`.
  Readouts: `P(lep=W)` `-0.066`, `P(any W-sjet)` `+0.113`,
  `P(both W-sjets)` `+0.043`, median W-sjet-sum-margin `+0.490`.
- **high→low inserted relative-pT**, matched subset `|delta mjj|<=10 GeV`, `n=294`:
  median `delta mjj≈0`, median `delta relpt=-0.360`, median `delta sumpt=-507 GeV`.
  Readouts: `P(lep=W)` `+0.735`, `P(any W-sjet)` `-0.861`,
  `P(both W-sjets)` `-0.680`, median W-sjet-sum-margin `-11.335`.
- Interpretation: relative hardness has a large causal effect even when pair mass is
  tightly matched. The effect is asymmetric: making a weak pair hard helps somewhat;
  making a hard pair weak almost destroys resolved W detection and hands W to the
  leptonic side.

R2 conclusion:

- Strong evidence that **relative hardness is causally important** for resolved W
  detection.
- Current evidence is **against `mjj` being a dominant causal variable** in the resolved
  detector. Real-pair swaps cannot fully rule out pair-mass computation, because donor
  swaps break event coherence and still alter individual masses/sum-pT, but the clean
  residual signal expected from a pair-mass circuit is not present here.
- Next: synthetic angle sweep at fixed individual sjet pT/mass/tag. This can change
  pair geometry/mass while leaving single-sjet features and HT nearly fixed, at the cost
  of possible directional/recoil OOD artifacts.

## R3 Design — Synthetic Angle Sweep

Goal: test whether resolved detection is sensitive to W-sjet pair geometry/mass when
single-sjet features are held fixed.

Interventions:

- **Co-rotation control**: rotate both truth-W sjets by the same `delta_phi`. This
  preserves individual sjet features, pair `mjj`, pair pT, pair sum-pT, and internal
  geometry; it changes only absolute direction relative to the rest of the event.
- **Relative-angle sweep**: keep the leading W-sjet fixed and rotate the subleading
  W-sjet to a grid of target `|delta_phi|` values. This preserves each W-sjet's pT, pz,
  E, mass, and tag, as well as scalar HT and pair sum-pT. It changes pair `mjj`, pair
  vector pT, and geometry relative to the rest of the event.

Epistemic caveat: pair `mjj` and pair vector pT are physically coupled under a fixed
individual-p4 angle sweep. Therefore this tests pair-geometry sensitivity, not pure
mass in isolation.

## R3 Results — Synthetic Angle Sweep

Script:

```bash
.venv/bin/python experiments/h1/resolved_r3_angle_sweep.py \
  --model thesis-ent1-bn1-d152 --skip 24 --batches 12
```

Same fresh slice, all resolved events used (`n=1979`).

Baseline:

- `P(lep=W)=0.1369`, `P(any W-sjet)=0.7115`, `P(both W-sjets)=0.4512`.
- median `mjj=154.5 GeV`, median pair-relative-pT `0.212`, median `|dphi|=1.167`.

Co-rotation control:

- Rotate both W-sjets together while preserving internal pair geometry.
- Small but nonzero direction/recoil effect:
  - `delta=0.50`: `P(lep=W)` `+0.002`, `P(any W-sjet)` `-0.004`.
  - `delta=1.57`: `P(lep=W)` `+0.008`, `P(any W-sjet)` `-0.026`.
  - `delta=pi`: `P(lep=W)` `+0.020`, `P(any W-sjet)` `-0.047`, median W-sjet-sum
    margin `-0.467`.
- Interpretation: absolute orientation relative to the rest of the event matters a
  little, so relative-angle sweeps have a direction/recoil confound. The size is much
  smaller than the relative-pT swap effects.

Relative-angle sweep:

- Leading W-sjet fixed; subleading W-sjet rotated to target `|dphi|`, preserving each
  W-sjet's pT, pz, E, mass, and tag.
- Median `mjj` moved from `95.3 GeV` at `|dphi|=0` to `256.1 GeV` at `pi`.
- Despite this large `mjj` movement:
  - `P(lep=W)` stayed almost fixed: from baseline `0.1369` to `0.137-0.140` across the
    whole sweep.
  - `P(any W-sjet)` stayed almost fixed: `0.708-0.715` except `0.704` at `pi`.
  - `P(both W-sjets)` declined with opening angle: `0.484` at `0`, `0.455` near `1.0`,
    `0.396` at `pi`.
  - Median W-sjet-sum-margin moved only mildly: about `+0.04` at `0`, `-0.14` at `pi`.

R3 conclusion:

- Strong evidence that the **lepton/neutrino resolved-channel verdict is not driven by
  pair mass or pair opening angle** in the way a human dijet-mass reconstruction would
  suggest.
- There is mild evidence that large opening angle weakens **two-sjet pair binding**
  (`both W-sjets` drops by ~5-6 points), but this barely propagates to the event-level
  channel verdict. This supports a picture where resolved success/failure is dominated
  by W-sjet individual/context-relative strength and readout competition, not an
  explicit pair-mass circuit.
- R2+R3 together make "model computes pair mass as the main resolved W score" unlikely.
Remaining live alternatives:
  1. weak pair binding exists but is not mass-like;
  2. individual W-sjet/context scores are enough for most resolved decisions;
  3. readout/broadcast mostly sees generic hadronic strength and does not require both
     W-sjets to bind as a single object.

## R4 Results — Direct W-sjet↔W-sjet Edge Knockout

Script:

```bash
.venv/bin/python experiments/h1/resolved_r4_sjet_edge_knockout.py \
  --model thesis-ent1-bn1-d152 --skip 24 --batches 12
```

Same fresh resolved slice (`n=1979`). Surgical hook validated against the standard
bottleneck hook (`max dev 0.00e+00` on the validation chunk).

Baseline:

- `P(lep=W)=0.1369`, `P(any W-sjet)=0.7115`, `P(both W-sjets)=0.4512`.
- Leading/subleading W-sjet claim rates: `0.6367 / 0.5260`.

Direct W-sjet-pair edge KO readouts:

- **All blocks/all heads bidirectional**: `P(lep=W)` `+0.0147`, `P(any W-sjet)`
  `-0.0248`, `P(both W-sjets)` `-0.1016`; lep flip `0.0268`.
- **Block 2 all heads bidirectional**: `P(lep=W)` `+0.0000`, `P(any W-sjet)`
  `+0.0061`, `P(both W-sjets)` `-0.1364`; leading `+0.0268`, subleading `-0.1572`.
- **b2h3 bidirectional**: `P(lep=W)` `+0.0000`, `P(any W-sjet)` `+0.0005`,
  `P(both W-sjets)` `-0.1587`; leading `+0.0222`, subleading `-0.1804`.
- Direction split for `b2h3`:
  - leading query → subleading key KO: `P(both W-sjets)` `+0.0167`, leading claim
    `+0.0222`, subleading unchanged.
  - subleading query → leading key KO: `P(both W-sjets)` `-0.1774`, subleading claim
    `-0.1804`, lep verdict unchanged.
- **b1h3 bidirectional**: `P(lep=W)` `+0.0147`, `P(both W-sjets)` `-0.0419`,
  lep flip `0.0197`.
- **b0h3 bidirectional**: essentially null despite high attention in R1.

R4 conclusion:

- There **is** a direct sjet-sjet causal path for pair binding: in block 2 head 3,
  the subleading W-sjet reads the leading W-sjet; cutting that edge removes about
  18 points of subleading W-claim rate and about 18 points of two-W-sjet joint claims.
- This path **does not materially drive the lep/nu channel verdict** on natural
  resolved events (`P(lep=W)` unchanged for b2h3, zero lep flips). The verdict readout
  can apparently rely on generic hadronic/W-sjet evidence without requiring the
  subleading token to bind as a clean second member of the W.
- Attention-only path hints were partly misleading: `b0h3` had high partner attention
  but null causal effect. This repeats the broader H1 lesson that attention mass is not
  causal weight.

Updated working model after R1-R4:

- Resolved W finding is **not** a human-like pair-mass reconstruction.
- The model mostly asks whether the event contains sufficiently hard/contextually
  plausible W-sjet evidence. Relative hardness has strong causal control.
- A weak pair-binding subcircuit exists, localized mainly to `b2h3` subleading→leading,
  but this is more important for the subleading sjet's own label than for the global
  channel decision.
- Failure modes therefore look like a combination of weak resolved W-sjet evidence and
  readout that does not require pair binding; not primarily missing pair-mass
  computation.

## Method Note — Correct-Event Restriction

Sid proposed studying only "correct" resolved events first to get a cleaner signal.
This makes sense, with caveats.

Usefulness:

- It targets the mechanism the model uses when it actually succeeds, instead of mixing
  successful resolved reconstruction with misses, one-sjet partial detections, and
  readout failures.
- It may sharpen weak paths like the `b2h3` pair-binding edge, which can be diluted when
  averaged over events where the model never formed a resolved W representation.
- It gives cleaner causal readouts: interventions can be measured as degradation from a
  known-good baseline.

Caveats:

- Conditioning on correctness creates selection bias. In R1, correct events are harder
  and have higher `mjj` than errors; a correct-only subset will overrepresent easy,
  high-relative-hardness topologies.
- "Correct" needs tiers, not one boolean:
  1. **channel-correct**: lep/nu take `none` in resolved truth events;
  2. **W-sjet-correct**: both truth-W sjets claim W;
  3. **clean-exclusive**: both truth-W sjets claim W, non-W jets do not claim W, lep/nu
     do not claim W.
- Results from clean-exclusive events should be validated on channel-correct partials
  and errors before calling them "the resolved mechanism."

Next recommended analysis: rerun R1/R4 summaries stratified by
`clean-exclusive`, `channel-correct but partial`, and `channel-error`. This should tell
us whether pair binding is a success-only subcircuit, while keeping the failure modes
visible.

## R5 Design — Success-Stratified Resolved Analysis

Goal: sharpen the signal by analyzing the model's **successful** resolved behavior
separately from partial detections and failures.

Definitions on resolved truth events:

- **channel-correct**: both lepton and neutrino predict `none` (class 0), i.e. the
  model assigns the W to hadronic objects rather than the leptonic side.
- **clean-exclusive**: channel-correct, both truth-W sjets predict W, and no non-W jet
  predicts W.
- **channel-correct partial**: channel-correct but not clean-exclusive. Split further
  by whether exactly one truth-W sjet claims W, zero truth-W sjets claim W, or non-W
  jets claim W.
- **channel-error**: lepton or neutrino predicts W in a resolved truth event.

Readouts:

- R1-style feature summaries and wire/value decompositions per stratum.
- R4-style direct W-sjet↔W-sjet edge KO per stratum, focusing on `b2h3` and all-head
  block controls.

Interpretation target:

- If direct pair binding is much stronger in clean-exclusive events, then `b2h3` is a
  success-path pair-binding circuit.
- If lep/nu verdict remains insensitive to pair-edge KO even in clean-exclusive events,
  then the global resolved verdict is still not reading a bound two-sjet object in a
  required way.

## R5 Results — Success-Stratified Resolved Analysis

Script:

```bash
.venv/bin/python experiments/h1/resolved_r5_success_strata.py \
  --model thesis-ent1-bn1-d152 --skip 24 --batches 12
```

Same fresh resolved slice (`n=1979`).

Baseline strata:

- `channel_correct=1694`, `channel_error=285`.
- `clean_exclusive=744`: both W-sjets claim W, no non-W jet claims W, lep/nu none.
- `channel_correct_partial=950`.
- Partial subgroups: `partial_one_wsj=504`, `partial_zero_wsj=299`,
  `partial_nonw_claim=549` (overlaps with the one/zero W-sjet categories).

Baseline feature differences:

- Clean-exclusive successes are much harder: median pair-relative-pT `0.367`, median
  W-sjet-sum-margin `+5.71`, median `mjj=186.5 GeV`.
- Channel-correct partials: median pair-relative-pT `0.164`, W-sjet-sum-margin `+0.40`,
  median `mjj=158.4`.
- Channel errors: median pair-relative-pT `0.175`, W-sjet-sum-margin `-4.09`,
  median `mjj=99.3`.

Wire decompositions at neutrino, by stratum:

- `b2h2` becomes much more W-sjet-specific in clean successes:
  - clean-exclusive: W-sjet attention `0.637`, W-sjet value `-0.470`, other-jet value
    `-0.267`, scalar `-0.730`.
  - partial: W-sjet attention `0.324`, W-sjet value `-0.173`, other-jet value `-0.362`,
    scalar `-0.525`.
  - channel-error: W-sjet value flips sign `+0.094`, other-jet value `+0.220`, scalar
    `+0.323`.
- `b2h3` clean successes show a strong W-sjet-vs-other-jet contrast:
  - clean-exclusive: W-sjet value `+0.539`, other-jet value `-0.255`, scalar `+0.288`.
  - partial: W-sjet value `+0.157`, other-jet value `-0.350`, scalar `-0.188`.
  - error: W-sjet value `+0.052`, other-jet value `-0.218`, scalar `-0.082`.
- `b1h3` remains broad hadronic-vs-leptonic evidence, not clean pair binding:
  clean and partial have similar all-jet values (`+1.067`, `+1.093`), while errors have
  much stronger leptonic negative contribution (`-0.683`).

Direct W-sjet edge KO by baseline stratum:

- In **clean-exclusive** events, `b2h3` subleading-query→leading-key KO has a large
  effect on pair binding:
  - `P(both W-sjets)` delta `-0.4946`.
  - subleading W-claim delta `-0.4946`.
  - median W-sjet-sum-margin delta `-1.963`.
  - `P(lep=W)` delta `+0.0000`, lep flip `0.0000`.
- `b2h3` bidirectional is identical on clean-exclusive events, confirming the causal
  direction is subleading query reading leading key.
- `b2_all_heads` on clean-exclusive is similar: `P(both)` delta `-0.4758`, no lep flip.
- `b1h3` on clean-exclusive has a smaller pair-binding effect and a small channel effect:
  `P(both)` `-0.1277`, `P(lep=W)` `+0.0403`.
- All-block/all-head direct pair-edge KO on clean-exclusive: `P(both)` `-0.4247`,
  `P(any W-sjet)` `-0.0726`, `P(lep=W)` `+0.0470`.

Partial and error strata:

- Channel-correct partials do **not** depend on the same direct `b2h3` edge; `b2h3`
  subleading→leading has `P(both)` delta `+0.0179`, `P(lep=W)` delta `0`.
- In `partial_one_wsj`, direct edge KO can increase two-sjet claiming (`b2h3` bidir
  `P(both)` `+0.1409`, all-block all-head `+0.2778`). This suggests some partial
  events are not simply missing a pair edge; they may be suppressed/competed by context.
- Channel errors are basically insensitive to direct pair-edge KO; this supports the
  view that failures are upstream weak evidence or readout competition, not just a
  missing direct sjet-sjet edge.

R5 conclusion:

- Sid's correct-event restriction was productive. In clean-exclusive successes, the
  pair-binding circuit becomes clear and causal: **`b2h3` lets the subleading W-sjet
  read the leading W-sjet, supporting its own W label**.
- However, even in clean-exclusive successes, cutting that edge does **not** make lep/nu
  take W. The global channel verdict is robust to losing the bound second sjet.
- Therefore the model has a success-path pair-binding subcircuit, but the global
  resolved decision is not mechanically dependent on reconstructing a two-sjet W object.
  It appears to be enough for the model to see strong hadronic/W-sjet evidence,
  especially from the leading/harder W-sjet and broader context.

Next candidate:

- Failure-rescue experiments stratified by channel-error and partial-zero/partial-one:
  boost pair-relative-pT, overwrite `b2h2/b2h3` clean-success wire values, or patch clean
  W-sjet streams into failures. This should distinguish "weak upstream evidence" from
  "downstream readout conflict" in the errors.

## R6 Design — Failure Rescue

Goal: distinguish weak upstream resolved evidence from downstream readout conflict.

Baseline strata:

- **channel_error**: resolved truth, lep or nu predicts W.
- **partial_zero**: channel-correct, no truth-W sjet claims W.
- **partial_one**: channel-correct, exactly one truth-W sjet claims W.
- **partial_nonw_claim**: channel-correct, at least one non-W jet claims W.

Interventions:

1. **Pair-hardness boost**: scale the two truth-W sjets' 3-momenta by factors
   `1.25`, `1.5`, `2.0`, preserving each sjet's invariant mass, direction, tag, and
   type. This should rescue failures if upstream relative hardness is the limiting
   variable.
2. **Clean-success scalar overwrite at lep/nu**: overwrite bottleneck scalars at both
   lepton and neutrino positions with clean-exclusive means:
   - `b2h2` only (hadronic W-found wire);
   - `b1h3` only (hadronic-vs-leptonic comparator);
   - `b2h3` only (pair-binding/secondary wire);
   - `b1h3+b2h2`;
   - all known verdict-ish wires `b1h3+b2h2+b2h0+b2h3`.

Readouts:

- For channel errors: rescue = lep and nu both become `none`.
- For partial-zero/partial-one: rescue = both truth-W sjets claim W, while the channel
  remains correct.
- Track whether interventions move jet labels, lep/nu labels, or both. Scalar
  overwrite at lep/nu should mostly test downstream sufficiency; pair-hardness boost
  tests upstream detection.

Caveats:

- Pair-hardness boosts are synthetic and can be OOD at large factors.
- Scalar overwrite proves readout sufficiency, not that the upstream model naturally
  computes those clean-success scalar values in failures.

## R6 Results — Failure Rescue

Script:

```bash
.venv/bin/python experiments/h1/resolved_r6_failure_rescue.py \
  --model thesis-ent1-bn1-d152 --skip 24 --batches 12
```

Same fresh resolved slice (`n=1979`).

Baseline groups:

- `channel_error=285`, median pair-relative-pT `0.175`.
- `partial_zero=299`, median pair-relative-pT `0.133`.
- `partial_one=504`, median pair-relative-pT `0.217`.
- `partial_nonw=549`, median pair-relative-pT `0.140`.
- `clean_exclusive=744`, median pair-relative-pT `0.367`.

Pair-hardness boost:

- Scaling the two truth-W sjets by `1.25`, `1.5`, `2.0` partially rescues channel
  errors:
  - channel-error `dChannelCorrect`: `+0.102`, `+0.204`, `+0.312`.
  - channel-error `dClean`: only `+0.004`, `+0.025`, `+0.042`.
  - channel-error W-sjet-sum-margin medians: `+0.405`, `+0.684`, `+1.027`.
- Boosting partial-zero events creates some W-sjet claims but not much clean rescue:
  - `dAnyW`: `+0.097`, `+0.164`, `+0.204`.
  - `dBothW`: `+0.007`, `+0.030`, `+0.030`.
  - `dClean`: `+0.007`, `+0.027`, `+0.023`.
- Boosting partial-one events increases two-sjet claiming somewhat:
  - `dBothW`: `+0.071`, `+0.099`, `+0.125`.
  - `dClean`: `+0.050`, `+0.071`, `+0.093`.
  - But `dAnyW` becomes negative at larger boosts (`-0.081`, `-0.157`), suggesting
    competition/over-hardness side effects.
- Boosting clean-exclusive events degrades clean object reconstruction:
  - `dClean`: `-0.124`, `-0.243`, `-0.462`.
  - `dBothW`: `-0.106`, `-0.199`, `-0.407`.
  - `dNonW`: `+0.046`, `+0.125`, `+0.245`.
- Interpretation: relative hardness is sufficient to move some failures toward
  hadronic-W evidence, but boosting is a blunt synthetic intervention. Too much
  hardness creates non-W competition and degrades clean successes.

Clean-success scalar overwrite at lep/nu:

- `b2h2_only`:
  - channel-error `dChannelCorrect=+0.909`, median lep-margin delta `-2.895`.
  - no jet-label changes (`dAnyW=dBothW=dNonW=0`) in all groups.
  - Interpretation: `b2h2` alone is almost sufficient to fix the downstream channel
    verdict, but not object-level resolved reconstruction.
- `b1h3_only`:
  - channel-error `dChannelCorrect=+0.407`, `dAnyW=+0.144`, `dBothW=+0.032`,
    median W-sjet-sum-margin `+1.358`.
  - Because `b1h3` is in block 1, overwriting lep/nu streams can affect later jet
    outputs through block-2 attention; this is a multi-hop effect, not a pure readout
    intervention.
- `b2h3_only`:
  - negligible channel rescue (`+0.014` on channel errors) and no jet-label rescue.
  - It can push lep margins in the wrong direction in partial groups.
- `b1h3+b2h2`:
  - channel-error `dChannelCorrect=+1.000`, median lep-margin delta `-4.004`.
  - object-level clean rescue remains small: `dClean=+0.035`, `dBothW=+0.032`,
    `dAnyW=+0.144`.
- `all_verdictish` (`b1h3+b2h2+b2h0+b2h3`):
  - same channel rescue as `b1h3+b2h2`; no meaningful extra object rescue.

R6 conclusion:

- Channel errors are largely **downstream verdict-wire failures** in the sense that
  clean-success `b2h2` plus `b1h3` values at lep/nu are sufficient to make all channel
  errors channel-correct.
- But they are not fully solved upstream: scalar overwrites barely create clean
  two-sjet reconstruction (`dClean=+0.035` in channel errors). This means a fixed
  channel verdict can coexist with weak or absent W-sjet object labels.
- Pair-hardness boosts provide upstream evidence and partially rescue channel errors,
  but not clean object reconstruction; they also degrade clean successes when too large.
- `b2h3` is not a failure-rescue wire for the channel verdict. Its role remains the
  success-path subleading-sjet pair-binding edge found in R5.

Updated explanation of failures:

- **Channel-error resolved events**: the lep/nu readout did not receive a strong enough
  clean hadronic-W verdict scalar, especially `b2h2`; providing that scalar fixes the
  channel. The underlying W-sjet object representation often remains weak.
- **Partial-zero/partial-one events**: failures are mostly upstream/object-level and
  not rescued by downstream readout overwrites. Hardness boosts help somewhat, but
  pair binding remains fragile.
- **Readout conflict vs weak signal**: both exist, at different levels. The channel
  error is downstream-fixable; the missing two-sjet object reconstruction is upstream
  weak-signal/competition-limited.

## R7 Design — Do Sjets Query Leptons/Neutrino/Large Jets To Decide "Am I W?"

Question: do leading/subleading/other sjets get their W-claim information by asking
whether other systems are claiming W, e.g. reading leptons/neutrino or large-R jets and
then claiming W if no other object/system does?

This is plausible under the broader context-relative scoring story. We need to separate:

- **attention/read presence**: an sjet attends to lep/nu or ljets;
- **causal dependence**: cutting those edges changes the sjet's W margin/claim;
- **sign of dependence**: reading a key may suppress or support a W claim.

Plan:

- Measure sjet-query attention by category for leading truth-W sjet, subleading
  truth-W sjet, and other sjets, split by clean-exclusive, partial, and channel-error
  resolved events.
- Causally knock out direct attention edges from sjet queries to:
  - lep+nu, lepton only, nu only;
  - H-ljet, all ljets;
  - other sjets;
  - partner W-sjet (for comparison to R5).
- Read out changes in leading/subleading/other-sjet W-claim rates, margins, and lep/nu
  channel verdict.

Interpretation:

- If cutting sjet→lep/nu or sjet→ljet edges strongly changes W-sjet claims, then those
  keys are part of the sjet's own claim computation.
- If clean successes are insensitive but partial/errors are sensitive, the context read
  may be a failure/competition path rather than the success path.
- If cutting edges to lep/nu or ljets does little, then resolved W-sjet claims mostly
  come from object features plus jet-context, not from asking the leptonic/large-jet
  systems whether W is already claimed.

## R7 Results — Sjet Queries To Lep/Nu And Large Jets

Script:

```bash
.venv/bin/python experiments/h1/resolved_r7_sjet_context_queries.py \
  --model thesis-ent1-bn1-d152 --skip 24 --batches 12
```

Same fresh resolved slice (`n=1979`): clean `744`, partial `950`, error `285`,
leading non-W sjet present in `1872` events.

Observational attention:

- W-sjet queries have large attention to lep/nu and large jets:
  - lead W-sjet → lep/nu top heads: `b1h2=0.943`, `b0h0=0.651`, `b2h2=0.532`.
  - sub W-sjet → lep/nu: `b1h2=0.999`, `b0h0=0.659`, `b2h2=0.584`.
  - both W-sjets → all ljets are near-total in `b2h1` (`~0.999`) and high in `b2h0`.
- Other-sjet queries look similar in broad strokes: high attention to lep/nu and large
  jets, and nontrivial attention to W-sjets in `b0h3/b2h3/b0h2/b1h3`.

Causal edge KOs, clean-exclusive events:

- **W-sjets → lep/nu is a major success-path input.**
  - Cutting both truth-W sjet queries to lep+nu, all blocks/all heads:
    - `dClean=-0.991`
    - `dLeadW=-0.935`, `dSubW=-0.961`
    - `dChannelCorrect=-0.406`
    - median lead/sub margin deltas `-6.176`, `-5.618`.
  - Cutting lead W-sjet → lep/nu alone:
    - `dClean=-0.946`, `dLeadW=-0.933`, `dSubW=-0.565`.
  - Cutting sub W-sjet → lep/nu alone:
    - `dClean=-0.960`, `dLeadW=-0.210`, `dSubW=-0.950`.
- **Where for lep/nu reads:**
  - `b0h0` is the largest single head: clean `dClean=-0.937`,
    `dLeadW=-0.832`, `dSubW=-0.848`, `dChannelCorrect=-0.468`.
  - `b1h3`: clean `dClean=-0.703`, especially subleading `dSubW=-0.659`.
  - `b2h2`: clean `dClean=-0.579`, `dLeadW=-0.243`, `dSubW=-0.534`,
    but `dChannelCorrect=0`, suggesting a late object-label support path without
    channel-verdict movement.
  - `b1h2` had very high attention but smaller causal effect (`dClean=-0.137`),
    another attention-not-causal reminder.
- **W-sjets → large jets also matter.**
  - Cutting both W-sjets → all ljets, all blocks/all heads:
    - clean `dClean=-0.788`, `dLeadW=-0.363`, `dSubW=-0.634`,
      `dChannelCorrect=-0.317`.
  - Cutting both W-sjets → H-ljet only:
    - clean `dClean=-0.612`, `dLeadW=-0.267`, `dSubW=-0.530`,
      `dChannelCorrect=-0.290`.
  - Main heads for large-jet reads in clean successes:
    - all ljets `b0h1`: `dClean=-0.519`;
    - H-ljet `b0h1`: `dClean=-0.464`;
    - all ljets `b1h1`: `dClean=-0.434`;
    - H-ljet `b1h1`: `dClean=-0.386`.
- **W-sjets → partner W-sjet remains important but smaller than lep/nu and ljet context.**
  - both W-sjets → partner, all blocks: clean `dClean=-0.445`,
    `dLeadW=-0.103`, `dSubW=-0.394`.
  - block-2 partner-only reproduces the R5 subleading effect:
    clean `dClean=-0.476`, `dSubW=-0.476`, channel unchanged.
- **W-sjets → other sjets has a competition-like role.**
  - Cutting both W-sjets → other sjets in clean events: `dClean=-0.375`,
    `dNonW=+0.335`.
  - This suggests reads involving other sjets help keep non-W sjet claims controlled,
    not just identify the W pair.

Partial/error strata:

- In partial events, cutting W-sjets → lep/nu still strongly damages lead/sub W labels
  (`dLeadW=-0.492`, `dSubW=-0.279`) and reduces channel correctness (`-0.303`), but
  `dClean` is near zero because these events are already not clean.
- In channel-error events, some context cuts can improve the channel:
  - both W-sjets → all_ljets, all blocks: `dChannelCorrect=+0.165`.
  - the large effect appears particularly in b0h0 all-ljet/H-ljet reads in errors
    (`dChannelCorrect` up to `+0.470` for all-ljets `b0h0` in the printed top-effects
    table).
  - Interpretation: in failures, large-jet/context reads can suppress or confuse the
    resolved W-sjet evidence; in successes, those same reads support clean labels.

Other non-W sjet query:

- The leading non-W sjet also reads W-sjets/ljets/lepnu causally.
- In clean events, cutting other-sjet → W-sjets:
  - `dNonW=+0.425`, `dOther1W=+0.358`, `dClean=-0.471`.
- This is strong evidence that non-W sjets use W-sjet context to avoid claiming W.
  It is closer to an exclusion/competition mechanism: reading the W-sjets suppresses
  non-W sjet W-claims.

R7 conclusion:

- Yes, sjets strongly query lep/nu and large jets. Direct W-sjet→lep/nu and W-sjet→ljet
  reads are **causally necessary for clean resolved W-sjet labels**.
- The sign is not simply "if nobody else is claiming W, I claim W." In clean successes,
  removing lep/nu or large-jet reads usually **destroys** W-sjet claims, so those reads
  provide positive contextual support or calibration for the W-sjet label.
- A more accurate description: W-sjet tokens compute "am I W?" from their own features
  plus event context, where lep/nu and large jets provide a strong contextual frame
  (likely channel/leptonic-vs-hadronic and H-vs-W contrast), while other sjets/W-sjets
  provide exclusion/competition signals.
- Non-W sjets do use W-sjet context suppressively: cutting other-sjet→W-sjet edges makes
  non-W sjet W-claims increase in clean events.

Updated mechanism sketch:

- Clean resolved success requires W-sjets reading:
  1. lep/nu context, especially `b0h0`, `b1h3`, `b2h2`;
  2. H/large-jet context, especially `b0h1`, `b1h1`;
  3. partner W-sjet context, especially `b2h3` for subleading binding;
  4. other-sjet context to keep non-W claims controlled.
- The global lep/nu verdict still does not require two-sjet pair binding, but clean
  object-level resolved reconstruction is much more context-dependent than R1-R6 alone
  made clear.

## R8 Design — Message Content On The Sjet Context Paths

Question: after R7 localized important direct reads, what information is actually
passed on those paths?

Approach:

- Decompose each head/query scalar into key-category contributions:

  `message(query, category, bXhY) = sum_key attention(query,key) * down_h(Wout_h V_h(key))`

  This is the scalar before the bottleneck is projected back up. Summing over all keys
  should reconstruct the cached head/query bottleneck scalar.
- Save both attention mass and scalar message contribution for lead W-sjet, sub W-sjet,
  and leading non-W sjet queries, across all blocks/heads and categories:
  lep/nu, lepton, neutrino, H-ljet, all ljets, W-sjets, other sjets, partner/self, etc.
- Split by baseline behavior: clean-exclusive, partial-one, partial-zero,
  partial-with-non-W, and channel-error.
- Probe message contents against named physics features:
  W-pair geometry/hardness, individual W-sjet features, lep/nu features,
  large-jet context, and global event counts/hardness.
- Also report message-only outcome probes. These are observational and are not causal
  sufficiency claims; they tell us whether the already-localized messages contain
  enough information to predict the model's object/channel behavior.

Interpretation discipline:

- A high message-feature correlation or R2 means the information is decodable from the
  message. It does not prove the model computes that named feature internally.
- Message categories can be confounded by event strata. If a lep/nu message correlates
  with W-pair hardness, that may reflect correlations in clean events rather than
  pair information literally being transmitted by lep/nu keys.
- PySR is installed, but symbolic regression is only worth running after the named
  probes show a compact target; otherwise it risks fitting a descriptive proxy rather
  than a mechanism.

## R8 Results — Message Content On The Sjet Context Paths

Script:

```bash
.venv/bin/python experiments/h1/resolved_r8_message_content.py \
  --model thesis-ent1-bn1-d152 --skip 24 --batches 12
```

Same fresh resolved slice (`n=1979`): clean `744`, partial `950`, error `285`,
partial-one `504`, partial-zero `299`, partial-with-non-W `549`.

Sanity check:

- The per-key scalar decomposition reconstructs the cached bottleneck scalar:
  max `|sum_key message - cached bottleneck| = 4.8e-7`.

Focused path message means:

- W-sjet → lep/nu:
  - `both lepnu b0h0`: clean `+1.586`, error `+1.316` (modest clean-error gap).
  - `both lepnu b1h3`: clean `-0.868`, error `-0.629`.
  - `both lepnu b2h2`: clean `+1.417`, error `+0.571` (large gap).
- W-sjet → H/ljet:
  - `both H_ljet b0h1`: clean `+1.125`, error `+1.324`.
  - `both H_ljet b1h1`: clean `-1.140`, error `-1.590`.
  - `both all_ljets b0h1`: clean `+0.888`, error `+1.380`.
  - `both all_ljets b1h1`: clean `-1.160`, error `-1.746`.
- Partner path:
  - `sub partner b2h3`: clean `+0.578`, partial-one `+0.276`,
    partial-zero `+0.025`, error `+0.061`.
  - `lead partner b2h3`: near zero everywhere (`clean=-0.043`, error `-0.008`).
- Leading non-W sjet reading W-sjets:
  - `other1 W_sjets b2h3`: clean `+0.501`, partial-zero `+0.012`, error `+0.053`.
  - `other1 W_sjets b1h3`: clean `+0.726`, error `+0.586`.

Largest clean-error message contrasts:

- The biggest contrasts include `msg_both_all_jets_b2h3`, `msg_both_W_sjets_b2h3`,
  `msg_both_lepnu_b2h2`, `msg_both_all_ljets_b1h1`, and several `b1h2` lep/nu
  messages.
- Caution: R7 found `b1h2` lep/nu attention had small causal effect despite large
  attention, so the `b1h2` contrasts are candidate correlates/redundant signals, not
  established mechanism.

Feature content:

- `b0h0` lep/nu messages are mostly decodable from lep/nu kinematics:
  - `both lepnu b0h0` top correlations: `lepnu_sumpt r=-0.69`,
    `lepnu_pt r=-0.68`, `lep_pt r=-0.65`.
  - Named-feature R2: all physics `0.886`, lep/nu-only `0.595`.
  - Because the message is attention times value, correlations with W-sjet tags also
    appear; we should split attention vs value in a follow-up before claiming pure
    lep/nu-value content.
- `b2h2` lep/nu messages carry behaviorally relevant W-sjet-support information:
  - `both lepnu b2h2` correlates with `wsj_sum_margin r=+0.61`,
    `clean r=+0.47`, `both_wsj r=+0.46`.
  - Feature R2 is modest: all physics `0.359`, best non-all/global `0.166`.
  - Interpretation: the late lep/nu path contains resolved-success information not
    captured by simple named features alone.
- H/ljet messages are strongly tag/context coded:
  - `both H_ljet b0h1`: `h_ljet_tag r=+0.89`, largejet-only R2 `0.313`.
  - `both H_ljet b1h1`: `h_ljet_tag r=-0.76`, largejet-only R2 `0.239`.
  - These paths also correlate with `best_other_margin` (`~0.43-0.52` in magnitude),
    consistent with ljet/H context participating in competition/control rather than
    simply declaring "W already claimed elsewhere".
- Partner `b2h3` is the cleanest low-dimensional path:
  - `sub partner b2h3` top correlations: `pair_relpt r=+0.81`,
    `pair_relsumpt r=+0.81`, `lead_pt/pt_max r=+0.77`, `pair_pt r=+0.76`.
  - Feature R2: all physics `0.771`, pair geometry `0.718`.
  - This supports the R5/R7 picture: the subleading sjet reads the leading sjet's
    candidate strength/relative hardness, not an obvious pair-mass formula.
- Non-W sjet reading W-sjets mirrors the partner-hardness signal:
  - `other1 W_sjets b2h3`: `pair_relpt r=+0.76`, pair geometry R2 `0.629`.
  - Together with R7's causal result (cutting other1→W-sjets increases non-W W-claims),
    this suggests non-W sjets receive a W-pair strength signal that suppresses their
    own W claim. The scalar sign alone should not be interpreted without the up
    projection/readout direction.

Message-only behavior probes:

- Focused messages predict model behavior well observationally:
  - clean AUC `0.883`;
  - channel-correct AUC `0.892`;
  - both-W-sjets AUC `0.878`;
  - lead/sub W AUC `0.892/0.884`.
- Lep/nu-support messages alone are also strong (`clean AUC=0.826`,
  `sub-W AUC=0.825`), consistent with R7's causal dependence on W-sjet→lep/nu reads.
- These probes are not causal reconstructions; they say the information is present in
  these messages.

Targeted PySR:

```bash
.venv/bin/python experiments/h1/resolved_r8_message_content.py \
  --model thesis-ent1-bn1-d152 --skip 24 --batches 12 \
  --run-pysr --pysr-target msg_sub_partner_b2h3 --pysr-iterations 40
```

This was run because `msg_sub_partner_b2h3` had high pair-geometry R2 and is a causal
success-path edge from R5/R7.

Best simple formulas:

- complexity 1: `pair_relsumpt`, loss `0.158`.
- complexity 5: `(pair_relpt * 2.611762) - 0.3225455`, loss `0.0899`.
- complexity 6: `pair_relpt * (log(lead_pt) - 4.4869614)`, loss `0.0783`.
- More complex formulas add mild `lead_m`, `pt_max`, and `pair_relsumpt` terms,
  with only modest loss improvement (`~0.069-0.074`).

Interpretation:

- Symbolic regression agrees with the named probes: the causal subleading-partner
  message is mostly a relative-hardness signal, with possible mild modulation by the
  leading sjet mass/normalization.
- It does **not** look like a pair-mass-window formula. This is another independent
  reason to deprioritize "the model computes pair mass" as the main resolved-W
  mechanism, while still allowing that mass-like features may weakly modulate some
  paths.

R8 updated mechanism:

- W-sjet object labels receive several kinds of direct context:
  1. lep/nu messages that carry lepton-side kinematics and late resolved-success
     signals (`b0h0`, `b2h2`);
  2. H/ljet messages that carry H-tag/large-jet context and competition information
     (`b0h1`, `b1h1`);
  3. a causal subleading→leading partner read in `b2h3` that is mostly relative
     hardness/candidate-strength information;
  4. non-W-sjet reads of W-sjets that appear to transmit W-pair strength for exclusion.
- The current evidence favors a context-relative candidate-strength and competition
  mechanism over an explicit two-sjet invariant-mass reconstruction.

Next useful tests:

- Split message into attention-mass vs value-only terms for the strongest paths,
  especially `b0h0` lep/nu and `b2h3` partner, to see whether the feature content
  lives in the value vector, attention gating, or their product.
- Do path/message replacement: clean partner message into partial-zero/error, and
  failure partner message into clean, with same-stratum controls. This would test
  causal sufficiency of the decoded message rather than just information presence.
- For the ljet/H context paths, patch H-tag-like value components or stratify by H-ljet
  tag to test whether the ljet message is directly controlling competition vs acting
  as a channel/topology context signal.

## R9 Design — Causal Replacement Of The Partner Message

Question: is the `b2h3` subleading-query→leading-key partner message just correlated
with clean resolved success, or is it causally sufficient to move failures toward
two-sjet W reconstruction?

Intervention:

- For block 2 head 3, compute the scalar partner-message contribution:

  `msg_sub_partner_b2h3 = attention(sub, lead) * down_h(Wout_h V_h(lead))`

- In a custom attention hook, alter only the block-2/head-3 bottleneck scalar at the
  chosen query token by:

  `scalar_total(query) += target_partner_message - current_partner_message`

  This keeps all other events, objects, heads, and non-partner key contributions
  unchanged, and changes the final head output only through the normal bottleneck-up
  direction.
- Primary query: subleading truth-W sjet. Direction control: leading truth-W sjet.

Conditions:

- Clean-message insertion: replace each event's sub-partner message with a random
  clean-event sub-partner message, and with the clean mean/p75.
- Failure-message insertion into clean events: replace with random partial-zero or
  channel-error sub-partner messages.
- Same-stratum shuffle: replace each event's sub-partner message with another message
  drawn/permuted from the same baseline stratum. This controls for distributional
  perturbation while removing event-specific alignment.
- Zero-message and PySR-simple formula controls:
  - zero tests scalar-only necessity in the same coordinate;
  - formula uses the R8 simple expression
    `(pair_relpt * 2.611763) - 0.32256523` as an analytical reconstruction candidate.

Predicted interpretations:

- If clean-message insertion rescues partial-zero/error events, the partner message is
  at least partially sufficient.
- If it does not rescue failures but failure-message insertion damages clean events,
  the path is necessary/fragile in successes but not sufficient by itself.
- If same-stratum shuffle is damaging, event-specific matching of the message matters;
  if it is benign, group-level message strength may be enough.
- If the PySR-simple replacement preserves behavior, the path is well described by the
  relative-hardness formula. If it damages behavior, the formula is only a proxy for
  the message, not a faithful replacement.

## R9 Results — Causal Replacement Of The Partner Message

Script:

```bash
.venv/bin/python experiments/h1/resolved_r9_partner_message_patch.py \
  --model thesis-ent1-bn1-d152 --skip 24 --batches 12
```

Same fresh resolved slice (`n=1979`): clean `744`, partial `950`, error `285`,
partial-one `504`, partial-zero `299`, partial-with-non-W `549`.

Sanity checks:

- Partner-message/total-scalar decomposition reconstructs cached scalar:
  max error `2.38e-7`.
- Surgical baseline exactly matches hooked baseline: max deviation `0.00`.
- Baseline sub-partner message means reproduce R8:
  - clean `+0.578`;
  - partial-zero `+0.025`;
  - partial-one `+0.276`;
  - error `+0.061`.

Clean-message insertion into failures:

- `sub_clean_mean`:
  - partial-one: `dClean=+0.117`, `dBoth=+0.208`, `dSub=+0.198`,
    median sub-margin `+0.524`.
  - partial-zero: `dAny=+0.244`, `dSub=+0.244`, but `dBoth=0`, `dClean=0`.
  - error: `dAny=+0.098`, `dSub=+0.102`, `dBoth=+0.004`, but
    `dChannel=0`, `dClean=0`.
- `sub_clean_p75` is a stronger insertion:
  - partial-one: `dClean=+0.153`, `dBoth=+0.274`, `dSub=+0.262`.
  - partial-zero: `dAny=+0.344`, `dSub=+0.344`, still `dBoth=0`, `dClean=0`.
  - error: `dAny=+0.165`, `dSub=+0.168`, `dBoth=+0.004`, `dChannel=0`.
- Interpretation: the sub-partner message is sufficient to make the subleading sjet
  claim W in some failures, especially partial-one events. It is not sufficient to
  create a full clean reconstruction when the leading W-sjet is also absent, and it
  does not affect the lep/nu channel verdict.

Failure-message insertion into clean events:

- `sub_pzero_random` into clean:
  - `dClean=-0.368`, `dBoth=-0.368`, `dSub=-0.368`,
    median sub-margin `-1.358`.
- `sub_error_random` into clean:
  - `dClean=-0.351`, `dBoth=-0.351`, `dSub=-0.351`,
    median sub-margin `-1.318`.
- `sub_zero` into clean:
  - `dClean=-0.375`, `dBoth=-0.375`, `dSub=-0.375`,
    median sub-margin `-1.453`.
- Interpretation: the scalar partner message is causally necessary for a large
  fraction of clean subleading W-sjet claims. This is a scalar-coordinate version of
  the R5/R7 edge-KO result.

Same-stratum/event-specific controls:

- `sub_same_stratum_shuffle` in clean events:
  - mean message delta is `~0`, but `dClean=-0.172`, `dSub=-0.172`.
- `sub_clean_random` in clean events:
  - mean message delta is also `~0`, but `dClean=-0.185`, `dSub=-0.185`.
- `sub_clean_mean` damages clean less (`dClean=-0.048`) and `sub_clean_p75` damages
  even less (`dClean=-0.026`), despite being cruder.
- Interpretation: event-specific alignment of the partner scalar matters. A clean-like
  distribution is not automatically faithful; random clean messages can push many
  events across threshold in the wrong direction even with zero mean delta. The scalar
  is not just a group-level "resolved success" flag.

Formula replacement:

- `sub_pysr_simple`, using `(pair_relpt * 2.611763) - 0.32256523`:
  - clean: `dClean=-0.035`, `dSub=-0.035`, median sub-margin `-0.115`.
  - partial-one: `dClean=+0.056`, `dBoth=+0.077`, `dSub=+0.063`.
  - partial-zero: `dAny=+0.104`, `dSub=+0.104`, still `dBoth=0`, `dClean=0`.
  - error: `dAny=+0.035`, `dSub=+0.035`, `dChannel=0`.
- Interpretation: the simple relative-pT formula is a reasonably faithful replacement
  for many clean events and has mild rescue effects, but it is weaker than clean-mean
  or clean-p75 insertion. It should be treated as a useful low-dimensional proxy, not
  a full reconstruction of the path.

Direction control:

- `lead_clean_submsg_random` patches the same clean sub-partner-message distribution
  into the **leading** W-sjet query's partner coordinate:
  - partial-zero: `dLead=+0.117`, `dSub=0`, `dClean=0`.
  - partial-one: `dLead=+0.022`, `dSub=0`, `dClean=+0.048`.
  - clean: `dLead=-0.034`, `dSub=0`, `dClean=-0.034`.
- Interpretation: the scalar coordinate acts locally on the patched query token. This
  supports the reading that sub-partner b2h3 controls the subleading W-sjet label,
  rather than globally broadcasting a resolved-W verdict.

R9 conclusion:

- The `b2h3` subleading partner message is **necessary and locally sufficient** for
  many subleading W-sjet object labels.
- It is **not globally sufficient** for clean resolved reconstruction: it does not fix
  the lep/nu channel verdict and cannot produce two W-sjet claims when the leading
  W-sjet is also missing.
- The failure mode is therefore factorized:
  - missing subleading claim in partial-one events can be partly repaired by this
    partner scalar;
  - partial-zero events need leading-sjet evidence too;
  - channel-error events still need downstream verdict-wire repair (`b2h2/b1h3` from
    R6), even if the subleading object label is helped.
- Analytical reconstruction: the relative-pT formula is directionally faithful but not
  exact. The partner message is mostly relative-hardness-like, but event-specific
  calibration still matters.

Updated mechanism after R9:

- The resolved W pair is assembled as object-local claims, not as one central
  pair-mass decision.
- The leading W-sjet claim appears upstream/independent of the `sub←lead b2h3` edge.
- The subleading W-sjet reads a leading-candidate-strength scalar in `b2h3`; this can
  tip the subleading token into W.
- The lep/nu channel verdict is downstream and separate: it can remain wrong even
  after a subleading W-sjet claim is created.

## Session Synthesis — Current Resolved qqbb W Mechanism

Current best description:

- The model does not appear to find resolved qqbb W cases by computing a central
  two-sjet invariant-mass decision and broadcasting it.
- Resolved W reconstruction is factorized across object-local claims and a separate
  channel verdict:
  1. leading W-sjet evidence is formed upstream from relative hardness/object/context
     features;
  2. subleading W-sjet evidence is helped by a direct `b2h3` read from the leading
     W-sjet, carrying mostly relative-hardness/candidate-strength information;
  3. lep/nu and H/large-jet context reads calibrate W-sjet object labels and
     competition;
  4. the lep/nu channel verdict is delivered downstream, mostly through `b2h2` and
     `b1h3`, and can be repaired without repairing object labels.

Strongest evidence:

- R2/R3: real swaps and angle sweeps do not support pair mass as the dominant causal
  variable; relative hardness has much stronger causal leverage than `mjj`.
- R4/R5: direct `b2h3` subleading-query -> leading-key edge is causally important for
  the subleading W-sjet label, especially in clean successes, but not for the lep/nu
  channel verdict.
- R6: clean-success `b2h2+b1h3` scalar overwrites fix channel errors but barely rescue
  clean two-sjet object reconstruction, separating channel readout from object binding.
- R7: W-sjet labels causally depend on direct reads from lep/nu and large jets; this
  argues against a simple "claim W if nothing else does" rule.
- R8: the causal partner path is mostly relative-hardness-like (`pair_relpt`/relative
  sum-pT), while large-jet paths are strongly H-tag/context coded.
- R9: scalar replacement of `msg_sub_partner_b2h3` is necessary and locally sufficient
  for many subleading W-sjet claims, but not globally sufficient for clean resolved
  reconstruction or channel repair.

Important caveats:

- We have a good local mechanism for the subleading partner-binding edge, but not yet
  a full reconstruction of the leading W-sjet claim.
- Lep/nu and ljet context paths are causally necessary, but R8 message correlations do
  not yet tell us whether their feature content lives in attention, value vectors, or
  the product.
- The PySR formula for the partner message is useful but incomplete: same-stratum
  shuffles and formula replacement show event-specific calibration still matters.
- Most causal interventions were on the same fresh validation slice (`skip=24`,
  batches `12`). We should eventually validate the final mechanism on another slice.

Most useful next-session experiments:

- R10: split key paths into attention-only vs value-only content for `b0h0` lep/nu,
  `b0h1/b1h1` large-jet context, and `b2h3` partner.
- R11: reconstruct the factorized circuit by combining leading-sjet evidence patching,
  `sub<-lead b2h3` partner patching, and lep/nu verdict-wire overwrites; test whether
  the pieces predictably compose.
- R12: find the leading W-sjet claim circuit, likely by running message replacement or
  edge KOs on the strongest R7 lep/nu and ljet context paths into the leading W-sjet.
- Cross-slice validation: rerun the key R5/R7/R8/R9 results on a new skip range before
  treating the mechanism as stable.

Session stopping point:

- We have moved from "there is a resolved-W pair path" to a more precise claim:
  `b2h3` provides a local, causal, relative-hardness-like partner signal that helps the
  subleading W-sjet claim W. The remaining unresolved part is how the leading W-sjet
  obtains its own claim and how the object claims compose with the downstream verdict
  wires into full clean resolved classification.

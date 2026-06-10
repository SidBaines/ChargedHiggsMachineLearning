# H1 v3 test plan — the "ask the jets" hypothesis (preregistered brainstorm)

> Drafted 2026-06-10 (late night), Sid + Claude, **before running any of these** —
> predictions below are preregistered. Status: for discussion; mark experiments
> RUN/SKIP as we go and record outcomes against the predictions.
>
> **STATUS 2026-06-11 (small hours)**: Wave 1 RUN (D2 ✓ timing, A1 ⚡ models diverge —
> thesis mass-bump@80, organism monotonic; B1/B2/B3 ✓, A6 ✓) and wave 2 RUN
> (A2 ✗ mass-not-sufficient, A1-pT ✓ pT dominates, A4 partial-sufficiency,
> A8 winner-take-all by pT, C1/C4 reader heads mapped, D4-lite: **full causal control
> of ν's verdict via the b2h2+b1h3 scalars; lockstep broken**). Results in the
> session log (wave-1 and wave-2 sections).
>
> **Wave 3 RUN (same night)**: A5 ✓ (tag = continuous Xbb-score, real gate; full
> H-disguise re-labels the W as H), E2 ✓ (candidacy ≈ pT↑↑ × mass-window × anti-tag,
> direction-blind), D4-lite@lep ✓ (parallel readers closed bidirectionally),
> A3-lite ✓ (resolved detector weak+unspecific), D6 ✓ (candidacy decodable at the
> EMBEDDING, AUC 0.93), D1c ✓ (verdict transits blk-1/2 reads; final-depth patch
> decouples jet label from verdict; embed-patch partial recovery = stage-B teaser).
> Stages (a) and (c) of the circuit are closed on the thesis model. **Next: Stage B
> (SB1-SB6 below).** New organism trained (0.8142) — universality reruns pending.

## The hypothesis under test

**H-A ("ask the jets")**: Each jet token computes its own *W-candidacy* — "am I (part
of) the hadronic W from the H⁺?" — from its kinematics (mass window ≈ m_W for ljets;
pair structure for sjets). The lep/ν tokens read the *aggregated* answer and set their
class by mutual exclusion: some jet claims the W ⇒ lep/ν = "none"; no jet claims ⇒
lep/ν = "W". A secondary continuous input (leptonic-side hardness) shifts the boundary.
The verdict is broadcast so lep and ν move in lockstep.

**Rivals it must beat:**
- **H-B ("parallel global")**: every token *independently* computes the channel verdict
  from raw jet information (residual streams converge to a shared event summary); the
  jets' own claims and the lep/ν verdict are siblings of one computation, not
  parent→child. (Round-1's ablation hyper-redundancy points this way!)
- **H-C ("lepton-side primary")**: lep/ν compute their own W-likeness from the leptonic
  system; the *jets* read that and claim the W only if the leptonic side declined.
  (Information flows the other way.)
- **H-D ("output bookkeeping")**: the XOR is shaped mostly by the loss/output structure
  (training saw exactly one W system per event), with no dedicated internal mechanism —
  exclusivity would then be brittle under interventions that never co-occur in data.

H-A vs H-B is the central discrimination and is fundamentally about **direction and
timing of information flow** (jets first, lep/ν later vs simultaneous). H-C is mostly
already disfavoured (lepton corruptions barely matter) but cheap to kill properly.

## Evidence already in hand (rounds 1–2d)

Coupling P(ν=lep) ≥ 0.95 under every intervention; jet-system swap moves the verdict
both directions; surgical truth-W removal ≫ sham (77%/66% vs 20%/4.5% boosted,
thesis/organism); prediction-level XOR ≈ 0.96–0.99; mediation: flips track whether the
model *re-finds* a jet-W (organism 72% vs 5.7%). All replicated across a 33× size gap.
Strata: qqbb = 76.9% boosted (W = one ljet, m ≈ 85 GeV) + 23.1% resolved (W = 2 sjets).

## Conventions for all experiments

- Models: **new organism first** (d20, 2 blocks, no LayerNorm ⇒ clean logit-lens;
  trained with full config provenance), then old organism (replication), then thesis
  model (does it scale?). Types: **3=ljet, 4=sjet** (verify kinematically at load!).
- Data: 20250321v1 val split; always report per-stratum (lvbb / qqbb-boosted /
  qqbb-resolved) and flag the resolved stratum's noisier truth (m_jj tail).
- Definitions: a token "claims W" iff argmax class = 2; claim strength = logit margin
  (W − none); "XOR violation" = event with 0 or ≥2 W-claimants among {jets, lep/ν pair}.
- Costs: S < 30 min, M ≈ half-day, L ≈ multi-day (mostly engineering).

---

## A. Input-level interventions (more ablations & dose-response)

**A1. W-ljet mass dose-response (boosted).** Set the truth-W ljet's mass to a grid
(0–250 GeV; adjust E at fixed p⃗), measure: that jet's claim rate, lep/ν flip rate.
*Why*: directly tests WHAT makes a jet claim — the natural candidate is a mass window.
*Predict (H-A)*: claim rate is a bump centred ~80–90 GeV; lep/ν flips rise symmetrically
as the jet leaves the window (both directions!), mirroring the claim curve.
*Readouts*: flat curve ⇒ mass isn't the feature (→ A5/E4 for tag/pT); threshold-not-bump
⇒ "heavy jet" detector, not W detector; lep/ν flips NOT mirroring claim curve ⇒ lep/ν
don't read the claim (hit to H-A, points to H-B). ⚠ Round 1 logged "ljet mass ×0.5 →
only 4.5% flips" — type-naming now suspect; this rerun adjudicates. **Cost S.**

**A2. Same dose-response on the H-ljet (two controls in one).** Drag the H-ljet's mass
in lvbb events toward 80 GeV. *Predict (H-A)*: the H-ljet starts claiming W at some
rate, and lep/ν flip toward "none" in lockstep — a *sufficiency* result. If the H-ljet
never claims regardless of mass ⇒ candidacy uses more than mass (tag, ΔR-to-lepton,
pT-balance), or the H-assignment "protects" it (ordering of decisions — interesting
either way). In qqbb-boosted, dragging H-ljet to 80 creates a *second* candidate → see A8.
**Cost S.**

**A3. Resolved-pair surgery.** In resolved events, rescale/rotate one truth-W sjet to
move m(jj) toward/away from 80 GeV at fixed single-jet properties.
*Predict (H-A, pair-aware)*: claim/flip tracks m(jj) window. *Alternative*: model never
binds the pair (only individual sjet features matter) — consistent with its weaker
resolved performance and the sham fragility; would mean the "resolved W detector" is
crude or absent. **Cost M** (needs care with which features the model derives).

**A4. W-candidate insertion (sufficiency, surgical).** Insert into lvbb events: (a) a
real truth-W ljet copied from a boosted qqbb event; (b) a synthetic ljet with m=80 and
typical pT/η; (c) controls: H-like ljet (m=125), soft sjet, padding-like row.
*Predict (H-A)*: (a)≈(b) ≫ (c); inserted jet claims W; lep/ν flip to "none" at a rate
comparable to (1 − sham) effects; XOR preserved (claims transfer, not duplicate).
*Readout*: (a)≫(b) ⇒ candidacy needs more than (m,pT) — context features; flips without
the inserted jet claiming ⇒ verdict reads raw jet content, not computed claims (H-B).
**Cost S–M.**

**A5. Tag-bit surgery.** First check what `tag` encodes per type (preprocessing:
INCLUDE_TAG_INFO). Then flip/zero tag on truth-W ljet vs H-ljet separately.
*Predict*: H-candidacy is tag-sensitive (h→bb̄), W-candidacy weakly/anti tag-sensitive.
Round 2 already saw small tag effects on the *verdict*; this targets *claims*. **Cost S.**

**A6. Slot-permutation control.** Move the W-jet between slots; verify nothing changes
(permutation invariance). Run once, cite forever for all rebuild-style experiments.
**Cost S.**

**A7. Graded fade-out.** Interpolate the truth-W ljet's features toward padding.
*Why*: distinguishes binary detection (sharp flip at threshold) from graded evidence
accumulation (smooth lep/ν margin shift). Margins, not just argmax, are the readout.
**Cost S.**

**A8. Two-claimant events.** Give a boosted qqbb event a second W-like ljet (insertion
or H-mass-drag). *Predict (H-A + global exclusivity)*: exactly one claims (which one —
better mass? higher pT? — maps the tie-break); lep/ν stay "none" regardless.
*Readout*: both claim ⇒ exclusivity is local per-jet thresholding, not enforced — and
H-D gains; lep/ν flip *more* with two candidates ⇒ aggregation is sum-like, not OR-like.
**Cost S–M.**

**A9. 2D decision surface: jet-W quality × lepton-W quality.** Grid scan (W-ljet mass
shift) × (lep+ν pT scale) on the same events; plot P(lep=W) contours.
*Why*: H-A says the verdict compares two candidate hypotheses → diagonal trade-off
contours. Jets-only decision ⇒ vertical contours; lepton-only ⇒ horizontal (H-C).
This is the cleanest *behavioral* signature of "compare both sides". **Cost M.**

**A10. m_T(lep, MET) test.** Construct corruptions that change lep–MET transverse mass
at fixed magnitudes (rotate MET φ relative to lepton) and vice versa.
*Predict (from round 2's angle-insensitivity)*: m_T does NOT matter, only magnitudes —
i.e. the model's "leptonic W quality" is cruder than a physicist's. If m_T *does*
matter, round 2's φ-randomization missed it (it randomized lepton φ, moving m_T — so a
null here would actually need reconciling with the 2.8% flip rate). **Cost S.**

**A11. Edge-of-distribution honesty check.** For every synthetic intervention above,
nearest-neighbour distance to real events in (m, pT) space of the modified jet; flag
results that live off-manifold (OOD artifacts masquerading as mechanism — cf. the
thesis model's XOR breakdown under mask-ALL-type3 in round 2c). **Cost S, methodological.**

---

## B. Output-level observational laws (no intervention, very cheap)

**B1. Claim bookkeeping across the full val set.** Distribution of #W-claimants per
event (jets + lep/ν as one unit), split by channel/stratum/correctness.
*Predict (H-A)*: sharply peaked at exactly 1, errors are claim *transfers*.
*Readout*: violations concentrated in resolved stratum (noisy truth) is fine; broad
violation spectrum ⇒ exclusivity is statistical, not mechanistic (H-D). **Cost S.**

**B2. Shared-score test (margins).** Event-by-event scatter: lep/ν W-margin vs the
best jet's W-margin. *Predict (H-A/H-B both)*: strong anti-correlation, even within
fixed true channel (i.e. driven by the model's belief, not the truth). A single shared
scalar read with opposite signs would give a tight line; two coupled-but-separate
scores give a blob with negative trend. Feeds E1 geometry. **Cost S.**

**B3. Error autopsy.** Characterize the ~5% of qqbb where lep wrongly = W: is the
truth-W jet's mass off-window? soft? Is there a claim anywhere?
*Predict (H-A)*: errors = W-detector misses (no claimant), with off-window W-jet
kinematics. *Bonus*: this converts the mechanism into a concrete performance-improvement
story (better W-finder ⇒ better reco), relevant for thesis corrections. **Cost S.**

---

## C. Attention-pattern analyses (observational, head-level)

**C1. Within-event contrast: lep/ν attention to W-ljet vs H-ljet (boosted).** The two
ljets in the same event are a perfect matched pair. Per block/head, both models.
*Predict (H-A)*: ≥1 late head where lep/ν attend W-ljet ≫ H-ljet; in lvbb the same
head's attention is diffuse/low. *Readout*: no differentiating head anywhere + verdict
still flips under D1d edge knockouts ⇒ value-content carries the signal with uniform
attention, or it routes via intermediate tokens (look at ν→lep, all→all). **Cost S.**

**C2. Attention tracks the claim, not the kinematics.** Regress (lep→jet attention) on
{jet's own W-claim margin} vs {raw jet mass/pT}: partial correlations.
*Why*: H-A says readers read the *computed candidacy*; H-B says everyone recomputes
from raw features. If attention (or message content, C4) correlates with claim margin
*beyond* what kinematics explain — e.g. in events where the claim is wrong —
that's parent→child flow. **Use model-error events as the discriminator set.** **Cost M.**

**C3. Redo round-1's channel-conditional attention per-stratum with corrected types**
(b1h3/b2h0/b2h3 story), + the reverse edges. Round-1 numbers are fine but were
interpreted with swapped jet names anywhere jets entered. **Cost S.**

**C4. Bottleneck-scalar census (thesis model).** For each head/block: AUC of the scalar
message (jet→ν, jet→lep, lep→ν) for (i) true channel, (ii) the sender-jet's claim.
Round 1 found b2h3→ν AUC 0.83 for channel; map them all, per stratum.
*Predict (H-A)*: messages FROM the claiming jet carry its candidacy; messages into ν
mirror lep's. **Cost S–M.**

---

## D. Activation-level causal interventions (the core)

**D1. Residual-stream patching suite.** Pairs: (clean event, surgically-W-masked event)
— the minimal corrupted pair we validated behaviorally — plus cross-event
channel-opposite pairs. Both directions (noising clean→corrupt and denoising
corrupt→clean). Sweep patch layer ℓ ∈ {embed, post-b0, post-b1(, post-b2)}.
- **D1a. Freeze lep/ν streams** (patch lep/ν tokens from clean run into corrupted run,
  at ℓ): find the layer after which lep/ν streams *already contain* the verdict.
  *Predict (H-A)*: verdict enters lep/ν streams late (after ≥1 block of jet
  processing); patching post-b1 (organism) blocks the flip; patching at embed does
  nothing. *H-B predicts*: same timing as jets (simultaneous convergence) — so compare
  against D1c timing; *the relative timing jets-vs-leptons is the discriminator*.
- **D1b. Verdict transplant**: patch lep/ν streams from a channel-opposite donor event;
  map the ℓ where the donor verdict sticks.
- **D1c. Patch the W-jet token only**: overwrite the truth-W-ljet's stream at ℓ with a
  non-candidate ljet's stream (H-ljet same event / lvbb donor). *Predict (H-A)*: by
  some ℓ* the jet's stream alone carries its candidacy, and patching it away flips
  lep/ν ≈ as much as input-level surgical masking; ℓ*(jet) < ℓ*(lep/ν) from D1a.
  *Readout*: if patching the W-jet's stream late does NOT flip lep/ν (but input masking
  does) ⇒ lep/ν read the jet EARLY or via other tokens ⇒ localize with ℓ sweep.
- **D1d. Path/edge patching**: restrict to specific edges — patch only K/V coming from
  the W-jet into (lep/ν) queries at block b, head h. Identify reader heads; then knock
  out *only* those edges and measure how much of the surgical-mask flip is reproduced.
  *Predict (H-A)*: a small set of late heads carries most of it in the organism;
  thesis model likely redundant across heads (round 1). Also include **lep↔ν edges**:
  knocking ν→lep edges everywhere is the sharpest test of *how* the lockstep works
  (does ν read lep, or do both read the same jet aggregate?). *Predict*: both read the
  aggregate (coupling survives ν→lep knockout via redundancy) — if instead coupling
  breaks, ν is a lep-follower (changes the story).
**Cost M (a,b,c) / M–L (d). The single most informative family — directly tests
direction-of-flow, the H-A/H-B/H-C discriminator.**

**D2. Logit-lens timing.** Decode every token's class logits from the residual stream
after each block with the final head (organism: no LN ⇒ exact).
*Predict (H-A)*: W-jet's claim crystallizes a block before the lep/ν verdict; in lvbb,
lep/ν margin rises only after jets have (negatively) resolved. *H-B predicts*
simultaneous rise everywhere; *H-C predicts* lep/ν first. Dirt cheap, run before D1 to
guide which ℓ to patch. **Cost S.**

**D3. Per-head ablation, redone right.** On the organism (8 heads): single + all pairs,
zero- vs mean-ablation, per-stratum, with **claim bookkeeping** as the readout (not just
accuracy): which ablation *creates XOR violations* (both-W or no-W events)?
*Predict (H-A)*: the aggregation/readout step has a locus — some ablation decouples
jets' claims from lep/ν verdicts rather than degrading everything uniformly.
*Readout*: pure uniform degradation again (as round 1) ⇒ aggregation is distributed ⇒
strengthens H-B for the thesis model; the organism may differ — that contrast is itself
informative about whether constraints/scale create the redundancy. **Cost S–M.**

**D4. Message overwrite ≫ message removal (thesis model).** Round 1 zero-ablated the
b2 heads' contributions and nothing flipped (redundancy). Stronger: *overwrite* all
late-block jet→ν (and jet→lep) bottleneck scalars with values from a channel-opposite
donor. *Predict*: redundant carriers all vote the same way, so flipping ALL votes flips
the verdict even though removing some didn't. *Readout*: still no flip ⇒ the verdict
truly doesn't transit these messages at all ⇒ it's already in the lep/ν streams earlier
(→ D1a timing). **Cost M.**

**D5. Zero vs mean vs resample ablation triplet.** For every ablation in D1/D3/D4 run
the three variants. Zero-ablation alone overstates effects (off-manifold); resample
(donor) ablation is the fair test. Methodological hygiene, folded into the others.

**D6. Targeted linear probes.** (i) jet tokens: "am I the truth-W?" per block;
(ii) lep/ν tokens: channel, per block; (iii) every token type: channel, per block
(maps the broadcast). *Predict (H-A)*: jet-candidacy decodable at block 0–1 (AUC≫0.9)
before lep/ν channel becomes decodable; eventually channel readable everywhere
(round-1 cos-convergence 0.65→0.88 anticipates this).
*Why probes failed before (thesis "no insights")*: they were generic class probes;
these are *named, hypothesis-driven* targets. Probe directions feed D7/E1. **Cost M.**

**D7. Steering with the probe direction.** Add ±α·(channel direction) into chosen
tokens' streams at chosen blocks. *Predict (H-A)*: steering the *jet aggregate /
broadcast layer* flips jets' claims AND lep/ν together (shared upstream); steering
lep/ν alone late flips them while jets keep claiming — i.e. we can *manufacture XOR
violations*, proving the exclusion lives upstream of the readout, not in the output
layer (kills H-D). Dose-response curves = how linear the verdict variable is. **Cost M.**

**D8. Causal-scrubbing-lite (capstone).** Write the full hypothesized graph
(jet kinematics → per-jet candidacy [b0–b1] → aggregate/broadcast → lep/ν readout
[last block]; leptonic-hardness as parallel input into the aggregate) and
resample-ablate everything off-path on the organism. *Predict*: ≥90% of lep/ν
assignment behavior retained. This is the "we found THE circuit" certificate; only
worth running once D1/D2/D6 have pinned the graph. **Cost L.**

---

## E. Representation geometry & feature attribution

**E1. Shared-direction geometry.** Cosine similarity between the channel-probe
directions at different token types, per block. *Predict (H-A)*: directions align
(or anti-align) after the broadcast block — one "channel" variable, signed reads.
Independent of (and cross-validating) B2. **Cost S** (given D6).

**E2. What feeds candidacy (DLA/occlusion on the W-jet).** Per-input-feature
attribution of the jet's own claim margin (occlude mass/pT/tag; direct-logit-attribution
through the object net). Bridges to A1/A2 dose-responses; cross-check the two.
*Predict*: mass dominates for ljets; for resolved sjets — open question (pair features
can't live in a single token's object-net; they must come from attention → if E2 shows
single-sjet features explain claims, the model never binds the pair → explains resolved
weakness). **Cost M.**

**E3. Universality check.** Re-run the headline tests (A1, B1–B3, D2) on: new organism,
old organism, thesis model, +2 fresh organism seeds (cheap to train now, ~2.5 h each on
CPU). *Predict*: mechanism universal (timing, jets-first), specific heads vary by seed.
A failure of universality on seeds would be a major caveat for the whole program.
**Cost M (mostly training time).**

**E4. PySR round 2, targeted.** Symbolic regression of the jet-candidacy logit (from
D6 probe or logit lens) against jet-level physics features (m, pT, |η|, tag, ΔR to
lepton/ljet, m_jj for sjet pairs). *Predict*: ljet candidacy ≈ window(m_J) with mild
pT dependence. This is the step that turns the circuit into *physics*; it's also where
the faithfulness/replacement test (plan §3 punchline) plugs in: substitute the fitted
expression for the candidacy computation and measure retention. **Cost M–L (needs pysr
installed — not in the venv yet).**

---

## F. Training-dynamics tests (free lunch: tonight's organism saves every epoch)

**F1. When does the mechanism assemble?** Run B1/B2/D2 (+A1 coarse) on checkpoints
0…29 of the new organism. *Predict (H-A)*: per-jet candidacy accuracy develops first;
the lep/ν↔jet anti-coupling and XOR sharpen after/with it; lep/ν verdict accuracy
*tracks* jet-candidacy accuracy epoch-by-epoch (it's derived). *Readout*: lep/ν
coupling appears *before* jet claims are any good ⇒ the coupling is a statistical prior
(class frequencies), not a read of the jets — favors H-B/H-D for early training, and
shows when (if ever) the mechanism switches. Nobody has this picture for a HEP
transformer; it's cheap and novel. **Cost S–M.**

---

## Suggested running order (for discussion)

| Wave | Experiments | Rationale |
|------|------------|-----------|
| 1 (tonight-scale, all S) | D2 logit-lens, B1, B2, B3, A1, A6 | Timing + dose-response + laws; D2/A1 alone already split H-A/H-B/H-C substantially |
| 2 | A2, A4, A8, C1, C4, A10 | Sufficiency + readout heads + tie-breaks |
| 3 | D1a–c, D6, D3 | The patching core, guided by wave-1 timing |
| 4 | D1d, D4, D7, A9, E1, E2, F1 | Edge-level localization, steering, geometry, dynamics |
| 5 | A3, E4, D8, E3 | Resolved-pair story, symbolic extraction, scrubbing certificate, universality |

**Decision points preregistered:** if D2/D1 show *simultaneous* verdict formation
across tokens → pivot the framing from "ask the jets" to "shared event summary" (H-B)
and target the summary's construction instead of jet→lep edges. If A1 shows no mass
window → chase the candidacy feature (A5/E2/E4) before any patching, since "claim"
needs an input-level handle for clean interventions.

---

## Wave 3 — REVISED after waves 1-2 (2026-06-11, thesis model only)

The readout stage (c) is solved (D4-lite). Remaining holes, in priority order:

1. **A5 tag surgery** (S) — the A2 refutation's prime suspect. First establish what
   `tag` encodes per type (preprocessing INCLUDE_TAG_INFO semantics). Then: lvbb
   H-ljet {m→80 only, tag→0 only, both}; boosted W-ljet {tag→1}.
   *Predict*: m→80+tag→0 ≫ m→80 alone for H-claims-W; W-ljet with tag→1 loses
   candidacy (or migrates to claiming H). If tag does nothing → the gate is
   relational context (ΔR to lepton? recoil?) → E2.
2. **D4-lite at the LEPTON position** (S) — overwrite the lep's b1h3/b2h2 scalars;
   *predict*: lep flips like ν did (≥99% with both wires swapped) while ν holds →
   completes the "parallel readers of shared evidence" claim causally, both directions.
3. **D6 probes + D1c patching: where candidacy lives in the JET's stream** (M) —
   probe "am I the truth-W?" at jet tokens per depth; patch the W-ljet's stream at
   depth ℓ with an H-ljet/donor stream. *Predict*: candidacy linearly decodable and
   committed by end of block 1 (claims crystallize blocks 0-1 per D2); patching it
   away flips ν via the wires. Localizes the stage (a)→(b) boundary.
4. **E2 feature attribution on candidacy** (S-M) — per-feature occlusion/DLA of the
   W-ljet's own claim margin; quantify the pT vs mass vs tag vs direction mix that
   waves 1-2 probed one-at-a-time.
5. **A3-lite resolved-stratum battery** (M) — rerun C1 + decomposition + D4-lite on
   resolved events: do the wires read the sjet PAIR? *Predict*: messier attention,
   weaker single-sender dominance — the mechanistic source of the resolved
   performance gap (ties to thesis Table RecoTransformerModifications cats 0-3).
6. *(parked while thesis-focused: F1 training dynamics, E3 universality/new-organism
   replication, A9 2D scan, A10 mT, D8 scrubbing-as-capstone.)*

## Stage B — the competition sub-circuit (preregistered 2026-06-11, not yet run)

A8 found: two real W-ljets ⇒ exactly one claims 86.5%, higher-pT wins 87%, and the
lepton goes MORE firmly to "none". How is winner-take-all implemented? Note the
"loser" effect must live in the jets' OWN streams (claims are per-token classifier
outputs), so competition ≠ reader-softmax artifact.

Mechanism hypotheses:
- **B-i lateral inhibition via attention**: jets attend each other in blocks 0-1;
  the higher-pT jet's presence suppresses the other's candidacy in its stream.
- **B-ii relative-context scoring**: each jet's candidacy score is computed against
  an event-level context (e.g. "am I the hardest non-H jet?") gathered by attention —
  no directed inhibition edge, but the same prediction at the output.
- **B-iii independent thresholding** (null): no interaction; apparent exclusivity is
  coincidental thresholds. (Disfavored already: P(both)=8% ≪ independent expectation.)

Experiments:
- **SB1 ΔpT dose-response** (S): insert the second W-ljet with controlled pT ratio r =
  pT_ins/pT_orig ∈ [0.5, 2]. *Predict (B-i/B-ii)*: P(both claim) peaks sharply at
  r≈1 (competition is comparative); P(ins wins) is a smooth sigmoid in r crossing 0.5
  at r≈1. *B-iii predicts* flat P(both).
- **SB2 lens timing on two-W events** (S): per-depth claim margins of winner vs loser.
  *Predict (B-i/B-ii)*: margins start TOGETHER (both look like W at embed/blk0) and
  diverge at the block where competition acts; *B-iii*: never together.
- **SB3 jet↔jet attention** (S): in two-W events, a(loser→winner) vs a(winner→loser)
  vs single-W baselines, per head. *Predict (B-i)*: asymmetric edge (loser attends
  winner); the suppression head is identifiable. *B-ii*: symmetric/diffuse.
- **SB4 edge knockout** (M): kill the jet↔jet attention edges between the two W-jets
  (per head / per block). *Predict (B-i)*: both claim — exclusivity broken between
  jets while lep/ν still read "W found". *B-ii*: knockout does little (context comes
  from everywhere).
- **SB5 stream patching of the loser** (M): patch the loser's post-blk0/blk1 stream
  from its single-W twin run (same event, inserted jet removed). Localizes WHEN the
  suppression lands in the loser's stream.
- **SB6 does the WINNER know it won?** (S): compare winner's claim margin in two-W vs
  single-W events. *Predict (B-i/B-ii)*: slightly reduced but robust (suppression is
  one-sided toward the loser); large reduction would suggest mutual inhibition with
  a threshold readout.

Bookkeeping: same battery on the H-side (two H-ljets, A8-style) would tell us whether
competition machinery is per-class or shared — park until W-side is mapped.

# 2026-06-21 — Adversarial audit of the interp claims for thesis Ch.7

Triggered by picking the interpretability work back up to finalise thesis Ch.7
(`PhDThesis/Chapters/07_EventSelectionLowLevel.tex`, the
`sec:...:Interpretability` section + appendix `ZD`). Before writing anything into
the thesis, we ran a **maximally critical** pass: three independent **Opus
(max-thinking)** reviewers, each told NOT to trust the repo's own confidence
labels/markdown, to read the actual scripts, re-derive the logic, **run code to
reproduce ≥1 headline number, and design+run NEW controls to try to break each
claim**. This log records what survived, what didn't, and the corrections to our
own docs that the audit forced.

> **One-line takeaway:** the *causal core* of the H1 story is real and reproduces
> exactly; but several "circuit" *framings* are partly architecture+linearity
> restated rather than discovered mechanism, the Phase-2 "+2.8-pt recipe lift" is
> a **weighting-comparison bug**, and the E4 "95.5% faithful" headline does not
> survive (the b2h2 half is a signal-only correlate blind to the mechanism). The
> thesis-relevant conclusions all still hold in a **narrower, honest** form.

## Method / provenance of this audit
- 3 background Opus agents, ~10 min each, on the local assets (thesis ckpt
  `20250512-093728`, signal memmaps, the 24-model Phase-2 suite, suite logs,
  archived wandb JSONs). All three reproduced the repo's headline numbers and then
  stress-tested them.
- Reviewer scratch scripts (session scratchpad, not committed):
  `audit_repro.py`, `audit_controls.py`, `audit_closure_and_b1h3.py`,
  `audit_control_adequacy.py`, `e4_attack{,2,3}.py`. Numbers below are theirs,
  reproduced from our own scripts.

---

## 1. MODEL PROVENANCE — RESOLVED (good news): the thesis "n_blocks=4" is a typo for 3

The thesis attributes its interp table/figures to a model with **n_blocks=4**,
d_m=152, n_heads=4, d_mlp=400, bottleneck-1. The repo dissects
`thesis-ent1-bn1-d152` = **3 blocks**. This had to be run down — if the repo were
interpreting a *different* model than the published one, every mechanistic finding
would fail to describe the thesis.

**Verdict: typo. The repo dissects the correct, published model.** Evidence:
- The checkpoint's `state_dict` has `attention_blocks.{0,1,2}.*` only → **3 blocks**;
  loads `strict=True`. Registry spec matches exactly.
- Run on val, the 3-block model reproduces the thesis `RecoTransformerModifications`
  **"Both" row to ~3–4 dp**: measured `[0.3097, 0.3709, 0.2917, 0.4959, 0.9587,
  0.9271]` vs thesis `[0.3101, 0.3709, 0.2917, 0.4957, 0.9587, 0.9272]`.
- `RunLowLevelInterp.py:319-340` (commented *"THIS IS THE ONE I USED FOR MAKING THE
  PLOTS IN MY THESIS (20250709)"*) builds this checkpoint with
  `num_attention_blocks=3`. Run name `…BN3…` = 3 blocks.
- No 4-block d152+ent+bn checkpoint was ever trained (exhaustive search of
  `output/`, `tmp_checkpoints/`, `docs/run_configs/`).

**→ Thesis fix: correct `n_blocks=4 → 3` in the section + the `ZD` appendix
captions.** Harmless typo, but must be fixed so an examiner doesn't infer the
interp describes a non-existent model.

### Two provenance sub-findings that correct our docs
- **The `[0.260,…]` vs `[0.310,…]` "discrepancy" is just two weightings of the SAME
  run.** Thesis table = wandb **`val_MC`** (raw MC / cross-section weights). Suite &
  the `[0.260,0.327,0.244,0.435,0.947,0.907]` numbers = wandb **`val`**
  (per-mass-equalised "proportional" `training_Wts`). Both are computed from one
  checkpoint each epoch (`TrainLowLevelReconstruction.py:489-490`). **This weighting
  distinction is the root of the Phase-2 bug in §2.**
- **The whole thesis table is `val_MC`**, and a **bottleneck-only legacy run DOES
  exist** (contradicting the suite log's "no legacy bottleneck-only run exists"):
  None=`20250510-123629`, Ent-only=`20250512-093602`, **Bn-only=`20250707-132308`**,
  Both=`20250512-093728` — each matching its table row in `val_MC`.
- `eval_thesis_numbers.py` actually reproduces `val_MC` to ~3e-4, **but its
  `WANDB_VAL_TARGETS` are the `val` numbers** → it prints spurious "+0.01–0.04"
  diffs and looks like a near-miss. Mislabelled targets; fix to `val_MC`.

---

## 2. INTERPRETABILITY-COST TRADEOFF — the "+2.8-pt recipe lift" is a weighting-comparison BUG

This is the result most likely to enter the thesis, and the part of our own docs
that was most wrong.

**The bug.** Suite logs are **MC-weighted** (`train_organism.py:236` passes
`MC_Wts`), so suite `PerfectRecoPct_all` is comparable to wandb **`val_MC`** (thesis
"both" = **0.8725**), NOT to `val` (0.8434). The Phase-2 doc + `suite_tradeoff_plots.py`
repeatedly compared **MC-weighted new runs against `val`-weighted thesis-era
numbers**. The w→MC gap for these models is *itself* ~2.7–2.9 pts — which we
misread as a training-recipe effect.

| Sub-claim (repo) | Under matched (MC) weighting | Verdict |
|---|---|---|
| "+2.8–2.9 pt recipe lift" | recipe Δ = none **+0.12**, ent **−0.13**, both **+0.63** pts → real recipe effect ≈ **0 to +0.6** | **ARTIFACT — drop** |
| "`d152_both_s0_LEGACY`=0.850 reproduces thesis 0.843 (+0.7)" | legacy-both 0.850 (MC) vs thesis **val_MC 0.8725** = **−2.2 pts**; legacy plateaus ~0.842 then jumps only at the final LR→0 step | **does NOT reproduce; drop as an anchor** |
| "marginal interp cost (both−none) ≈1.2 pts, recipe-stable" | winner (MC) **−1.23** (SE 0.17, 95% CI ±0.33); thesis-era (MC val_MC) **−1.74** | **HOLDS (soften to ~1.2–1.7 pts; don't lean on the legacy run)** |
| "cost concentrated in cats 0–3, cats 4–5 untouched, ~20% rel" | winner both−none per-cat: c0 −9.7, c1 −5.6, c2 −9.0, c3 −7.6, **c4 −0.05, c5 −0.7**; relative cat0 −21%, cat2 −22%, cat3 −13%; **seed-stable** | **HOLDS (robust)** |
| "30 epochs ≈ converged" | constrained runs still rising *faster* than `none` at ep30 (both +5.7e-3/ep vs none +2.8e-3) | cost is an **UPPER BOUND** |

**What's thesis-safe (and what the thesis actually needs):** the broken thesis
claim is *"[simplifications] without degrading performance too much."* The fix is
the **constraint cost** (both vs none, one recipe/weighting), which survives:
> the bottleneck+entropy simplifications cost **≈1.2 pts** overall (winner recipe,
> 3 seeds, CI ±0.3) / **≈1.7 pts** (thesis-era), **entirely in the rare categories
> 0–3** (~13–22% relative); cats 4–5 (~86% of events) are unaffected; an upper
> bound at 30 epochs.

Crucially, **the thesis doesn't need the suite at all for commit 1** — the existing
`RecoTransformerModifications` table (None→Both, val_MC) already shows the same
thing: cats 4–5 change ≤1 pt, cats 0–3 drop 16–29%. The suite's value is the
3-seed error bars (a commit-2 nice-to-have), NOT the "recipe lift" narrative.

**Defects to fix in repo code (for when we draw real figures):**
- `suite_tradeoff_plots.py` plots MC-weighted (winner curve, legacy ◆) and
  `val`-weighted (thesis-era ■, thesis ★) points on **the same axis** → the apparent
  vertical "recipe lift" is mostly the weighting jump. **Do not use as drawn.**
- `eval_thesis_numbers.py` mislabelled targets (see §1).
- 1 seed for the legacy run is inadequate; combined with the weighting bug + the
  end-of-schedule jump, it's unusable as a thesis anchor.

---

## 3. CAUSAL CIRCUIT CLAIMS (C1, C2, C4) — core reproduces exactly; soften the framings

All headline numbers reproduced within noise on the narrative slice, a fresh slice
(batches 25-30), and by re-running the red-team scripts.

### C2 — "weigh the jets" (the channel verdict = best hadronic-W claim vs leptonic-W evidence) — **STRONGEST, thesis-grade**
- mask true W-jet → lepton claims W **0.772**; insert real W-jet → P(ins=W)=**0.318**,
  P(lep=W|ins claims)=**0.057** vs **0.965** otherwise (tight event-by-event coupling);
  cross-system dose sigmoid crossing **r≈1.4**; LW1 ρ(pT(lepW)/HT)=**−0.795**,
  ρ(mT)=**−0.004**.
- **Free-slot confound killed:** only **3/8240** lvbb events lack a free slot.
- **Donor-composition account is predictive, not post-hoc:** flip rate by donor
  relative-pT quartile = .001/.052/.387/.832.
- Object-level surgery (mask/insert) is much closer to manifold than scalar
  overwrites. **Verdict: YES.** Only trim: "evidence" = vector-sum *hardness* (mT≈0,
  φ+180° weakens it), not a W-mass-like quantity.

### C1 — "ν–lep lockstep is shared wiring, not ν reading the lepton" — **soften to "shared readout"**
- Reproduced: P(ν=lep)=**0.9954**; overwrite both wires @ν → ν flips **0.998**, lep
  holds; @both → lockstep restored; own-class placebo **0**.
- **New control (on-manifold):** overwrite ν's wires with a *real opposite-class
  event's* (b1h3,b2h2) → still flips **0.907** (vs 0.999 for the class-mean). Effect
  is real, off-manifold-ness inflates it ~9 pts.
- **Control that FAILS: "norm-matched random kick 2.1%" is the wrong null.** A random
  152-d direction is ~orthogonal to the 1-d readout. The fair null — equal-norm kick
  *along the W-vs-none readout direction*, orthogonalised against the wires — flips
  **0.699** (0.998 at 2× norm). So *any* readout-ward push of equal norm flips ν; the
  wires are the **most efficient lever**, not a unique one.
- **Verdict: SOFTEN.** "ν and lep share a near-linear readout over correlated
  residuals (not ν←lep); the two readout-aligned scalars are the most efficient
  causal lever (on-manifold flip ≈0.91)." Drop the random-kick number as evidence of
  specificity.

### C4 — "two wires carry the verdict (~80-90%); +12% secondary; 12-scalar set CLOSED" — **soften**
- Per-key decomposition is exact & clean: **b2h2 is jets-only** (lep/ν keys
  +0.000/+0.001; ljets +1.192, sjets +0.338); **b1h3 = signed lep+ν(−) vs jets(+)
  comparator**. RT2 flow split 0.816/0.907; +b2h0+b2h3 → 0.021 (the ~12% path). These
  are real.
- **"Closed 12-scalar set" is architecturally FORCED, not a finding.** With
  `bottleneck_attention=1`, each head's entire residual write is rank-1
  (scalar × fixed `w_up`); the only attention-path input to a token is its 12
  scalars. Proof: clamp the 12 → ν invariant (≤4e-4) under ×5 scaling / total jet
  masking. So "more wires hide elsewhere" was never an option.
- **AUC-vs-prediction (0.996) is near-circular** (a causally-upstream scalar must
  correlate with its own downstream output). Report **AUC-vs-truth** (b2h2 **0.987**,
  b1h3 **0.980**) instead — the non-circular version, also high.
- **Single-wire KO is asymmetric:** b2h2 alone flips **0.727**, b1h3 alone **0.106**,
  both **0.998** (synergy). b2h2 is dominant; b1h3 co-required mainly in lvbb.
- Causal power ≠ AUC (b2h3 has the largest attention but flips 0.014).
- **Verdict: SOFTEN.** Keep wire identities/roles + the RT2 80-90%/12% split; reword
  closure as architectural; report AUC-vs-truth; note b2h2 ≫ b1h3.

---

## 4. SOFTER CLAIMS (C3, C5, C6) and E4 FAITHFULNESS

### C3 — candidacy = (predominantly) relative hardness × m²-window × anti-tag — **YES**
- RT4 (causal, no edge surgery): rest×2 (W untouched) **0.924→0.573**; whole×0.5
  **0.749**; W×0.5 0.486; rest×0.5 0.970. Four arms jointly refute pure-absolute. Tag
  surgery flips W↔H. DSID-stratified replication (27-37 pp).
- Caveat to keep: whole×0.5 is not pure survival (0.924→0.749) → **"predominantly**
  relative", not "purely" (residual abs-scale/OOD sensitivity). Demote the
  "AUC 0.93 decodable at embedding" line (decodable ≠ used).

### C5 — competition = per-jet context scoring; b2h2 is the comparator — **positive YES; negative SOFTEN**
- SB4: tie both-claim 0.745; KO b2h2 restores loser **0.935** (heads 0/1/3:
  0.006/0.004/0.043); direction-resolved o→i 0.139 vs i→o 0.791 (self-suppression);
  no-renorm 0.844; surgical hook validated to 0.0 dev. **b2h2 = the comparator AND
  the broadcaster** is a real single-head causal result (not over-fitting).
- **Negative claim "no inhibition edge" must soften:** established only in the
  post-softmax *attention-edge* framing. Our own LW1 shows suppression living
  multi-hop: KO lep→candidate makes the candidate concede **0.905→0.151** without
  cutting the candidate's own edges. So inhibition exists; it's just not a single
  directed edge. Report the **84–93% renorm band**.

### C6 — cross-scale divergence — **YES as existence/caution; "across scale" over-claims**
- A1 mass-dose: **thesis non-monotonic** (m_W bump), **both organisms (2025 + 2026
  independent retrain) monotonic** → divergence replicates across 2 organisms. RT3:
  thesis mass-penalty vs organism pT-first. RT4 (relative hardness) replicates on the
  organism → *contextual scaffold shared, feature parameterisation diverges*.
- **Confound:** thesis differs from organism in scale AND depth AND bottleneck AND
  d_model → "doesn't transfer across **scale**" over-attributes. Honest:
  "**these two capacities** parameterise the same mechanism differently; universality
  is a per-finding empirical question." n=2 is not a scale law.

### E4 — "two formulas, 95.5% faithful" — **b1h3 half thesis-grade; b2h2 half FAILS; do not headline**
- Reproduced: replace-both ν==intact **0.9553**, ν==truth **0.9309**, mean-floor
  **0.6951** (≈ lvbb base rate → uninformative).
- **The decisive test the repo never ran (reviewer ran it): b2h2's formula is blind
  to the mechanism C3 proves is core.** Sweep the W-jet mass: real b2h2 shows the m_W
  bump (−0.31→−1.66→−0.72 @ 40/80/150 GeV), real claim 0.78→0.95→0.82; **the formula
  is flat at −0.766.** Sweep tag W-like→H-like: real claim 0.94→0.58; **formula
  flat.** The formula's R² is dominated by a **leptonic** feature (pT(lepW) alone
  R²=0.648) though the wire is **jet-fed** (all jet features R²=0.435). It survives
  "95.5%" only because true W-jets are already W-like in-distribution.
- Nuance (so we don't over-correct): lep×2 *does* move the real b2h2 +0.68 via
  multi-hop (jets absorb leptonic context, b2h2 reads jets) → "pT(lepW) is
  non-causal" is too strong; it's a proxy for a real multi-hop pathway. Either way it
  is **not** the wire's input law.
- "95.5%" is agreement-with-intact; truth cost is **2.4 pts** (0.955→0.931).
  Readout-only (jets stay neural) → "semi-symbolic reconstruction algorithm" is
  overstated.
- **Verdict: b1h3 reduction (single tanh of the leptonic/hadronic hardness ratio,
  R²=0.835, wire genuinely lep-fed: lepton −0.73, ν −0.58) is thesis-grade. The b2h2
  "faithfulness" headline is NOT — present b2h2 as behavioral compression that
  exploits a signal-only correlation and is provably blind to the mass/tag axes.**

---

## 5. The unifying theme (and what to foreground in the thesis)
**Several "circuit" claims are partly architecture + linearity restated, not
discovered mechanism:** the closed 12-scalar set (forced by the rank-1 bottleneck);
C1 wire-specificity (any equal-norm readout-ward push flips ν ~70%+); the AUC census
(correlates, not causes); E4-b2h2 (a correlate blind to the real inputs). A
transformer-literate examiner would read these as architecture, not mechanism.

**But the causal core is genuinely strong and reproducible:** bidirectional
overwrite (C1), object-level insert/mask with tight event-by-event coupling and a
predictive donor account (C2), the exact jets-only/comparator decompositions and the
RT2 flow split (C4), RT4 relative-hardness (C3), and the single-head comparator KO
(C5). **Foreground the interventions; demote the closure, the random-kick null, the
AUC-vs-prediction census, and the E4-b2h2 formula.**

---

## 6. Implications for thesis Ch.7 (the plan, agreed with Sid 2026-06-21)
Two-commit approach (so scope vs the post-viva corrections can be decided later):
- **Commit 1 — cleanup + restructure, NO new science.** Restructure into
  (i) Motivation & methodology [+ add activation patching; group methods
  correlational vs interventional], (ii) Network simplification [+ the corrected
  cost claim from the *existing* val_MC table — dominant cats unchanged, cats 0–3
  cost 16–29%], (iii) Results & outlook [attention win; ablation→appendix summary;
  nulls reframed non-apologetically with the "relational computation" reason;
  interp-specific outlook]. Plus: fix n_blocks 4→3; remove the two `\note{}` TODOs;
  soften the novelty footnote; rename `tmp`-named attention figures.
- **Commit 2 — one *example* investigation, framed as "an example of what you can do
  on an example model" (NOT strong new results).** The **neutrino puzzle**: ν tracks
  lep (0.995) → activation patching refutes "ν reads lep" → mask/insert shows the
  verdict follows the jets → puzzle dissolves (this *answers Ch.7's own robustness
  question*, L403-405). Tight in main text (≈3 paras + 1 clean regenerated figure),
  detail in a new appendix. **Limitations stated plainly.** Excluded: E4-b2h2
  "faithfulness", the closed-set claim, the "no inhibition edge" negative, the recipe
  lift.

## 7. Corrections this audit forces on OUR docs (do as docs-only)
- `HANDOFF_2026-06-12.md` top banner ("recipe lift ~+2.9 pts", "Phase-2 CLOSED")
  and `2026-06-15_suite-d152-results-and-entropy-mismatch.md` (the +2.8 recipe-lift
  conclusion, "no legacy bottleneck-only run exists") are **superseded by this log**
  (weighting bug). Added pointer banners.
- `experiments/h1_narrative.py` scoreboard overstates: Claim 4 "closure" (it's
  architectural) and Claim 1 controls (random-kick is the wrong null). Re-word when
  next touched.
- Repo-code TODOs: redraw `suite_tradeoff_plots.py` in a single weighting; fix
  `eval_thesis_numbers.py` targets to `val_MC`.

## Learnings
- **Don't trust run-name labels OR aggregate metrics across weightings.** The
  Phase-2 "recipe lift" was MC-weighted-new vs proportional-weighted-old. **Always
  state which weighting** (`val` = per-mass-equalised `training_Wts`; `val_MC` = raw
  cross-section `MC_Wts`); the thesis tables are `val_MC`. → candidate for
  CLAUDE.md / experiments/README conventions.
- **Negative mechanistic claims need on-manifold controls AND the right null.** "No
  inhibition edge" / "wires are specific" both weakened under controls the original
  pass didn't run (multi-hop suppression; readout-direction null).
- **A rank-1 bottleneck makes "closed set" claims trivial** — clamping all per-head
  scalars removes the entire attention path by construction.
- **High R² ≠ faithful** (E4-b2h2): always cross-check a fitted formula against the
  causal decomposition AND sweep the mechanism variables the formula can't see.
- **Feature-informed PySR ≠ prospective.** Treat the formulas as post-hoc until
  re-fit on a fresh slice under a nested protocol.
- The repo's own logs were, encouragingly, mostly well-hedged at the *log* level; the
  overstatements were in the *narrative/handoff headlines*. Trust the careful
  wording, not the headline.

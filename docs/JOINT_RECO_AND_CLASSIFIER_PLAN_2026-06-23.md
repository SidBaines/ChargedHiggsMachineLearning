# Joint reconstruction + signal/background classification — plan & progress log (2026-06-23)

> New workstream (Sid + Claude). **Question:** can a *single* transformer do both
> (a) object-level **reconstruction** (per-object `none/H/W`) and (b) event-level
> **signal-vs-background classification** — and how does the reco↔classification
> performance trade off with scale (width/depth) and dataset size?
>
> This doc is the **living plan AND the running progress log**: append to "Results (live)"
> and "Session log" as we go; edit the plan sections when we pivot. Sister docs:
> [`RECO_TRAINING_PLAN_2026-06-22.md`](RECO_TRAINING_PLAN_2026-06-22.md) (the parallel
> minimal/best reco-net workstream — shares the harness and baselines) and
> [`RESEARCH_PLAN_2026-06.md`](RESEARCH_PLAN_2026-06.md) (the umbrella plan).

## Goals (two, separable — same as the reco workstream)

1. **Interpretability:** does adding the classification objective *clarify* or *muddy* the
   reconstruction circuit (and vice versa)? Is there a single clean network that does both?
2. **Performance:** for each task, does sharing a backbone *help* (positive transfer /
   regularization) or *hurt* (capacity competition / task interference) vs the single-task
   baselines — and how does that change with scale and dataset size?

The headline deliverable is a **trade-off surface**: (reco PerfectRecoPct, classification
AUC/Z) as a function of (architecture size, training config, dataset size), against
single-task baselines.

## Decisions settled this session (2026-06-23)

- **Data sourcing:** plan around **copying pre-made background memmaps** from the Seagate
  (Sid to confirm they exist, which processes, how many events). Fall back to regenerating
  from ROOT with `preprocessing-scripts/preprocessLowLevel.py` only if no compatible
  memmaps exist. ⚠ **Background is not on the laptop** — see "Data requirements".
- **Compute:** **local MPS, dominant-background subset first** (keep it tractable; cap
  model size ~d152/overnight). RunPod (MCP available) parked for the full-background /
  large-model sweep if/when MPS becomes the bottleneck.
- **Metrics:** **ROC-AUC** (primary, cheap, already implemented) + **Asimov significance Z**
  (physics-facing). The **downstream expected limit** (`fig:TransformerVsOriginalExpectedLimits`,
  lxplus pipeline — see [`LIMITS_PIPELINE_AND_MODEL_PROVENANCE.md`](LIMITS_PIPELINE_AND_MODEL_PROVENANCE.md))
  is the **gold-standard** test but too heavy to run per-model — **recorded here as the
  finalist-only / future sanity check, deliberately deferred.**
- **Build order:** stand up **both** structural approaches (single-head + two-head) and
  compare **head-to-head on the same data/scale grid** from day one.

## Background: what the code already gives us (verified 2026-06-23)

- **One backbone, two modes.** `TestNetwork` (`models/models.py`) already supports both:
  `is_reconstruction_model=True` → per-object logits `[B, n_obj, 3]` = `none/H/W`;
  `is_reconstruction_model=False` → mean-pool non-pad objects → event logits `[B, 3]` =
  `{background, qqbb, lvbb}` (this is exactly what `TrainLowLevelClassifier.py` uses).
  → a **two-head** model = keep the per-object head + add the pooled head + return both
  (small change). A **single-head** model needs **no architecture change at all**.
- **Loader already handles signal+background.** `dataloaders/lowleveldataloader.py`:
  `signal_only` flag; event-level label at `batch_samples[:,0,0]` (`0=bkg, 1=qqbb, 2=lvbb`);
  per-object truth at `x[...,-1]` (`{0,1,2,3}→{none,H,W-had,W-lep}`, collapsed to `{0,1,2}`);
  signal/background **MC-weight balancing already built in** (signal scaled by
  `1/n_sig · Σw_bkg/Σ|w|`, background by `Σw/Σ|w|`). So background support ≈ "set
  `signal_only=False`, point at background memmaps, mask the reco loss for background."
- **Reco harness** `experiments/train_organism.py` is the base to extend (it has all the
  arch/optim/`--keep-cats` knobs and the winner recipe). Reco loss = per-object weighted
  CE (`metrics/lowlevelrecometrics.py:HEPLossWithEntropy`); classifier loss = event-level
  weighted CE (`metrics/lowlevelmetrics.py`).
- **Scientific hook:** H1 established the reco net's lep/ν *"W vs none"* verdict **is** the
  global channel decision (lvbb vs qqbb). So signal = "event has a real H + W", background
  = "neither" → in the single-head framing **P(signal) ≈ P(≥1 object is H or W)** falls
  straight out of the object scores; and the event head's `qqbb`/`lvbb` split is *already
  computed inside the reco head*. Joint training is therefore a direct probe of both goals.

## The two structural approaches

### S — Single-head ("background = all-none")
- **Architecture:** unchanged reco net (per-object `none/H/W`). Signal events keep their
  truth object labels; **background events: every object labelled `none`.**
- **Event score (readout, parameter-free):** `P(signal) = readout(object scores)`. Default
  primary readout = **soft-OR / max over objects of `P(object ∈ {H,W})`**; we may also try
  `P(≥1 H-candidate) · P(≥1 W-candidate)`. These are *fixed* functions → the single-head's
  classification number is an *honest lower bound* on what the representation supports.
- **Fairer companion (also baseline #3):** freeze the backbone, fit a **tiny logistic on a
  few object-score summary stats** (max H-prob, max W-prob, counts above threshold) — an
  upper bound that isolates "representation quality" from "crude readout".
- **Pros:** zero arch change; reco head stays *identical* to the current baselines (clean
  comparison); exercises the full data pipeline immediately; physically principled.
- **Cons:** fixed readout may under-sell; background's many `none` objects could swamp the
  signal reco signal → needs loss balancing (lean on the loader's sig/bkg weighting + an
  explicit reco-vs-background-none weight).

### T — Two-head (per-object reco head + pooled event head)
- **Architecture:** shared backbone → (i) per-object `none/H/W` head (trained on **signal
  only**; background masked out of the reco loss), (ii) pooled event head → **3-way
  `{bkg,qqbb,lvbb}`** (matches the existing classifier baseline; collapse to binary for
  AUC). Trained on **all** events.
- **Loss:** `L = L_reco(signal) + λ · L_event(all)`; default `λ=1`, tune if one task starves.
- **Training-schedule sub-variants** (the axes Sid called out — run *after* the basic
  interleaved version works):
  | id | schedule | backprop | what it isolates |
  |---|---|---|---|
  | **T-INT** | interleaved batches (alternate reco/event updates) | full (both heads → backbone) | default multi-task; balanced gradient |
  | **T-JOINT** | both losses every batch (summed) | full | simplest; baseline for T-INT |
  | **T-SEQ-full** | reco→convergence, *then* event | full (fine-tune whole net) | does the event task degrade reco? (forgetting) |
  | **T-SEQ-frozen** | reco→convergence, freeze backbone, train event head only | head only | = frozen-backbone probe = "free discrimination" |
  | **T-INT-detach** | interleaved, event head reads **detached** representation | reco→backbone only | does the event task *need* to reshape the backbone? |
- **Pros:** explicit event optimisation; clean separate metrics; exposes the channel
  connection; the sub-variants directly answer "interleave vs sequential" and "head-only vs
  whole-net backprop".
- **Cons:** new plumbing (dual head, dual loss, mixed-batch masking); task-balance to tune.

## Baselines to beat (reference points)

1. **Standalone reco net** — the existing reco organisms / d152 `none` (winner recipe):
   d152 b3 h4 = **0.891 overall** (cat0–5: 0.456/0.472/0.410/0.602/0.960/0.936);
   d20 b2 h4 = 0.847. (From `RECO_TRAINING_PLAN_2026-06-22.md`.) Joint models must be read
   against the **same-size** reco baseline.
2. **Standalone low-level classifier** — `TrainLowLevelClassifier.py` retrained on the same
   signal+background. **Number not yet established on this data** → produce it as a baseline
   (AUC + Z). README_metrics shows illustrative AUC ~0.90–0.93 for qqbb — treat as a
   placeholder until measured.
3. **Frozen-backbone probe** (= S-companion / T-SEQ-frozen) — how much sig/bkg
   discrimination is "free" in a reco-only representation. Interpretability-relevant floor.

## Metrics & evaluation protocol

- **Reconstruction:** `PerfectRecoPct` per category cat0–5 (MC-weighted), as today
  (`metrics/lowlevelrecometrics.py:HEPMetrics`). Log train AND val per-cat every eval.
- **Classification:** **ROC-AUC** (sig vs bkg; also per-channel qqbb/lvbb and mass-window
  variants that already exist) + **Asimov Z** = `sqrt(2((s+b)ln(1+s/b) − s))` from the
  MC-weighted score distribution. ⚠ confirm/wire Z against the existing
  `max_bkg_levels` / `signal_acceptance_levels` hooks in `HEPMetrics` — *don't assume it's
  already implemented*.
- **Gold standard (deferred):** downstream expected limit. Record finalists' model paths so
  we can run the lxplus pipeline on the best 1–2 models later. **Not per-run.**
- **Comparability:** keep `n_splits=2`, the winner recipe, and the candidate size ladder so
  numbers line up with the reco workstream's baselines. Fresh preregistered eval slices use
  **val batches 25+** (`--skip 24`) per the data-slice convention.

## Data requirements (the blocker)

- **On laptop:** signal only (`tmp_data_20250321v1_signal/`, DSIDs 510115–124, ~2.81M ev).
- **Needed:** background memmaps in the **same format** (`(n_ev, max_objs+2, n_vars+1)`;
  row0=event label, row1=weight/dsid, rows2+=objects). Sid to bring from Seagate and
  confirm: (i) which background processes exist, (ii) event counts, (iii) that the
  per-object **feature layout matches** the signal memmaps (same `N_Real_Vars_In_File`,
  same column order, tag present). **Format mismatch is the #1 risk** — verify before any
  joint run.
- **Subset for prototyping:** start with the **dominant background process(es)** (identify
  the largest-yield DSIDs from the memmaps on arrival; ttbar is typically dominant in
  HplusWh). Scale to the full cocktail later (RunPod if needed).
- **Regeneration fallback:** `preprocessing-scripts/preprocessLowLevel.py` is on the laptop;
  needs the background **ROOT files** (drive/heppc) if no compatible memmaps exist.

## The scale / dataset-size sweep (the trade-off surface)

Reuse the reco candidate ladder for comparability. Two axes + dataset size:

- **Size ladder:** d20/b2/h4 · d48/b2/h4/mlp192 · d96/b3/h4/mlp384 · **d152/b3/h4/mlp400**
  (· d256 if MPS allows). Probe **depth** (b2↔b3↔b4) and **width** separately at a couple
  of points (depth is the high-interp-cost / high-leverage knob per the reco workstream).
- **Dataset-size fractions:** {≈10%, 30%, 100%} of available train events — does more data
  resolve interference?
- **Cells:** at each (size, config ∈ {S, T-INT}, data-fraction) record reco PerfectRecoPct
  (per cat) + classification AUC + Z, plus the single-task baselines at that size.
- Start **small + subset** (d20/d48, dominant-bkg subset) to get the first head-to-head
  cheaply; expand the grid once the pipeline + a first signal are trusted.

## Hypotheses & decision criteria (what to watch; when to pivot/continue)

**Registered hypotheses:**
- H-A (transfer×scale): negative transfer (interference) at small d_model, → neutral/positive
  as capacity grows. *Evidence:* joint-minus-baseline gap (per task) shrinks/flips with size.
- H-B (readout vs representation): single-head S ≈ two-head T at large scale (shared
  representation dominates) but T > S at small scale (dedicated readout helps). *Evidence:*
  S-vs-T AUC gap vs size; S's frozen-logistic companion closes most of the gap ⇒ it's
  readout not representation.
- H-C (free discrimination): a reco-only backbone already separates sig/bkg well
  (frozen-probe AUC ≫ 0.5), because the channel decision lives in the reco head. *Evidence:*
  T-SEQ-frozen / S-companion AUC.
- H-D (interp): joint training makes the channel / W-candidacy circuit *more* explicit
  (cleaner attention, stronger probes) — or measurably muddier. *Evidence:* re-run the H1
  probes/attention summaries on a joint model vs the reco-only model at matched size.

**Continue down a path if:** the joint model **matches or beats both** single-task
baselines at some scale (free or positive transfer), OR trades a small reco cost for a
worth-it classification gain (or vice versa) — *and* the trade improves with scale.

**Pivot / down-weight a path if:** **persistent negative interference at all scales** (joint
strictly worse than both baselines everywhere) ⇒ the "one network does both" thesis is weak
for that variant — document *where* it breaks (which task, which categories, which scale).
S-specific pivot: if the fixed readout caps S far below T *and* the frozen-logistic
companion doesn't recover it, S's representation is genuinely worse → favour T.

## Phased schedule

### Phase 0 — plumbing, signal-only (NOT blocked on data; start now)
- 0a. Extend the harness (fork/extend `train_organism.py` → `train_joint.py`) for: a dual-head
  `TestNetwork` (return reco + event logits), the dual loss with mask + `λ`, and the
  single-head `none`-for-background mode + readout. Add `--mode {single,twohead}`,
  `--schedule {int,joint,seqfull,seqfrozen,intdetach}`, `--lambda-event`.
- 0b. **Smoke-test using signal as pseudo-background** (relabel a held-out signal DSID's
  objects to `none` and flag it `bkg`) — exercises every code path (mixed batches, masking,
  both heads, both losses, both metrics) **without** real background.
- 0c. Asimov-Z + weighted-AUC metrics implemented, unit-tested, and wired into the harness
  ✅. Still TODO (Phase 1): run `TrainLowLevelClassifier.py` smoke so baseline #2 is ready to
  fire the moment background lands.

**Phase 0 status (2026-06-23): BUILT + SMOKE-TESTED signal-only ✅** — see "Results (live)".

### Phase 1 — get background (Sid) + verify format
- Copy dominant-background memmaps from Seagate; verify feature layout vs signal (Risk #1).
- Establish baseline #2 (standalone classifier) and baseline #3 (frozen probe).

### Phase 2 — first head-to-head (one scale, subset)
- At d20 and d48 on the dominant-bkg subset: run S, T-INT, both single-task baselines.
  First read on transfer sign + S-vs-T gap. Decide whether to widen the grid.

### Phase 3 — scale & dataset-size sweep
- Fill the trade-off surface (size ladder × {S, T-INT} × data-fraction). 2–3 seeds at
  finalists (per-cat seed noise ±0.005–0.016 from the reco workstream).

### Phase 4 — two-head schedule variants
- T-JOINT / T-SEQ-full / T-SEQ-frozen / T-INT-detach at the best 1–2 sizes. Answers
  "interleave vs sequential" and "head-only vs whole-net backprop".

### Phase 5 — synthesis & interp
- Trade-off figure; H-D interp comparison (joint vs reco-only at matched size); pick
  finalist(s) for the deferred downstream-limit gold-standard check.

## Open questions / risks
- **R1 (format):** background memmap layout must match signal exactly — verify first.
- **R2 (imbalance):** real proportions are background-dominated; rely on the loader's
  sig/bkg weighting + tune S's reco-vs-`none` balance and T's `λ`.
- **R3 (readout fairness):** S's fixed readout vs T's learned head — always report S's
  frozen-logistic companion alongside, so S-vs-T compares representations not readouts.
  **Tooling done** (`experiments/eval_single_readout.py`): report both the parameter-free
  softor/max (lower bound) and the fitted fair readout (upper bound) for every S model.
- **R4 (MPS scale):** signal+background ≈ doubles data; the subset-first plan mitigates;
  RunPod is the escape hatch.
- **R5 (channel labels):** event head is 3-way `{bkg,qqbb,lvbb}` — confirm the qqbb/lvbb
  split in background events is well-defined (background → `bkg` class only).
- **R6 (significance weighting — MUST fix before trusting any classification number):** the
  harness currently feeds `abs(MC_Wts)` to AUC/Asimov-Z (sklearn rejects negative sample
  weights) and does **not** apply per-DSID luminosity/expected-yield normalization. So the
  Phase-0 AUC/Z are NOT physical. When background lands: (a) normalize per-DSID to expected
  yields using the loader's `weight_sums`/`abs_weight_sums`; (b) for the Asimov *yield* sums
  use **signed** MC weights (negatives are physical); keep abs only where sklearn's AUC
  forces it (or switch to the manual weighted-AUC in `metrics/lowlevelmetrics.py`).
- **R7 (event pooling masks padding):** the new dual-head event pooling (and the harness)
  **mask padding** out of the mean (correct). The *legacy* `is_reconstruction_model=False`
  classifier path in `models.py` does NOT mask (sums padding features). Left unchanged for
  baseline reproducibility — but decide, when establishing baseline #2, whether to re-train
  the standalone classifier with masked pooling for an apples-to-apples comparison.

## Results (live)

### Phase 0 — harness built + smoke-tested signal-only (2026-06-23)
Built via codex-driven-development (Codex implementer + Claude spec/quality reviewers). All
committed on `sid-fable-experiments`.

**What exists now:**
- `models/models.py` — `TestNetwork` gains optional `add_event_head` / `num_event_classes`
  (default off → byte-identical; new params init **after** existing modules so seeds +
  checkpoints are unaffected). Dual-head `forward` returns `{"reco":[B,n_obj,3],
  "event":[B,n_event]}` with **padding-masked** event pooling.
- `experiments/joint_primitives.py` — `event_score_from_object_logits` (softor/max readout),
  `weighted_roc_auc`, `asimov_z`, `best_asimov_z`, plus the **fair-readout** pieces
  (`event_summary_features` → 8 per-event object-score summaries that keep H and W separate;
  `FairReadout`/`fit_fair_readout` = StandardScaler + logistic). 15 unit tests.
- `experiments/eval_single_readout.py` — post-hoc evaluator for a trained single-head (or
  reco) checkpoint: fits the fair readout on train, reports **parameter-free softor/max
  (lower bound)** vs **fair-readout logistic (upper bound)** as AUC + Asimov Z + coefficients.
  `--ckpt <path>`; `--pseudo-bkg-dsid` for signal-only. (= the S-companion / baseline-#3 probe.)
- `experiments/train_joint.py` — the harness: `--mode {single,twohead}`,
  `--schedule {joint,int,intdetach,seqfull,seqfrozen}`, `--lambda-event`, `--event-classes`,
  `--pseudo-bkg-dsid`, `--seq-split`, `--readout`, plus all the `train_organism` knobs. Dual
  loss (background-masked reco + weighted event CE), the five schedules, signal-only reco
  metrics + classification AUC/Z, config.json + checkpoints.
- Tests: `tests/test_model_dualhead.py` (5), `tests/test_joint_primitives.py` (15),
  `tests/test_train_joint_smoke.py` (8 combos), `tests/test_eval_single_readout_smoke.py`
  (train→eval integration). All subprocess-runs self-clean `output/`. All green. (`pytest`
  isn't in `.venv`; run files directly with `.venv/bin/python tests/<file>.py`.)

**Smoke matrix (CPU, d8/b1/h2/no-mlp, DSID 510124 relabelled pseudo-background):** single,
single+max-readout, twohead×{joint, joint-binary, int, intdetach, seqfull, seqfrozen} all run
to completion with finite loss + a metrics line. ⚠ **the numbers are NOT physical** (510124 is
really signal; metrics are unnormalized — see R6). This phase only proves the plumbing.

**Run examples (real run, once background data is in place):**
```
# two-head, interleaved, all-cats, MPS:
.venv/bin/python experiments/train_joint.py --mode twohead --schedule int --d-model 48 \
    --blocks 2 --num-heads 4 --event-classes 3 --lambda-event 1.0
# single-head (background=all-none), softor readout:
.venv/bin/python experiments/train_joint.py --mode single --readout softor --d-model 48 --blocks 2
# smoke any config signal-only (pseudo-bkg, CPU):
.venv/bin/python experiments/train_joint.py --smoke --device cpu --no-wandb --mode twohead \
    --schedule joint --pseudo-bkg-dsid 510124 --d-model 8 --blocks 1 --num-heads 2 --no-mlp
```

### Trade-off surface (to fill once background lands)
*(append per run; mirror the reco plan's table style: model | params | mode | schedule |
data-frac | reco all + cat0–5 | AUC | Z | vs-baseline.)*

## Session log
- 2026-06-23 — Plan written. Confirmed laptop has **signal only** (background absent →
  hard blocker); `TestNetwork` already does both modes (dual-head = small change,
  single-head = no change); loader already supports signal+background + MC weighting;
  `preprocessLowLevel.py` present for regeneration fallback. Decisions: pre-made memmaps
  from Seagate / local-MPS-subset-first / AUC+Z (limit deferred) / build both S+T in
  parallel. Phase 0 (signal-only plumbing) is unblocked and can start immediately.
- 2026-06-23 (pm) — **Phase 0 BUILT + smoke-tested** via codex-driven-development (4 tasks:
  dual-head model, metric primitives, `train_joint.py` harness, integration smoke; each gated
  by an independent Claude spec+quality review). Dual-head `TestNetwork`, `joint_primitives`
  (readout/AUC/Asimov-Z), the full mode×schedule harness, and 26 tests (5+13+8) — all green
  signal-only with a pseudo-background DSID. Review caught + fixed a padding-pooling
  inconsistency (event head now masks padding, model.forward == harness). Logged R6
  (significance weighting is unnormalized / abs-weighted — must fix before trusting numbers)
  and R7 (legacy classifier pooling unmasked) as the key things to resolve when real
  background arrives. **Next = Phase 1: Sid brings background memmaps; verify format (R1).**
- 2026-06-23 (pm) — Added the **single-head fair-readout** companion (`event_summary_features`
  + `FairReadout` in `joint_primitives.py`; `experiments/eval_single_readout.py`): reports the
  parameter-free softor/max readout (lower bound) vs a fitted logistic on object-score
  summaries (upper bound), so the single-head S-vs-T comparison is about representation, not
  readout crudeness (resolves R3 tooling). Unit + train→eval integration smoke, green
  signal-only. Both readouts will be reported per S model once real background lands.

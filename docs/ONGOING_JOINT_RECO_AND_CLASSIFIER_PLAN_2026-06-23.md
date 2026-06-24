# Joint reconstruction + signal/background classification — plan & progress log (2026-06-23)

> New workstream (Sid + Claude). **Question:** can a *single* transformer do both
> (a) object-level **reconstruction** (per-object `none/H/W`) and (b) event-level
> **signal-vs-background classification** — and how does the reco↔classification
> performance trade off with scale (width/depth) and dataset size?
>
> This doc is the **living plan AND the running progress log**: append to "Results (live)"
> and "Session log" as we go; edit the plan sections when we pivot. Sister docs:
> [`ONGOING_RECO_TRAINING_PLAN_2026-06-22.md`](ONGOING_RECO_TRAINING_PLAN_2026-06-22.md) (the parallel
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
   d20 b2 h4 = 0.847. (From `ONGOING_RECO_TRAINING_PLAN_2026-06-22.md`.) Joint models must be read
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

### First local queue — d20/b2/h4, 10 ep, MPS, PSEUDO-background (2026-06-23)
> ⚠️ **PSEUDO-BACKGROUND (DSID 510124 relabelled as bkg). The reco PerfectRecoPct numbers
> are real; every classification AUC/Z below is DIAGNOSTIC ONLY — not physics** (no real
> background; weights unnormalised + abs-weighted — see R6). This queue's purpose was to
> exercise the harness end-to-end at real scale and sanity-check the schedule semantics.

| run | reco all | cat0 | cat1 | cat2 | cat3 | cat4 | cat5 | AUC* | Z* | train_loss |
|---|---|---|---|---|---|---|---|---|---|---|
| single (softor) | 0.788 | 0.103 | 0.164 | 0.072 | 0.241 | 0.909 | 0.858 | 0.946 | — | 0.212 |
| twohead joint | 0.805 | 0.070 | 0.106 | 0.036 | 0.220 | 0.935 | 0.884 | 0.968 | — | 0.443 |
| twohead int | 0.803 | 0.035 | 0.100 | 0.019 | 0.185 | 0.937 | 0.883 | 0.950 | — | 0.241 |
| twohead intdetach | 0.805 | 0.077 | 0.099 | 0.036 | 0.181 | 0.937 | 0.885 | 0.880 | — | 0.397 |
| twohead seqfull | **0.075** | 0.000 | 0.004 | 0.000 | 0.001 | 0.121 | 0.003 | 0.956 | — | 0.274 |
| twohead seqfrozen | 0.812 | 0.112 | 0.161 | 0.065 | 0.238 | 0.938 | 0.883 | 0.855 | — | 0.574 |

(* AUC = pseudo signal/bkg; Z values ~1600–2100, omitted as meaningless under pseudo-bkg + R6.)

**Reco baselines for context** (reco-only, 30 ep winner recipe): d20 b2h4 = **0.847**, d152 = 0.891.

**Takeaways (plumbing + schedule-semantics validation; NOT physics):**
- **Reco ≈ 0.79–0.81 at d20/10 ep** — undertrained vs the 0.847 30-ep d20 baseline (expected: 10 vs
  30 ep). Cats 4/5 strong (~0.88–0.94); cats 0–3 weak (undertrained). Two-head variants slightly
  edge single-head on reco; differences are within undertraining noise — **don't rank yet**.
- **`seqfull` catastrophically forgot reconstruction (reco 0.075).** Exactly the failure mode this
  schedule probes: phase-2 full-backprop on the event loss reshaped the backbone and destroyed the
  reco head. **`seqfrozen` (freeze backbone in phase 2) preserved reco best (0.812).** The schedules
  behave as designed — and this previews a real **forgetting risk** to watch for `seqfull`/`seqfull`-like
  setups once background is real.
- **Fair vs lower-bound readout (single-head, eval on its checkpoint):** softor AUC 0.945, max 0.943,
  **fair-readout logistic 0.944** — on this pseudo task the fitted readout ≈ the parameter-free one, i.e.
  the crude readout is *not* the bottleneck here. (Coefficients lean on max/sum p_H & p_W as intended.)
  We'll re-judge on real background — the tooling now reports both per model.
- **Perf:** after the `best_asimov_z` vectorisation, each run is training-bound (~2 min) not
  validation-bound (~30–50 min before). The first two rows (single, joint) were trained pre-fix
  (correct, just slow); the four two-head schedules were re-run post-fix.

### First REAL-background queue — d20/b2/h4, 8 ep, MPS (2026-06-24) ✅ COMPLETE (8/8 + fair-readout)
> **First real background.** Signal 510115–124 + the full background cocktail (ttbar
> 410470/41xxxx, W/Z+jets 700xxx, diboson 363xxx; ~20.5M bkg events) copied locally to
> `tmp_data_20250321v1_all/` and verified. d20 b2 h4, 8 epochs, 3-class event head, λ=1,
> `--real-bkg --position-split`. Run concurrently with another agent's MPS job (≈3× slower
> wall-clock; results unaffected).
>
> ⚠️ **Caveats:** (1) **AUC is the comparative metric; absolute Asimov Z is UNCALIBRATED**
> (abs MC weights, no per-DSID luminosity normalization — R6). (2) **Position split** (not
> eventNumber): the background memmaps have systematically *even* eventNumbers, so the
> eventNumber split dumped ~91% of background (incl. all ttbar) into val — position split
> gives a balanced 50/50 per-DSID split. Signal is statistically equivalent either way, so
> reco is ~comparable to the eventNumber-split reco baselines. (3) W&B off (unattended).

**Data + harness changes this session:**
- `train_joint.py` gains `--data-dir`, `--real-bkg` (include all `dsid_*.memmap`,
  `signal_only=False`), `--position-split`, a complete greppable `JOINTSUMMARY` per-epoch
  line, and a metrics-buffer sized to the signal val count (avoids ~4 GB waste / OOM with the
  12M-event val). `eval_single_readout.py` gained the same real-bkg flags. New:
  `experiments/run_joint_real_bkg_mps.sh` (the 8-run queue), `parse_joint_results.py`
  (logs→table), `verify_real_bkg_data.py` (R1 live check). **All uncommitted — pending Sid.**
- **Bug found + fixed** (`metrics/lowlevelrecometrics.py`): the per-object CE built its target
  with `one_hot(...)` **without `num_classes`**, so an all-background / all-`none` batch (which
  occurs in single-head mode once signal indices deplete near an epoch end) produced a
  1-column target vs 3-class logits → `cross_entropy` crash. Pinned `num_classes` to the logit
  count — a **no-op for every signal-containing batch** (so it can't change the reco baseline
  or the two-head runs). Two-head mode never hit it (it skips reco when signal-weight is 0).

**Results** (all 8 runs complete; table auto-built by `parse_joint_results.py`). Grouped:
baselines, two-head schedules, single-head.

| run | role | reco all | lvbb | qqbb | cat0 | cat1 | cat2 | cat3 | cat4 | cat5 | AUC | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pure_reco (λ=0) | base #1 reco | **0.856** | 0.903 | 0.761 | 0.212 | 0.308 | 0.206 | 0.404 | 0.953 | 0.919 | 0.500† | 0.147 |
| standalone classifier | base #2 clf | 0.000 | — | — | — | — | — | — | — | — | **0.9842** | 0.218 |
| twohead joint (λ=1) | T-JOINT | 0.841 | 0.895 | 0.731 | 0.156 | 0.227 | 0.128 | 0.280 | 0.951 | 0.912 | **0.9839** | 0.371 |
| twohead int | T-INT | 0.839 | 0.893 | 0.728 | 0.146 | 0.218 | 0.116 | 0.282 | 0.950 | 0.909 | 0.9832 | 0.201 |
| twohead intdetach | T-INT-detach | 0.840 | 0.891 | 0.738 | 0.153 | 0.218 | 0.126 | 0.356 | 0.947 | 0.907 | 0.9591 | 0.284 |
| twohead seqfrozen | T-SEQ-frozen | 0.852 | 0.904 | 0.747 | 0.181 | 0.275 | 0.186 | 0.380 | 0.957 | 0.909 | 0.9688 | 0.308 |
| twohead seqfull | T-SEQ-full | 0.462 | 0.630 | 0.123 | 0.004 | 0.151 | 0.019 | 0.030 | 0.670 | 0.160 | 0.9833 | 0.223 |
| single-head softor | S | 0.797 | 0.849 | 0.691 | 0.129 | 0.234 | 0.142 | 0.250 | 0.901 | 0.866 | 0.9822 | 0.135 |

† pure_reco AUC 0.500 is correct — its event head is never trained at λ=0. (Uncalibrated
Asimov Z ~0.9k–2.0k omitted as meaningless in absolute terms — R6.)

**Single-head readout (S, `eval_single_readout` on its ckpt, 246k fit / 246k eval events):**
parameter-free softor **0.9816**, parameter-free max **0.9823**, fair-readout logistic
**0.9818** — i.e. the fitted readout ≈ the parameter-free one. The crude soft-OR is **not** the
bottleneck; the readout already extracts ~all the object-score discrimination.

**Reco anchors:** same-pipeline reco-only (λ=0) d20 b2h4 8ep = **0.856** (best); separate reco
workstream d20 b2h4 30ep = 0.847.

**Headline findings (real background):**
- **Near-free positive transfer at d20 (H-A).** One `twohead joint` transformer does both at
  once: reco **0.841** + AUC **0.9839**. Vs the *same-pipeline* single-task anchors (pure_reco
  0.856; standalone 0.9842) it costs a **small reco tax (−0.015)** for **~zero classification
  cost (−0.0003)**. Clear *yes* to "one network does both" at this scale — full classification
  rides ~free on top of reco for a slight reco dip. `int` matches `joint` (0.839/0.9832) —
  interleave-vs-joint barely matters.
- **S vs T (H-B): T ≥ S, and the readout is not the limiter.** Single-head S matches T on
  classification (0.982 vs 0.984; and its fair-readout = parameter-free, so *not* readout-bound)
  but pays a **bigger reco tax** (S reco 0.797 vs T 0.841) — forcing the shared reco head to emit
  all-`none` on background degrades signal reco more than T's background-masked reco loss. So at
  d20 two-head is the better structure: same discrimination, better reco.
- **Free discrimination is real but partial (H-C).** A reco-only representation already separates
  sig/bkg well: `seqfrozen` (freeze reco backbone, train only event head) **0.969**;
  `intdetach` (event head reads detached backbone) **0.959**. Letting classification *reshape*
  the backbone adds **+0.015–0.025** to reach 0.984. So the channel decision largely lives in the
  reco representation, with a modest gain from dedicated backbone adaptation.
- **Sequential full-backprop forgets reco (forgetting risk).** `seqfull` (reco → then full-backprop
  event) **collapsed reco to 0.462** (from the ~0.85 phase-1 level) while keeping AUC 0.983;
  `seqfrozen` (freeze in phase 2) preserved reco at **0.852**. Milder than the pseudo-bkg run's
  near-total 0.075, but the same catastrophic-forgetting mechanism — avoid unfrozen sequential
  fine-tuning if reco must survive.
- Reco per-category pattern matches the reco workstream: cat4/cat5 strong (~0.90–0.96), cat0–3
  weak (the hard categories); lvbb (0.85–0.90) > qqbb (0.69–0.76).
- **Next:** scale axis (d48/d96 — does the −0.015 reco tax shrink/flip with capacity? H-A) and
  the dataset-size axis; fix **R6** (signed-weight, per-DSID luminosity-normalized Z) before any
  physics/significance claim; W&B on for the publishable sweep; H-D interp (joint vs reco-only).

### Old-vs-new Sʙ₂₀₀ comparison — reco+classification, native event set each (2026-06-24)
> **The headline thesis comparison (Ch 7 Fig-4 style):** signal yield retained when **200
> background events survive** (Sʙ₂₀₀), per H⁺ mass, **qqbb and lvbb separately**, with each
> method evaluated on **its own native event set**. This is the fair cross-method metric (ROC-AUC
> is *not* comparable across methods that admit different event sets — Ch 7).

**The two methods (corrected understanding — thesis Ch 5/6 = OLD, Ch 7 = NEW):**
- **OLD** = hand-cut, heuristic **C++ reconstruction** (ν from MET+W-mass; large/small-R jets →
  H/W by mass windows + Xbb-tag; keep only categories 0 & 3; split into reco-lvbb/reco-qqbb) →
  **per-channel high-level DNN** on ~7 engineered variables. Throws away most events; reco
  efficiency <60%. Numbers = the stored `sig_rems_{qqbb,lvbb}.pkl` ("High-level Joint/Parametrised
  NN") — already computed, on the OLD method's cut-based native sample.
- **NEW** = one transformer on **low-level objects**, reco (per-object none/H/W → 6 topologies)
  + event classification, on **all** preselected events; events are kept if the reco produces a
  valid topology (`utils.check_valid`) — that validity *is* the native selection (it removes
  ~41% of background; the thesis quotes "59% of background remains", which our `check_valid`
  reproduces: reco-valid frac ≈ **0.59** for joint/int). The thesis's own NEW number is stored as
  `"Low-level NN (transformer-reco inputs)"` (a **d200** transformer) — used here as a reference.

**How OUR (tonight's d20) numbers are computed** (`experiments/sb200_oldvsnew.py`): each model's
**own reco head → `check_valid` → valid events = native set**; reco-**invalid** events are set to
pure-background `[1,0,0]` so they're rejected by the working point rather than yield-scaled away
(keeps the per-DSID expected-yield normalization correct); then **Sʙ₂₀₀ via the identical
`lowlevelmetrics.HEPMetrics.compute_signal_selection_metrics`** that produced the stored numbers
(channel = argmax P(qqbb)/P(lvbb); threshold on P(bkg) for 200 expected bkg; signal of the true
channel retained). Full val, per-DSID expected-yield normalized (signal at ref σ×BR=1 pb — same
convention as the stored numbers, so directly comparable).

**Sʙ₂₀₀ — qqbb (signal kept @ 200 bkg), per H⁺ mass [TeV]:**

| model | 0.8 | 1.0 | 1.4 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| **MAX available** (all truth-qqbb signal, no reco) | 11068 | 11851 | 12359 | 12133 | 11431 |
| OLD High-level Joint | 114 | 956 | 2584 | 3246 | 3260 |
| OLD High-level Parametrised | 287 | 811 | 2480 | 3727 | 4200 |
| *ref* Low-level transformer-reco (thesis, d200) | 745 | 1983 | 4426 | 7296 | 8662 |
| **NEW ours twohead_joint (d20)** | **407** | **1635** | 3958 | 6859 | **8500** |
| NEW ours twohead_int (d20) | 359 | 1525 | 3934 | 6712 | 8435 |
| NEW ours single_softor (S, d20) | 208 | 1175 | 3599 | 6291 | 8066 |
| NEW ours twohead_seqfrozen | 7 | 80 | 1672 | 5324 | 8410 |
| NEW ours twohead_intdetach | 4 | 67 | 971 | 3611 | 7169 |
| NEW ours twohead_seqfull | 259 | 679 | 1266 | 1203 | 740 |

**Sʙ₂₀₀ — lvbb:**

| model | 0.8 | 1.0 | 1.4 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| **MAX available** (all truth-lvbb signal, no reco) | 20705 | 23432 | 24821 | 25168 | 25145 |
| OLD High-level Joint | 121 | 728 | 4020 | 6622 | 7220 |
| OLD High-level Parametrised | 205 | 1071 | 4025 | 7171 | 7879 |
| *ref* Low-level transformer-reco (thesis, d200) | 2317 | 5651 | 11156 | 17392 | 21402 |
| **NEW ours twohead_joint (d20)** | 518 | 2991 | 8800 | 15911 | 21055 |
| NEW ours twohead_int (d20) | 409 | 2612 | 8833 | 16019 | 21105 |
| NEW ours single_softor (S, d20) | 427 | 2578 | 8510 | 15718 | 20666 |

Plot: `output/joint_real_bkg_20260624-010237/sb200_oldvsnew.png` (2 panels qqbb/lvbb; old dashed,
thesis-ref dotted, ours solid). reco-valid frac: joint 0.594, int 0.593, intdetach 0.627,
seqfrozen 0.646, **seqfull 0.261**.

**Findings:**
- **Our NEW method beats the OLD high-level method at *every* mass**, biggest at low mass where the
  hand-cut reco is weakest (lvbb 0.8 TeV: old ~120–205 → ours ~520; qqbb 0.8 TeV: old 114–287 → 407).
- **At high mass our tiny d20 model ≈ the thesis d200 reference** (qqbb 3 TeV 8500 vs 8662; lvbb
  21055 vs 21402). The low-mass shortfall vs d200 is the expected cost of a much smaller, 8-epoch
  model on the hard regime → **scale-up (d48/d96) is the clear lever** to close it.
- **Reco quality drives the NEW method's selection.** `seqfull` (forgot reco; reco-valid frac only
  **0.26**) collapses (qqbb plateaus ~1200, *drops* to 740 at 3 TeV). `seqfrozen`/`intdetach` are
  fine at high mass but fall apart at low mass (qqbb 0.8 TeV: 7 / 4) — a frozen/detached
  representation can't make a pure low-mass selection. So a good shared reco is what makes the
  unified model's event selection work — direct support for the "one network does both" thesis.

**Single-head S ("background = all-none") — now included.** `single_softor` has no 3-class event
head, so its **channel comes from its reco** (`check_valid` → valid_qqbb/valid_lvbb) and its
**discriminant is the parameter-free softor readout** (P(bkg)=1−softor); fed through the *same*
metric (reco-channel rather than classifier-argmax — a documented difference). Result: the
**zero-extra-parameter single-head also beats the OLD method** (qqbb 0.8 TeV 208 vs old 114/287;
lvbb 0.8 TeV 427 vs old 121/205) and is **close to the two-head at high mass** (qqbb 3 TeV 8066 vs
8500) — despite only **~23% of its events being reco-valid** (vs ~59% for two-head): its reco head,
trained on background-as-all-none, reconstructs far fewer valid topologies, shrinking its native
set. So most of the discrimination is in the shared representation, not a dedicated readout (echoes
the AUC-level H-B finding).

**MAX-available reference (the ceiling).** The black line = full **truth-channel** signal yield per
mass (no reco/selection), per-DSID expected-yield normalized. Fractions kept at 200 bkg: two-head
**~74% qqbb / ~84% lvbb at 3 TeV**, but only **~3–4% at 0.8 TeV** — vs the OLD method's ~1% at
0.8 TeV. So the new method's biggest *relative* gain is at low mass, yet **large headroom remains
everywhere** (esp. low mass) → motivates scale-up + the dataset-size axis. (MAX is ~flat in mass
because all signals are normalized to σ×BR=1 pb; the rising "kept" curves reflect easier separation
at high mass.)

### Scale-up: d152 best-method run + Sʙ₂₀₀ (2026-06-25) ✅
> **Best method, trained to completion at the reco-winner scale.** two-head **joint** (λ=1),
> **d152 b3 h4 mlp400** (672k params), real bkg + position split, **winner recipe** (lr 1e-3→5e-7
> cosine, ramp warmup 100, wd 1e-6, batch 4096), **30 epochs**, W&B run `hor26ayx`, ckpt
> `output/20260624-163646_TrainingOutput/models/Nplits2_ValIdx0/chkpt29_85320.pth`.

**Training result (epoch 29, full val):** AUC **0.989**, reco_all **0.884** (lvbb 0.917, qqbb 0.816),
cats: cat0 0.385 · cat1 0.624 · cat2 0.649 · cat3 0.670 · cat4 0.989 · cat5 0.947; train_loss 0.195.
- **Near-free transfer *tightens* with scale (H-A):** joint reco 0.884 vs reco-only d152 ≈ 0.891
  → ~0.007 tax (vs ~0.015 at d20). AUC 0.989 > d20 joint / standalone (0.984).
- **The hard reco categories (the bottleneck) jumped ~2–4×** vs d20 joint (cat0 0.16→0.39,
  cat1 0.23→0.62, cat2 0.13→0.65, cat3 0.28→0.67) — exactly the regime tied to low-mass Sʙ₂₀₀.

**Sʙ₂₀₀ scale-up (signal kept @ 200 bkg, per H⁺ mass [TeV], native set each):**

qqbb:
| model | 0.8 | 1.0 | 1.4 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| MAX available | 11068 | 11851 | 12359 | 12133 | 11431 |
| OLD High-level Joint | 114 | 956 | 2584 | 3246 | 3260 |
| OLD High-level Parametrised | 287 | 811 | 2480 | 3727 | 4200 |
| *ref* Low-level transformer-reco (thesis d200) | 745 | 1983 | 4426 | 7296 | 8662 |
| NEW ours **d20** joint | 407 | 1635 | 3958 | 6859 | 8500 |
| NEW ours **d152** joint | **1369** | **2501** | **4707** | 7267 | 8658 |

lvbb:
| model | 0.8 | 1.0 | 1.4 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| MAX available | 20705 | 23432 | 24821 | 25168 | 25145 |
| OLD High-level Joint | 121 | 728 | 4020 | 6622 | 7220 |
| OLD High-level Parametrised | 205 | 1071 | 4025 | 7171 | 7879 |
| *ref* Low-level transformer-reco (thesis d200) | 2317 | 5651 | 11156 | 17392 | 21402 |
| NEW ours **d20** joint | 518 | 2991 | 8800 | 15911 | 21055 |
| NEW ours **d152** joint | **3108** | **6219** | **11223** | 17126 | 21706 |

Plot: `output/joint_real_bkg_20260624-010237/sb200_oldvsnew_scaleup.png`. reco-valid frac:
d152 0.596 ≈ d20 0.594 (≈ thesis "59% bkg remains") — validity selection unchanged; the gain is
pure reco+classification quality.

**Findings:**
- **Scale-up closes the low-mass gap.** d20→d152 at 0.8 TeV: qqbb 407→**1369** (3.4×), lvbb
  518→**3108** (6×); at high mass d152≈d20 (both saturate near the references). So the d20
  shortfall really was capacity in the hard low-mass regime, as hypothesized.
- **d152 matches-or-beats the thesis d200 transformer-reco reference:** ties at high mass (qqbb
  3 TeV 8658 vs 8662; lvbb 21706 vs 21402) and **exceeds at low mass** (qqbb 0.8 TeV 1369 vs 745;
  lvbb 3108 vs 2317). ⚠ The thesis ref was computed on a *different* data version (30-object,
  reco-applied/split sample) and pipeline, so treat it as a sanity reference, not a strict
  apples-to-apples — the encouraging read is our 15-object joint d152 is in that model's league
  or better.
- **Dominates the OLD hand-cut + high-level-NN method everywhere**, hugely at low mass (qqbb
  0.8 TeV ~5–12×; lvbb ~15–26×).
- **Fraction of available signal kept** (vs MAX): d152 ~**12% qqbb / ~15% lvbb at 0.8 TeV**
  (d20 ~3%, OLD ~1%) and ~**76% / ~86% at 3 TeV** — big low-mass gain, headroom still remains
  → further scale / data-fraction and the R6-calibrated significance are the next levers.

### Reproduce / monitor (local tooling)
- **Sʙ₂₀₀ old-vs-new:** `experiments/sb200_oldvsnew.py [run_root] --models … [--extra label=<TrainingOutputDir>] --device mps --plot p.png`
  (reads `sig_rems_{qqbb,lvbb}.pkl` for OLD + thesis-ref; computes ours). Also
  `experiments/signal_at_fixed_bkg.py` (combined-signal Sʙ_N, no channel split / no reco-validity —
  a simpler diagnostic).
- **Real-bkg runner:** `experiments/run_joint_real_bkg_mps.sh` (env `EPOCHS`/`D_MODEL`/…) →
  `output/joint_real_bkg_<stamp>/{manifest.log,logs/*.log}`; parse with
  `experiments/parse_joint_results.py <run_root>`. The dashboard now also auto-discovers
  `output/joint_real_bkg_*/logs/*.log`.
- **Runner scripts:** `experiments/run_joint_first_set_mps.sh` (full set: `single_softor` +
  fair-readout eval + 5 two-head schedules) and `experiments/run_joint_remaining_mps.sh` (the 4
  two-head schedules only — the post-fix re-run). Env knobs: `EPOCHS`, `BATCH_SIZE`,
  `PSEUDO_BKG_DSID`. Sequential, MPS, run under `caffeinate` for long queues; logs +
  `manifest.log` under `output/joint_first_set_<stamp>/`.
- **Live dashboard:** `.venv/bin/python experiments/dashboard.py --port 8765`
  (→ http://localhost:8765) — self-refreshes every 4 s and auto-discovers
  `output/joint_first_set_*/logs/*.log` (no restart needed when a new queue starts).
- **W&B:** the pseudo-background runs used `--no-wandb` (diagnostics, not publishable).
  **Turn W&B back on** (drop `--no-wandb`) for real-background comparison runs.
- **Checkpoints/configs** for completed runs are under `output/<stamp>_TrainingOutput/`
  (gitignored, local); e.g. the single-head checkpoint the fair-readout eval used =
  `output/20260623-155505_TrainingOutput/models/Nplits2_ValIdx0/chkpt9_3430.pth`.
- **Gotcha:** `train_joint.py` stdout is block-buffered through `tee`, so the log file (and the
  dashboard) update in chunks and can briefly look "stalled" mid-run — verify with `ps`
  (CPU% / state `R`) before assuming a hang. (The earlier multi-hour "stall" was the O(N²)
  `best_asimov_z`, now vectorised.)

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
- 2026-06-23 (eve) — **First local pseudo-background queue** (another agent kicked it off;
  handover in `HANDOVER_JOINT_RECO_CLASSIFIER_2026-06-23.md`): d20/b2/h4, 10 ep, single +
  5 two-head schedules. **Found + fixed an O(N²) `best_asimov_z`** (looped `np.unique(scores)`
  in python → ~20 min on the 1.4M-event val, which dominated each run's wall-time and looked
  like a stall); vectorised to O(N log N) (sort + cumulative weight sums), identical result
  (brute-force regression test), 1.4M events 20 min → 0.16 s; re-ran the 4 two-head schedules
  fast (~2 min each). Results table above (PSEUDO-bkg = diagnostic only). Key validation:
  **`seqfull` catastrophically forgot reco (0.075)** while **`seqfrozen` preserved it (0.812)** —
  the schedules behave as designed; fair-readout ≈ parameter-free on this pseudo task. Helper
  scripts (`run_joint_first_set_mps.sh`, `run_joint_remaining_mps.sh`, `dashboard.py` patch)
  still uncommitted pending Sid's call. **Next still = Phase 1 (real background; R1 format check).**
- 2026-06-24 (overnight) — **Phase 1+2+4 done on REAL background.** Sid's bulk copy had missed
  the memmaps (`heppc_recovered/data/` empty); copied the 12 GB `20250321v1` subdir locally
  (`tmp_data_20250321v1_all/`, signal symlinked + 40 bkg memmaps), byte-verified vs a size
  snapshot. **R1 confirmed** (background layout matches signal; loader maps bkg→label 0).
  Found the background eventNumbers are systematically **even** → eventNumber split dumped ~91%
  of bkg (all ttbar) into val → added **`--position-split`** (balanced 50/50). Adapted
  `train_joint.py` (`--data-dir/--real-bkg/--position-split`, JOINTSUMMARY line, signal-sized
  metrics buffer) + `eval_single_readout.py`; new `run_joint_real_bkg_mps.sh` /
  `parse_joint_results.py` / `verify_real_bkg_data.py`. Ran an **8-run d20/b2/h4 8-ep queue**
  (concurrent with another agent's MPS job, coordinated via `~/Documents/tmp.20260624.md`).
  **Fixed a real bug** in `metrics/lowlevelrecometrics.py` (one_hot lacked `num_classes` → crash
  on all-`none` batches in single-head mode; no-op for signal batches). **Headline: near-free
  positive transfer at d20** — joint matches both single-task anchors (reco 0.841 vs 0.856;
  AUC 0.984 vs 0.984) for a −0.015 reco tax; T≥S (S not readout-bound but pays more reco tax);
  frozen reco-rep gives 0.96–0.97 "free" discrimination (+0.02 from backbone reshaping);
  seqfull forgets reco (0.46) vs seqfrozen (0.85). All AUC comparative; **Z uncalibrated (R6)**.
  Results table + findings above. **Code + metrics fix uncommitted — pending Sid's review.**
  **Next:** scale (d48/d96) + dataset-size axes; fix R6; W&B on; H-D interp.
- 2026-06-24 (pm) — **Old-vs-new Sʙ₂₀₀ comparison** (the thesis Ch-7 Fig-4 metric). Read the
  thesis (Ch 5/6 = OLD hand-cut C++ reco + per-channel high-level DNN; Ch 7 = NEW low-level
  transformer = ours) and the comparison code to correct my model-of-the-methods. The OLD method's
  Sʙ₂₀₀ is already stored in `sig_rems_{qqbb,lvbb}.pkl` (incl. the thesis's own NEW d200 number,
  `"Low-level NN (transformer-reco inputs)"`). Built `experiments/sb200_oldvsnew.py`: scores our
  models with their own reco→`check_valid` (native set; reco-invalid→pure-bkg) + the *identical*
  `lowlevelmetrics.HEPMetrics` Sʙ₂₀₀ → directly comparable to the stored numbers. **Validated**
  (our twohead_joint ≈ thesis d200 at high mass; reco-valid frac 0.59 ≈ thesis "59% bkg remains").
  **Result: our NEW d20 model beats the OLD high-level method at every mass** (biggest at low mass),
  **matches the thesis d200 at high mass**, low-mass shortfall = small-model cost → scale-up next.
  Reco quality drives the selection (seqfull's forgotten reco → valid frac 0.26 → collapses).
  Single-head S excluded (no 3-class head; needs reco-channel+softor) — TODO. Needed `np.trapz`→
  `np.trapezoid` shim (numpy upgrade). **Committing the joint real-bkg + comparison code now.**
- 2026-06-24 (pm, follow-up) — Added to `sb200_oldvsnew.py`: (1) the **single-head S** (channel from
  its reco via `check_valid`, softor discriminant → pseudo-3-class probs into the same metric) — it
  **also beats OLD** and tracks the two-head at high mass despite only ~23% reco-valid (vs ~59%);
  (2) a **MAX-available** reference line = full truth-channel yield (no reco), per-DSID
  expected-yield normalized. Confirmed the metric counts **only correct-truth-channel** signal
  (target==2 qqbb / ==1 lvbb, truth set pre-reco). Two-head keeps ~74%/84% (qqbb/lvbb) of available
  at 3 TeV, ~3–4% at 0.8 TeV (OLD ~1%) → headroom everywhere, biggest relative gain at low mass.
  Table/plot updated above.
- 2026-06-25 — **Scale-up: best-method d152 run to completion.** two-head joint, d152/b3/h4/mlp400,
  winner recipe, 30 ep (~3.5 h compute; resumed cleanly through a lid-close suspend), W&B `hor26ayx`.
  **AUC 0.989, reco_all 0.884** (near reco-only d152 0.891 → transfer tax shrinks 0.015→0.007 with
  scale), **hard reco cats cat0–3 jumped ~2–4×** vs d20. Added `--extra` to `sb200_oldvsnew.py` to
  score models outside a run_root; ran the Sʙ₂₀₀ scale-up. **d152 closes the low-mass gap** (0.8 TeV
  qqbb 407→1369, lvbb 518→3108), **matches/beats the thesis d200 ref** (ties high mass, exceeds low
  mass — caveat: ref is a different 30-obj data version), dominates OLD everywhere. Plot
  `sb200_oldvsnew_scaleup.png`. Next: further scale / data-fraction; **R6** calibrated significance
  before any limit-level claim.

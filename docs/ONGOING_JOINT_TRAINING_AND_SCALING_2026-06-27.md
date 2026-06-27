# Joint reco+classifier — training runs & scaling surface (2026-06-27)

> **Active workstream (performance half of the joint reco+classifier project).** Spun out
> of [`JOINT_RECO_AND_CLASSIFIER_PLAN_2026-06-23.md`](JOINT_RECO_AND_CLASSIFIER_PLAN_2026-06-23.md)
> (now wrapped up — read it for the plan, decisions, baselines, hypotheses, and the d20
> head-to-head + d152 scale-up that this doc continues from). Interp work lives in the
> sister doc [`ONGOING_JOINT_INTERP_2026-06-27.md`](ONGOING_JOINT_INTERP_2026-06-27.md).
> Shares the harness + reco ladder with
> [`ONGOING_RECO_TRAINING_PLAN_2026-06-22.md`](ONGOING_RECO_TRAINING_PLAN_2026-06-22.md).

## The question this doc owns

The **trade-off surface**: (reco PerfectRecoPct per cat, classification AUC/Z, physics-facing
Sʙ₂₀₀) as a function of **(architecture size, depth, dataset fraction)**, against the
single-task baselines — and how transfer (joint vs single-task) changes along those axes.

## Where we are (from the wrapped-up plan doc)

- **d20/b2/h4** — full head-to-head done (S, T-INT, T-JOINT, 4 schedules, both single-task
  anchors). Headline: **near-free positive transfer** (joint reco 0.841 vs reco-only 0.856;
  AUC 0.984 vs 0.984); T≥S; seqfull forgets reco, seqfrozen preserves it.
- **d152/b3/h4/mlp400** — two-head **joint** trained to completion (winner recipe, 30 ep,
  W&B `hor26ayx`, ckpt `output/20260624-163646_TrainingOutput/.../chkpt29_85320.pth`).
  **AUC 0.989, reco_all 0.884** (transfer tax shrinks 0.015→0.007 with scale); **hard reco
  cats jumped 2–4×**; Sʙ₂₀₀ closes the low-mass gap and matches/beats the thesis d200 ref.

So we have **two scale points** (d20, d152) for the two-head joint, and a full schedule/approach
sweep only at d20. The d152 cell is **not complete** (see baselines below).

## In flight (background MPS queue, launched 2026-06-27)

`output/joint_d152_interp_20260627-231344/` (runner `experiments/run_joint_d152_interp_mps.sh`,
`--no-wandb`, ~8–9 h total, ~4 h/run @ ~8.5 min/epoch). Both d152/b3/h4/mlp400, 30 ep, winner
recipe, real bkg, position split — **matched to the joint d152** so each isolates one variable:

1. **`pure_reco_control`** (`--schedule joint --lambda-event 0.0`) — the matched-split
   **reco-only baseline #1 at d152** (event objective off → reco-only backbone, same
   data/split/recipe/seed). Doubles as the leakage-free H-D control (see interp doc).
2. **`single_softor`** — the **single-head S at d152** (S-vs-T at scale; baseline for H-B).

These were prioritized because they also serve the interp work. **Still missing from the d152
cell:** the **standalone classifier #2 at d152** (`--schedule seqfull --seq-split 0.0
--lambda-event 1.0`, matched masked pooling) and the **frozen probe #3** (have the d20 read +
the interp-doc probe sweep; a d152 `seqfrozen` would give the training-side number).

## Proposed scaling surface (suggestions)

### Size ladder (width) — reuse the reco candidate ladder for comparability
| d_model | blocks | heads | mlp | params (reco) | status |
|---|---|---|---|---|---|
| 20  | 2 | 4 | 200 | ~0.1 M | ✅ full sweep |
| 48  | 2 | 4 | 192 | ~0.1 M | ⬜ **next rung** |
| 96  | 3 | 4 | 384 | ~0.3 M | ⬜ |
| 152 | 3 | 4 | 400 | ~0.67 M | ◑ joint done; cell incomplete |
| 256 | 3 | 4 | ~512 | ~1.9 M | ⬜ if MPS allows (else RunPod) |

At each rung run **{reco-only, standalone-classifier, single-head S, two-head joint}** so every
point has its baselines. That yields the transfer-vs-size curve cleanly (we already see the tax
shrink 0.015→0.007 from d20→d152 — d48/d96 fill the curve and test H-A's monotonicity).

### Depth axis (separate, high-leverage / high-interp-cost)
Probe **b2 ↔ b3 ↔ b4** at one or two widths (e.g. d96, d152). Depth is the knob most tied to
the interp circuit (per the reco workstream), so depth points pay double (perf + interp).

### Dataset-fraction axis
{≈10 %, 30 %, 100 %} of train events at one or two sizes (d48 + d152) → "does more data resolve
interference?" (H-A). Cheapest new science per GPU-hour after the d48/d96 rungs.

### Schedule variants at scale
The 5 schedules (joint/int/intdetach/seqfull/seqfrozen) were only swept at d20. Re-run the
informative ones (seqfull = forgetting, seqfrozen = free discrimination, intdetach = does the
event task need to reshape the backbone) at **d152** to confirm the d20 schedule story holds with
capacity.

## Suggested priority order

1. **Finish the d152 cell** — standalone-classifier #2 (+ optional seqfrozen #3). [in queue: control + S]
2. **d48 + d96 full rungs** ({reco, classifier, S, joint}) → the transfer-vs-size curve.
3. **Dataset-fraction axis** at d48 + d152.
4. **Depth probe** (b4) at d96/d152, and the schedule variants at d152.
5. **d256** (RunPod if MPS-bound) — the top of the ladder.
6. **R6 calibration** before any of these numbers become a *significance/limit* claim
   (currently AUC = comparative; absolute Z uncalibrated — abs weights, no per-DSID luminosity
   normalization; Sʙ₂₀₀ *is* per-DSID expected-yield normalized and is the trustworthy metric).

## Recipe, runners, compute

- **Winner recipe (settled):** lr 1e-3 → 5e-7 **cosine**, **ramp** warmup 100, wd 1e-6,
  batch 4096, no entropy penalty, seed 0. (The 3e-4/log/legacy-warmup variants are superseded —
  legacy warmup was a +3 pt bug; see the reco plan.)
- **Harness:** `experiments/train_joint.py` (`--mode {single,twohead}`,
  `--schedule {joint,int,intdetach,seqfull,seqfrozen}`, `--lambda-event`, `--d-model/--blocks/
  --num-heads/--d-mlp`, `--real-bkg --position-split --data-dir tmp_data_20250321v1_all`).
  **`--d-mlp` defaults to 200 — set it explicitly** (d152 wants 400). Checkpoints save at
  epoch 0/25/final; config.json written alongside.
- **Runners:** `run_joint_real_bkg_mps.sh` (full 8-run set, env `D_MODEL/BLOCKS/HEADS/EPOCHS`),
  `run_joint_d152_interp_mps.sh` (the current 2-run d152 queue). Run under `caffeinate -i`;
  lid-close suspends + resumes cleanly (wall-clock advances).
- **MPS cost:** d152 ≈ 8.5 min/epoch (11.6 M train / position split) → ~4 h for 30 ep. Smaller
  rungs proportionally faster. **Monitor:** `experiments/dashboard.py --port 8765`
  (auto-discovers `output/joint_*`); parse logs with `parse_joint_results.py <run_root>`.
- **Data:** `tmp_data_20250321v1_all/` (10 signal symlinks + 40 real bkg memmaps, ~12 GB,
  gitignored). **`--position-split` is mandatory** with real bkg (background eventNumbers are all
  even → eventNumber split dumps ~91 % of bkg into val).

## Session log
- 2026-06-27 — Doc spun out of the wrapped-up joint plan. Launched the d152 interp-support queue
  (`joint_d152_interp_20260627-231344`: `pure_reco_control` + `single_softor`). Laid out the
  scaling surface (size ladder d20→d256, depth b2/b3/b4, data-fraction) + priority order. Next
  hands-on rung after the queue: **d48 + d96 full rungs**, then dataset-fraction.

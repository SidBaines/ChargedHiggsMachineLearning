# 2026-06-22/23 — Minimal "normal-transformer" reco organisms: plan, harness, boosted-only convergence sweep, error-profile tooling

> New workstream (separate from the interp programme): train **plain** reconstruction
> transformers — NO entropy penalty, NO attention bottleneck — toward (G1a) a minimal
> full-parity organism, (G1b) a simplest-viable organism, (G1c) a **boosted-only (cats 4,5)**
> organism, and (Goal 2) best-possible performance. Plan doc:
> [`RECO_TRAINING_PLAN_2026-06-22.md`](../RECO_TRAINING_PLAN_2026-06-22.md).
> Reference bar = the **d152 `none`** baseline (671k, all-cats, winner recipe):
> all 0.891, per-cat cat4 0.960 / cat5 0.936 (the rest 0.46/0.47/0.41/0.60).

## What we did

- **Data audit** (`tmp_audit_categories.py`, gitignored): 2.81M signal events; cats 0–3 =
  **15.3%** of events, 4–5 = **84.7%** (cat4 60.4%, cat5 24.3%). **raw% ≈ MC%** in every
  category → hard cats are *rare, not down-weighted*; they're also *not data-starved*
  (36k–173k events each). ⇒ loss-reweighting would be capacity-*reallocation*, not de-biasing.
- **Extended `experiments/train_organism.py`** (additive, defaults preserve old behaviour):
  flags `--num-heads/--dropout/--weight-decay/--batch-size/--embedding-size/--no-mlp/
  --warmup-steps/--features/--object-net-layers/--layer-norm/--keep-cats`. `--keep-cats 4,5`
  zeros the loss weight of non-kept categories (eval still covers all). Also: per-epoch val
  now uses a **representative random subset** (`shuffle_batch=True`) so the wandb curve isn't
  biased; checkpoints saved **every 25 ep** (was every epoch).
- **Boosted-only (G1c) sweeps**, normal transformer + winner recipe (`--entropy-weight 0
  --lr 1e-3 --lr-low 5e-7 --schedule cosine --warmup-mode ramp`):
  `run_min_boosted.sh` (30-ep grid) → `run_queue_b.sh` (cat5 probe + A2) →
  **`run_g1c_converge.sh`** (the keeper: width×depth grid, **300 ep**, attention-only).
- **Local progress dashboard** `experiments/dashboard.py` (stdlib server, parses
  `tmp_suite/*.log`, groups by experiment, progress bars + per-cat evals + sparklines) with
  two alternative designs `dashboards/v2.html` (sortable table/heatmap) and `v3.html`
  (gap-to-target leaderboard), built by subagents; cat-neutral fields (`margins`,
  `key_margin`, both-cat `sparks`).
- **Error-profile tool** `experiments/eval_error_breakdown.py`: per-category miss-type
  breakdown (resolved-H hallucination, hadronic-W, lost-H-jet, H↔W swap, lepton/ν err),
  weighted, saved as `error_profile.json` per run. Validated: reproduces the d152 reference
  0.960/0.936 exactly.

## Key results

**Boosted-only frontier @ 2 blocks (300 ep, converged, full-val):**

| d_model | params | cat4 | cat5 |
|---|---|---|---|
| 8 | 807 | 0.966 | 0.913 |
| 12 | 1.6k | 0.970 | 0.920 |
| 16 | 2.7k | 0.973 | 0.925 |
| 24 | 5.8k | 0.975 | 0.932 |
| 32 | 10k | 0.974 | 0.930 |
| 48 | 22k | 0.975 | 0.932 |
| 64 | 38k | 0.977 | 0.9355 |

- **cat4 is solved at every size** (≥0.966 > 0.960). **cat5 is the binding constraint** and
  **plateaus ~0.93 with width** — even a 38k-param d64 only grazes 0.9355.
- **DEPTH cracks cat5, cheaply**: `d16b3` (3 blocks, **3.8k params**, still finishing) is
  provisionally at cat4 0.972 / cat5 ~0.936 and rising — beating d64 on cat5 with ~10× fewer
  params. So depth ≫ width for the binding category (matches the multi-block "ask-the-jets"
  comparator from the interp work). → minimal organism heading to ~3–4k params, 3 blocks,
  attention-only.

**Why cat4 is easy / cat5 is hard (error-breakdown, generalist vs specialist):**
- Generalist (d152, all-cats) cat4 misses: **22.7% are resolved-Higgs hallucinations**
  (small-R jet → H). Boosted specialists (d24, d64): **0.0%** — they *structurally can't*,
  never having seen a resolved Higgs. Removing the resolved categories deletes that whole
  error mode ⇒ cat4 jumps 0.960→0.977. This is a **structural** (category-removal) effect,
  not a label-mix effect (reweighting alone can't drive it to *exactly* 0%).
- cat5 misses are dominated by the **H↔W large-R-jet swap (~35–39%)** + lost-H-jet — for
  *both* generalist and specialist (≈unchanged). Specialization only removes the minor
  resolved hallucinations (8%/3.5%→0%). cat5's hardness is the **intrinsic same-type
  (ljet-vs-ljet) discrimination** — untouched by category removal, hence width-insensitive
  and depth-sensitive.

## Learnings / gotchas (for future sessions)

- **Last-epoch "uptick" on the curves is a MEASUREMENT ARTIFACT, not learning.** The harness
  evals a 30-batch *subset* every epoch but the *full* val set on the final epoch. Verified
  (`tmp_eval_uptick.py`): for `d20_none_s0`, full-val `all` = 0.8508 at *both* ep28 and ep29
  (real late gain ≈ 0.000); the entire +0.0096 jump is the subset→full switch. Direction is
  metric-dependent (subset reads `all`/cat4 *low* → uptick; cat5 *high* → downtick). **Trust
  the final full-val number; the dashboard "best" is a noisy upper estimate.** No low-LR tail
  improvement is being missed.
- **For boosted runs, cat4 AND cat5 are equally the target** — rank/select by the *binding*
  (worst) margin, not cat5 alone.
- **Bigger nets converge in MORE epochs** (cat5 plateau: d8 ~ep132, d12 ~ep198) — do NOT
  extrapolate a small model's convergence epoch to bigger ones, and remember it's entangled
  with the cosine schedule length. All 2-block runs did converge within 300 ep, so 300 was a
  safe (generous) budget; verify per-run ("still rising in the last ~20%?") rather than
  blind-trimming.
- **`--entropy-weight 0` is required** for a plain transformer — the trainer default is 1e-2.
- **Unattended overnight on a MacBook**: detached daemon (`nohup … & disown`) + `caffeinate
  -is` (idle+system sleep, on AC) survives terminal/Claude close — but `caffeinate` can't stop
  **lid-close** sleep (needs clamshell). The sweep ran ~5 runs/night cleanly on AC, lid open.
- `experiments/eval_error_breakdown.py:profile_outdir()` is importable → wiring an auto
  error-profile at end of each training run is a ~3-line hook (deferred until the sweep ends,
  to avoid editing the harness mid-flight).

## Open / next

- Finish the 3-block runs (`d24b3`, `d32b3`); then wire the auto error-profile hook into
  `train_organism.py` + profile the 3-block runs.
- Pending Sid's go: **d8b3/d12b3** probe to pin the smallest 3-block organism.
- Then G1a/G1b all-cats sweep (capacity bisect d32→d96) and the Goal-2 scaling ladder.
- Measure a **boosted-only-trained large** model (e.g. d152) as the true boosted *ceiling*
  (the 0.960/0.936 bar is the all-cats generalist, a generous reference, not the ceiling).

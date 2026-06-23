# Reconstruction-net training plan — minimal organisms + best performance (2026-06-22)

> New workstream (Sid + Claude). Train **normal** reconstruction transformers (NO
> entropy penalty, NO attention bottleneck) for two ends:
> **Goal 1** minimal nets we can fully mechanistically interpret; **Goal 2** the best
> performance we can get at any size/architecture. Adaptive schedule — react to results.
> Harness: `experiments/train_organism.py` (extended 2026-06-22) + per-goal queue scripts.

## Targets (agreed)

- **G1a — full-parity organism:** smallest normal transformer matching **d152 `none`
  (0.891 overall)** across *all* categories, including the hard cats 0–3.
- **G1b — simplest-viable organism:** interpretability-simplicity first (prefer
  attention-only / ≤2 blocks / ≤3 heads); report the per-category cost honestly.
- **G1c — boosted-only organism (cats 4 & 5 only):** minimal net that matches d152 on
  cat4/cat5 only; we don't care about 0–3. Trained with `--keep-cats 4,5`. Likely the
  smallest + cleanest interp target of all.
- **Goal 2 — best performance:** maximize PerfectRecoPct (driven by cats 0–3; 4–5 are
  saturated). Any size/architecture, but **local MPS only** → realistic ceiling ≈ d256
  (overnight), d384 a multi-day stretch, d512 would need a GPU (parked).

## Ground truth (measured this session)

**Metric:** `PerfectRecoPct` = MC-weighted fraction of events with *every* non-pad object
correct. Per-object 3-way (none/H/W). Reported per **6 reco categories** cat0–5 =
{resolved-H, boosted-H} × {resolved-had-W, leptonic-W, boosted-had-W}; thesis groups
hard/rare 0–3 vs easy/dominant 4–5.

**Data (`tmp_data_20250321v1_signal/`, 10 signal DSIDs, 2.81M events; n_splits=2 →
~1.40M train / 1.40M val):**

| cat | name | raw % | MC % | raw count |
|---|---|---|---|---|
| 0 | resolved-H, resolved-W | 3.13 | 3.09 | 87,760 |
| 1 | resolved-H, leptonic-W | 6.17 | 6.17 | 173,212 |
| 2 | resolved-H, boosted-W | 1.30 | 1.32 | 36,634 |
| 3 | boosted-H, resolved-W | 4.68 | 4.69 | 131,307 |
| 4 | boosted-H, leptonic-W | 60.43 | 60.29 | 1,697,041 |
| 5 | boosted-H, boosted-W | 24.30 | 24.44 | 682,287 |

- **raw% ≈ MC%** → hard cats are rare, *not* down-weighted. Reweighting = capacity
  reallocation (trade 4–5 for 0–3), not de-biasing.
- Cats 0–3 (15.3%) are rare but **not data-starved** → scaling lifts them.
- Cats 4–5 (84.7%) near-saturated even at d20 → boosted-only organism can be tiny.

**Baselines (normal transformer = `none`; winner recipe; 30 ep; MC-weighted):**

| model | params | all | cat0 | cat1 | cat2 | cat3 | cat4 | cat5 |
|---|---|---|---|---|---|---|---|---|
| d152 b3 h4 mlp400 | 671k | 0.891 | 0.456 | 0.472 | 0.410 | 0.602 | 0.960 | 0.936 |
| d20 b2 h4 mlp200 | 20k | 0.847 | 0.178 | 0.246 | 0.159 | 0.371 | 0.952 | 0.913 |

→ **the entire capacity gap is in cats 0–3.** Regime = **underfitting** (1.4M events vs
≤1M params; suite noted curves still rising at ep30), so training-length/capacity are the
levers; regularization only matters at the very top of the scaling ladder.

**Winner recipe (confirmed by LR sweep):** Adam, peak LR 1e-3 → 5e-7 **cosine**, **linear
ramp** warmup (100 steps), wd 1e-6, batch 4096, 30 ep. (Legacy const-1e-3 warmup was a
bug; recipe alone = +3 pts.) Flag string:
`--entropy-weight 0 --lr 1e-3 --lr-low 5e-7 --schedule cosine --warmup-mode ramp`.

**Candidate sizes (params):** d8/b1/h2/nomlp 519 · d16/b1/h2/nomlp 1.6k · d16/b2/h2/nomlp
2.7k · d24/b2/h2/nomlp 5.8k · d20/b2/h4/nomlp 4.1k · d32/b2/h4/nomlp 10k · d48/b2/h4/mlp192
59k · d64/b3/h4/mlp256 154k · d96/b3/h4/mlp384 345k · **d152/b3/h4/mlp400 671k** ·
d256/b4/h8/mlp1024 3.2M · d384/b6/h8/mlp1536 10.8M · d512/b6/h8/mlp2048 19.2M.

## Design space, ranked by leverage

| knob | effect on cats 0–3 (the headroom) | interp cost | used by |
|---|---|---|---|
| training length (30→60–100 ep) | likely large (curves rising) | none | all — test first, ~free |
| capacity (d_model) | large | low–moderate | all |
| loss reweighting (upweight cats 0–3) | moderate–large (trades 4–5) | none | G1a, Goal 2 |
| #blocks (depth) | moderate–large | **high** (composition) | scaling; minimize for interp |
| MLP present + d_mlp | moderate | **high** (superposition) | scaling; drop for interp |
| #heads | small–moderate | moderate | minimize for interp |
| pre-LN in blocks (code change) | stability enabler at depth/width | low | scaling only |
| LR re-tune per scale | moderate | none | scaling |

Interpretability-cost ordering (minimize for organisms): **blocks > MLP-presence ≈ heads >
d_model.** Prefer attention-only, ≤2–3 blocks, ≤2–4 heads, small d_model.

## Schedule

### Phase A — calibrate & instrument  *(done 2026-06-22)*
- A0 data audit → table above. A1 harness extended (`--num-heads/--dropout/--weight-decay/
  --batch-size/--embedding-size/--no-mlp/--warmup-steps/--features/--object-net-layers/
  --layer-norm/--keep-cats`); train+val per-category already logged. Smoke-tested.
- A2 (pending): take d152 `none` to ~100 ep → is "longer training" a free lever for all
  goals? Fold the answer into every subsequent run.

### Phase B — the three organisms (Goal 1)
- **G1c (boosted-only) FIRST** (cheapest, newest, cleanest): `run_min_boosted.sh` grid
  d∈{8,16,24,32}×blocks{1,2}×heads2×{attn-only,+small MLP}, `--keep-cats 4,5`. Find the
  smallest matching d152 cat4=0.960/cat5=0.936. Confirm winner with 3 seeds.
- **G1a/G1b (all-cats):** bisect capacity between d20 (fails 0–3) and d152. Width sweep
  d∈{32,48,64,96} at b2 h4 +MLP; find the knee where cats 0–3 reach ~d152. Then probe
  interp-simplifications one at a time, costed per-category (attn-only, heads 4→2,
  blocks 2↔3). G1b = simplest acceptable; G1a = smallest hitting full parity. 2–3 seeds
  at finalists (per-cat seed noise ±0.005–0.016).
- **Adaptation:** attn-only collapses 0–3 → keep small MLP; 2 blocks can't reach parity →
  allow 3; reweighting recovers 0–3 cheaply → fold in. **Under/overfit:** tiny vs 1.4M
  events ⇒ pure underfitting → ensure enough epochs (A2), don't regularize.

### Phase C — best performance (Goal 2)
- Ladder from d152 anchor, one knob at a time, then combine: width d152→d256(→d384),
  depth 3→4(→6), heads 4→8; add **pre-LN** if deep+wide destabilizes. Then best capacity ×
  {longer schedule (A2)} × {cat-reweighting} × {2-point LR re-tune per scale}.
- **Adaptation / stopping:** watch train−val per-cat gap (first opens in cats 0–3, fewest
  events) → early-stop + dropout 0.1 / wd. Stop when a capacity doubling buys <~0.3 pts or
  the gap opens. **MPS time:** d256 ≈ overnight/60ep; d384 multi-day; d512 parked (GPU).

## Over/underfitting protocol
Log train AND val PerfectRecoPct per category every eval; select checkpoints by val.
- val rising + train≈val → underfitting → more epochs / capacity.
- val plateaued, train≫val (esp. cats 0–3) → overfitting → early-stop / dropout / wd.
Keep n_splits=2 for comparability with the baselines above (optionally test 80/20 only for
the final best model, re-evaluating baselines on the same split if so).

## Results (live) — see [`logs/2026-06-23_minimal-reco-organism-sweep.md`](logs/2026-06-23_minimal-reco-organism-sweep.md) for full detail

**G1c boosted-only CONVERGENCE frontier** (`run_g1c_converge.sh`, 300 ep, attn-only,
`--keep-cats 4,5`; ref d152 cat4=0.960 / cat5=0.936):

| d_model @ 2 blocks | params | cat4 | cat5 |
|---|---|---|---|
| 8 | 807 | 0.966 | 0.913 |
| 16 | 2.7k | 0.973 | 0.925 |
| 24 | 5.8k | 0.975 | 0.932 |
| 32 | 10k | 0.974 | 0.930 |
| 48 | 22k | 0.975 | 0.932 |
| 64 | 38k | 0.977 | 0.9355 |
| **16 @ 3 blocks** | **3.8k** | **0.972** | **~0.936 (rising, finishing)** |

- **cat4 solved at every size; cat5 is the binding constraint** and **plateaus ~0.93 with
  width** (even 38k-param d64 only grazes 0.9355). **DEPTH cracks cat5 cheaply** — d16b3
  (3.8k params) provisionally clears both → minimal organism ≈ 3–4k params, 3 blocks.
- **Why (error-breakdown, `eval_error_breakdown.py`):** the all-cats generalist loses
  **22.7%** of its cat4 misses to *resolved-Higgs hallucination* (sjet→H); boosted
  specialists do this **0.0%** (structural, not label-mix) → cat4 easy. cat5 misses are
  dominated by the **H↔W large-R-jet swap (~35–39%)** for *both* → intrinsic same-type
  discrimination, depth-limited.
- **Convergence:** bigger → later (cat5 plateau d8 ~ep132, d12 ~ep198); all 2-block runs
  converged within 300 ep. Don't extrapolate convergence-epoch across sizes.
- **Gotcha:** last-epoch "uptick" = subset→full-val measurement artifact (real late gain
  ≈0); trust the final full-val number.

## Session log
- 2026-06-22 — Plan written; data audited (cats 0–3 rare-not-starved, raw≈MC); harness
  extended with arch/optim/keep-cats knobs + smoke-tested; G1c 30-ep grid + Queue B launched.
- 2026-06-23 — **Boosted-only 300-ep convergence sweep** (frontier above): cat4 solved
  everywhere, cat5 plateaus ~0.93 with width but **depth (d16b3, 3.8k) clears both**.
  Error-profile tooling (`eval_error_breakdown.py`, stored per-run JSON) shows cat4's ease =
  removed resolved-Higgs error mode; cat5's hardness = intrinsic H↔W jet swap. Uptick =
  measurement artifact. Dashboard (`dashboard.py` + v2/v3) for live monitoring. Full write-up:
  [`logs/2026-06-23_minimal-reco-organism-sweep.md`](logs/2026-06-23_minimal-reco-organism-sweep.md).

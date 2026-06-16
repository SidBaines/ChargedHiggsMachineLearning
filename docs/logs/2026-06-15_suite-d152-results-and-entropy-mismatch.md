# 2026-06-15 — d152 suite results (+ a false-alarm entropy-weight scare, RESOLVED 06-16)

## ✅ RESOLVED 2026-06-16 — the "entropy-weight mismatch" was a false alarm

**The earlier worry below was wrong, caused by a misread run-name label.** The
thesis reconstruction training script (`TrainLowLevelReconstruction.py:369`, the
active line) uses **`entropy_weight=1e-2, target_entropy=0`** — which is EXACTLY what
the new suite `ent`/`both` runs used (`entropy_weight=0.01`, confirmed in their
`config.json`). The `YesEnt1` in the thesis run name is an **old index label, NOT
weight=1.0**; the "~100× stronger" inference came from applying the NEW naming
convention (`YesEnt{weight:g}`) to an OLD run name. So:

- **The comparison is NOT confounded by entropy weight.** New `ent`/`both` match the
  thesis training config.
- **The queued legacy runs need NO change** — entropy 0.01 + legacy LR recipe is the
  correct comparison. They were launched as-is on 2026-06-16.
- **The legacy run is itself the empirical confirmation:** if `d152_both_s0_LEGACY`
  lands ≈ 0.843 (the thesis number), that confirms thesis entropy = 1e-2 and proves
  the winner-vs-legacy `both` gap (0.879 vs 0.843, ~+3.6 pts) is **pure training
  recipe** → the "interpretability cost was largely under-training" hypothesis holds.
- Minor residual: the active line is 1e-2 while a commented alternative is 1e-3
  (`target_entropy=log(2)`); if the legacy run does NOT reproduce ~0.843, revisit
  whether the thesis used 1e-3. Best evidence is 1e-2.

The original conclusions (winner recipe lifts hard cats; `none` +2.8 pts) stand.

---

## (superseded) ⚠️ original worry — entropy-weight mismatch — see RESOLVED above

The completed winner-recipe `ent` and `both` runs used `entropy_weight = 0.01`
(logged as `YesEnt0.01`). I initially read the thesis `YesEnt1` label as weight 1.0
(~100× stronger) and flagged the comparison as confounded. **This was incorrect** —
the thesis training code uses 1e-2 (see RESOLVED section). Kept for the record of how
the run-name label misled the analysis.

## What ran (and what's paused)

Full **winner-recipe d152 suite**: `{none, ent, bn1, both} × seeds {0,1,2}`, 30
epochs, MPS, cosine LR 1e-3→5e-7 with ramp warmup. All 12 finished
(2026-06-14, ~36–77 min each; mild thermal throttling late). **Paused before the 2
legacy comparability runs** for a machine cooldown (`d152_both_s0_LEGACY`,
`d20_ent_s0_LEGACY` remain — but see fix #3 above). `run_suite.sh` is resumable
(skips any log with a "done in" marker).

Overnight reboot note: the first launch (2026-06-13) was interrupted by an OS update
reboot after the 3 `d152_none` seeds; relaunched 2026-06-14 13:33 and completed the
rest.

## Results — winner-recipe d152, final epoch, mean over 3 seeds

| Condition | entropy wt | all | cat0 | cat1 | cat2 | cat3 | cat4 | cat5 |
|---|---|---|---|---|---|---|---|---|
| none | 0    | 0.8911 | 0.456 | 0.472 | 0.410 | 0.602 | 0.960 | 0.936 |
| bn1  | 0    | 0.8864 | 0.403 | 0.443 | 0.361 | 0.570 | 0.961 | 0.935 |
| ent  | 0.01 | 0.8842 | 0.410 | 0.444 | 0.366 | 0.573 | 0.958 | 0.931 |
| both | 0.01 | 0.8789 | 0.358 | 0.416 | 0.319 | 0.526 | 0.959 | 0.929 |

Seed spreads are tight (all: none 0.0011, ent 0.0052, bn1 0.0019, both 0.0053).
Cost vs unconstrained lives entirely in cats 0–3 (cats 4–5 untouched, ≤0.1 pt):
`both` overall −1.2 pts, but cat0 −9.7, cat2 −9.0, cat3 −7.6 (~20% relative). The
"nearly free" framing is true *on aggregate* (cats 4–5 dominate the event mix), not
in the categories the constraints actually affect.

Note: `ent`/`both` use `entropy_weight=0.01`, which **matches the thesis training
config** (`TrainLowLevelReconstruction.py:369`). (Earlier draft wrongly feared a
mismatch — see RESOLVED section at top.)

Curves had **not plateaued at epoch 30** (still gently rising) → mild under-training;
longer runs may narrow the gaps.

## Results — old (thesis-era, legacy recipe, 1 seed each) vs new

From `docs/run_configs/` wandb summaries (per category):

| Condition | recipe | entropy wt | all | cat0 | cat1 | cat2 | cat3 | cat4 | cat5 |
|---|---|---|---|---|---|---|---|---|---|
| none | old (legacy) | 0 | 0.8631 | 0.366 | 0.412 | 0.329 | 0.529 | 0.951 | 0.920 |
| none | new (winner) | 0 | **0.8911** | 0.456 | 0.472 | 0.410 | 0.602 | 0.960 | 0.936 |
| ent  | old `YesEnt1_NoBn` | ~1 | 0.8583 | 0.342 | 0.398 | 0.312 | 0.506 | 0.950 | 0.917 |
| ent  | old `YesEnt2_NoBn` | ~2 | 0.8630 | 0.362 | 0.413 | 0.334 | 0.521 | 0.952 | 0.920 |
| ent  | new | 0.01 | 0.8842 | 0.410 | 0.444 | 0.366 | 0.573 | 0.958 | 0.931 |
| bn1  | new | 0 | 0.8864 | 0.403 | 0.443 | 0.361 | 0.570 | 0.961 | 0.935 |
| both | old `YesEnt1_YesBn` = **THESIS** | ~1 | 0.8434 | 0.260 | 0.327 | 0.244 | 0.435 | 0.947 | 0.907 |
| both | old `YesEnt2_YesBn` | ~2 | 0.8541 | 0.305 | 0.382 | 0.299 | 0.467 | 0.950 | 0.915 |
| both | new (winner) | 0.01 | **0.8789** | 0.358 | 0.416 | 0.319 | 0.526 | 0.959 | 0.929 |

No legacy bottleneck-only (`NoEnt_YesBn`) run exists in the archive.

### Old-vs-new comparison (entropy weight matches; see RESOLVED at top)
- **`none` ↔ `none`** (no entropy, no bottleneck — only recipe + seeds differ):
  recipe effect = **+2.8 pts overall**, concentrated in hard cats (cat0 +8.9,
  cat2 +8.0, cat3 +7.3; cats 4–5 flat). The new recipe genuinely lifts the hard cats.
- **`both` ↔ thesis**: new 0.879 vs thesis 0.843 (+3.6). Both use `entropy_weight=1e-2`
  + bottleneck-1, so this is a recipe + (small) seed-vs-single-run effect — pending the
  `d152_both_s0_LEGACY` run (entropy 1e-2 + legacy recipe) to confirm it lands ≈ 0.843
  and pin the gap to the recipe.
- The "~1" / "~2" entropy-weight labels in the old-vs-new table are the **`YesEnt1` /
  `YesEnt2` run-name indices, NOT weights** — actual reconstruction entropy weight is
  1e-2 (active) per the training code. Treat those column entries as labels only.

## Reproduce

- Suite logs: `tmp_suite/d152_*.log` (per-epoch val dicts; full dict every epoch).
- Checkpoints + per-run `config.json`: `output/2026061[34]-*_TrainingOutput/`.
  (entropy weight confirmed from `config.json`: `"entropy_weight": 0.01`.)
- Parser + plots: `experiments/suite_tradeoff_plots.py`
  → `tmp_plots/suite_d152_overall.png`, `..._by_category.png`,
  `..._final_by_category.csv`. Re-runs in seconds; picks up legacy runs when present.
- Old numbers: `docs/run_configs/*DSSARVTSBN3*.json` (wandb summaries).

## Learnings

- **`run_suite.sh` `ent`/`both` use the trainer default `entropy_weight=1e-2`, which
  MATCHES the thesis** (`TrainLowLevelReconstruction.py:369`). No `--entropy-weight`
  override is needed for a thesis comparison.
- **Don't trust run-name labels as hyperparameters.** `YesEnt1`/`YesEnt2` are old
  index labels, not weights; reading "1" as weight=1.0 (via the new `YesEnt{weight:g}`
  convention) sent a whole turn down a false "100× mismatch" path. Check the actual
  training code / config, not the name.
- **Archived wandb JSONs are summary-metrics only** — the `config` block is empty, so
  thesis hyperparameters cannot be recovered from them; use the training code.
- The bottleneck setting *does* round-trip: new `config.json` `bottleneck_attention=1`
  matches the thesis `YesBn`.
- Per-category > aggregate for the interpretability-cost story: aggregate hides the
  ~20% relative hit in cats 0–3 because cats 4–5 saturate and dominate the mix.
- 30 epochs under-trains slightly at d152 under the winner recipe (curves still rising).

# 2026-06-15 — d152 suite results + ENTROPY-WEIGHT MISMATCH (next to-do)

## ⚠️ NEXT TO-DO (read first if resuming — the current comparison is confounded)

The completed winner-recipe `ent` and `both` runs used **`entropy_weight = 0.01`**
(the trainer default; logged as `YesEnt0.01` in their config.json). The **thesis
model is `YesEnt1`** — almost certainly entropy weight **1.0, i.e. ~100× stronger**.
So the headline "new constrained model beats the thesis model / interpretability is
nearly free / the thesis cost was just under-training" is **NOT yet supported** — the
`ent`/`both` arms are barely penalised compared with the thesis.

**Do this next:**

1. **Confirm the thesis entropy weight.** It could NOT be recovered from the wandb
   archive (the JSONs in `docs/run_configs/` are summary-metrics only; the `config`
   block is empty — `{"magic": ...}`). Get it from the thesis text / original
   training code / Sid's memory. Working assumption: `YesEnt1` = `entropy_weight 1.0`.
2. **Re-run the `ent` and `both` arms at the thesis weight**, e.g.
   `--entropy-weight 1` (×3 seeds each, d152, winner recipe). Then the old-vs-new
   `both` comparison becomes clean and actually answers "how much of the thesis
   interpretability cost was under-training vs the constraint itself."
3. **The queued legacy comparability runs need the same fix.** As written,
   `run_suite.sh` builds `d152_both_s0_LEGACY` with the default entropy (0.01), so it
   will **not** reproduce the thesis 0.843 and won't adjudicate the under-training
   hypothesis. Either add `--entropy-weight 1` to the LEGACY lines or interpret them
   only as "recipe effect at weak entropy."

What still stands without caveat: the **`none`** (unconstrained) old-vs-new
comparison is clean, and the **`bn1`** result (bottleneck setting matches the thesis:
`bottleneck_attention = 1`).

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

⚠ Caveat: `ent`/`both` here use entropy 0.01, so their cost is an underestimate of
the thesis-strength penalty. See NEXT TO-DO.

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

### Clean vs confounded
- **`none` ↔ `none` is clean** (only recipe + seeds differ): recipe effect = **+2.8
  pts overall**, concentrated in hard cats (cat0 +8.9, cat2 +8.0, cat3 +7.3; cats 4–5
  flat). Trustworthy result: the new recipe genuinely lifts the hard categories.
- **`both` ↔ thesis is confounded**: new 0.879 vs thesis 0.843 (+3.6) mixes the
  better recipe AND a ~100× weaker entropy penalty. Cannot attribute to recipe.
- **`ent` rows** similarly mix recipe + weaker penalty.

## Reproduce

- Suite logs: `tmp_suite/d152_*.log` (per-epoch val dicts; full dict every epoch).
- Checkpoints + per-run `config.json`: `output/2026061[34]-*_TrainingOutput/`.
  (entropy weight confirmed from `config.json`: `"entropy_weight": 0.01`.)
- Parser + plots: `experiments/suite_tradeoff_plots.py`
  → `tmp_plots/suite_d152_overall.png`, `..._by_category.png`,
  `..._final_by_category.csv`. Re-runs in seconds; picks up legacy runs when present.
- Old numbers: `docs/run_configs/*DSSARVTSBN3*.json` (wandb summaries).

## Learnings

- **`run_suite.sh` `ent`/`both` rely on the trainer default `entropy_weight=1e-2`**,
  which is NOT the thesis setting. Any thesis-comparison run must pass
  `--entropy-weight 1` (pending confirmation of the thesis value). This is the trap
  that silently confounded the first comparison.
- **Archived wandb JSONs are summary-metrics only** — the `config` block is empty, so
  thesis hyperparameters (entropy weight, block count, LR) cannot be recovered from
  them. Need the thesis text / original code for those.
- The bottleneck setting *does* round-trip: new `config.json` `bottleneck_attention=1`
  matches the thesis `YesBn`.
- Per-category > aggregate for the interpretability-cost story: aggregate hides the
  ~20% relative hit in cats 0–3 because cats 4–5 saturate and dominate the mix.
- 30 epochs under-trains slightly at d152 under the winner recipe (curves still rising).

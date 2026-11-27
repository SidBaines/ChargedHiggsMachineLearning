# Four-angle reconstruction confirmation

Exact reconstruction improved from **88.445% to 89.171%** on 136,039 held-out signal events. The paired signed MC-weighted gain was **+0.726 percentage points** (SE 0.121 pp, 95% CI +0.489 to +0.963 pp). This independently confirms the direction of the exploratory residue-6 result (88.581% to 89.125%). Four-angle averaging rescued 1,868 events and regressed 1,006. It changed at least one nonpadding object assignment in 7,285 events.

## Sample and estimation

The frozen width-152 joint checkpoint was `20260624-163646/chkpt29_85320.pth`. The baseline predicted at zero rotation. The comparison averaged per-object softmax probabilities at 0, π/2, π and 3π/2 before taking the argmax. All transverse momenta, including the missing-momentum object, rotated together. Reconstruction counts exact agreement with truth across every nonpadding object after merging leptonic and hadronic W labels.

This confirmation used signal from row residue 4 modulo 8, with arbitrary sampler seed 8147. Of 140,001 sampled candidates, 3,962 also appeared in the earlier single-angle flip audit (9,999 signal events, requested 10,000, seed 2701). All those rows were excluded. The analyzed events have zero overlap with that audit and with the exploratory residue-6 four-angle study. The confirmation includes 49,807 negative-weight events.

The paired estimate is the signed-weight mean of the per-event change in exact correctness. It targets the nonaudit residue-4 population. Within each DSID, the ratio’s influence uses the signed weight times the difference between the paired correctness change and its weighted mean, with an indicator for the evaluated slice. The variance comes from the within-DSID influence variance with a finite-population correction. The normal intervals are pointwise estimates of sampling uncertainty for this finite population. The sample is signal-only, so this confirmation establishes the reconstruction effect.

## Reconstruction by topology and signal mass

The first row is the primary comparison. The slice intervals are unadjusted for multiple comparisons. Rescues and regressions count raw events.

| Slice | Events | Native (%) | Four angles (%) | Difference in pp [95% CI] | Rescues | Regressions |
|---|---:|---:|---:|---:|---:|---:|
| All signal | 136,039 | 88.445 | 89.171 | +0.726 [+0.489, +0.963] | 1,868 | 1,006 |
| 0 · resolved H, resolved W | 4,331 | 44.923 | 45.127 | +0.203 [-2.387, +2.793] | 212 | 126 |
| 1 · resolved H, leptonic W | 8,532 | 49.183 | 50.121 | +0.938 [-0.698, +2.573] | 217 | 161 |
| 2 · resolved H, boosted W | 1,752 | 40.237 | 47.471 | +7.234 [+1.910, +12.558] | 61 | 43 |
| 3 · boosted H, resolved W | 6,348 | 60.354 | 62.215 | +1.861 [+0.023, +3.698] | 229 | 148 |
| 4 · boosted H, leptonic W | 81,909 | 95.306 | 95.645 | +0.339 [+0.111, +0.566] | 666 | 312 |
| 5 · boosted H, boosted W | 33,167 | 92.121 | 93.372 | +1.251 [+0.781, +1.721] | 483 | 216 |
| 0.8 TeV | 12,111 | 78.412 | 79.577 | +1.165 [+0.103, +2.227] | 322 | 172 |
| 0.9 TeV | 12,898 | 81.571 | 82.706 | +1.135 [+0.127, +2.143] | 237 | 154 |
| 1.0 TeV | 13,413 | 85.118 | 85.369 | +0.250 [-0.570, +1.070] | 248 | 123 |
| 1.2 TeV | 13,880 | 86.958 | 87.433 | +0.475 [-0.317, +1.267] | 224 | 110 |
| 1.4 TeV | 14,011 | 88.847 | 90.095 | +1.249 [+0.498, +1.999] | 200 | 113 |
| 1.6 TeV | 14,192 | 89.201 | 89.927 | +0.727 [+0.035, +1.419] | 168 | 92 |
| 1.8 TeV | 14,038 | 92.213 | 92.922 | +0.709 [+0.087, +1.331] | 145 | 70 |
| 2.0 TeV | 14,010 | 92.588 | 92.776 | +0.188 [-0.418, +0.793] | 118 | 80 |
| 2.5 TeV | 13,887 | 92.658 | 93.436 | +0.778 [+0.207, +1.348] | 97 | 63 |
| 3.0 TeV | 13,599 | 94.763 | 95.458 | +0.695 [+0.172, +1.218] | 109 | 29 |

All six topology and ten mass point estimates are positive. The intervals include zero for topologies 0 and 1 and at 1.0, 1.2 and 2.0 TeV. Topology 2 has the largest estimated gain and the fewest events.

## Reproduction

Run from the worktree root.

```sh
source_repo=/Users/sidbaines/Documents/PhD/20260608_ChargedHiggsMachineLearning_heppc-work
MPLCONFIGDIR=tmp_readouts_mpl tmp_physics_env/bin/python \
  experiments/physics_rotation_confirmation.py \
  "$source_repo/tmp_data_20250321v1_all" \
  "$source_repo/output/20260624-163646_TrainingOutput" \
  output/physics_interp/rotation_average_confirmation_reproduced.json \
  --candidate-n 140000 --seed 8147 --audit-n 10000 --audit-seed 2701 --batch 512
```

- [Confirmation script](../../experiments/physics_rotation_confirmation.py)
- [Confirmation sample counts and numerical results](../results/2026-11-27/rotation_average_confirmation.json)
- [Exploratory four-angle results](../results/2026-11-27/rotation_average.json)
- [Earlier frozen single-angle audit](../results/2026-11-27/rotation_frozen.json)

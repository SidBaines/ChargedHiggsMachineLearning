# Compact joint students and scalar substitution — 27 November 2026

Soft teacher targets improved absolute-yield-weighted AUC and signed perfect reconstruction on the expanded confirmation sample in all three paired seeds. They reduced hard event-label accuracy on both pilot comparison splits. The students remain roughly 6–8 percentage points behind the teacher in signed perfect reconstruction. Reconstruction cross-entropy was still falling in five of the six arms between 1,200 and 1,600 steps, with a near-flat sixth curve. The training budget and smaller supervision sample leave the effect of rank one on asymptotic capacity unresolved.

## Paired training

Both arms used the same joint `TestNetwork` with residual width 48, three blocks, four heads per block, width-96 MLPs and 61,302 stored parameters, including unused legacy bottleneck biases. Each head sends one scalar per query token before expansion back into the residual stream. The attention forward hooks were active throughout training and inference. Relative to the repository’s legacy hook, the maximum initial logit difference was 2.38e-7 for reconstruction and 1.19e-7 for event classification.

Every arm saw the same 48,004 unique events from row residue 1 modulo 8, with matched initialization and batch order within each seed. The model took 1,600 updates with batch size 128, AdamW, gradient norm clipping at 1, weight decay 1e-6, a 30-update learning-rate ramp, and cosine decay from 0.002 to 0.0004. The labels-only arm summed mean object cross-entropy over non-padding tokens of signal events and mean three-class event cross-entropy over all events. The soft arm used half that joint cross-entropy plus half the temperature-squared joint KL against the frozen width-152 joint teacher at temperature 2, with the same head-specific inclusion rules. Optimization used equal event weights. Evaluation uses signed expected-yield weights for reconstruction and tails, and absolute expected-yield weights for AUC.

The 80-update throughput benchmark used a tiny real-data draw. The pilot used 15,002 events per residue and logged common checkpoints at 0, 1, 160, 400, 800, 1,200 and 1,600 updates. At the final checkpoint, the soft-minus-labels change in signed perfect reconstruction was −0.75, +1.84, and +2.29 percentage points on residue 4 and +1.36, +1.89, and +1.29 on residue 6, in seed order. The corresponding approximate paired MC standard errors were 1.24, 1.00, and 1.21 percentage points on residue 4 and 1.18, 0.99, and 1.19 on residue 6. On both splits all six soft-versus-hard comparisons improved weighted AUC and decreased hard event-label accuracy. The logged hard event loss fluctuated at late checkpoints. [Learning curves](../results/2026-11-27/students/training.json) · [Paired pilot deltas](../results/2026-11-27/students/paired_differences.json) · [Pilot evaluation by seed](../results/2026-11-27/students/evaluation.json)

## Larger held-out tail evaluation

Calibration residue 2 and confirmation residue 6 each contain 355,000 background and 15,999 signal events and exactly match the readout experiment’s sample IDs and weights. They are disjoint. The expanded samples overlap the corresponding small pilot draws by 1,789 and 1,643 events, respectively. The expanded evaluation reused the fixed models, and each arm’s native reconstruction determines its validity and reconstructed channel. The event discriminant is signal log-sum-exp minus the background logit. A separate signed-yield cut was fit on calibration background per reconstructed channel and applied unchanged to confirmation. All cuts bracketed 200 on calibration.

| Arm | Perfect reco | Weighted AUC | Calibration B lvbb / qqbb | Confirmation B lvbb | Confirmation B qqbb | Confirmation S lvbb | Confirmation S qqbb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher | 88.09% | 0.9896 | 197.6 / 198.4 | 242.4 ± 39.2 (880) | 176.1 ± 34.2 (1259) | 122393 ± 6420 | 55972 ± 4194 |
| 2701 labels | 80.57% | 0.9673 | 199.4 / 200.0 | 155.3 ± 35.2 (270) | 167.8 ± 38.9 (679) | 68965 ± 4801 | 33802 ± 3229 |
| 2701 soft + labels | 81.63% | 0.9778 | 197.4 / 202.8 | 266.1 ± 45.0 (612) | 177.2 ± 36.1 (1033) | 88862 ± 5486 | 38883 ± 3508 |
| 2702 labels | 80.82% | 0.9683 | 200.1 / 199.5 | 160.9 ± 34.0 (283) | 173.6 ± 36.2 (571) | 64180 ± 4700 | 33549 ± 3182 |
| 2702 soft + labels | 81.69% | 0.9749 | 198.8 / 200.0 | 219.2 ± 40.7 (410) | 196.2 ± 39.0 (1020) | 80551 ± 5233 | 39002 ± 3550 |
| 2703 labels | 80.36% | 0.9693 | 197.0 / 200.7 | 116.7 ± 29.3 (235) | 203.9 ± 38.2 (714) | 68049 ± 4698 | 32612 ± 3235 |
| 2703 soft + labels | 81.23% | 0.9791 | 195.5 / 197.4 | 212.5 ± 42.4 (542) | 223.1 ± 40.5 (1237) | 82272 ± 5262 | 37807 ± 3495 |

Uncertainties are square roots of the selected sum of squared event weights and condition on the calibrated cuts. Parentheses give raw confirmation background counts. In seed order, distillation increased qqbb signal by 5,081 ± 1,972, 5,453 ± 1,973 and 5,194 ± 1,994 at signed background changes +9 ± 34, +23 ± 30 and +19 ± 35. Those paired uncertainties use the events selected by just one arm. The lvbb signal gains of 19,898 ± 2,909, 16,371 ± 2,780 and 14,223 ± 2,823 were accompanied by background changes +111 ± 34, +58 ± 35 and +96 ± 37, leaving the matched-background lvbb gain unresolved. The frozen teacher retained more signal in both channels. [Expanded tail yields and all six truth topologies](../results/2026-11-27/students_sb200/results.json)

Expected signal yields by mass, rounded to the nearest event, require that the event’s reconstructed and true channels agree.

| Reco channel | Mass (GeV) | Teacher | 2701 labels | 2701 soft + labels | 2702 labels | 2702 soft + labels | 2703 labels | 2703 soft + labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lvbb | 800 | 4448 | 257 | 487 | 246 | 206 | 344 | 158 |
| lvbb | 900 | 6660 | 132 | 242 | 1014 | 1079 | 145 | 814 |
| lvbb | 1000 | 7115 | 2451 | 3424 | 1450 | 2735 | 2019 | 3282 |
| lvbb | 1200 | 6040 | 2241 | 3776 | 1604 | 2489 | 2034 | 2743 |
| lvbb | 1400 | 8413 | 3603 | 4510 | 3102 | 3972 | 3322 | 4055 |
| lvbb | 1600 | 15054 | 9143 | 10359 | 9095 | 9757 | 8316 | 10525 |
| lvbb | 1800 | 17349 | 10756 | 12494 | 10995 | 12150 | 10605 | 11340 |
| lvbb | 2000 | 17654 | 11015 | 14560 | 10588 | 12907 | 11523 | 12895 |
| lvbb | 2500 | 18923 | 13281 | 19317 | 11616 | 16275 | 13566 | 17138 |
| lvbb | 3000 | 20738 | 16087 | 19692 | 14469 | 18979 | 16174 | 19320 |
| qqbb | 800 | 1048 | 132 | -68 | 246 | -120 | 292 | 283 |
| qqbb | 900 | 2136 | -271 | -322 | 78 | 123 | -51 | -462 |
| qqbb | 1000 | 3251 | 1646 | 1578 | 1522 | 2016 | 1648 | 1613 |
| qqbb | 1200 | 2838 | 1570 | 2752 | 1647 | 1849 | 1685 | 2031 |
| qqbb | 1400 | 4828 | 1767 | 2173 | 1218 | 1720 | 1348 | 2000 |
| qqbb | 1600 | 7862 | 5154 | 6698 | 5773 | 6870 | 5613 | 6514 |
| qqbb | 1800 | 7742 | 4665 | 5906 | 4243 | 6165 | 4321 | 4735 |
| qqbb | 2000 | 8359 | 5158 | 6188 | 5215 | 6030 | 4231 | 5158 |
| qqbb | 2500 | 7851 | 5522 | 5737 | 5477 | 5290 | 6118 | 6412 |
| qqbb | 3000 | 10057 | 8461 | 8242 | 8129 | 9059 | 7407 | 9523 |

Signed perfect reconstruction within each truth topology on the expanded confirmation sample.

| Truth topology | Teacher | 2701 labels | 2701 soft + labels | 2702 labels | 2702 soft + labels | 2703 labels | 2703 soft + labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H resolved / W resolved | 46.7% | 8.8% | 6.9% | 8.8% | 9.2% | 4.7% | 9.2% |
| H resolved / W leptonic | 41.9% | 9.3% | 19.6% | 10.0% | 10.7% | 12.4% | 21.2% |
| H resolved / W merged | 31.6% | 4.9% | 6.7% | 6.6% | 5.1% | -0.2% | 7.8% |
| H merged / W resolved | 54.8% | 18.8% | 18.4% | 19.0% | 29.1% | 30.4% | 23.5% |
| H merged / W leptonic | 94.7% | 92.4% | 92.8% | 91.7% | 92.3% | 90.7% | 91.0% |
| H merged / W merged | 94.1% | 88.3% | 89.9% | 90.9% | 90.8% | 89.6% | 91.3% |

The largest deficits are in resolved topologies. The −0.2% estimate in the labels-only seed 2703 is a signed-weight ratio with a negative selected numerator.

## Formula substitution

The discovery student was the minimum summed hard validation loss among seeds 2701 and 2702 over both arms. Within that student, the first 1,072 development events selected the block and head with the largest rise in summed hard reconstruction and event cross-entropy under replacement by the scalar’s mean. A disjoint subset of 3,072 development events fitted an additive expression with at most 12 terms from a fixed physics basis using orthogonal matching pursuit. The same algorithm was applied to the corresponding distilled arm and independently to both objectives in held-out seed 2703. Each fitted formula replaces the scalar at every query position before expansion and subsequent layers. Identity replay reproduced both heads exactly. [All formulas, ablated heads and per-dose results](../results/2026-11-27/students/formula.json)

| Arm | Block/head (zero based) | Held-out residue | Scalar R² | Whole assignment agreement | Event label agreement | Signed perfect reco, intact / formula / scalar mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2701 labels | 1 / 2 | 4 | 0.754 | 78.10% | 97.46% | 80.84% / 77.47% / 46.54% |
| 2701 labels | 1 / 2 | 6 | 0.756 | 78.52% | 97.46% | 82.57% / 78.18% / 47.10% |
| 2703 labels | 2 / 0 | 6 | 0.847 | 77.38% | 96.53% | 80.79% / 75.97% / 72.72% |
| 2701 soft + labels | 0 / 0 | 4 | 0.567 | 76.69% | 95.22% | 80.10% / 80.68% / 79.38% |
| 2701 soft + labels | 0 / 0 | 6 | 0.563 | 76.29% | 95.09% | 83.92% / 83.00% / 82.51% |
| 2703 soft + labels | 2 / 3 | 6 | 0.953 | 94.57% | 96.79% | 82.08% / 82.56% / 79.46% |

Each split provides 384 matched intervention events with at least two large-R jets above 150 GeV in transverse momentum and 40–150 GeV in mass. The jet closest to the W mass is edited at fixed three-momentum. Tag doses interpolate to a low- or high-quartile donor from a distinct event of the same jet type, matched in mass, pseudorapidity and log transverse momentum. The table shows means at four endpoints. Each cell lists the scalar change for intact / formula and the target-jet W probability change for intact / formula / constant-mean replacement. The rest of the dose series, other object and event classes, hard-label flips, scalar changes on unedited tokens, and zero-effect identities are in the linked outputs.

| Arm | Residue | Edit | Scalar delta intact / formula | W probability delta intact / formula / scalar mean |
| --- | ---: | --- | ---: | ---: |
| 2701 labels | 4 | 40 GeV | -0.0567 / 0.0000 | 0.0804 / 0.0823 / 0.0909 |
| 2701 labels | 4 | 150 GeV | -0.0073 / 0.0000 | -0.1079 / -0.1064 / -0.1217 |
| 2701 labels | 4 | Low tag | 0.1525 / 0.1532 | 0.1666 / 0.1583 / 0.1806 |
| 2701 labels | 4 | High tag | -0.2804 / -0.1945 | -0.2640 / -0.2538 / -0.2775 |
| 2701 labels | 6 | 40 GeV | -0.0497 / 0.0000 | 0.0706 / 0.0702 / 0.0813 |
| 2701 labels | 6 | 150 GeV | -0.0027 / 0.0000 | -0.1015 / -0.1029 / -0.1166 |
| 2701 labels | 6 | Low tag | 0.1369 / 0.1454 | 0.1276 / 0.1232 / 0.1433 |
| 2701 labels | 6 | High tag | -0.2859 / -0.2023 | -0.2449 / -0.2325 / -0.2535 |
| 2703 labels | 6 | 40 GeV | 0.0831 / 0.3222 | 0.0592 / 0.0453 / 0.0774 |
| 2703 labels | 6 | 150 GeV | 0.1463 / 0.1662 | -0.0989 / -0.1070 / -0.1124 |
| 2703 labels | 6 | Low tag | 0.4193 / 0.4380 | 0.1143 / 0.1191 / 0.1678 |
| 2703 labels | 6 | High tag | -0.5228 / -0.5349 | -0.2330 / -0.2553 / -0.2949 |
| 2701 soft + labels | 4 | 40 GeV | 0.0031 / 0.0104 | 0.0802 / 0.0806 / 0.0806 |
| 2701 soft + labels | 4 | 150 GeV | -0.0486 / -0.0258 | -0.1070 / -0.1158 / -0.1289 |
| 2701 soft + labels | 4 | Low tag | -0.0712 / -0.0604 | 0.1561 / 0.1631 / 0.1316 |
| 2701 soft + labels | 4 | High tag | 0.0794 / 0.0767 | -0.2269 / -0.2628 / -0.2741 |
| 2701 soft + labels | 6 | 40 GeV | 0.0032 / 0.0099 | 0.0773 / 0.0745 / 0.0763 |
| 2701 soft + labels | 6 | 150 GeV | -0.0512 / -0.0264 | -0.1041 / -0.1032 / -0.1172 |
| 2701 soft + labels | 6 | Low tag | -0.0634 / -0.0574 | 0.1319 / 0.1342 / 0.0984 |
| 2701 soft + labels | 6 | High tag | 0.0841 / 0.0798 | -0.2229 / -0.2443 / -0.2587 |
| 2703 soft + labels | 6 | 40 GeV | -0.0154 / 0.0000 | 0.0606 / 0.0603 / 0.0583 |
| 2703 soft + labels | 6 | 150 GeV | 0.0488 / 0.0000 | -0.0919 / -0.0909 / -0.0906 |
| 2703 soft + labels | 6 | Low tag | -0.0002 / 0.0000 | 0.1361 / 0.1361 / 0.1419 |
| 2703 soft + labels | 6 | High tag | 0.0208 / 0.0000 | -0.2173 / -0.2169 / -0.2208 |

The best scalar fit was for distilled seed 2703. The formula preserved 94.57% of full assignments and signed truth accuracy was 0.48 percentage points higher. Its scalar stayed flat under every mass dose. At 150 GeV the intact scalar rose by 0.099 on lepton or missing-momentum tokens, 0.052 on large jets and 0.078 on small jets. The candidate W-probability drop was about 0.09 even under constant-mean replacement. In the labels-only confirmation seed the 40 GeV edit produced a substantially overstated scalar change, +0.322 versus +0.083. The distilled discovery formula retained some mass variation, with substantial errors by query type. The tag response varied by seed and query type. These discrepancies rule out these expressions as faithful accounts of the chosen scalars under the tested changes, even where the wider network’s response survived.

## Commands

Run from the worktree with the pinned Python environment. Set `DATA` to the full memmap directory and `JOINT` to the original width-152 training output directory. The large-tail command reuses the readout cache for the frozen teacher and verifies its sampled IDs and weights before scoring students. Student checkpoints and per-event arrays are generated locally under `output/physics_interp/students/` and `output/physics_interp/students_sb200/`.

```sh
MPLCONFIGDIR=output/physics_interp/mpl_cache tmp_physics_env/bin/python -u experiments/physics_students.py --data "$DATA" --teacher "$JOINT" \
  --benchmark --updates 80 --out output/physics_interp/students_benchmark
MPLCONFIGDIR=output/physics_interp/mpl_cache tmp_physics_env/bin/python -u experiments/physics_students.py --data "$DATA" --teacher "$JOINT" \
  --updates 1600 --train-signal 32000 --train-background 16000 --eval-signal 5000 --eval-background 10000
MPLCONFIGDIR=output/physics_interp/mpl_cache tmp_physics_env/bin/python -u experiments/physics_students.py --data "$DATA" --teacher "$JOINT" --formula-only
MPLCONFIGDIR=output/physics_interp/mpl_cache tmp_physics_env/bin/python -u experiments/physics_students_sb200.py --data "$DATA" \
  --teacher-cache tmp_readouts_cache
```

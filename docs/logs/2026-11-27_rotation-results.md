# Azimuthal rotations and joint training, 27 November 2026

A global azimuthal rotation changes every transverse momentum, including the missing-momentum object, while preserving their relative geometry. Tags, particle types, truth assignments and longitudinal components stay fixed. This is an input-space test. Detector material and acceptance can have real azimuthal structure.

## Frozen checkpoint and decision rule

The first audit used row residue 4, 9,999 signal events and approximately 35,000 background events. The metric that triggered training was the unweighted fraction of signal events with any non-padding object assignment flipped. Thresholds were determined independently on residue 2 for selection flips. The two control and assessment working points use the same predicted reconstruction validity and channel rules. A log-odds change measures a change in the signal-to-background score.

| Rotation | Assignment flip, S / B (%) | Selection flip, S / B (%) | Mean absolute log-odds change, S / B |
|---|---:|---:|---:|
| +0° | 0.00 / 0.00 | 0.00 / 0.00 | 0.000 / 0.000 |
| +45° | 6.21 / 36.06 | 5.08 / 0.25 | 0.456 / 0.886 |
| +90° | 7.74 / 43.41 | 6.93 / 0.33 | 0.602 / 1.182 |
| +180° | 8.47 / 47.34 | 7.95 / 0.36 | 0.702 / 1.368 |
| -90° | 7.49 / 43.64 | 6.64 / 0.32 | 0.601 / 1.191 |

## Matched d152 fine-tunes

All nine arms started from the same June checkpoint. Each fine-tune used 144,000 sampled odd rows (48k signal and 96k background requested), 1,600 updates at batch 1,024, Adam, peak learning rate 0.0002 and a cosine decay. These are short adaptations of the same pretrained network, with paired minibatch randomness across input treatments. They are not independent pretraining seeds. Both losses used the production DSID training factors with inverse inclusion corrections. Ordinary and augmented use absolute azimuth. Augmentation applies an independent uniform rotation to each event each time it is sampled. The relative representation encodes lepton-relative sine and cosine; initial sine weights are copied from the original absolute-angle input and initial cosine weights are zero. On this one-lepton event selection the lepton supplies the coordinate reference.

The evaluation used 30,000 signal events and 120,002 background events on residue 6, with 120,002 separate calibration background events on residue 2. The reference was re-evaluated on exactly those rows. AUC is measured on all events with absolute yield weights. Reconstruction percentages use signed signal yields. Rotation-flip rates are the unweighted mean over four angles (+45°, +90°, 180°, −90°). The reported background uncertainty is √Σw² at the calibrated cut and excludes cut uncertainty.

| Arm | Perfect reco (%) | AUC | Rotated signal / background assignment flips (%) | Test background lvbb / qqbb |
|---|---:|---:|---:|---:|
| Frozen joint | 88.581 | 0.98950 | 7.458 / 43.009 | 162 ± 55 / 80 ± 41 |
| 2701 ordinary | 86.698 | 0.98602 | 11.519 / 55.429 | 253 ± 72 / 185 ± 64 |
| 2701 augmented | 87.438 | 0.98774 | 7.328 / 40.417 | 264 ± 69 / 144 ± 56 |
| 2701 relative | 86.121 | 0.98523 | 0.000 / 0.000 | 197 ± 61 / 123 ± 50 |
| 2702 ordinary | 86.384 | 0.98604 | 11.474 / 55.307 | 277 ± 76 / 188 ± 64 |
| 2702 augmented | 87.635 | 0.98766 | 7.274 / 40.596 | 330 ± 83 / 147 ± 59 |
| 2702 relative | 86.441 | 0.98529 | 0.000 / 0.000 | 146 ± 52 / 60 ± 39 |
| 2703 ordinary | 86.563 | 0.98600 | 11.406 / 55.383 | 259 ± 74 / 76 ± 53 |
| 2703 augmented | 87.586 | 0.98775 | 7.369 / 40.550 | 224 ± 68 / 144 ± 57 |
| 2703 relative | 86.219 | 0.98515 | 0.001 / 0.001 | 109 ± 45 / 102 ± 49 |

## Frozen four-angle control

After inspecting the first ordinary fine-tune, we added an inference-only comparison on the same residue-6 rows. We averaged the object probability vectors and event class probability vectors over 0°, 90°, 180° and 270°, then selected objects, assigned the channel and calibrated the working point from the averaged outputs. The price is four forward passes. For rotation checks we sampled the same first 16,000 shuffled events per arm; +90° permutes the four orientations, while 0.37 radians tests an angle outside that set.

| Frozen arm | Perfect reco (%) | AUC | Signal / background flips at 90° (%) | Signal / background flips at 0.37 rad (%) | Test B, lvbb / qqbb |
|---|---:|---:|---:|---:|---:|
| native | 88.581 | 0.98950 | 7.20 / 43.12 | 3.80 / 27.09 | 162 ± 55 / 80 ± 41 |
| mean_four | 89.125 | 0.99031 | 0.00 / 0.00 | 1.65 / 14.21 | 189 ± 58 / 166 ± 58 |

## Topologies and mass-specific signal yields

| Arm | cat0 (%) | cat1 (%) | cat2 (%) | cat3 (%) | cat4 (%) | cat5 (%) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen joint | 43.57 | 35.90 | 30.80 | 56.07 | 95.28 | 94.28 |
| 2701 ordinary | 42.17 | 31.08 | 29.56 | 51.76 | 94.12 | 91.38 |
| 2701 augmented | 43.31 | 35.05 | 28.86 | 58.58 | 94.47 | 91.35 |
| 2701 relative | 29.89 | 29.26 | 17.87 | 55.83 | 93.41 | 92.18 |
| 2702 ordinary | 41.75 | 33.01 | 27.38 | 50.98 | 93.83 | 90.87 |
| 2702 augmented | 43.01 | 34.79 | 29.07 | 58.50 | 94.37 | 92.52 |
| 2702 relative | 29.11 | 26.00 | 18.36 | 52.66 | 94.50 | 91.91 |
| 2703 ordinary | 41.48 | 32.53 | 25.59 | 55.98 | 93.57 | 91.41 |
| 2703 augmented | 42.36 | 31.13 | 35.17 | 56.03 | 94.79 | 92.06 |
| 2703 relative | 28.47 | 30.00 | 19.39 | 52.17 | 94.02 | 91.73 |
| Frozen, four angles | 39.07 | 38.34 | 30.92 | 59.78 | 96.02 | 93.85 |

**lvbb, signed signal retained at the independently calibrated cut**. Columns give different signal hypotheses in TeV. Each cell reports yield ± √Σw², rounded to whole events.

| Arm | 0.8 | 0.9 | 1.0 | 1.2 | 1.4 | 1.6 | 1.8 | 2.0 | 2.5 | 3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen joint | 2832 ± 649 | 4059 ± 807 | 6373 ± 934 | 4862 ± 1075 | 8784 ± 1284 | 13857 ± 1486 | 14634 ± 1577 | 12138 ± 1671 | 20999 ± 1792 | 19985 ± 1852 |
| 2701 ordinary | 3268 ± 657 | 4736 ± 804 | 5713 ± 901 | 3970 ± 982 | 6867 ± 1126 | 10693 ± 1302 | 11384 ± 1368 | 9596 ± 1477 | 17517 ± 1632 | 16561 ± 1719 |
| 2701 augmented | 4460 ± 741 | 5175 ± 908 | 6456 ± 1026 | 5959 ± 1146 | 8086 ± 1279 | 13454 ± 1469 | 14826 ± 1576 | 12585 ± 1638 | 20204 ± 1777 | 19061 ± 1845 |
| 2701 relative | 3242 ± 667 | 4174 ± 811 | 6128 ± 914 | 7037 ± 1068 | 6983 ± 1255 | 14200 ± 1441 | 15272 ± 1559 | 11340 ± 1656 | 21725 ± 1819 | 20261 ± 1887 |
| 2702 ordinary | 3742 ± 656 | 4064 ± 795 | 5877 ± 875 | 4648 ± 977 | 6290 ± 1135 | 10003 ± 1306 | 11811 ± 1398 | 9935 ± 1498 | 17288 ± 1639 | 17415 ± 1739 |
| 2702 augmented | 4022 ± 716 | 4428 ± 864 | 6162 ± 970 | 5606 ± 1110 | 8007 ± 1269 | 13055 ± 1463 | 14743 ± 1545 | 12397 ± 1636 | 20962 ± 1785 | 20194 ± 1852 |
| 2702 relative | 2473 ± 595 | 2914 ± 729 | 4188 ± 812 | 6019 ± 995 | 6465 ± 1206 | 13802 ± 1402 | 15237 ± 1525 | 10909 ± 1620 | 20547 ± 1776 | 20081 ± 1862 |
| 2703 ordinary | 3219 ± 663 | 4489 ± 806 | 5638 ± 890 | 3844 ± 986 | 5498 ± 1138 | 10394 ± 1319 | 12429 ± 1394 | 10145 ± 1488 | 17631 ± 1644 | 17080 ± 1734 |
| 2703 augmented | 3504 ± 674 | 3944 ± 805 | 5482 ± 899 | 5032 ± 1028 | 7291 ± 1189 | 11965 ± 1363 | 14071 ± 1496 | 11612 ± 1574 | 19802 ± 1742 | 19585 ± 1815 |
| 2703 relative | 2078 ± 538 | 2747 ± 665 | 3670 ± 735 | 5635 ± 887 | 6602 ± 1083 | 11976 ± 1270 | 13065 ± 1415 | 9887 ± 1517 | 18799 ± 1716 | 19367 ± 1812 |
| Frozen, four angles | 3498 ± 698 | 4746 ± 859 | 6262 ± 1000 | 6005 ± 1156 | 9748 ± 1367 | 16203 ± 1579 | 16708 ± 1653 | 12974 ± 1738 | 21921 ± 1844 | 21465 ± 1900 |

**qqbb, signed signal retained at the independently calibrated cut**. Columns give different signal hypotheses in TeV. Each cell reports yield ± √Σw², rounded to whole events.

| Arm | 0.8 | 0.9 | 1.0 | 1.2 | 1.4 | 1.6 | 1.8 | 2.0 | 2.5 | 3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen joint | 988 ± 428 | 1510 ± 508 | 2521 ± 569 | 2535 ± 691 | 4096 ± 813 | 6177 ± 976 | 7201 ± 1045 | 7435 ± 1151 | 8736 ± 1272 | 9307 ± 1247 |
| 2701 ordinary | 725 ± 403 | 1454 ± 473 | 2063 ± 529 | 1827 ± 616 | 3432 ± 702 | 4785 ± 827 | 6398 ± 905 | 6348 ± 1017 | 7517 ± 1143 | 7885 ± 1132 |
| 2701 augmented | 725 ± 436 | 1673 ± 517 | 1543 ± 532 | 1953 ± 622 | 3791 ± 717 | 6339 ± 878 | 6357 ± 957 | 6717 ± 1040 | 7785 ± 1160 | 8802 ± 1156 |
| 2701 relative | 384 ± 368 | 959 ± 430 | 1517 ± 459 | 1574 ± 544 | 2390 ± 678 | 5439 ± 842 | 6028 ± 921 | 5924 ± 1018 | 7379 ± 1138 | 8578 ± 1166 |
| 2702 ordinary | 796 ± 429 | 1476 ± 508 | 1998 ± 567 | 2305 ± 673 | 3128 ± 727 | 5474 ± 881 | 6934 ± 952 | 6118 ± 1055 | 7680 ± 1189 | 8594 ± 1184 |
| 2702 augmented | 740 ± 427 | 1233 ± 486 | 1480 ± 513 | 1829 ± 618 | 3111 ± 674 | 6241 ± 882 | 6807 ± 961 | 6582 ± 1036 | 7819 ± 1186 | 8459 ± 1156 |
| 2702 relative | 797 ± 343 | 308 ± 347 | 1349 ± 424 | 1480 ± 535 | 2379 ± 651 | 4457 ± 803 | 5495 ± 888 | 5381 ± 989 | 6521 ± 1112 | 7589 ± 1121 |
| 2703 ordinary | 754 ± 412 | 1401 ± 452 | 1672 ± 513 | 1472 ± 614 | 2729 ± 677 | 5074 ± 811 | 5860 ± 898 | 5354 ± 987 | 7088 ± 1147 | 8413 ± 1106 |
| 2703 augmented | 994 ± 434 | 1216 ± 516 | 1392 ± 529 | 2126 ± 618 | 3284 ± 694 | 5755 ± 857 | 5594 ± 938 | 6469 ± 1020 | 7478 ± 1152 | 8537 ± 1139 |
| 2703 relative | 971 ± 349 | 355 ± 375 | 1555 ± 486 | 1502 ± 530 | 2857 ± 678 | 5127 ± 833 | 6155 ± 901 | 5717 ± 991 | 6815 ± 1118 | 7501 ± 1132 |
| Frozen, four angles | 1435 ± 501 | 1623 ± 613 | 3560 ± 663 | 3320 ± 782 | 5159 ± 908 | 7121 ± 1050 | 8496 ± 1136 | 8797 ± 1221 | 8942 ± 1326 | 9964 ± 1307 |

## Reproduction

The executable scripts take paths to the full background-inclusive memmaps and the June joint checkpoint directory. A fresh output directory is required for changed training settings. The sampler seed is 2701. Versioned scripts use the repo’s pinned dependencies.

~~~sh
python experiments/physics_rotation.py "$DATA" "$JOINT" --signal 10000 --background 35000 --out output/physics_interp/rotation_frozen.json
python experiments/physics_rotation_train.py "$DATA" "$JOINT" --out output/physics_interp/rotation_train --device mps --steps 1600 --batch 1024
python experiments/physics_rotation_average.py "$DATA" "$JOINT" output/physics_interp/rotation_average.json
~~~

Full-precision results and per-angle responses are available for the [fine-tunes](../results/2026-11-27/rotation_train/), [frozen single-angle audit](../results/2026-11-27/rotation_frozen.json) and [frozen four-angle comparison](../results/2026-11-27/rotation_average.json). Fine-tuning writes checkpoints locally under `output/physics_interp/rotation_train/`.

The [separate four-angle confirmation](2026-11-27_rotation-confirmation.md) used 136,039 signal events disjoint from the residue-6 comparison and the original frozen rotation audit. The source-level duplicate check and its effect on the cut comparisons are documented in the [session report](2026-11-27_physics-interpretable-networks.md).

# Frozen paired jet interventions

Assessment comprised 200,001 events. The fixed selection yielded 37,204 ambiguous-jet events, 34,822 of which had all three matched donor jets. The separate geometry selection yielded 6,480 events. See the [paired diagnostics](../results/2026-11-27/interventions/paired.json) for thresholds and counts. Source IDs are in the local per-event arrays under `output/physics_interp/interventions/`.

## Jet edits

Unweighted per-event means. W-margin changes are in logits. Swap, verdict flip and signal columns are percentage points. The joint-minus-reco gap is event-paired with an approximate 95% interval. Leptonic flips are measured among events with exactly one lepton and one neutrino.

| Arm | N | Joint Δ W margin | Reco Δ W margin | Joint − reco difference [95% CI] | Joint winner swap | Reco winner swap | Joint leptonic flip | Reco leptonic flip | Joint Δ signal p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 37,204 | +0.000 | +0.000 | +0.000 [+0.000, +0.000] | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| mass_half | 37,204 | +0.748 | +0.910 | -0.162 [-0.176, -0.148] | +4.271 | +3.825 | +2.054 | +1.981 | +0.791 |
| mass_full | 37,204 | +1.628 | +2.082 | -0.454 [-0.478, -0.430] | +8.986 | +9.268 | +3.631 | +3.779 | +0.677 |
| pt_down | 37,204 | -0.620 | -0.742 | +0.122 [+0.106, +0.137] | +4.879 | +3.836 | +3.486 | +3.395 | -2.574 |
| pt_up | 37,204 | +0.790 | +0.968 | -0.178 [-0.194, -0.163] | +5.042 | +3.992 | +3.814 | +3.594 | +2.583 |
| tag_neutral | 34,822 | +0.000 | -0.000 | +0.000 [-0.000, +0.001] | +0.146 | +0.126 | +0.037 | +0.052 | +0.001 |
| tag_down | 34,822 | +0.790 | +1.125 | -0.335 [-0.349, -0.321] | +5.296 | +4.948 | +2.237 | +2.234 | -2.336 |
| tag_up | 34,822 | -0.749 | -1.134 | +0.385 [+0.371, +0.399] | +5.293 | +4.684 | +2.481 | +2.306 | +2.123 |

Tag matching used signed changes of the continuous Xbb score. Errors are mean absolute donor–target kinematic differences.

| Tag arm | Mean tag shift | Δ log pT | Δη | Δ mass (GeV) |
|---|---:|---:|---:|---:|
| tag_neutral | -0.0001 | 0.0310 | 0.0625 | 2.4917 |
| tag_down | -1.1784 | 0.0147 | 0.0294 | 1.1724 |
| tag_up | +1.1562 | 0.0154 | 0.0306 | 1.2077 |

## Geometry and head restoration

The geometry edit changes mean H–W proxy ΔR by -0.522 toward and +0.570 away. Both proxies match labelled H and hadronic W in 57.8% of the selected qqbb events. Rotating one jet also changes its absolute azimuth and its separation from the rest of the event.

Values below are percentages of the mean corruption error recovered. Object error is probability total variation averaged over real tokens. The event column uses absolute signal-probability error. Negative values indicate a larger error after restoration. Paired b1h0-minus-control confidence intervals are in the model JSON files.

| Arm | Joint object toward | Joint object away | Joint signal toward | Joint signal away | Reco object toward | Reco object away |
|---|---:|---:|---:|---:|---:|---:|
| geometry | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| identity_patch | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| b1_all | +53.07 | +43.09 | +43.76 | +39.92 | +65.74 | +56.47 |
| b0h0 | +11.56 | +9.61 | +9.18 | +8.56 | +5.76 | +3.16 |
| b0h1 | +7.76 | +7.63 | +6.57 | +6.43 | +1.66 | +0.95 |
| b0h2 | +5.24 | +2.90 | +3.42 | +3.48 | +18.61 | +23.43 |
| b0h3 | +17.54 | +22.72 | +17.98 | +22.89 | +10.91 | +6.24 |
| b1h0 | +36.00 | +23.60 | +29.86 | +20.12 | +3.58 | +4.76 |
| b1h1 | +2.82 | +2.62 | +1.79 | +0.26 | +16.73 | +13.09 |
| b1h2 | +11.58 | +10.35 | +7.05 | +13.50 | +2.30 | +3.82 |
| b1h3 | +1.09 | +3.69 | +2.28 | +3.69 | +44.46 | +33.04 |
| b2h0 | +14.20 | +9.38 | +4.93 | +6.46 | +3.90 | +6.54 |
| b2h1 | +8.06 | +7.61 | +7.96 | +5.19 | +10.47 | +7.52 |
| b2h2 | +29.65 | +25.63 | +20.07 | +17.95 | +33.16 | +27.96 |
| b2h3 | +5.95 | +7.73 | +9.27 | +8.65 | +11.08 | +5.11 |

## Inclusive baseline

Perfect reconstruction uses signed Monte Carlo weights and all legitimate signal topologies.

| Truth topology | N | Joint | Matched reco |
|---|---:|---:|---:|
| all_signal | 99,999 | 88.25% | 88.37% |
| cat0 | 3,134 | 43.70% | 46.16% |
| cat1 | 6,106 | 41.53% | 40.97% |
| cat2 | 1,278 | 38.06% | 37.12% |
| cat3 | 4,610 | 63.20% | 65.95% |
| cat4 | 60,388 | 95.65% | 95.39% |
| cat5 | 24,483 | 92.80% | 93.26% |

Joint native signal score calibrated independently per reconstructed channel at 200 expected background.

| Reco channel | Score cut | Calibration B | Assessment B ± √Σw² | Assessment B count |
|---|---:|---:|---:|---:|
| lvbb | 0.99908751 | 205.5 | 118.5 ± 52.5 | 151 |
| qqbb | 0.99846643 | 195.1 | 131.8 ± 52.0 | 250 |

| Mass (TeV) | lvbb yield ± √Σw² [count] | qqbb yield ± √Σw² [count] |
|---|---:|---:|
| 0.8 | 2630.80 ± 325.10 [549] | 1158.40 ± 238.69 [273] |
| 0.9 | 3330.14 ± 420.98 [883] | 1924.04 ± 283.91 [407] |
| 1 | 5252.58 ± 487.90 [1,198] | 2337.86 ± 324.76 [528] |
| 1.2 | 5945.89 ± 596.70 [1,806] | 3910.18 ± 389.69 [737] |
| 1.4 | 8862.07 ± 689.49 [2,424] | 4393.92 ± 459.25 [1,059] |
| 1.6 | 10694.94 ± 778.59 [3,085] | 5800.74 ± 536.61 [1,406] |
| 1.8 | 12541.13 ± 856.45 [3,624] | 6705.65 ± 580.90 [1,649] |
| 2 | 15457.28 ± 909.85 [4,198] | 7060.13 ± 602.49 [1,769] |
| 2.5 | 16628.77 ± 1001.83 [5,049] | 8089.87 ± 650.70 [2,099] |
| 3 | 21355.64 ± 1038.26 [5,459] | 6567.40 ± 676.05 [2,211] |

The checkpoint with zero event loss has an untrained event head. Both native repeated forward passes and both corrupted-context restoration controls were identical at the logit level.

## Exploratory truth breakdown of the edited jet

The mass ranking often picks a truth Higgs jet as the edit target. Truth is used here for evaluation only. The target W margin moves in the predicted direction within each truth group. Entries are unweighted joint / matched reco mean deltas in logits.

| Truth label | Mass/pT N | Tag N | Mass to 80.4 | pT up | Tag down | Tag up |
|---|---:|---:|---:|---:|---:|---:|
| Unmatched | 6,383 | 6,207 | +0.641 / +0.706 | +0.930 / +0.909 | +0.164 / +0.257 | -0.217 / -0.321 |
| Higgs | 15,065 | 13,670 | +2.823 / +3.809 | +0.632 / +1.031 | +1.291 / +1.890 | -1.046 / -1.680 |
| Hadronic W | 2,652 | 2,423 | +1.436 / +1.483 | +1.075 / +0.990 | +0.437 / +0.447 | -0.689 / -0.735 |


## Reproduction

Run the [intervention script](../../experiments/physics_interventions.py) with the full data directory, the final joint checkpoint directory from 24 June and the matched reconstruction-only directory from 27 June. The main assessment requested 100,000 signal and 100,000 background events with seed 2701. The pilot used smaller samples and has [separate numerical summaries](../results/2026-11-27/interventions_pilot/). All head restoration summaries are in the [joint](../results/2026-11-27/interventions/joint.json) and [reconstruction-only](../results/2026-11-27/interventions/reco.json) JSON files. Per-event arrays are generated locally under `output/physics_interp/interventions/` and `output/physics_interp/interventions_pilot/`.

~~~sh
python experiments/physics_interventions.py --data-dir "$DATA" --joint "$JOINT" --reco "$RECO" --n-signal 100000 --n-background 100000 --out output/physics_interp/interventions
~~~

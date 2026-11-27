# Frozen physics readout results, 27 November 2026

Experiment 4 tested whether explicit W/H reconstruction features can replace the event head of the width-152 joint model. The physics-only MLP reduced valid-event discrimination and signal retention, especially at lower masses. The pooled linear refit recovered native discrimination. Adding explicit physics features to pooled inputs gave mixed results across readout families and held-out splits.

## Setup

- Frozen joint model `20260624-163646`, final `chkpt29_85320.pth`. Fit/calibration/evaluation/confirmation are rows 0/2/4/6 modulo 8 within each DSID. The joint model trained on odd rows. The full fit sample contained 20,001 signal and 55,002 background events. Each other split contained 15,999 signal and 355,000 background events. Sampler seed 2701 and readout seeds 1421, 1422 and 1423 were implementation choices. All learned arms used the same splits and paired minibatch order within each full-run seed.
- The common validity gate and channel came from the frozen reconstruction argmax, using `utils.check_valid`. The three-class score is `logsumexp(lvbb, qqbb) - background logit`, which ranks events by signal probability. The 29 explicit inputs include reconstructed candidate masses, transverse momentum relative to jet HT, angles, tags, assignment margins and presence flags. Candidates and features use the same algorithm on signal and background. The pooled input is the native masked mean of the final object representations. A check on 128 held-out events reproduced the native forward exactly and changing truth left scores, channels and features unchanged.
- Fitted arms used the same three-class cross-entropy, inclusion-corrected absolute training weights with equal total signal/background fit weight, and AdamW with cosine decay. Features were standardized on fit events. The native head uses the original end-to-end training.
- For each arm and seed, calibration chose one fixed cut per reconstructed channel at the first crossing of 200 signed expected background, choosing the nearer of the two attainable yields. All masses share that channel cut. Signal yield requires matching reconstructed and true channels. Yields use signed MC weights scaled by inverse sampling fraction within DSID. AUC uses absolute yield weights and the tables report it on reco-valid events. The JSON also contains all-event and channel-sliced AUC. `sumw2` uses the squared scaled weights. Its square root quantifies MC variability for a fixed cut and excludes calibration-cut uncertainty.

## Pilot

The pilot used fit/calibration/evaluation only, one arbitrary seed (1421) and 20 fit epochs. It established feasibility. Sparse high-weight background made the nominal working point unreliable.

| Arm | Evaluation AUC | Calibration B lvbb / qqbb | Realized B lvbb / qqbb |
|---|---:|---:|---:|
| Native event head | 0.98719 | 85.3 / 346.9 | 1130.5 / 333.3 |
| Physics MLP | 0.97088 | 300.1 / 35.2 | 241.4 / 387.6 |
| Pooled MLP | 0.98259 | 241.7 / 35.7 | 836.1 / 321.2 |
| Physics + pooled MLP | 0.98347 | 279.0 / 77.0 | 263.1 / 352.8 |
| Wide physics MLP | 0.97169 | 321.5 / 87.6 | 240.5 / 416.5 |
| Pooled linear refit | 0.97534 | 98.1 / 97.4 | 86.4 / 747.7 |

## Full run and linear controls

AUC is the arithmetic mean over three fits where applicable. For the realized confirmation background columns the entry is **B (sumw2, raw n)** for the first listed seed, 1421. The native head is fixed. These representative numbers are from the same runs as the per-mass table below. All exact seed-level yields, cut values, background counts and uncertainties for both held-out splits are available in the linked tables.

| Arm | Parameters | Evaluation AUC | Confirmation AUC | Confirmation B lvbb (sumw2, n) | Confirmation B qqbb (sumw2, n) |
|---|---:|---:|---:|---:|---:|
| Native event head | 459 | 0.98642 | 0.98630 | 242.4 (1,534, 880) | 176.1 (1,168, 1,259) |
| Physics MLP | 11,982 | 0.97497 | 0.97529 | 272.7 (1,610, 1,317) | 185.8 (1,249, 1,304) |
| Pooled MLP | 11,859 | 0.97955 | 0.98021 | 170.1 (1,042, 766) | 186.1 (1,274, 1,084) |
| Physics + pooled MLP | 11,843 | 0.97946 | 0.97998 | 251.0 (1,672, 898) | 243.3 (1,684, 1,109) |
| Pooled linear refit | 459 | 0.98629 | 0.98667 | 218.5 (1,380, 862) | 202.0 (1,305, 1,408) |
| Wide physics MLP | 47,985 | 0.97503 | 0.97530 | 274.1 (1,618, 1,383) | 151.8 (1,012, 1,044) |
| Physics linear, post hoc | 90 | 0.96672 | 0.96740 | 225.5 (1,265, 1,475) | 201.6 (1,168, 3,178) |
| Physics + pooled linear, post hoc | 546 | 0.98639 | 0.98673 | 231.2 (1,428, 922) | 200.6 (1,328, 1,574) |

The two post hoc linear controls followed the finding that a refitted pooled linear head separated signal from background better than the larger pooled MLP. Physics + pooled linear reproduced pooled-linear AUC to within 0.0001 on both splits. The pooled MLP lowered training cross-entropy to about 0.064 versus 0.166 for pooled linear, with poorer held-out AUC. This is consistent with overfitting under the tested training recipe. Quadrupling the physics MLP’s parameter count changed confirmation AUC by about 0.000002. The available evidence covers this set of observables and these classifier families.

## Signal by charged-Higgs mass

Each row is a separate signal hypothesis. Interpret the expected signal yield per mass. Entries are signed S ± √sumw2 for the fixed native head and the physics MLP at seed 1421, with the same calibration cuts applied to evaluation and confirmation. The results tables contain the other arms and seeds.

| Mass (TeV) | Evaluation lvbb native / physics | Evaluation qqbb native / physics | Confirmation lvbb native / physics | Confirmation qqbb native / physics |
|---|---:|---:|---:|---:|
| 0.8 | 2,616 ± 954 / 352 ± 321 | 1,519 ± 623 / -230 ± 373 | 4,448 ± 1,092 / 732 ± 432 | 1,048 ± 607 / 407 ± 305 |
| 0.9 | 1,526 ± 1,269 / -38 ± 588 | 1,856 ± 847 / 1,161 ± 570 | 6,660 ± 1,283 / 1,759 ± 649 | 2,136 ± 788 / -203 ± 527 |
| 1.0 | 6,884 ± 1,437 / 2,549 ± 786 | 1,920 ± 888 / 460 ± 624 | 7,115 ± 1,456 / 3,390 ± 840 | 3,251 ± 838 / 1,563 ± 553 |
| 1.2 | 7,614 ± 1,599 / 3,488 ± 1,175 | 4,146 ± 1,029 / 1,063 ± 854 | 6,040 ± 1,679 / 2,930 ± 1,234 | 2,838 ± 1,018 / 776 ± 766 |
| 1.4 | 10,799 ± 1,874 / 5,921 ± 1,502 | 7,430 ± 1,294 / 4,558 ± 1,116 | 8,413 ± 1,919 / 4,195 ± 1,561 | 4,828 ± 1,171 / 4,016 ± 1,056 |
| 1.6 | 8,201 ± 2,115 / 4,677 ± 1,859 | 7,864 ± 1,438 / 6,890 ± 1,263 | 15,054 ± 2,227 / 8,670 ± 1,909 | 7,862 ± 1,439 / 7,360 ± 1,314 |
| 1.8 | 12,947 ± 2,287 / 9,687 ± 2,056 | 6,723 ± 1,557 / 5,387 ± 1,420 | 17,349 ± 2,359 / 13,441 ± 2,117 | 7,742 ± 1,495 / 5,229 ± 1,276 |
| 2.0 | 18,813 ± 2,372 / 14,651 ± 2,174 | 7,099 ± 1,687 / 5,833 ± 1,532 | 17,654 ± 2,415 / 13,718 ± 2,238 | 8,359 ± 1,627 / 6,656 ± 1,444 |
| 2.5 | 13,677 ± 2,622 / 11,201 ± 2,453 | 9,135 ± 1,678 / 8,722 ± 1,526 | 18,923 ± 2,565 / 16,002 ± 2,434 | 7,851 ± 1,791 / 6,224 ± 1,656 |
| 3.0 | 26,723 ± 2,682 / 22,391 ± 2,507 | 6,130 ± 1,750 / 5,877 ± 1,565 | 20,738 ± 2,612 / 19,612 ± 2,432 | 10,057 ± 1,824 / 10,106 ± 1,617 |

At these cuts the native evaluation background was 179.1 in lvbb (sumw2 1,139, n 858) and 141.2 in qqbb (770, n 1,270). For physics seed 1421 it was 208.4 (1,161, n 1,242) and 125.3 (805, n 1,275). Realized background and signal both varied across arms and seeds. This limits comparisons of small yield differences at a nominal common working point. Signed weights and low effective counts leave several low-mass signals unresolved, including qqbb at 0.9 TeV on confirmation for the physics MLP.

## Failure slices

The ambiguity split uses the smaller of the minimum assigned-object H and W logit margins. The unweighted median over valid fit events was 3.105. At 1.8 TeV on confirmation in qqbb, the native head retained 1,828 ± 696 and the physics MLP (1421) retained 829 ± 435 in the lower-margin group. In the upper-margin group the yields were 5,915 ± 1,323 and 4,400 ± 1,200, respectively. This example shows a larger proportional loss at lower margins and a larger absolute loss at higher margins. The main JSON provides this breakdown for every mass, channel and seed.

The 410xxx top group provided most of the retained background. For the native head on confirmation it contributed 198.5 of 242.4 lvbb and 142.7 of 176.1 qqbb. The corresponding physics seed 1421 contributions were 210.1 of 272.7 and 151.4 of 185.8. Other groups and individual DSID contributions are in the JSON.

Signed MC-weighted perfect object reconstruction for the shared joint checkpoint was 0.879 on evaluation and 0.881 on confirmation. On confirmation, categories 0–5 were 0.467, 0.419, 0.316, 0.548, 0.947 and 0.941. The category-0 evaluation estimate was 0.279. Each held-out sample had about 510 category-0 events and the signed estimates fluctuate. See the counts and sumw2 by truth topology in the results.

## Reproduction

Run from the worktree root in the pinned environment. `source_repo` is the original checkout containing the data and trained checkpoint. Use a fresh output directory if preserving the archived results.

```sh
source_repo=/Users/sidbaines/Documents/PhD/20260608_ChargedHiggsMachineLearning_heppc-work
checkpoint="$source_repo/output/20260624-163646_TrainingOutput/models/Nplits2_ValIdx0/chkpt29_85320.pth"
readout() {
  tmp_physics_env/bin/python -m experiments.physics_readouts \
    --data-dir "$source_repo/tmp_data_20250321v1_all" --ckpt "$checkpoint" \
    --device cpu --threads 2 --inference-batch 1024 --fit-batch 1024 \
    --sample-seed 2701 --cache-dir tmp_readouts_cache "$@"
}
readout --out output/physics_interp/readouts_reproduced.json \
  --n-fit 20000 55000 --n-calibration 16000 355000 --n-test 16000 355000 \
  --epochs 40 --seeds 1421 1422 1423
readout --out output/physics_interp/readouts_linear_controls_reproduced.json \
  --n-fit 20000 55000 --n-calibration 16000 355000 --n-test 16000 355000 \
  --epochs 40 --seeds 1421 1422 1423 --arms physics_linear physics_pooled_linear
readout --out output/physics_interp/readouts_pilot_reproduced.json --pilot \
  --n-fit 2500 6500 --n-calibration 2500 10000 --n-test 2500 10000 \
  --epochs 20 --seeds 1421
```

The archived pilot consumed the same random stream for weight initialization and shuffling. The current script uses an independent shuffle generator to align data order across architectures. It reproduces the pilot protocol with a different training order, so the archived pilot figures remain the record of that initial run.

- [Readout script](../../experiments/physics_readouts.py) and [shared sampler and calibration](../../experiments/physics_samples.py)
- [Pilot per-arm results](../results/2026-11-27/readouts_pilot.json)
- [Full per-seed results](../results/2026-11-27/readouts.json)
- [Post hoc linear control results](../results/2026-11-27/readouts_linear_controls.json)

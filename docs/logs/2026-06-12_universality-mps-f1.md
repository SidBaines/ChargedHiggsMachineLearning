# 2026-06-11/12 (night) — MPS on the M5, organism universality battery, F1 launch

Agreed plan (research-plan session line 2026-06-11): universality → faithfulness →
write-up, with MPS unlocked for training. This session: MPS validated end-to-end,
the full Stage-B/A4b battery rerun on the fresh organism, SB4 upgraded to
depth-relative blocks (new per-block resolution on BOTH models), the organism's
comparator localized at its operating point, F1 launched, and a second-seed
organism trained on MPS.

## MPS on the M5: VALIDATED (the old-Mac crash does not reproduce)

- Basic ops + `nn.MultiheadAttention` with `key_padding_mask`: clean.
- **Thesis-model forward parity vs CPU (with the bottleneck hook): max logit dev
  6.5e-5, argmax agreement 1.000000.**
- `train_organism.py --smoke --device mps`: 50 steps, loss 1.01→0.71, all finite.
- Full 30-epoch run launched (`--seed 1`, new flag): healthy through training,
  ~2x CPU speed with eval included (epoch ~3 min vs ~5 CPU); val tracking the
  seed-0 run (0.799 @ ep10 vs 0.814 final seed-0). Output
  `output/20260611-234355_TrainingOutput` (+ a `234100` false-start dir from the
  same minute — ignore). **MPS is now the default training device on this Mac**;
  keep the existing NaN guards.

## Universality battery: organism (ent1-d20-2blk, fresh ckpt 0.8142) vs thesis

Same scripts, same fresh slice (val batches 19-24). `a4b` was parameterized
(--model/--skip/--batches + BN; also fixed its REPO path from the tmp promotion);
`stageb_sb4` rewritten **depth-relative** (last block vs earlier; per-block + all
per-head rows) — reran on both models.

### UNIVERSAL (same structure on both models)

| property | thesis | organism |
|---|---|---|
| cross-system verdict = graded sigmoid in pT(cand)/pT(lepW) (A4b) | crossing r≈1.4 | crossing r≈1.25, sharper |
| broadcast tight: P(lep=W \| ins claims) vs (no claim) | 0.06 / 0.96 | 0.10 / 0.92 |
| "30% insertion flip" = donor composition (quartile sweep) | 0.001→0.83 | 0.000→0.85 |
| candidacy constitutively relational: KO early W→ctx kills claims | 0.000 | 0.026 |
| candidacy context read mostly at block 0 (solo KO b0 alone) | 0.002 | (early=b0) 0.026 |
| competition = SELF-suppression via own read of rival | direction-resolved KO | KO c1→c2 alone restores 0.063→**0.908** |
| competition machinery ≡ solo-suppression machinery (head overlap) | {b2h0,h2,h3} both | {b0h2,b0h3} both |
| emergent exclusivity (ties → both claim) | 0.745 | 0.880 |
| suppression timing: silent early absorption, claim crystallizes at last block | SB2 | SB2 (2-block compressed) |

### DIVERGENT (different implementation/parameters)

| property | thesis | organism |
|---|---|---|
| comparator location | **dedicated late read: b2h2 alone** (KO restores 93%; b1 jet↔jet irrelevant — NEW: KO b0 alone also restores 93%, so the 2 stages are b0-visibility + b2h2-read) | **block-0 read itself carries it** (KO b0 restores 0.91 at r=1.6; b1 jet↔jet irrelevant; distributed across b0 heads: h2 0.71, h0/h1/h3 0.27-0.38 single-KO) |
| comparator sharpness (SB1 half-suppression) | ~10-15% pT gap | ~30-60% gap (both-claim plateau 0.70+ for r∈[0.85,1.3]) |
| coherence handicap (sym vs asym crossing) | ≈7% pT | ≈0 (sym ≡ asym) |
| readout attention asymmetry (loser→winner) | large (b2: 0.72-0.99 vs ~0) | none (attention symmetric) |
| SB6 winner margin cost | −3.2 | −1.6 |
| lepton default when no jet claims (solo ctx-KO) | lepton TAKES the W (P(lep=W)→0.70) | lepton does NOT (→0.03) |
| rest×2 solo suppression strength | 31% | 53% |

### Reading of the divergence

The organism solves competition with ONE mechanism (context-relative scoring at
the b0 read, integrated by b1) — pure B-ii. The thesis model has the same base
mechanism PLUS a dedicated late sharpener (b2h2) that resolves near-ties the
organism simply doesn't resolve (it lets both claim until ~30% gaps). Capacity
appears to buy: a sharper comparator, a coherence prior, and a "somebody must be
the W" default (lepton takeover) — all refinements on top of a shared scaffold.
The PRINCIPLE (relative hardness, self-suppression, emergent exclusivity, tight
broadcast) is universal; the IMPLEMENTATION (where/how sharp) is not.

Caveats: organism tie-XOR stats are n≈26 (ties barely break — use the r=1.6 cell
for its competition); the b2h2-vs-b0 contrast is between models of different
depth, so "late vs early" is partly an architecture-shape statement.

## F1 formation dynamics — RESULT: the circuit is born in epoch 0; training is calibration

`experiments/h1/f1_formation_dynamics.py` (new): probes every checkpoint with a
compact battery (task, lockstep, mask-flip broadcast, rest×2 relativity, tie-both,
SB6 cost, cross-system doses) → `tmp_plots/f1_formation_20260610-203258.{csv,png}`.

On the seed-0 organism (30 checkpoints, 1/epoch):
- **Everything qualitative exists at checkpoint 0** (after 343 steps): task 0.90,
  lockstep 0.96 (→0.99 by ep1 and frozen), broadcast mask-flip ~0.45, rest×2
  suppression 0.62, tie-both 0.84, cross-system sigmoid present.
- **What training does is calibrate**: the cross-system threshold tightens fast
  (candidate at r=1.0 claims 0.61 @ep0 → 0.19 @ep1 → 0.07 @ep4, then frozen) and
  the SB6 competition cost deepens smoothly (−0.60 → −1.62 over ~15 epochs) —
  competition pressure strengthens long after the qualitative structure is fixed.
- Limitation: epoch granularity hides the true formation (it all happens inside
  epoch 0). A dedicated run checkpointing every ~20 steps of epoch 0 would give
  the real birth order — worth doing on the seed-1 MPS run config. Also rerun this
  battery on seed-1 when done (seed-robustness of the timeline).

## Artifacts

- `experiments/h1/f1_formation_dynamics.py` (new), `a4b` parameterized,
  `stageb_sb4` depth-relative v3, `train_organism.py --seed`.
- `tmp_org_asym_ko.py` (gitignored): the organism r=1.6 comparator localization.
- Second-seed organism training on MPS → `output/20260611-234355_TrainingOutput`.

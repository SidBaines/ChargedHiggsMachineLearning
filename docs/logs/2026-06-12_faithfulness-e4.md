# 2026-06-12 — E4 faithfulness: the verdict readout extracted to two formulas

Step 2 of the agreed plan (universality → faithfulness → write-up). PySR 1.5.10 +
Julia bootstrapped into `.venv`. Thesis model, fresh val slice (batches 19-24;
fit on 19-22, all numbers below on held-out 23-24).

## E4 part 1 — symbolic extraction (`experiments/h1/e4_wire_extraction.py`)

Targets: the two verdict-wire bottleneck scalars at the ν position. Features: 15
NAMED event-level physics scalars chosen from the H1 findings (lepton/MET/lepW
magnitudes, jet HT, lead/sublead-ljet pT/m²/tag, best-W-window-ljet pT/tag,
hardness ratios) — the "informed retry" after round-1 PySR searched blind and
found nothing.

- **b1h3 (hadronic-vs-leptonic comparator): R²=0.835 with a 6-symbol formula —
  `−4.57·tanh(pT(lepW)/HT(jets)) + 1.25`** — a saturating function of the single
  ratio LW1's correlation scan flagged (ρ=−0.80). Beats 15-feature ridge (0.829).
- **b2h2 ("W found"): R²≈0.83-0.85** (ridge 0.850, PySR 0.835; c=8:
  `0.39·(pT(lepW) − √(HT−2.6))`). ⚠ Mechanistic caution: the FIT leans on
  leptonic features although the per-key decomposition (LW1) shows b2h2 is
  purely jet-fed — in signal events leptonic and jet-side hardness are
  kinematically correlated, so regression finds proxies. Formulas describe
  BEHAVIOR; the decomposition remains the ground truth about INPUTS.

## E4 part 2 — in-network replacement (`experiments/h1/e4_wire_replacement.py`)

The wires are OVERWRITTEN at the ν and lepton positions with the formula values
(wave2 override-hook math; everything else neural). Held-out metrics:

| condition | ν == intact | ν == truth | P(ν=lep) |
|---|---|---|---|
| intact | 1.0000 | 0.9551 | 0.9941 |
| replace b1h3 (c=6 formula) | 0.9890 | 0.9534 | 0.9919 |
| replace b2h2 (c=8 formula) | 0.9631 | 0.9373 | 0.9915 |
| **replace BOTH (simple, c=6+8)** | **0.9551** | **0.9307** | 0.9880 |
| replace BOTH (c=22 formulas) | 0.9636 | 0.9402 | 0.9897 |
| mean-control BOTH (info destroyed) | 0.6946 | 0.6936 | 0.9846 |

**Headline: the channel-verdict readout of the 677k-param thesis transformer can
be replaced by two closed-form physics expressions — one of them a single tanh of
pT(lepW)/HT — keeping 95.5% of its decisions (96.4% with the complexity-22
versions) at a 2.4-pt truth-accuracy cost; the mean-control floor is 69.5% (the
lvbb base rate), so the formulas recover ~85-88% of the wire information.**
Lockstep survives replacement (the formulas drive both readers identically —
by construction, which is also why P(ν=lep) stays 0.99: same formula at both
positions, per LW1's identical decompositions).

## Caveats / what this is NOT yet

- Replacement is at the READOUT only: jets still compute candidacy neurally and
  the b2h0/b2h3 secondary path (~12%) stays neural. A full "semi-symbolic
  reconstruction algorithm" needs the jet-side claim formula too (the per-jet
  relative-hardness × m²-window × tag score) — next E4 stage.
- Formulas fit at ν, applied at ν+lep (justified: identical decompositions).
- Single model (thesis); organism version pending (no bottleneck → fit the claim
  logit directly).

## Learnings

- Feature-informed PySR works where blind PySR failed (round 1): the difference
  was knowing WHICH ~15 scalars to offer, which took the whole H1 circuit
  programme to learn. "Interpretability first, then symbolic regression."
- High R² ≠ mechanistic alignment (b2h2's leptonic-proxy fit) — always cross-read
  extraction against causal decomposition.
- PySR API: `temp_equation_file=True` conflicts with `output_directory`.

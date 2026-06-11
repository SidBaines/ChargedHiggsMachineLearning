# `experiments/` — interpretability experiments on the reconstruction transformer

Research code from the 2026-06 interpretability push (Sid + Claude). Everything here
was written and run on 2026-06-10/11 against the **thesis model**
(`thesis-ent1-bn1-d152`: entropy-penalty + bottleneck-1, 677k params) and the **d20
proto-organism** (`ent1-d20-2blk`, 20k params), both loaded via `models/registry.py`.

**Read alongside the docs — the code is the "how", these are the "what/why":**
- [`docs/H1_ASK_THE_JETS_TEST_PLAN.md`](../docs/H1_ASK_THE_JETS_TEST_PLAN.md) —
  preregistered hypotheses (H-A…H-D), experiment IDs (A1, B2, D4, SB1, …), status.
- [`docs/logs/2026-06-10_planning-and-data-recovery.md`](../docs/logs/2026-06-10_planning-and-data-recovery.md)
  — results per round/wave (sections "H1 round 1/2/2c/2d", "WAVE 1/2/3", "b2h3
  autopsy"). Experiment IDs in script docstrings match the test plan.
- [`docs/RESEARCH_PLAN_2026-06.md`](../docs/RESEARCH_PLAN_2026-06.md) — where H1 sits
  in the overall programme.

## The headline result these scripts established

(as revised by the 2026-06-11 red-team pass —
[`docs/logs/2026-06-11_red-team-h1.md`](../docs/logs/2026-06-11_red-team-h1.md))

The lep/ν "W vs none" assignment is a **global channel decision**: each jet computes
a W-candidacy score from its **hardness relative to the event** (not absolute pT —
doubling everything *except* the W drops its claim rate 0.92→0.57; holds on both
models), an m²-window and an anti-Xbb tag. Candidates are **score-ranked with no
enforced exclusivity**: identical candidates BOTH claim W in ~74% of events, and the
thesis model penalises H-ward masses so heavily that (pT 450, m 80) beats
(pT 600, m 110) ~90/10, while the d20 organism ranks by pT everywhere — the
cross-scale feature divergence found in wave 1 extends to the competition rule.
The verdict is delivered to the lepton and neutrino **in parallel** through
bottleneck scalars: **b2h2** ("W found", sign-coded, single-sender from the W-jet)
and **b1h3** (hadronic-vs-leptonic comparator) carry ~80-90% of the causal flow,
with **b2h0+b2h3** a real ~12% secondary path (clamping all 12 scalars at ν
reproduces its verdict exactly — the set is architecturally closed). Overwriting the
two primary scalars at the ν position flips its verdict in 99.9% of events while the
lepton holds (and vice versa), and the swap is wire-specific (own-class placebo 0%,
norm-matched random kick 2%); the famous P(ν=lep)≈1 lockstep is nothing but shared
wiring.

## Prerequisites (gitignored, machine-local)

| path (repo root) | what | how to recreate |
|---|---|---|
| `.venv/` | python env | `pip install -r requirements.txt` (frozen 2026-06-10; reproduced thesis numbers to 4 s.f.) |
| `tmp_data_20250321v1_signal/` | signal memmaps of dataset `20250321v1` (1.4 GB, DSIDs 510115-124 + norm files) | rsync from heppc `/data/atlas/baines/20250321v1_…` or from the Seagate copy |
| `tmp_checkpoints/` | thesis + organism checkpoints (local mirror, ~3 MB) | `scp baines@heppc402.ph.qmul.ac.uk:Code/ChargedHiggs_ExperimentalML/output/{20250512-093728,20250709-095246}_TrainingOutput/models/Nplits2_ValIdx0/chkpt*.pth …` (or Seagate `heppc_recovered/`) |
| `tmp_plots/` | output plots | regenerated on run |

All scripts resolve the repo root from their own location — run them from anywhere:
`.venv/bin/python experiments/h1/wave1_behavior.py --model thesis-ent1-bn1-d152`.

## Conventions (get these wrong and everything silently lies)

- **Object types: `{0: electron, 1: muon, 2: neutrino, 3: LJET, 4: SJET, 5: padding}`.**
  Rounds 2-2c were first run with 3/4 swapped — see the "round 2d" log section for the
  correction and the kinematic sanity check that now guards it (ljets: m~115 GeV,
  pT~535; sjets: m~9, pT~62). Verify against data before trusting any analysis.
- **Truth labels** (`x[...,-1]` = `trueInclusion`): `{0: none, 1: H, 2: W-hadronic,
  3: W-leptonic}`; training collapses to model classes `{0,1,2,2}`. `x[...,5]` is
  `recoInclusion` (the cut-based algorithm's choice — unused here, earmarked for H4).
- **Channel strata**: lvbb = lepton truth 3; qqbb splits into **boosted** (W = exactly
  one truth-2 ljet, 76.9%) and **resolved** (W = exactly two truth-2 sjets, 23.1%).
- **Scaling**: 4-momenta are `(px,py,pz,E)/1e5` (MeV); GeV value = unit × 100. No mean
  subtraction (`SCALE_DATA`, not `NORMALISE_DATA`). Ljet `tag` is a continuous
  Xbb-like score (H-ljets median +3.0, W-ljets −2.6) — not a bit.
- **Bottleneck models**: `TestNetwork.forward()` does NOT apply the bottleneck; either
  `load_model(..., register_bottleneck_hook=True)` or attach
  `hook_attention_heads(..., bottleneck_attention_output=1)` (what these scripts do —
  it also caches per-head weights/outputs). Hook math conventions (whose biases are
  dropped where) are documented in `h1/wave2_wires.py`'s docstring.

## Script inventory

### top level
| script | what | key result (log section) |
|---|---|---|
| `h1_narrative.py` / `.ipynb` | **START HERE for the findings** — the full H1 story as a playable notebook: claims, evidence, causal status, ruled-out alternatives, confidence scoreboard; regenerates the key experiments + figures live (~5-10 min CPU). `.ipynb` ships executed (plots inline); the `.py` is the jupytext source | synthesis of rounds 1-2, waves 1-3, red-team, Stage B |

| script | what | key result (log section) |
|---|---|---|
| `train_organism.py` | trains the d20/2-block entropy-penalty organism locally with full config serialization (`--smoke`, `--device`); includes the MPS NaN guards (`sanitize_padding`, grad-clip, non-finite skip) | new organism: val PerfectRecoPct 0.8142 vs old 0.8055 ("organism attempt 2/3" + wave-3 sections). **MPS crashed the machine — train on CPU** |
| `eval_thesis_numbers.py` | full-val eval of the thesis model on 20250321v1 | reproduced thesis Table values to ~4 s.f. ("PHASE 0 GATE PASSED") |
| `validate_registry.py` | strict-loads every registry variant + forward pass | needs the Seagate mounted (`DEFAULT_CHECKPOINT_ROOT`); 8/8 recovered variants pass |

### `h1/` — the neutrino-pairing / channel-decision investigation (chronological)
All take `--model` / `--batches` (and `--ckpt-override` where noted) unless marked
thesis-only. Experiment IDs refer to the test plan.

| script | experiments | what it established |
|---|---|---|
| `round1_behavior_attention.py` | round 1 | P(ν=lep)=0.995; channel-conditional attention; b2h3 scalar AUC 0.83 (superseded by round2e) |
| `round1_head_ablation.py` | round 1 | per-head subtraction ablations — "no single head matters" (later explained by wave-2 D4-lite: removal ≠ reversal) |
| `round2_jet_swap_margins_grid.py` | A-pre/B-pre | jet-system swap (verdict follows the jets), margin analysis (flips = borderline events only), corruption grid. ⚠ first run had sjet/ljet names swapped — fixed; numbers valid |
| `round2b_count_vs_content.py` | — | ljet count vs content: adding an H-like ljet ≠ faking a W; coherence matters |
| `round2c_strata_mediation.py` | — | truth-topology strata; surgical-vs-sham masking; XOR; mediation (flips track "did any jet claim") |
| `round2d_type_forensics.py` | — | **the type-mapping forensics** (3=ljet, 4=sjet) + strata are perfectly physical; no model needed |
| `round2e_message_split.py` | (C4 seed) | truth×prediction split of bottleneck scalars into ν → found the real verdict wires **b2h2 (AUC vs pred 0.996), b1h3 (0.989)**; b2h3's modes = ljet-multiplicity + e/μ. thesis-only |
| `wave1_behavior.py` | B1 B2 B3 A1 A6 | claim bookkeeping (XOR), shared-score anti-correlation (r≈−0.96, holds within strata), error autopsy (misses are SOFT W-jets), **A1 mass dose-response: thesis = bump@80 GeV, organism = monotonic** (cross-scale divergence), permutation control |
| `wave1_logit_lens.py` | D2 | exact logit-lens (no LayerNorm); jets claim a block before lep/ν verdict; lepton's own prior informative-but-overridden. Gotcha in-file: snapshot streams before re-calling `classifier()` |
| `wave2_dose_insertion.py` | A2 A1-pT A4 A8 | mass-not-sufficient; **pT dominates candidacy**; insertion sufficiency (real W-ljet claims 34% in lvbb); two-claimant winner-take-all, **higher-pT wins ~90%** ⚠ A8's pT-vs-mass comparison superseded by `redteam_rt3_wta.py` (real-W pairs have degenerate masses) |
| `wave2_wires.py` | C1 C4 D4-lite | reader-head taxonomy (W-pointing vs H-pointing); per-key decomposition of ν's scalars (b2h2 single-sender = W-ljet; b1h3 = two-sided comparator); **scalar-overwrite: ν flips 99.8%, lockstep breaks**. thesis-only |
| `wave3_tag_occlusion.py` | A5 E2 | tag forensics + surgery (full H-disguise re-labels W as H) + the candidacy occlusion table (pT ≫ mass ≈ tag ≫ direction) ⚠ ranking conflates perturbation sizes; "pT" is largely RELATIVE hardness (`redteam_rt4_relpt.py`); "direction-blind" fails under tied competition (`redteam_rt3_wta.py`) |
| `wave3_wires_positions_resolved.py` | D4-lite@lep, A3-lite | **bidirectional parallel-reader proof** (swap @both restores lockstep in flipped state); resolved stratum: detection weak + unspecific (the cats-0-3 gap, mechanistically) |
| `wave3_probe_patch.py` | D6 D1c | candidacy linearly decodable **at the embedding** (AUC 0.93); stream-patching: verdict transits blk-1/2 reads; final-depth patch decouples jet label from verdict (0 flips); embed-patch partial recovery (stage-B teaser). thesis-only |

### `h1/redteam_*` — the 2026-06-11 adversarial pass
All run on val batches **13-18** (the waves used 1-12; loader slices are disjoint by
construction, so this is a clean held-out set). Results + verdict:
[`docs/logs/2026-06-11_red-team-h1.md`](../docs/logs/2026-06-11_red-team-h1.md).
`rt3`/`rt4` take `--model`; the rest are thesis-only.

| script | what it tested | outcome |
|---|---|---|
| `redteam_rt6_data.py` | split parity (float32 eventNumbers), slice composition/disjointness, strata exhaustiveness, types kinematics | all clean |
| `redteam_rt1_wires.py` | D4-lite specificity: census replication out-of-sample; per-scalar swap controls; own-class placebo; norm-matched random + non-wire-direction kicks; RT8 XOR independence null | certificate survives (placebo 0%, random kick 2.1%); census replicates; "fully determined" → ~95%; XOR-ok mostly base rates (report φ≈−0.8) |
| `redteam_rt2_clamp.py` | sufficiency direction: input-side verdict flips (surgical mask / W insertion) with ν's wires clamped to clean values | wires carry ~80-90% of flow (restoration 81.6% / 90.7%); lep-side mirror symmetric |
| `redteam_rt2b_bypass.py` | which scalars carry the residual | b2h0+b2h3 (14.3%→2.1%); clamping all 12 = exactly 0 flips (architecturally closed) |
| `redteam_rt3_wta.py` | factorial two-candidate (pT, m) competition + symmetric tie | exclusivity is emergent (ties → both claim 74%); thesis = H-ward-mass-dominated ranking, organism = pT-first; coherence wins ties 88/12 (thesis) |
| `redteam_rt4_relpt.py` | absolute vs relative pT gating (complement scaling + per-DSID observational) | candidacy = RELATIVE event hardness, both models |

### `h1/stageb_*` — Stage B: the competition sub-circuit (2026-06-11, RESOLVED)
All on the fresh val slice (batches 19-24; `--skip 18`). Verdict: **weak B-i dead,
B-ii established & localized** — competition is per-jet context-relative scoring:
b0/b1 silent context absorption + the **b2h2 comparator** (the verdict-broadcast
head), each stage causally necessary; each jet suppresses only ITSELF via its own
read of the rival. Results: `docs/logs/2026-06-11_stage-b-competition.md` +
test-plan addendum.

| script | experiments | what it established |
|---|---|---|
| `stageb_sb3_attention.py` | SB3 | only-block-2 jet↔jet asymmetry (loser→winner b2h0/b2h2/b2h3); b0-1 symmetric/diffuse; competition-asymmetry heads ≡ solo-rest×2 mediation heads (shared machinery) |
| `stageb_sb2_lens_timing.py` | SB2 SB6 | big gaps resolve blk0-1, near-ties flip AT blk2; winner pays −3.3 margin vs solo twin (rates hid it) |
| `stageb_sb4_edge_knockout.py` | SB4 | surgical edge KO (validated by-hand recompute incl. bottleneck): tie loser restored 93% by b2h2 ALONE; direction-resolved ⇒ no winner→loser edge; b0+b1 KO also restores (two-stage); solo KO ⇒ candidacy constitutively relational |
| `stageb_sb1_dose_response.py` | SB1 | sym design crosses r=1.00 exactly; asym r≈1.07 ⇒ coherence handicap ≈7% pT; half-suppression at ~10-15% pT gap |

SB5 skipped — superseded by SB2+SB4.

| `a4b_lvbb_insertion_dose.py` | A4b | resolves the A4 "only ~30% flip" puzzle: broadcast is tight (P(lep=W\|ins claims)=0.06 vs 0.96); the average was donor composition; cross-system dose at pT=r×pT(lepW) is a clean sigmoid crossing r≈1.4 ⇒ the verdict is a **graded hadronic-vs-leptonic comparison** (claim-2 sharpened: "ask the jets" → "weigh the jets against the leptonic side") |

### What's next
Universality reruns of Stage B on the freshly trained organism
(`output/20260610-203258_TrainingOutput`, use `--ckpt-override`) — RT3 says it
ranks pT-first; is the two-stage last-block-comparator structure preserved?
F1 training-dynamics across its 30 checkpoints (when does the comparator form?).
E4 PySR extraction of the candidacy formula (**include event-context features** —
RT4; needs `pysr`/Julia). H-side competition battery (parked in the test plan).

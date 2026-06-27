# Joint reco+classifier — interpretability log (2026-06-27)

> **Active workstream (interp half of the joint reco+classifier project).** Spun out of
> [`JOINT_RECO_AND_CLASSIFIER_PLAN_2026-06-23.md`](JOINT_RECO_AND_CLASSIFIER_PLAN_2026-06-23.md)
> (now wrapped up). Training / scaling work lives in the sister doc
> [`ONGOING_JOINT_TRAINING_AND_SCALING_2026-06-27.md`](ONGOING_JOINT_TRAINING_AND_SCALING_2026-06-27.md).
> Builds on the **H1 investigation** ([`h1_narrative.py`](../experiments/h1_narrative.py),
> `experiments/h1/`, logs in `docs/logs/`) and the `interp/` toolkit.

## The question this doc owns (goal #1)

Does adding the **classification objective clarify or muddy the reconstruction circuit** (and
vice versa)? Is there a single clean network that does both? Concretely (registered hypotheses):

- **H-C (free discrimination):** a reco-only backbone already separates sig/bkg well, because the
  channel decision lives in the reco head. **✅ Confirmed + localized (2026-06-26).**
- **H-D (interp):** joint training makes the channel / W-candidacy circuit *more* explicit, or
  measurably muddier. **◑ First read done (2026-06-26):** *neither* — see below.
- **H1-generality (new):** does the H1 mechanistic story (graded had-vs-lep comparator, verdict
  broadcast to lep/nu) hold in a **joint-trained, larger, no-bottleneck** model — or was it an
  artifact of the specific `thesis-ent1-bn1-d152` model? ⬜ Open.

## The models under study

| name | dir | what it is |
|---|---|---|
| **joint d152** | `output/20260624-163646_TrainingOutput` (`chkpt29_85320.pth`) | two-head joint, reco+event, the scale-up run |
| **reco-only d152** | `output/20260613-234627_TrainingOutput` (`chkpt29_10290.pth`) | reco-only, signal-only, eventNumber split |
| **matched control d152** | `output/joint_d152_interp_20260627-231344/.../pure_reco_control` (training) | reco-only on the joint's *exact* data/split (leakage-free H-D control) |
| **single-head S d152** | same run-root, `single_softor` (training) | the S approach at scale |
| *(H1 reference)* | `thesis-ent1-bn1-d152` via `models.registry.load_model` | the original H1 model (**has bottleneck attention**) |

⚠ **Our d152 models have `bottleneck_attention: null`** — so H1's deepest tool (the per-head
bottleneck-*scalar* "verdict wire" b1h3/b2h2 histograms) does **not** transfer directly. The
attention-by-type, head/MLP ablation, activation-patching, DLA, and behavioural-intervention
tools all still apply.

## Results

### H-D first read — per-layer linear-probe decodability sweep (2026-06-26) ◑
Tool: **`experiments/hd_interp.py`** (mirrors `TestNetwork._backbone_features` to capture the
residual stream at emb/b0/b1/b2 — the `interp/residual_analysis.py` `ResidualStreamExtractor`
block-hook **cannot fire** on this arch's manually-applied `ModuleDict` blocks; probe =
StandardScaler + logistic, clean 60/40 split, test ROC-AUC). Joint d152 vs reco-only d152.
**Full tables + plot** are in the wrapped-up plan doc's "H-D interp" section; outputs
`output/hd_interp.{json,png}`. Headline:

- **Channel (lvbb/qqbb) decodability is UNCHANGED** by joint training at every layer & locus
  (lepton/neutrino/pool; joint−reco |Δ| ≤ 0.006, no trend, not a ceiling artifact). The reco
  circuit is left intact.
- **H-C confirmed + localized:** the reco-only backbone (never saw a background event) already
  separates sig/bkg at **weighted-AUC 0.95 (emb) → 0.978 (b2)**.
- **The classification objective adds *late, additive* structure:** joint sig/bkg separability is
  higher and the gap **grows monotonically with depth** (≈0 at emb → +0.012 / +0.008-wt at b2) —
  the event loss reshapes the **late** (pooling-adjacent) layers without disturbing early
  channel-building. → **"clarify-not-muddy / compatible in representation space"**, the
  interp-level twin of the near-free-transfer perf result.

**Caveats:** (1) reco-only used a *different* train/val split (eventNumber) than joint (position)
→ minor event-level leakage one side; leakage can only *shrink* the measured sig/bkg gap, so the
joint advantage is conservative. The **matched control** (now training) removes this entirely.
(2) "joint" bundles event head + event loss + real-bkg training = the model as deployed.

### Round 1 — head-role map (attention-by-object-type) + decomposition sanity (2026-06-27)
Tool: **`experiments/joint_mech_round1.py`** (reuses `hd_interp` load/loaders +
`interp.activations.hook_attention_heads`, CPU, signal slice). **Key enabler verified:** the
no-LayerNorm + linear-readout architecture means `hook_attention_heads`' per-head decomposition
is **exact** — `max|Σ_head contrib + bias − module output| = 1.3e-6`. So later rounds can do
*exact* additive direct-logit-attribution (no LN normalization to fight).

Per-head attention by object-type group {lep, nu, ljet, sjet} on signal events — several heads
have clean, physically-suggestive roles (raw matrices in `output/joint_mech_round1.json`):
- **b2h3 = lepton↔neutrino mutual head:** lep→nu **0.87**, nu→lep **0.93**. Almost certainly the
  H1 "lockstep" wiring (lep & nu coupled), now seen directly in *this* (no-bottleneck, joint) model.
- **b1h0 = "weigh the jets" candidate:** lep→ljet **0.72**, nu→ljet **0.67** — lep/nu read the
  large-R (hadronic-W candidate) jet. Echoes H1's "verdict = compare hadronic claim vs leptonic".
- **b2h1 = hadronic aggregator:** ljet→ljet 0.90, sjet→ljet 0.88 (jets pool onto the large-R jet).
- **b0h3 = same-type/self head:** lep→lep 0.56, ljet→ljet 0.62, sjet→sjet 0.79 (diagonal).
- **b1h3 / b2h2 = jets→lepton** read-back heads.
These map a plausible circuit: b0 builds per-type/local features → b1 routes the hadronic
candidate to lep/nu (and back) → b2 couples lep↔nu and aggregates the hadronic system. Next
rounds test this causally + via exact DLA.

### Round 2 — systematic map: exact DLA + causal ablation (2026-06-27)
Tool: **`experiments/joint_mech_round2.py`** (uses `joint_mech_lib.manual_forward`, verified to
reproduce the model to 3.7e-5 and to telescope the residual exactly). Two views per component
(12 heads + 3 MLPs + embedding): **DLA** = signed write to a decision direction (reco W-vs-none /
H-vs-none at true-W/H objects; event sig-vs-bkg / qqbb-vs-lvbb via the pooled readout), and
**ablation** = Δ when zeroed (reco object-accuracy on signal; event sig/bkg AUC on all-data).
Baseline: reco obj-acc **0.963**, event AUC **0.989**. Raw: `output/joint_mech_round2.json`.

Findings:
- **The MLPs do the decision-writing; attention does the routing.** `mlp2` (final block) has by
  far the largest DLA to *both* readouts (reco W-none +5.3, H-none +6.3, event sig-bkg −5.1) and
  the largest reco-ablation cost (Δacc **0.048**). `mlp1` carries the H and qqbb/lvbb splits
  (H-none +1.7, qqlv +0.7). The embedding already strongly encodes per-object class from raw
  kinematics (reco W-none +1.1, H-none +2.3) — attention then refines it.
- **Classification is distributed & robust; reco is more localized.** No single *head* ablation
  drops event AUC by >0.011 (biggest: **b0h2** 0.011, b0h3 0.008, mlp0 0.010) — consistent with
  the H-D "sig/bkg is free/distributed" result. Reco accuracy is more sensitive (mlp2 0.048,
  b0h3 0.019, mlp1 0.018, b0h2 0.014).
- **Reco-specialized heads (DLA W-vs-none):** b2h3 (+1.18, the lep↔nu head), b2h2 (+0.98), b2h1
  (+0.55), b1h0 (+0.31, "weigh-the-jets"). b2h1 writes a strong *negative* H-none (−2.2) at true-H
  objects (the hadronic aggregator suppresses/normalizes the H logit there — flagged, not yet
  explained). The qqbb/lvbb split is written mainly by mlp1(+0.7)/mlp2(−1.8) and b2h2 (−0.5→lvbb).
- ⚠ DLA-event is averaged over signal+bkg (sign = direction; magnitude = how much it writes);
  not yet class-split. Treat signs cautiously until cross-checked.

### Round 3 — physics-quantity probing sweep (2026-06-27) ⭐ ⚠ PARTLY REVISED by Round 4
> **Read Round 4 first for the corrected story.** Round 3's raw R² values stand, but the
> *interpretation* changed after an Opus critique + baseline controls: the high m(Wh)/parent-mass
> R² is largely an input-energy-scale + 10-point mass-grid confound (NOT "the network computes
> mass"); most scalar physics is available from the input featurization, not built by attention.
> The one cleanly attention-*computed* quantity is ΔR(H,W). See Round 4.
Tool: **`experiments/joint_mech_round3.py`** (Ridge/logistic probes on the residual stream at
{emb,b0,b1,b2} × {pooled, lepton, leading-ljet}, 16k signal events; targets computed from raw
4-vectors + truth labels in `joint_mech_lib.physics_quantities`). Report test R² (regression) /
AUC (channel). Raw: `output/joint_mech_round3.json`.

**What the network represents (pooled / event-level):**
| quantity | emb | b0 | b1 | b2 | read |
|---|---|---|---|---|---|
| **m(Wh) resonance [GeV]** | 0.73 | 0.78 | **0.80** | 0.78 | latent mass reco — strong |
| **parent H⁺ mass (DSID) [TeV]** | 0.69 | 0.76 | **0.78** | 0.77 | encodes the mass *hypothesis* |
| mH_truth | 0.50 | 0.50 | 0.47 | 0.47 | moderate |
| mW_truth | 0.27 | 0.29 | 0.28 | 0.32 | weak |
| sum_btag | 0.95 | 0.96 | 0.87 | 0.77 | front-loaded, decays w/ depth |
| dR(H,W) boost | 0.17 | 0.31 | **0.36** | 0.33 | built up by attention |
| HT / MET / lead_pt | ~0.95 | ~0.95 | ~0.90 | ~0.88 | strong (sums) |
| n_obj/n_sjet/n_ljet | ~0.90 | **~0.97** | ~0.93 | ~0.88 | strong |
| **is_lvbb (channel) AUC** | 0.981 | 0.984 | 0.991 | **0.995** | most decodable; present at emb |

**Findings:**
- **Latent mass reconstruction.** m(Wh) and the parent H⁺ mass hypothesis are linearly decodable
  at R²≈0.78–0.80 — the model effectively *computes the resonance mass* (a quadratic invariant-mass
  combination of object 4-vectors), though it's only trained on per-object reco + event class. The
  mass scale is presumably the key signal/bkg + mass-hypothesis discriminant, so it falls out.
- **Attention aggregates global info onto tokens at block 0.** At the lepton token, event-level
  quantities are absent at emb (m(Wh) 0.15, HT 0.18, n_obj ≈0) then **jump after one block**
  (m(Wh) 0.74, HT 0.92, MET 0.94, sum_btag 0.88) — block-0 attention broadcasts the event summary
  onto every token. Same at the ljet token. (At emb the ljet trivially knows its own pt: R²=1.00.)
- **b-tag is read early, then transformed.** sum_btag/max_btag peak at the embedding (it's an input
  feature) and *decay* with depth — the net consumes flavour info early and repackages it.
- **Late-layer compression toward the decision.** Most general kinematic R² *peaks at b0–b1 and
  decays by b2*, while the channel AUC *keeps rising* to b2 — the final layer trades general
  kinematic fidelity for sharper task-relevant (channel/sig-bkg) structure. Mirrors Round 2
  (mlp2 = the decision-writer) and the H-D "late, task-specialized" picture.
- ⚠ Caveat: pooled sums (HT, n_obj, sum_btag) are partly linearly available already at emb because
  pooling = mean of object features; the *computed* quantities (m(Wh), dR, channel) and the
  per-token aggregation jumps are the load-bearing evidence.

### Round 4 — corrective controls after Opus critique (2026-06-27) ⭐
An Opus subagent reviewed rounds 1–3 (read `interp/`, ran its own controls) and flagged: (i)
m(Wh) probing is confounded by raw-input energy scale + the 10-DSID mass grid; (ii) zero-ablation
can overstate importance; (iii) Round-2 event-DLA signs are bkg-dominated (88% bkg, not class-split)
and dropped the attn-bias term. **`experiments/joint_mech_round4.py`** runs the controls; raw:
`output/joint_mech_round4.json`.

**A. Physics with baselines (pooled R²; what's available from inputs vs computed by attention):**
| target | raw-4vec | hand-feats | emb | b2 | Δ(b2−emb)=attn | Δ(emb−raw)=featurize |
|---|---|---|---|---|---|---|
| m(Wh) resonance | 0.56 | 0.78 | 0.73 | 0.78 | +0.05 | +0.16 |
| parent mass (DSID) | 0.50 | 0.72 | 0.69 | 0.77 | +0.08 | +0.19 |
| mH_truth | 0.03 | 0.09 | 0.50 | 0.47 | **−0.03** | +0.47 |
| mW_truth | 0.01 | 0.13 | 0.27 | 0.32 | +0.05 | +0.27 |
| **dR(H,W)** | 0.01 | 0.05 | 0.17 | 0.33 (b1 **0.36**) | **+0.16** | +0.17 |
| sum_btag | 0.01 | 1.00 | 0.95 | 0.77 | **−0.19** | +0.94 |
| channel (AUC) | 0.53 | 0.96 | 0.98 | 0.995 | +0.01 | +0.45 |

Within a *fixed* DSID, m(Wh) R² collapses to **0.14–0.32** (raw 0.06–0.19) — confirming the
cross-DSID R²≈0.78 was mostly mass-grid/energy-scale, not event-by-event mass reconstruction.

**Corrected physics story:**
- **RETRACT "latent mass reconstruction."** m(Wh)/parent-mass are largely linearly available from
  raw inputs + the discrete mass grid; the network adds only +0.05/+0.08, and within-hypothesis
  mass is weakly represented (R²~0.2).
- **The one quantity attention genuinely *computes* is the relational ΔR(H,W)** (emb 0.17→b1 0.36,
  +0.16) — a cross-system geometric relation that can't be read off pooled inputs. mW and
  parent-mass get small genuine attention gains (+0.05/+0.08).
- **mH, channel, mW, b-tag are *featurization*-level**, not attention-built: the big jump is
  raw→emb (per-object Lorentz m²/pt/η/φ + type-embedding + pooling), then flat/declining through
  the blocks (mH −0.03, b-tag −0.19, channel +0.01). The model reads these from its input encoding;
  attention mostly *routes/broadcasts* them (the Round-3 per-token emb→b0 jumps) and sharpens the
  channel late.
- So: **attention's role here is aggregation/routing + a little relational computation (ΔR), not
  building new scalar physics.** Honest and still interesting (relational ΔR is the cleanest circuit lead).

**B. Mean- vs zero-ablation (Δ reco-acc / Δ event-AUC; mean = per-position dataset mean):**
- Importances **hold or grow** under mean-ablation — they are NOT zero-ablation off-distribution
  artifacts. mlp2 reco Δacc 0.048→**0.086** (dominant); late heads b2h1 0.007→**0.017**, b2h2
  0.010→0.014 grow (load-bearing late, as expected); early b0h2/b0h3 stay ~0.013–0.015.
  (The critic's spot-check that b0h3 *vanishes* under mean-ablation did NOT reproduce with
  per-position mean-ablation — likely a global-mean vs per-position-mean difference.)
- **Classification stays distributed/robust** under mean-ablation too (max single-component event
  Δauc: b0h2 0.008, mlp2 0.008, mlp0 0.007). Confirms Round 2 + H-D.
- **Corrected Round-2 takeaway:** MLPs (esp. mlp2) write the decision; among heads the load-bearing
  ones are b0h2/b0h3 (early, build features) and b2h1/b2h2 (late, write the verdict); reco is
  component-localized, classification is distributed. (Event-DLA *signs* from Round 2 remain
  un-trustworthy until class-split — treat magnitudes only.)

## Planned (next, foreground)

1. **Mechanistic localization of the late sig/bkg structure** — **head + MLP ablation** on the
   joint vs reco-only d152, measuring **dual impact** per component: Δ(event-head / probe sig-bkg
   AUC) vs Δ(reco PerfectRecoPct / channel decodability). Classifies each head as
   reco-specialized / classification-specialized / shared, and reveals whether joint training
   **repurposed** a head the reco-only model leaves "free". (Head-level ablation inside the fused
   `nn.MultiheadAttention` needs care — see `interp/ablation.py`; block/MLP-level ablation is
   trivially well-defined as a coarser fallback.)
2. **Attention-by-type comparison** (`interp/attention.py` `AttentionAnalyzer`) — did any head's
   attention pattern change between the two models?
3. **H1-generality / channel-mechanism check** — replay the H1 *behavioural* interventions
   (jet-swap, W-jet mask, cross-system dose-response from `h1_narrative.py`) on **both** joint and
   reco-only d152: does the verdict still follow the jets, with the same graded threshold? If
   identical → the channel mechanism is preserved (consistent with the probe) **and** H1
   generalizes beyond its bottleneck model.
4. **Re-run the H-D probe sweep with the matched control** (leakage-free) once it finishes
   training → bulletproofs the sig/bkg-gap number.

## Tooling
- **`experiments/hd_interp.py`** — the probe sweep (channel @ lep/nu/pool + sig/bkg @ pool, per
  layer). `--joint <dir> --reco <dir> --device mps [--channel-batches N --sigbkg-batches N]`.
- **`interp/`** — `activations.py` (`ActivationCache`, `hook_attention_heads`),
  `attention.py` (`AttentionAnalyzer`), `causal_analysis.py` (`ActivationPatcher`,
  `DirectLogitAttributor`), `residual_analysis.py` (probes), `ablation.py`.
- **Model/data idiom** — `build_model(cfg, device)` from `config.json` + latest `chkpt*.pth`;
  signal slice `tmp_data_20250321v1_signal/` (eventNumber split), sig+bkg
  `tmp_data_20250321v1_all/` (position split). See `hd_interp.py` for the exact pattern.
- **H1 reference** — `experiments/h1_narrative.py` (the full mechanistic narrative + the
  behavioural-intervention code to replay).

## Session log
- 2026-06-26 — **H-D first read** (`hd_interp.py`): channel circuit unchanged; classification adds
  late additive sig/bkg structure; H-C confirmed. Committed `aa81433`. (Full write-up in the
  wrapped-up plan doc.)
- 2026-06-27 — Doc spun out. Next = mechanistic localization (head/MLP ablation, dual impact) +
  H1-generality behavioural replay; matched control training in the background for the
  leakage-free re-run.

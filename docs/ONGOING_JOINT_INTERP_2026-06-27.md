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

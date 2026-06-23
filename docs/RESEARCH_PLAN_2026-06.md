# Interpretability research plan — DRAFT for discussion (2026-06-10)

> Joint plan (Sid + Claude). Goal hierarchy, agreed 2026-06-10:
> 1. **Research for its own sake**: what are the models actually learning? Are they
>    computing physics-relevant quantities? Can we find *circuits* that reliably explain
>    mechanisms?
> 2. Cleaner results/plots → possible thesis-correction upgrades (thesis submitted, viva
>    passed, corrections in progress).
> 3. Eventually: clean visuals / interactive demos for the public GitHub repo.
>
> Compute: Mac MPS now; **M5 Pro / 48 GB arriving ~1 day**; RunPod as fallback.
> Status inputs: thesis ch. 7 + App. ZD (read 2026-06-10); repo survey (2026-06-09);
> `docs/SESSION_FINDINGS_2026-06-08.md`.

## Where the thesis left interpretability (the launch point)

- Two novel-in-HEP simplifications worked: **attention bottleneck (n_bn=1)** and
  **attention-entropy penalty**; cost ~5–10% (bottleneck) to ~25% (both) in categories
  0–3, negligible in the dominant categories 4–5 (Table `RecoTransformerModifications`).
- **Attention-pattern analysis** (type→type heatmaps, per-category) made it into the
  thesis; the thesis model is entropy+bottleneck-1, n_blocks=3, d_m=152, n_heads=4
  (the original "n_blocks=4" here was the same typo as the thesis caption — see the
  2026-06-21 audit log).
- **Robustness studies** gave real mechanistic hints: rotating the *lepton* φ/η destroys
  cats 0–3; rotating the *neutrino* barely matters ⇒ the neutrino's class is inherited
  from the lepton, not from its own angles.
- **Head ablation**: no single head matters for cats 4–5 ⇒ redundancy.
- Probes / DLA / PySR: tried, "no additional insights yet" — the open frontier.

## Phase 0 — Foundations (unblock + trust the setup)

0.1 Finish the ROOT-data transfer to Seagate (in progress; signal DSIDs arrive last —
    verify file count + sizes vs cluster when rsync exits).
0.2 Recover from heppc while access lasts: repo-trained checkpoints (`output/`), wandb
    dirs, and ideally the derived memmaps (exact thesis training inputs). Commands below.
0.3 wandb relogin (`wandb login --relogin`) → pull old run configs/curves; archive the
    hyperparameters of the thesis models.
0.4 Regenerate memmaps from the 20250313 ROOT files with `preprocessLowLevel.py` on the
    new Mac; resolve the **type-encoding (3 vs 5) question** as part of this.
0.5 Retrain the reconstruction net (thesis config) + sanity-check against thesis numbers
    (Table `tab:EventReconstructionComparison` category counts; attention heatmaps look
    like `Pictures/AttentionPatterns/`). This validates the whole local pipeline.

## Phase 1 — Enablers (only the code work the research needs)

1.1 `requirements.txt` pinned from the working venv.
1.2 **Serialize model+data config as JSON next to every checkpoint** (kills checkpoint
    archaeology forever; we've been bitten twice).
1.3 Model-variant **registry** replacing the nine `if 0:` blocks in `RunLowLevelInterp.py`.
1.4 Smoke test: 50 training steps on the local 20k subset, run before/after refactors.
1.5 De-duplicate `mechinterputils.py` vs `interp/activations.py` (incl. the `wefwef`
    breakage); keep `interp/` as the one true library.
   (Deferred, not blocking: plotting consolidation, lvbb/qqbb compare-script merge.)

## Phase 2 — Model organisms + the tradeoff curve

2.1 Train a **suite** of models with serialized configs, spanning:
    constraint ∈ {none, entropy, bottleneck-1, both} × size ∈ {thesis-size, minimal}.
    "Minimal" = smallest (1–2 blocks, 1–2 heads, small d_m) that still solves the easy
    categories (4–5) — our fully-reverse-engineerable *model organisms*.
2.2 Produce the missing headline figure: **performance vs interpretability-constraint
    curve** (PerfectRecoPct / per-category efficiency vs entropy weight & n_bn), with
    seeds for error bars. (Also upgrades thesis Table `RecoTransformerModifications`.)

## Phase 3 — Circuit hunting (the core)

Concrete, falsifiable hypotheses first; generic dashboards second.

- **H1 — the neutrino-pairing circuit** — **v1 REFUTED, v2 ESTABLISHED behaviorally
  (2026-06-10, rounds 1-2)**: there is no lepton→[head]→ν message circuit; the lep/ν
  "W vs none" verdict is a *global channel decision* (lvbb vs qqbb) computed from the
  **jet system** — primarily the presence of a *hadronic-W candidate* (a W-mass-window
  ljet in ~77% of qqbb, an sjet pair in ~23%; see round-2d type-label correction),
  secondarily the lepton+MET magnitude — then broadcast, with lep+ν
  inheriting it in lockstep (P(ν=lep)≥0.95 under every intervention tried). Identical
  mechanism in the 677k thesis model and the 20k organism. **Remaining for H1: localize
  the circuit** that computes the channel score — preregistered test battery (~30
  experiments, waves 1-5) in [`H1_ASK_THE_JETS_TEST_PLAN.md`](H1_ASK_THE_JETS_TEST_PLAN.md);
  merges into H2/H3 below.
- **H2 — angular-proximity heads**: heads computing ΔR/Δφ/Δη(query, key) — prior PySR
  hints + entropy models. Fit attention *logits* (not post-softmax weights) as functions
  of pairwise physics features; quantify R² per head.
- **H3 — Higgs-candidate identification**: is "the large-R jet in the mass window /
  Xbb-tagged / far from lepton" computed and routed? Probe for these named quantities in
  the residual stream (targeted probes, not generic class probes — likely why probes
  found nothing before).
- **H4 — algorithm comparison**: the old cut-based reco is a *known* algorithm. Where
  does the transformer's decision agree with it, and what extra signal does it use where
  they disagree? (The disagreements are where the 2× efficiency gain lives.)
- **Faithfulness test (the punchline if it works)**: replace a head's attention/bottleneck
  with its fitted symbolic expression and measure performance retention. Treat early
  wins as readout-level behavioral compression until the jet-side algorithm is also
  extracted under a prospective/nested validation protocol.

Method notes: do all of this on the *minimal + constrained* models first, then check the
findings transfer to the thesis-size model. Use truth labels to build counterfactual
event pairs for patching. PySR round 2 targets bottleneck scalars + attention logits with
event filtering by category, learning from why round 1 underwhelmed.

## Phase 4 — Synthesis & presentation

- Canonical results notebook (tradeoff curve, circuit figures, attention showcases).
- Interactive demo idea for the repo: per-event display (objects in η-φ) with attention
  overlays + circuit annotations. Only worth building once Phase 3 has content.
- Optional: thesis-correction plots (the `\note{}` gaps in ch. 7, e.g. missing qqbb
  ROC-AUC, now reproducible locally).

## Open questions / risks

- MPS training throughput for the full background dataset (classifier) — reconstruction
  (signal-only, ~400k events) is comfortably fine; classifier (2.7M bkg) may want RunPod.
- Old heppc checkpoints may use configs we can only partially reconstruct (wandb helps).
- Memmap regeneration must reproduce the thesis preprocessing exactly for sanity checks
  (flags encoded in DATA_PATH strings; cross-check vs wandb configs).

## Session log

One line per session; details in `docs/logs/` (convention in `CLAUDE.md`).

- 2026-06-23 — **MPS training speed-ups (Tier 1, no param/epoch changes)**: profiled the train loop — it's overhead-bound (tiny model), with cost in the per-sample object-shuffle loop, a per-epoch memmap-column re-read, a discarded `loss_dict` (+disabled penalties), and a by-hand attention hook. Tier 1 (bit-identical, verified `|Δloss|=0`; organism entropy-grad ~7e-10): gate `loss_dict`/penalties (`build_loss_dict=False`), cache the train/val split, drop the hook when entropy is off (`hook_attention_heads`→new lightweight `hook_attention_weights_only` when on). **Compute/step organism 58.7→24.7 ms, joint 40.3→19.0 ms.** Tier 2 (deterministic, RNG-stream-changing): vectorised object shuffle (`next()` 14.2→3.7 ms/batch) + seeded `np.random.shuffle` → training **now fully deterministic** (old `random.shuffle` was unseeded), seed-0 A/B before/after confirms no degradation. **End-to-end/step organism ~72.7→28.4 ms (2.6×), joint ~54.3→22.7 ms (2.4×).** → [`logs/2026-06-23_mps-training-speedups.md`](logs/2026-06-23_mps-training-speedups.md).
- 2026-06-23 — **NEW workstream: joint reco + signal/background classifier** (one transformer, both tasks). Scoped the design space and settled initial strategy: build **both** structural options head-to-head — *single-head* (`background=all-none`, no arch change) and *two-head* (per-object reco head + pooled event head, with interleave/sequential & full/head-only-backprop sub-variants). Confirmed `TestNetwork` already does both modes and the loader already supports signal+background+MC-weighting; **background data is NOT on the laptop** (hard blocker → pre-made memmaps from Seagate). Metrics = AUC + Asimov Z (downstream limit deferred as gold-standard). Local-MPS subset-first. Phase 0 (signal-only plumbing) unblocked. Plan/progress log: [`JOINT_RECO_AND_CLASSIFIER_PLAN_2026-06-23.md`](JOINT_RECO_AND_CLASSIFIER_PLAN_2026-06-23.md).
- 2026-06-22/23 — **NEW workstream: minimal plain-transformer reco organisms** (no entropy/no bottleneck). Boosted-only (cats 4,5) 300-ep **convergence sweep**: cat4 solved at every size, cat5 plateaus ~0.93 with width but **depth (d16b3, 3.8k params) clears both**; error-profiling shows cat4's ease = *removed* resolved-Higgs error mode (generalist 22.7%→specialist 0.0%), cat5's hardness = intrinsic H↔W large-R-jet swap (~35–39%, both). Extended `train_organism.py` + live `dashboard.py` (+v2/v3) + `eval_error_breakdown.py` (per-run JSON profiles). Plan: [`RECO_TRAINING_PLAN_2026-06-22.md`](RECO_TRAINING_PLAN_2026-06-22.md); log: [`logs/2026-06-23_minimal-reco-organism-sweep.md`](logs/2026-06-23_minimal-reco-organism-sweep.md).

- 2026-06-21 — **Adversarial audit of the interp claims for thesis Ch.7** (3 Opus max reviewers, reproduced every headline + ran new controls). Provenance RESOLVED: thesis "n_blocks=4" is a **typo for 3**, repo dissects the correct published model. **Phase-2 "+2.8-pt recipe lift" is a WEIGHTING-COMPARISON BUG** (MC-weighted new vs `val`-weighted old) — real recipe effect ≈0; but the **constraint cost survives** (~1.2 pts winner / ~1.7 thesis-era, all in cats 0-3, upper bound). C2 "weigh the jets" strongest (thesis-grade); C1/C4 soften (shared readout; closure is architectural; report AUC-vs-truth); **E4 b2h2 "faithfulness" FAILS** (formula blind to the mass/tag mechanism), b1h3 half holds. Agreed thesis plan: commit 1 = cleanup/restructure (no new science), commit 2 = the neutrino-puzzle *example* investigation. → [`logs/2026-06-21_thesis-ch7-interp-audit.md`](logs/2026-06-21_thesis-ch7-interp-audit.md).
- 2026-06-16 — **Phase-2 CLOSED**: legacy comparability runs done (`d152_both_s0_LEGACY`=0.850 reproduces thesis 0.843), **Phase-2.2 tradeoff figure drawn** (`suite_tradeoff_plots.py` → `tmp_plots/suite_d152_tradeoff.png`); refined conclusion = recipe lift (~+2.9 pts) and constraint cost (~1.2 pts) are SEPARABLE & recipe-stable. Discussed the **next-experiment menu** (A jet-side faithfulness / B H4 algorithm comparison [lean] / C suite-as-population / D robustness gate) — grounded that the cut-based decisions are already in the data (`recoInclusion=x[...,-2]`); **decision pending**. → [`logs/2026-06-15_suite-d152-results-and-entropy-mismatch.md`](logs/2026-06-15_suite-d152-results-and-entropy-mismatch.md).
- 2026-06-14/16 — **Phase-2 d152 winner-recipe suite DONE** (12 runs; full per-category tables) + **merged the two codex branches** (analysis-fixes, resolved-qqbb). A 2026-06-15 "entropy-weight confound" scare turned out to be a **FALSE ALARM** (2026-06-16): I misread the thesis `YesEnt1` run-name as weight 1.0, but the training code uses `entropy_weight=1e-2`, matching the suite runs — comparison is valid. Clean results: `none` old-vs-new +2.8 pts (recipe lifts hard cats 0-3); 2 legacy comparability runs launched 2026-06-16 to pin the winner-vs-legacy `both` gap to the recipe. Parser/plots: `experiments/suite_tradeoff_plots.py`. → [`logs/2026-06-15_suite-d152-results-and-entropy-mismatch.md`](logs/2026-06-15_suite-d152-results-and-entropy-mismatch.md).
- 2026-06-08 — Repo/branch/data investigation after long gap → `docs/SESSION_FINDINGS_2026-06-08.md` (predates the logs convention).
- 2026-06-09/10 — Codebase + thesis ch.7 survey; found live heppc rsync; wrote this plan; agreed interp-first goals → [`logs/2026-06-10_planning-and-data-recovery.md`](logs/2026-06-10_planning-and-data-recovery.md).
- 2026-06-10 (night) — MPS training crashed the Mac → organism relaunched on CPU (healthy, ~5 min/epoch); **H1 round 2: channel decision reads the jet system** (sjet-system coherence + lepton/MET magnitude; replicates on the 20k organism) → same log, "H1 round 2" section.
- 2026-06-10/11 (late night) — H1 rounds 2c-2e + preregistered test plan + **waves 1-3 run**: "ask the jets" structure confirmed; candidacy ≈ pT×mass-window×anti-tag (computed per-object); verdict wires b2h2+b1h3 found and causally seized (lockstep = shared wires, broken & restored at will); resolved-mode weakness explained; A1 mass-window diverges across scale; new organism trained (0.8142). Code → `experiments/` (+README). → same log, wave sections + `docs/H1_ASK_THE_JETS_TEST_PLAN.md`.
- 2026-06-12 (pm) — **Suite d20 half done** (interpretability ~free at small scale, −0.65 pts for both constraints; constraints reduce seed variance 3×; d152 half PAUSED — resume `run_suite.sh`) + **HANDOFF written**: [`HANDOFF_2026-06-12.md`](HANDOFF_2026-06-12.md) is now the START-HERE doc. → [`logs/2026-06-12_suite-d20-half.md`](logs/2026-06-12_suite-d20-half.md).
- 2026-06-12 — **E4 faithfulness MILESTONE**: two readout-level formulas for the dominant verdict wires (b1h3 ≈ −4.57·tanh(pT(lepW)/HT)+1.25, R²=0.84) were spliced back in-network: **95.5% of decisions kept with the two simple formulas** (true-mean floor 69.5%); jet-side claim formula + prospective validation = next E4 stage. Also: LR sweep (legacy recipe undertrains ~3pts; suite recipe = 1e-3 cosine ramp), Phase-2 suite queue launched (26 runs), formation crisis confirmed real under sane schedule. → [`logs/2026-06-12_faithfulness-e4.md`](logs/2026-06-12_faithfulness-e4.md).
- 2026-06-11/12 (night) — **MPS validated on M5** (forward parity 6.5e-5; full training run stable, ~2× CPU; now default) + **universality battery on the organism**: the PRINCIPLE is universal (relative-hardness self-suppression, tight broadcast, emergent exclusivity, constitutively-relational candidacy), the IMPLEMENTATION diverges (organism: b0-read comparator, ~30-60% resolution, no coherence prior, no lepton-takeover default; thesis: b0-visibility + dedicated b2h2 sharpener — NEW: b1 jet↔jet irrelevant on both) + **F1: the circuit is born in epoch 0; training is calibration** (threshold tightens ep0-4, competition cost deepens ep0-15). Second-seed organism trained on MPS. → [`logs/2026-06-12_universality-mps-f1.md`](logs/2026-06-12_universality-mps-f1.md).
- 2026-06-11 (eve) — **Stage B run & RESOLVED (thesis model, fresh val slice)**: no separate directed winner→loser inhibition edge found; B-ii strongly supported & localized — competition = per-jet context-relative scoring, two-stage circuit (b0/b1 silent context absorption + **b2h2 comparator**, the same head that broadcasts the verdict); each jet suppresses only itself via its own read of the rival; sym/asym dose-response: comparator ~10% pT resolution, coherence handicap ≈7% pT; SB5 superseded. Also: **full wandb archive pulled** (550 runs → `docs/run_configs/`; Phase 0.3 closed). → [`logs/2026-06-11_stage-b-competition.md`](logs/2026-06-11_stage-b-competition.md).
- 2026-06-11 (pm) — **New MacBook (M5 Pro/48GB) migration**: audit of what survived the copy (all critical assets: data, checkpoints, organism, experiments — Stage B unblocked WITHOUT Seagate); uv + python 3.12 + venv rebuilt from frozen requirements; Phase-0 gate re-run on the new machine; list of what's still on the old Mac/Seagate → [`logs/2026-06-11_new-mac-m5-setup.md`](logs/2026-06-11_new-mac-m5-setup.md).
- 2026-06-11 — **Red-team of waves 1-3** (controls + factorials, unseen val slice): core circuit account SURVIVES (census/flip rates replicate out-of-sample; wire-swap is specific — placebo 0, norm-matched random kick 2%; parallel readers confirmed in the sufficiency direction). Corrections: candidacy = RELATIVE event hardness (not absolute pT); exclusivity = emergent score-ranking, ties → both claim; thesis model ranks mass-window-first H-ward while the organism ranks pT-first (scale divergence extends to the competition rule); wires = 2 primary + 2 secondary (b2h0/b2h3, ~12% of flow). **Update Stage-B preregs before running.** → [`logs/2026-06-11_red-team-h1.md`](logs/2026-06-11_red-team-h1.md).
- 2026-06-12 — **Resolved qqbb W circuit session started**: initial read-in found only suggestive resolved-specific evidence so far (weak/unspecific W-sjet-pair readout, poor two-sjet bookkeeping, no causal pair-mass test yet); plan is fresh-slice baselines, pairwise probes, pair-mass surgery, and path/readout-conflict interventions. → [`logs/2026-06-12_resolved-qqbb-w-circuit.md`](logs/2026-06-12_resolved-qqbb-w-circuit.md).

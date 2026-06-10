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
  thesis; the thesis model is entropy+bottleneck-1, n_blocks=4, d_m=152, n_heads=4.
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
  **jet system** — primarily the *relational coherence of the sjet system* (hadronic-W
  candidate), secondarily the lepton+MET magnitude — then broadcast, with lep+ν
  inheriting it in lockstep (P(ν=lep)≥0.95 under every intervention tried). Identical
  mechanism in the 677k thesis model and the 20k organism. **Remaining for H1: localize
  the circuit** that computes the channel score (per-block/per-head patching on the
  organism) — merges into H2/H3 below.
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
  with its fitted symbolic expression and measure performance retention — an extracted,
  semi-symbolic reconstruction algorithm. Strong, novel result for HEP.

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

- 2026-06-08 — Repo/branch/data investigation after long gap → `docs/SESSION_FINDINGS_2026-06-08.md` (predates the logs convention).
- 2026-06-09/10 — Codebase + thesis ch.7 survey; found live heppc rsync; wrote this plan; agreed interp-first goals → [`logs/2026-06-10_planning-and-data-recovery.md`](logs/2026-06-10_planning-and-data-recovery.md).
- 2026-06-10 (night) — MPS training crashed the Mac → organism relaunched on CPU (healthy, ~5 min/epoch); **H1 round 2: channel decision reads the jet system** (sjet-system coherence + lepton/MET magnitude; replicates on the 20k organism) → same log, "H1 round 2" section.

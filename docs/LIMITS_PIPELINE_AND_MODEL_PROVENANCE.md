# Expected-limits plot: pipeline trace + model provenance (2026-06-16)

> Investigation by an agent on the thesis side, tracing how the thesis figure
> `fig:TransformerVsOriginalExpectedLimits` (`PhDThesis/Pictures/TransformerNetworks/
> comparison_and_ratio4.v2.pdf`) was produced, so a **new (fixed) reconstruction model**
> can be swapped in and the limits re-derived. Numbers were reproduced end-to-end and
> match the thesis plot exactly (§3). Incorporates two corrections from the repo-side
> agent (flagged inline).
>
> Sibling reference (on the `heppc-work` branch, not this one):
> `docs/CLUSTER_AND_DATA_GUIDE.md` — heppc402 access, the `/data/atlas/baines` datasets,
> the `piienv`, and the qqbb ROC-AUC bugfix. This note is the lxplus/limits counterpart.

---

## 0. The distinction that matters: RECONSTRUCTION vs CLASSIFIER models

The limits plot depends on **two different kinds of transformer**, and they have very
different provenance. Keep them separate:

- **Reconstruction model** — assigns each object to {h-decay, W-decay, neither}. This is
  the model **this repo trains/interprets**, and the one being **retrained (fixed
  warm-start)**. In scope here.
- **Classifier models** — per-channel signal-vs-background (lvbb, qqbb), 2-fold each.
  **CORRECTION (repo-side agent): the classifiers are entirely OUTSIDE this repo's work** —
  never trained or interpreted here, no classifier checkpoints in this repo, and nothing on
  HF covers them. Their provenance (and the caption `n_blocks` discrepancy in §5) is the
  classifier team's to resolve. The earlier `CompareAllModels*` / qqbb-ROC work
  (`heppc-work` branch) used *separate* low-level classifiers and is a different thread.

Both are needed to make the limits plot. The new reco model alone does **not** reproduce it
— the (now-missing) classifiers are also required (§7, open question).

## 1. The pipeline (verified end-to-end, 2026-06-16)

```
RECO transformer (2-fold)  +  CLASSIFIER transformers (lvbb/qqbb, 2-fold)   [§4-5]
   → ApplyRecoAndClassifierToRoot.py   (heppc402; loads the 6 checkpoints)
   → decorated per-DSID, channel-split ROOT, tree "Events"
        actually-used copy that fed the fit:
        /eos/user/l/lubaines/CODE_HplusWh/TRExFitter/data/20250322v7_Root/   (on EOS — SURVIVES)
        (heppc402 also still has /data/atlas/baines/20250322v*_Root + 20250327v*_Root)
   → TRExFitter  (stat-only Asimov, var m_Wh, m_h-sideband CRs + NN-score SRs × {0,1+ b-tag},
                  POI mu_XS, FitType SPLUSB, FitRegion CRSR, LimitType ASYMPTOTIC / CL_s)
        transformer ("New") : /eos/user/l/lubaines/CODE_HplusWh/TRExFitter3/NewMethodFit5/   (SURVIVES)
        baseline   ("Old")  : /eos/user/l/lubaines/CODE_HplusWh/TRExFitter3/OldMethodFit/    (SURVIVES)
        per mass: .../Hp_allfits/Hp<MASS>_bkg/Limits/Asymptotics/myLimit.root
                  (TTree "stats": exp_upperlimit, exp_upperlimit_plus1/2, _minus1/2, obs_upperlimit)
   → limit-comparison plot  → comparison_and_ratio4.v2.pdf
```

Where each stage runs: reco/classifier training + apply-to-ROOT on **heppc402**; TRExFitter
fit on **lxplus** (also runnable on the **Mac via Docker** — `Work/HplusWh/TrexFitterStuff/
exampleTrexFitterDockerRunningScript.txt` uses `gitlab-registry.cern.ch/trexstats/trexfitter:latest`);
limit extraction + plotting on the Mac.

The `TRExFitter3/` dir also has `*_NewHighLevelNet*` runs = the "transformer reco +
high-level DNN classifier" mixed strategy the thesis mentions performs worse (not plotted).

## 2. lxplus access (how to get in non-interactively)

- CERN user is **`lubaines`**; EOS home `/eos/user/l/lubaines`, analysis tree
  `/eos/user/l/lubaines/CODE_HplusWh/`.
- **`kinit` / Kerberos does NOT log you into lxplus anymore** — lxplus offers only
  `publickey,keyboard-interactive`, so a ticket alone is rejected. Use **SSH connection
  multiplexing**: do ONE interactive login (password + 2FA) to open a master socket, then
  reuse it. The thesis-side `~/.ssh/config` lxplus block now has
  `ControlMaster auto` / `ControlPath ~/.ssh/sockets/cm-%r@%h:%p` / `ControlPersist 12h`.
  Open it with `ssh -fN lxplus.cern.ch` (forks after auth); then `ssh -o BatchMode=yes
  lxplus.cern.ch '<cmd>'` works with no prompt for 12h, pinned to one node.
- More permanent: register an SSH public key in the CERN account portal (removes the daily
  interactive step). Not yet done.

## 3. Reproduced numbers — they match the thesis plot exactly

Expected 95% CL upper limit on σ×BR (POI `mu_XS`; with xsec×BR≡1 pb normalisation):

| Mass [TeV] | OldMethodFit (baseline) | NewMethodFit5 (transformer) | New/Old |
|---|---|---|---|
| 0.8 | 0.01164 | 0.003493 | 0.300 |
| 0.9 | 0.00841 | 0.002908 | 0.346 |
| 1.0 | 0.005679 | 0.002236 | 0.394 |
| 1.2 | 0.00353 | 0.001608 | 0.456 |
| 1.4 | 0.002452 | 0.001278 | 0.521 |
| 1.6 | 0.001765 | 0.001024 | 0.580 |
| 1.8 | 0.001324 | 0.0008319 | 0.628 |
| 2.0 | 0.001111 | 0.0007529 | 0.678 |
| 2.5 | 0.0007507 | 0.0005163 | 0.688 |
| 3.0 | 0.0006537 | 0.0003856 | 0.590 |

These reproduce the blue (Baseline) and red (Transformer) curves and the New/Old ratio
panel of `comparison_and_ratio4.v2.pdf` bit-for-bit. ⇒ the thesis plot = `NewMethodFit5`
(transformer) vs `OldMethodFit` (baseline), fed by `20250322v7_Root`.

## 4. Reconstruction model provenance (IN scope for this repo)

The model behind the "New" (transformer) limit, per `ApplyRecoAndClassifierToRoot.py`
(and its `.bak20250322` twin) on the `heppc-work` branch:

- fold0: `output/20250321-124811_TrainingOutput/models/Nplits2_ValIdx0/chkpt4_27415.pth`
- fold1: `output/20250321-124953_TrainingOutput/models/Nplits2_ValIdx1/chkpt4_27440.pth`
- arch: `DeepSetsWithResidualSelfAttentionVariableTrueSkipReco`, `num_attention_blocks=3`,
  `include_mlp=False`, `hidden_dim=200`, `num_heads=4`, `embedding_size=10` — matches the
  thesis caption for `fig:TransformerVsOriginalExpectedLimits` (3 blocks, 4 heads, no MLP,
  d_m=200).

**CORRECTION (repo-side agent) to my earlier "wandb pruned to 25 runs, none from March":**
that was wrong from the repo side. This branch holds a local archive of **209 reco run
configs** in `docs/run_configs/` (Phase-0.3 pull, 2025-03-02 → 2026-06-10) that **includes
both March reco runs**: `20250321-124811_DSSARVTS3.json`, `20250321-124953_DSSARVTS3.json`.
Caveat (verified here): those JSONs are **summary-metrics only** — the `config` block is
empty (`{"magic":{"enable":true}}`), so they confirm the run existed + final val numbers
(e.g. 124811: run_id `h7gkkyok`, `state: failed`, epoch 29 / step 164490,
`train/Pct_MisPredict_all`≈0.022) but **NOT the hyperparameters**. (The "25" I saw was the
live `wandb/` dir on heppc402's clone, which is pruned — not this archive.)

Notable details to chase for the warm-start question:
- The run `state` is **"failed"** (both folds worth checking) — may relate to the warm-start bug.
- The **applied checkpoint is `chkpt4_…` (epoch 4)** although the run trained to epoch 29 —
  i.e. an early checkpoint was used for the plot, not the final one.
- The **checkpoints themselves are gone** from heppc402 (`output/2025032*` dirs pruned) AND
  the Mac (searched `~/Documents/PhD/Work` + mounted volumes; only later May/Jul-2025 reco
  checkpoints and the Mar-11/13 mech-interp scratch exist locally). Possibly on the Seagate
  drive (not mounted) or EOS (not yet found). Architecture is fully known regardless.

## 5. Classifier model provenance (OUT of scope here)

Per the apply script, the "New" limit also used per-channel classifiers (2-fold each):

- lvbb: `output/20250322-033930/lvbb_…ValIdx0/chkpt24_330225.pth` (heads=4, "Modified"),
  `output/20250322-005359/lvbb_…ValIdx1/chkpt24_332175.pth` (heads=2)
- qqbb: `output/20250322-011505/qqbb_…ValIdx0/chkpt24_229275.pth` (heads=2),
  `output/20250322-011528/qqbb_…ValIdx1/chkpt24_229925.pth` (heads=2)
- arch in the apply script: `…VariableTrueSkipClass[Modified]`, `num_attention_blocks=3`,
  `include_mlp=True`, `hidden_dim=256`, `hidden_dim_mlp=256`, `embedding_size=10/16`.

**CORRECTION (repo-side agent):** classifiers are **not this repo's work** — no checkpoints
here, none in the `run_configs` archive, nothing on HF. Checkpoints gone from heppc402 + Mac.
**Caption discrepancy for the classifier team:** the thesis caption says classification
transformers had `n_blocks=2`, but the apply script instantiates `num_attention_blocks=3`.
Needs resolving by whoever owns the classifiers.

## 6. What survives, and where

| Artifact | Location | Status |
|---|---|---|
| Reco run configs (summary only) | this branch `docs/run_configs/20250321-12481{1,3}_DSSARVTS3.json` | ✅ |
| Reco + classifier **checkpoints** | heppc402 `output/2025032*` & Mac | ❌ pruned/gone (arch known) |
| Decorated ROOT that fed the fit | EOS `…/TRExFitter/data/20250322v7_Root/` (+ heppc402 `/data/atlas/baines/20250322v*_Root`,`20250327v*_Root`) | ✅ |
| Transformer fit + limits | EOS `…/TRExFitter3/NewMethodFit5/…/myLimit.root` | ✅ |
| Baseline fit + limits | EOS `…/TRExFitter3/OldMethodFit/…/myLimit.root` | ✅ |
| TRExFitter configs | EOS `…/TRExFitter3/<run>/Hp<MASS>_all.config` | ✅ |

## 7. Recreating the plot with the NEW (fixed) reconstruction model

Only the "New" (transformer) side changes; `OldMethodFit` (cut-based baseline) is untouched.

1. **New reco checkpoints** + the (currently missing) **classifier checkpoints** →
   `ApplyRecoAndClassifierToRoot.py` on heppc402 → new decorated ROOT in the **same schema**
   as `20250322v7_Root` (per-DSID, channel-split `dsid_<DSID>_<chan>.root`, tree `Events`,
   branches incl. NN scores, reco `m_Wh`, `weights`, `dsid`).
2. Copy to EOS `…/TRExFitter/data/<newtag>_Root/`.
3. Run TRExFitter with the `NewMethodFit5` configs (`Hp<MASS>_all.config`), repointing
   `NtuplePath` → the new ROOT. (Locally via the TRExFitter Docker image, or on lxplus.)
4. Extract `exp_upperlimit` per mass from `…/Limits/Asymptotics/myLimit.root`; re-plot vs
   the unchanged `OldMethodFit` curve.

## 7b. The 2025 fit/plot code lives on EOS in `TRExFitter3/` (found 2026-06-16)

The Stage-3→7 code that the local repos lack is on EOS at
`/eos/user/l/lubaines/CODE_HplusWh/TRExFitter3/`:
- `runFitOverAllMassesNewMethod.sh` (and `…NewMethod2.sh`, `runFitOverAllMasses.sh`) — loop
  TRExFitter over all 10 mass points for the transformer ("New") strategy; `multiRunLoopScript.sh`.
- `GetLimitValues.py` — read `…/Limits/Asymptotics/myLimit.root` → limit-value tables.
- `PlotRat.py` — generates the `comparison_and_ratio*.pdf` (the 2-panel limit+ratio plot);
  `PlotRatXSecLimits_HpWh.py` is a variant. (The local `Work/HplusWh/ExclusionLimitPlottingCode/`
  is the OLDER 2023 published-analysis version — not these.)
- Output plots also live here: `comparison_and_ratio{0..4}.pdf`, `…4.v2.pdf`, and named
  variants (`…_BestOldMethodVsTransformer.pdf`, `…_BestOldMethodVsNew.pdf`, `…_OldVsOldBetter.pdf`).

Note: EOS `comparison_and_ratio4.v2.pdf` (md5 `5f07b6f…`) matches the **Feb-2025 IoP**
version; the **thesis** copy (`07b48da…`) differs in bytes — i.e. the thesis figure is a
**re-render of the same limit values** (the extracted `NewMethodFit5`/`OldMethodFit` numbers
match the thesis curves exactly, §3), not a byte-copy of the EOS file.

**Checkpoints are NOT on EOS** either (searched `/eos/user/l/lubaines` for `chkpt4_27415`
etc. — no hits). So reco + classifier checkpoints are gone from heppc402, Mac, and EOS;
only the unmounted Seagate drive remains untested.

⚠️ **These scripts exist only on EOS.** Given the checkpoints were already lost this way,
consider vendoring `PlotRat.py`, `GetLimitValues.py`, `runFitOverAllMassesNewMethod.sh` (+ a
representative `Hp1600_all.config`) into this repo so the recreation recipe is reproducible.

## 8. Open questions

- Was the **warm-start bug** in the **reconstruction** training, the **classifier**
  training, or both? (The plot's reco run is `state: failed`, epoch-4 checkpoint — suggestive.)
- **Classifier dependency:** recreating the limits needs classifiers. Their checkpoints are
  gone and out of this repo's scope — re-train, recover, or freeze the old decorated ROOT's
  classifier columns? Needs a decision with the classifier owner.
- Where is the **new reco model** to be pulled from (HF when ready, else which heppc402 run)?
- Caption `n_blocks` (2 vs 3) for the classifiers — classifier team.

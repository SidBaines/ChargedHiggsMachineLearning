# 2026-06-11 (evening) — Stage B: the competition sub-circuit (SB1-SB4, SB6)

Session on the new M5 Mac (setup: `2026-06-11_new-mac-m5-setup.md`). Ran the
preregistered Stage-B battery (`docs/H1_ASK_THE_JETS_TEST_PLAN.md` §Stage B, with
post-red-team amendments) on the **thesis model** (`thesis-ent1-bn1-d152`).
All experiments on a **fresh val slice: batches 19-24** (waves used 1-12, red-team
13-18; disjoint by construction). Boosted-qqbb events with pad slot(s), n≈3087.
Scripts: `experiments/h1/stageb_sb{1,2,3,4}_*.py` (SB6 rides inside SB2).

Also this session: **full wandb archive pulled** (Phase 0.3 closed) — see §wandb below.

## Headline

**No separate directed-inhibition edge is needed; B-ii is strongly supported and
localized in this model.** The edge-knockout evidence did not find a winner→loser
suppression path in the attention-edge framing. "Competition" between W-candidates
is sufficiently explained by the same context-relative scoring machinery that RT4
found for solo candidacy, applied per-jet: each jet scores its own hardness
*against the context it can see*. The circuit is two-stage, and each observed stage
is causally necessary under the tested interventions:

1. **Blocks 0-1 — silent context absorption**: jets read the event (including each
   other) symmetrically and diffusely. No margin effect visible yet (SB2), but cut
   these reads and candidacy itself collapses (SB4).
2. **Block 2 — the comparator/readout, carried almost entirely by b2h2** — the SAME
   head that broadcasts "W found" to the lepton and neutrino. The marginally-weaker
   candidate's claim flips here.

Exclusivity, ties, the winner's margin cost, and solo rest×2 suppression are all
one mechanism: relative scoring against visible context. Cut a comparable rival
from a jet's view and it claims again ("you're the hardest you can see"); cut all
context and it has no score at all.

## SB3 — jet↔jet attention forensics (`stageb_sb3_attention.py`)

- Claim bookkeeping replicates RT3 out-of-sample (tie → both 0.745; orig wins 85%
  of XOR; asym 600-vs-300 cells deterministic).
- **Only block 2 shows a big jet↔jet asymmetry**: a(loser→winner) ≈ 0.79/0.72/0.92-0.99
  on b2h0/b2h2/b2h3 vs a(winner→loser) ≈ 0.00/0.38/0.06. Identical in mirrored
  cells → tracks outcome, not insertion position/coherence. Blocks 0-1: symmetric,
  diffuse (differences ≤0.14).
- **RT-amendment overlap test hits exactly**: top-3 heads by competition asymmetry
  = top-3 by solo-rest×2 suppression mediation = {b2h0, b2h2, b2h3}. Shared
  machinery → B-ii.
- Solo600 reference: the W-jet itself attends context heavily at b0/b1 (0.86-1.0)
  but barely at b2 (it is the attended one); under rest×2 its b2 context-attention
  rises (+0.3-0.5) — it starts "reading" the now-harder rivals.

## SB2 — lens timing (`stageb_sb2_lens_timing.py`)

- **Large hardness gaps resolve early**: the same (300,80) candidate has blk1+ lens
  margin 0.44 solo but −0.75 with a (600,80) rival present — context suppression
  already at blocks 0-1 (matches solo rest×2 timing: suppressed events diverge from
  blk0+).
- **Near-ties resolve AT block 2**: tie-cell XOR events — both candidates claim
  after blk1 (P≈0.84-0.93, margins 1.1-1.9); block 2 flips the loser (P→0.00,
  margin → −0.3..−0.5) while keeping the winner. Exactly where SB3's attention
  asymmetry lives.
- Candidacy development (solo600): embed −0.40 → blk0+ 0.22 → blk1+ 3.54 → blk2+ 4.87.

### SB6 (ride-along) — the winner pays too

Winner's final margin in the tie vs its solo twin (same kinematics): median 4.87 →
1.51, per-event delta −3.3. The claim rate hid this (0.9725→0.9316). NOT the mild
one-sided suppression the original prereg guessed — competition costs both
candidates margin; the weaker one just crosses threshold. (Refines RT3's
0.980→0.937 observation.)

## SB4 — causal edge knockout (`stageb_sb4_edge_knockout.py`)

Method: faithful by-hand recompute of each attention block (per-head out-proj +
bottleneck math mirrored from `interp/activations.py`), with selected post-softmax
edges zeroed + row renormalized. Inactive-knockout output validated against the
library hook (max dev 0.0). No-renorm robustness arm agrees.

Tie cell (600,80)v(600,80):

| knockout | both claim | loser restored\|XOR |
|---|---|---|
| baseline | 0.745 | — |
| b2 o↔i all heads | 0.958 | **0.929** |
| b2 o↔i **head 2 only** | 0.960 | **0.935** |
| b2 o↔i head 0 / head 3 only | 0.743 / 0.744 | 0.006 / 0.043 |
| b0+b1 o↔i all heads ("control") | 0.971 | 0.934 |
| b0+b1+b2 o↔i | 0.992 | 0.997 |

- **b2h2 alone is the comparator** (h0/h3 knockouts do nothing despite b2h3 having
  the LARGEST loser→winner attention — attention mass ≠ causal influence; b2h3's
  value path doesn't drive the claim, consistent with its round-2e role).
- **Direction-resolved KO**: cutting o→i changes only orig's claim; cutting i→o
  changes only ins's. Each jet's suppression flows through its OWN read of the
  rival — no winner→loser inhibition edge exists. (Loser restored 0.139 vs 0.791
  for o→i vs i→o reflects that baseline losers are usually the ins jet.)
- **The b0+b1 "control" also restores** — not a failed control but the key finding:
  the b2h2 comparator needs the relative-standing evidence accumulated at b0/b1.
  Both stages individually necessary.
- P(lep=W) unchanged (0.006) under ALL tie knockouts: the lep/ν verdict does not
  route through jet↔jet edges (parallel-reader picture intact). With b0+b1+b2 cut,
  both jets claim near-independently (0.995/0.993 ≈ solo rates).

Solo conditions (KO W→all-context):

| condition | P(W claims) | P(lep=W) |
|---|---|---|
| solo600 | 0.9725 | 0.019 |
| solo600, KO b2 W→ctx | 0.8626 | 0.019 |
| solo600, **KO b0+b1 W→ctx** | **0.0000** | **0.697** |
| rest×2 | 0.6767 | 0.237 |
| rest×2, KO b2 W→ctx | 0.3282 | 0.237 |
| rest×2, KO b0+b1 W→ctx | 0.0000 | 0.734 |

- Candidacy is **constitutively relational**: a jet cut off from the event at b0/b1
  cannot claim at all — and the lepton then claims W instead (the coherent
  "no hadronic W found → lvbb" downstream behavior, not off-manifold garbage).
- Context KO does NOT undo rest×2 suppression (it worsens it): the suppression is
  not a removable "inhibition signal" riding on the context reads — the context
  reads ARE the score computation.

## SB1 — dose-response (`stageb_sb1_dose_response.py`)

c1 fixed (600,80); c2 at (600r,80); both W-tagged. Two designs per the amended
prereg: **asym** (real in-event W vs foreign insert — RT3-style) and **sym** (real
W masked to PAD; two foreign candidates — no coherence confound by construction).

- **Sym crossing at r = 1.00 exactly** (P(c2 wins|one) = 0.5014 at r=1) — design
  validated, the comparator is unbiased between two equally-foreign candidates.
- **Asym crossing at r ≈ 1.07** → the in-event coherence handicap ≈ **7% in pT**
  (at r=1.00 the foreign candidate wins only 13% of XOR — replicates RT3's 12%).
- **Comparator width**: both-claim peaks at r=1 (0.72-0.75) and falls to half by
  |Δln pT| ≈ 10-15%, to ~zero by ~40-60%. Sharp but graded — score-ranking with a
  ~10% pT resolution, not a hard argmax.
- P(lep=W) falls monotonically with r (0.012→0.0003): a harder competitor anywhere
  strengthens the global "hadronic W found" verdict. Broadcast picture again.

## SB5 — skipped (superseded)

The prereg goal (localize WHEN suppression lands in the loser's stream) was
answered by SB2 (timing: blk2 for ties, blk0-1 for big gaps) + SB4 (causal carrier:
b2h2; evidence accumulated at b0/b1). Stream-patching would add little; park unless
a specific question resurfaces.

## Status of the B-hypotheses

- **B-iii** (independent thresholds): dead since RT3.
- **B-i strong** (mutual inhibition, forced winner): dead since RT3 (ties → both).
- **B-i weak** (a separate directed winner→loser suppression edge): strongly
  disfavoured in the tested attention-edge framing — direction-resolved KO shows
  each jet only suppresses ITSELF via its own reads; no winner→loser path was found.
  This does not rule out every possible multi-hop or token-state description of
  suppression, but it removes the need for a distinct directed-inhibition mechanism.
- **B-ii** (context-relative scoring): **strongly supported and localized** — b0/b1
  context absorption (necessary, silent) + b2h2 comparator (necessary, decisive),
  shared with solo candidacy scoring AND with the lep/ν verdict broadcast (b2h2 is
  the "W found" wire). One mechanism for: exclusivity, tie behavior, winner's
  margin cost, solo relative-hardness, channel verdict.

## wandb archive (Phase 0.3 closed)

- Logged in on the new Mac; pulled ALL runs of the 4 PhD-relevant projects
  (entity `luke-sid-baines-blank`): `HEP-Transformers-TruthMatchingReco` (206,
  reco/thesis), `HEP-Transformers` (341, classifier era), `ChargedHiggs_ExperimentalML`
  (2), `20250716_tmp` (1). 2 runs failed to pull (server-side errors on long-dead runs).
- **Configs + final summaries** → `docs/run_configs/` (flat for the reco project,
  matching the existing 9 files; subdirs for the other projects). ~13 MB total.
  Verified: no API keys/emails/hostnames/machine paths in the JSONs (metadata block
  deliberately excluded). NOTE: 2025-era runs logged NO hyperparameters to wandb
  (config = wandb "magic" flag only) — their value is run names (encode config) +
  full final metrics; hyperparam archaeology still rests on the registry.
- **History curves** (~306 MB csv.gz) → `tmp_wandb_history/` (gitignored;
  re-pullable). Puller: `tmp_pull_wandb.py` (gitignored, resumable; skips existing).

## Learnings

- **Attention mass ≠ causal weight**: b2h3 has the largest loser→winner attention
  (0.92-0.99) and zero causal effect on the claim; b2h2 (0.72) carries ~everything.
  Always pair attention forensics with edge knockout.
- **A "control" that fires can be the finding**: KO b0+b1 was preregistered as the
  expected-null control; it restoring claims revealed the two-stage structure.
- The surgical-attention recompute (SB4) validates to 0.0 dev against the library
  hook including the bottleneck path — reusable pattern for any future edge-level
  intervention (the hook in `stageb_sb4_edge_knockout.py` is self-contained).
- Claim *rates* hide margin effects (SB6): report margins alongside rates.
- Sym-vs-asym insertion designs cleanly separate mechanism from coherence handicap;
  the handicap is now a number (~7% pT-equivalent) rather than a confound.

## Addendum (same session): the narrative notebook

Sid asked for the whole H1 story (claims / evidence / certainty / ruled-out
alternatives / causal status / plots) as a playable artifact →
**`experiments/h1_narrative.py`** (jupytext percent format, repo `# %%` convention)
+ **`experiments/h1_narrative.ipynb`** (executed, plots inline). Six claims with a
confidence scoreboard; live reproductions on the fresh slice matched the archived
numbers (lockstep 0.9954; W-mask lep-flip 0.771 ≈ round-2c's 77%; lvbb insertion
claims 0.318 ≈ A4's 34%; wires AUC-vs-truth 0.987/0.97; SB1/SB2/SB4 reproduce).
Env additions for notebooks: `jupytext`, `ipykernel`, `nbconvert` (in `.venv`;
not yet in `requirements.txt` — refreeze when convenient).

## Addendum 2 (same session): A4b — Sid's claim-2 challenge, resolved

Sid spotted that the lvbb-insertion flip rate (~30%, A4) looked too low for
"the verdict follows the jets". Diagnostic (`experiments/h1/a4b_lvbb_insertion_dose.py`,
fresh slice): **nothing missing — the verdict is a GRADED hadronic-vs-leptonic
comparison.** (a) Broadcast tight: P(lep=W|ins claims)=0.06 vs 0.96, P(ν=lep)=0.98.
(b) The 30% was donor composition: flip rate by donor-pT/lepW-pT quartile =
0.001/0.05/0.39/0.83 (median lvbb leptonic-W pT 575 GeV). (c) Dose at
pT=r×pT(lepW): clean sigmoid crossing **r≈1.4**, saturating 0.93/0.96 — a hadronic
candidate must beat the leptonic W by ~40% pT to steal a true lvbb event (ties go
to the genuinely-W-like leptonic side; the cross-system analogue of SB1's
comparator, and literally b1h3's job). (d) Direction control: own-H-jet direction
≈ foreign — no large cross-system coherence effect. Explains the mask(77%)/insert
asymmetry: masking removes all hadronic evidence; inserting must BEAT live
leptonic evidence. **Claim 2 sharpened: "ask the jets" → "weigh the best hadronic
claim against the leptonic-W evidence; threshold scales with leptonic hardness."**
Narrative notebook claim 2 + scoreboard updated; A4b promoted to `experiments/h1/`.

**Learning**: averages over heterogeneous intervention strengths (random donors)
can masquerade as mechanism ceilings — always dose-control before concluding a
flip rate is "low".

## Addendum 3 (same session): LW1 — where the hadronic-vs-leptonic comparison
## lives and what "leptonic-W evidence" is made of

Sid asked: where is the leptonic side read from, and how is it calculated?
(`experiments/h1/lw1_leptonic_evidence.py`, fresh slice, thesis model.)

**Where (1) — at the readout, b1h3 literally IS the comparator.** Per-key value
decomposition of b1h3 at the ν position in clean lvbb (sanity: keys sum to the
cached scalar to 1e-6): **lepton −0.73, ν −0.58, ljets +0.06, sjets +0.15** — a
signed sum with leptonic keys pushing lvbb-ward and jet keys qqbb-ward. Identical
at the lep position (parallel readers again). b2h2 by contrast is purely jet-fed
(ljets 1.19, sjets 0.34, lepton/ν 0.000) — the one-sided hadronic-claim wire.

**Where (2) — the lepton learns the hadronic claim via b2h2 (+b1), and the
candidate learns the leptonic evidence both directly and MULTI-HOP.** At r=2.0
(lepton normally concedes, keeps 0.10): KO lep→ins all blocks → lep keeps 0.99 —
and the **candidate then concedes too (0.90→0.15)** despite none of ITS edges
being cut: it reads the now-confident lepton's stream and gives up. Competition
is bidirectional through token states, not just direct edges. Block-resolved:
b2-only restores lep to 0.62 (b2h2 alone 0.55), b1-only 0.24 (b1h3 alone 0.16).
KO lep+ν→ins: both keep W 0.99, candidate 0.02. Cutting only ν→ins splits the
lockstep (ν 0.98 vs lep 0.53) — shared wiring, separate inputs, as claimed.
D1 caveat: at r=1.0 no single edge-cut FREES the candidate (suppression is
distributed across its context reads; renormalization redistributes attention,
so "remove the leptonic keys" ≠ "remove the leptonic evidence" at the jet query).

**What (variables) — magnitudes only, lepton-weighted, no W-structure:**
- mT(lep,ν) correlation with both wires ≈ **0.00**; rotating ν to φ+180° (which
  RAISES mT but shrinks |p_lep+p_ν|) WEAKENS the evidence (lep keeps 0.68→0.54)
  → the model uses the vector-sum magnitude, not W-likeness. (Confirms the thesis
  robustness hint "ν angles irrelevant" — angles only matter through the sum.)
- Best single correlate of b1h3@ν: **pT(lepW)/HT(jets), ρ=−0.795** (vs −0.55 for
  pT(lepW) alone) — the comparator tracks a leptonic-to-hadronic hardness RATIO,
  the cross-system twin of the jets' relative-hardness candidacy.
- Both tokens contribute individually (lep×1.3 OR MET×1.3 each push lep-keeps
  0.68→0.83/0.85; both×1.3→0.93, both×0.77→0.29), with extra lepton weight beyond
  the vector sum (swap-magnitudes test 0.68→0.62; kill-MET hurts far less than
  the vector-sum prediction: 0.37 observed vs ~0.07 predicted from the A4b
  sigmoid at the implied r_eff≈2.6).

**Learnings**: (i) per-key value decomposition + edge KO disagree informatively —
decomposition shows WHERE a scalar's value comes from; KO shows what the network
DOES when a read is removed (renorm + multi-hop effects included). Use both.
(ii) Multi-hop: cutting reader→source edges can flip the SOURCE via the reader's
changed stream — interpret single-edge KOs with the full loop in mind.

## Next steps

- **Universality**: rerun SB1-SB4 on the fresh organism
  (`output/20260610-203258_TrainingOutput`, `--ckpt-override`) — expect pT-first
  ranking (RT3) but is the two-stage b-last-block comparator structure preserved?
- **F1 training dynamics**: when does the comparator form across the organism's 30
  checkpoints?
- **E4 PySR**: extract the candidacy formula with event-context features (RT4) —
  needs `pysr` (Julia) installed.
- **H-side competition** (parked in prereg): does the H-candidate selection share
  the b2h2-style comparator or use its own head?
- Thesis-model story is now arguably complete enough to draft the Phase-4 synthesis
  figure set for H1 (channel decision: candidacy → competition → broadcast).

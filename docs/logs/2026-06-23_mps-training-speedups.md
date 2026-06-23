# 2026-06-23 — MPS training speed-ups (no parameter/epoch changes, determinism kept)

> Goal: make `experiments/train_organism.py` and `experiments/train_joint.py` train
> **faster on this Mac's MPS** without touching anything that changes the model
> (size, depth, #epochs, LR, …) and while keeping determinism. Split into **Tier 1**
> (bit-identical, safe to apply blind) and **Tier 2** (deterministic but the RNG stream
> changes — needs one before/after metric-curve check). This log covers Tier 1 (done +
> committed) and will be extended with Tier 2 results.

## Where the time actually went (profiled, batch=4096, d20/2-block, MPS)

The model is tiny (~20k params) → training is **overhead-bound**, not matmul-bound.
Measured per-step / per-epoch costs *before* any change:

| Component | Cost | Note |
|---|--:|---|
| dataloader `next()` | 14.2 ms/batch | **11.2 ms is the per-sample object-shuffle Python loop** (4096× `torch.randperm`) |
| `_reset_indices()` (per loader, per epoch) | ~534 ms | 268 ms re-reading a memmap column (**constant!**) + ~334 ms `random.shuffle` |
| compute: pure model + CE floor | 24.6 ms/step | irreducible model work |
| compute: + `isvalid`+`correlation` | +7.4 ms | computed every step, **both penalties disabled → discarded** |
| compute: + attention hook (entropy off) | +8.3 ms | the by-hand attention re-derivation |
| compute: + entropy loop (entropy on) | +18 ms | organism default; mostly the heavy hook + per-head `.item()` syncs, **not** the entropy maths |

Two discoveries while profiling:
1. The loss always built a `loss_dict` that **both training scripts discard**, and that
   dict forces ~8 per-head `.item()` device syncs/step plus the disabled penalties.
2. The per-epoch sample shuffle uses **unseeded `random.shuffle`** (the scripts seed
   `torch`+`np` but never `random`), so current training is **already not fully
   deterministic** run-to-run.

## Tier 1 — what we changed (bit-identical, verified)

1. **`metrics/lowlevelrecometrics.py`** — `HEPLossWithEntropy.forward` gained
   `build_loss_dict=True`. Training passes `False`, which skips `isvalid_loss`,
   `correlation_loss`, and the discarded `loss_dict` (incl. its per-head `.item()`
   syncs). The entropy term still runs when `entropy_loss=True`. Default `True` keeps
   every other caller bit-identical.
2. **`dataloaders/lowleveldataloader.py`** — `_reset_indices` no longer re-reads the
   memmap event-number column each epoch; the constant train/val split is computed once
   (`_compute_base_indices`) and cached, then copied + reshuffled per epoch.
3. **`interp/activations.py`** — new `hook_attention_weights_only`: stores just the
   per-head attention weights `nn.MultiheadAttention` already returns
   (`need_weights=True, average_attn_weights=False`), instead of re-deriving attention by
   hand (the per-head *output* reconstruction is never read by the entropy loss).
4. **`experiments/train_{organism,joint}.py`** — register **no hook** when entropy is off
   and no bottleneck (joint default → model output bit-identical), the **lightweight**
   hook when entropy is on (organism default), and the **full** `hook_attention_heads`
   only when `--bottleneck` is set (where it must rewrite the forward). Both pass
   `build_loss_dict=False`.

## Correctness (verified this session)

- Joint / entropy-off, no hook: total loss **bit-identical, `|Δ|=0`**, and the model
  forward output is bit-identical with vs without the hook (`max|Δ|=0`).
- Organism / entropy-on, light hook: loss value `|Δ|=0`; the per-head attn weights are
  bit-identical to the old by-hand ones (`max|Δ|=0`), but the **entropy gradient matches
  only to ~7e-10** (different autograd path). Numerically equivalent, **not literally
  bit-for-bit** — flagged honestly; over a long Adam run the organism trajectory could
  in principle drift. (If strict bit-identity for organism is ever required, keep the
  full hook.)
- End-to-end `--smoke` runs pass for: organism (light hook), joint (no hook), organism
  `--bottleneck 4` (full hook), and `--device cpu`.

## Key results: before → after (MPS, batch 4096, d20/2-block)

| | Before | After | Speedup |
|---|--:|--:|--:|
| Compute/step — organism (entropy on) | 58.7 ms | **24.7 ms** | 2.4× |
| Compute/step — joint (entropy off) | 40.3 ms | **19.0 ms** | 2.1× |
| `_reset_indices` / epoch / loader | 534 ms | **239 ms** | 2.2× |
| **End-to-end/step** (compute + 14 ms dataloader), organism | ~72.7 ms | **~38.7 ms** | **1.9×** |
| **End-to-end/step**, joint | ~54.3 ms | **~33.0 ms** | **1.6×** |

Net ≈ **1.6–1.9× wall-clock**, no parameter/epoch changes. The joint win exceeded the
prediction because dropping the hook also lets `nn.MultiheadAttention` use its fused path
(no `need_weights=True`).

## Learnings / gotchas

- **The dataloader is now the dominant per-step cost** (~14 ms ≈ 40% of a step) → Tier 2
  (vectorise the object shuffle 11.2→0.69 ms; seed+speed the index shuffle) now buys a
  *larger relative* gain than it would have before Tier 1.
- The `Edit` tool kept failing on this repo's loss/loader because **blank lines carry 8
  trailing spaces** and `# comment` lines have exact indentation; precise multi-line edits
  here are more reliable via a small Python `str.replace` script (kept in scratchpad).
- The loss `forward` *required* the hook to have populated the cache even when entropy was
  off — with an empty cache the discarded `loss_dict` did `total_entropy_loss.item()` on
  an `int` and crashed. Gating `loss_dict` (Tier 1 #1) is what makes "no hook" safe.
- `random.shuffle` on a 170k numpy array is ~34 ms (Python-level); `np.random.shuffle` is
  ~2 ms **and** is seeded → Tier 2 will make it both faster and *more* deterministic.

## Status / next

- **Tier 1: DONE + committed** (this log + the 5 code files).
- **Tier 2: in progress** — vectorised object shuffle (`argsort(rand)`, same semantics) +
  `random.shuffle`→`np.random.shuffle`. Deterministic but RNG-stream-changing, so will
  run a short before/after to confirm the loss/val curves match before committing.
  Results to be appended here.

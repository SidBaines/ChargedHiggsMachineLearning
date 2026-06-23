#!/bin/bash
# Queue B (2026-06-22, wandb ON -> project HEP-Transformers-TruthMatchingReco):
#  (1) G1c cat5-closing probe: isolate which lever lifts boosted-hadronic cat5 toward the
#      d152 ref 0.936 (epochs vs width vs depth vs MLP), still boosted-only (--keep-cats 4,5).
#      Anchor from the 30-ep grid: d32 b2 h4 attn-only = cat4 0.970 / cat5 0.918.
#  (2) A2 free-lunch probe: d152 `none` normal transformer to 100 epochs, ALL categories,
#      to test whether longer training lifts the hard cats 0-3 for free (curves were still
#      rising at 30 ep). Informs every goal's epoch budget.
# Resumable: skips any tag whose log already says "done in". Logs -> tmp_suite/<tag>.log
# Usage:  caffeinate -i bash experiments/run_queue_b.sh
cd "$(dirname "$0")/.."
mkdir -p tmp_suite
PY=.venv/bin/python
# NB: no --no-wandb => each run logs to wandb.
WIN="--device mps --lr 1e-3 --lr-low 5e-7 --schedule cosine --warmup-mode ramp --entropy-weight 0"

run () {  # tag, then trainer args...
  local tag=$1; shift
  if grep -q "done in" "tmp_suite/$tag.log" 2>/dev/null; then echo "skip $tag (done)"; return; fi
  echo "=== $tag ($(date +%H:%M)) ==="
  $PY experiments/train_organism.py "$@" > "tmp_suite/$tag.log" 2>&1
  grep -oE "'val/PerfectRecoPct_all(_cat[0-5])?': tensor\([0-9.eE+-]+\)" "tmp_suite/$tag.log" | tail -7
}

# (1) cat5-closing probe (boosted-only) — one lever each off the d32 b2 h4 anchor
run g1c_d32b2h4_ep80    $WIN --keep-cats 4,5 --d-model 32 --blocks 2 --num-heads 4 --no-mlp --epochs 80
run g1c_d48b2h4_nomlp   $WIN --keep-cats 4,5 --d-model 48 --blocks 2 --num-heads 4 --no-mlp
run g1c_d32b3h4_nomlp   $WIN --keep-cats 4,5 --d-model 32 --blocks 3 --num-heads 4 --no-mlp
run g1c_d32b2h4_mlp128  $WIN --keep-cats 4,5 --d-model 32 --blocks 2 --num-heads 4 --d-mlp 128

# (2) A2 free-lunch probe (all categories, normal d152, long)
run a2_d152_none_ep100  $WIN --d-model 152 --blocks 3 --num-heads 4 --d-mlp 400 --epochs 100 --seed 0

echo "===== QUEUE B DONE ($(date +%H:%M)) ====="
for f in tmp_suite/g1c_*.log tmp_suite/a2_*.log; do
  [ -f "$f" ] && echo "$(basename "$f" .log): $(grep -oE "'val/PerfectRecoPct_all(_cat[0-5])?': tensor\([0-9.eE+-]+\)" "$f" | tail -7 | tr '\n' ' ')"
done

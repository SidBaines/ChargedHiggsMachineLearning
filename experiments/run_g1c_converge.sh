#!/bin/bash
# G1c CONVERGENCE sweep: minimal BOOSTED-ONLY (cats 4,5) organisms trained to completion.
# 300 epochs each (way more than the ~80 that was near-converged) so results are
# capacity-limited, not under-training-limited. Normal transformer, ATTENTION-ONLY,
# winner recipe (cosine over the full 300 ep), wandb ON.
# Target: match the d152 all-cats reference  cat4=0.960  cat5=0.936.
# Resumable: skips any tag whose log already says "done in". Logs -> tmp_suite/<tag>.log
# Usage:  caffeinate -i bash experiments/run_g1c_converge.sh
# NB: smoke-test the harness first after the 2026-06-22 eval/ckpt tweaks:
#   .venv/bin/python experiments/train_organism.py --smoke --device mps --no-mlp \
#       --num-heads 4 --d-model 16 --blocks 2 --keep-cats 4,5 --entropy-weight 0
cd "$(dirname "$0")/.."
mkdir -p tmp_suite
PY=.venv/bin/python
WIN="--device mps --lr 1e-3 --lr-low 5e-7 --schedule cosine --warmup-mode ramp --entropy-weight 0 --keep-cats 4,5 --epochs 300 --num-heads 4 --no-mlp"

run () {  # tag, then trainer args...
  local tag=$1; shift
  if grep -q "done in" "tmp_suite/$tag.log" 2>/dev/null; then echo "skip $tag (done)"; return; fi
  echo "=== $tag ($(date +%H:%M)) ==="
  $PY experiments/train_organism.py "$@" > "tmp_suite/$tag.log" 2>&1
  grep -oE "'val/PerfectRecoPct_all_cat[45]': tensor\([0-9.eE+-]+\)" "tmp_suite/$tag.log" | tail -2
}

# width frontier @ depth 2 (attention-only, 4 heads)
run cvg_d8b2h4   $WIN --d-model 8  --blocks 2
run cvg_d12b2h4  $WIN --d-model 12 --blocks 2
run cvg_d16b2h4  $WIN --d-model 16 --blocks 2
run cvg_d24b2h4  $WIN --d-model 24 --blocks 2
run cvg_d32b2h4  $WIN --d-model 32 --blocks 2
run cvg_d48b2h4  $WIN --d-model 48 --blocks 2
run cvg_d64b2h4  $WIN --d-model 64 --blocks 2
# depth 3 at small width (does an extra block close cat5 cheaper than width?)
run cvg_d16b3h4  $WIN --d-model 16 --blocks 3
run cvg_d24b3h4  $WIN --d-model 24 --blocks 3
run cvg_d32b3h4  $WIN --d-model 32 --blocks 3

echo "===== G1c CONVERGENCE SWEEP DONE ($(date +%H:%M)) ====="
for f in tmp_suite/cvg_*.log; do
  echo "$(basename "$f" .log): $(grep -oE "'val/PerfectRecoPct_all_cat[45]': tensor\([0-9.eE+-]+\)" "$f" | tail -2 | tr '\n' ' ')"
done

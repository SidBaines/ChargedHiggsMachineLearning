#!/bin/bash
# Goal 1c: minimal BOOSTED-ONLY (cats 4,5) interpretable reco organism.
# Normal transformer (no entropy / no bottleneck), winner recipe, train only on cats 4,5.
# Target: match d152 `none`  cat4=0.960  cat5=0.936.
# Resumable: skips any tag whose log already says "done in". Logs -> tmp_suite/<tag>.log
# Usage:  caffeinate -i bash experiments/run_min_boosted.sh
cd "$(dirname "$0")/.."
mkdir -p tmp_suite
PY=.venv/bin/python
RECIPE="--device mps --no-wandb --entropy-weight 0 --lr 1e-3 --lr-low 5e-7 --schedule cosine --warmup-mode ramp --keep-cats 4,5"

run () {  # tag, then trainer args...
  local tag=$1; shift
  if grep -q "done in" "tmp_suite/$tag.log" 2>/dev/null; then echo "skip $tag (done)"; return; fi
  echo "=== $tag ($(date +%H:%M)) ==="
  $PY experiments/train_organism.py "$@" > "tmp_suite/$tag.log" 2>&1
  grep -oE "'val/PerfectRecoPct_all_cat[45]': tensor\([0-9.eE+-]+\)" "tmp_suite/$tag.log" | tail -2
}

#    tag                   d_model blocks heads  mlp?
run minB_d8b1h2_nomlp   $RECIPE --d-model 8  --blocks 1 --num-heads 2 --no-mlp
run minB_d16b1h2_nomlp  $RECIPE --d-model 16 --blocks 1 --num-heads 2 --no-mlp
run minB_d16b2h2_nomlp  $RECIPE --d-model 16 --blocks 2 --num-heads 2 --no-mlp
run minB_d24b2h2_nomlp  $RECIPE --d-model 24 --blocks 2 --num-heads 2 --no-mlp
run minB_d32b2h4_nomlp  $RECIPE --d-model 32 --blocks 2 --num-heads 4 --no-mlp
run minB_d16b2h2_mlp64  $RECIPE --d-model 16 --blocks 2 --num-heads 2 --d-mlp 64

echo "===== MIN-BOOSTED GRID DONE ($(date +%H:%M)) ====="
for f in tmp_suite/minB_*.log; do
  echo "$(basename "$f" .log): $(grep -oE "'val/PerfectRecoPct_all_cat[45]': tensor\([0-9.eE+-]+\)" "$f" | tail -2 | tr '\n' ' ')"
done

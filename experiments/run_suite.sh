#!/bin/bash
# Phase-2 suite: constraint {none, ent, bn1, both} x size {d20-2blk, d152-3blk}
# x seeds {0,1,2}, winner recipe (1e-3 -> 5e-7 cosine, ramp warmup), MPS.
# Plus 2 legacy-recipe comparability runs. Resumable: skips tags whose log says done.
# Usage: bash experiments/run_suite.sh   (logs -> tmp_suite/<tag>.log)
cd "$(dirname "$0")/.."
mkdir -p tmp_suite
PY=.venv/bin/python
RECIPE="--device mps --no-wandb --lr 1e-3 --lr-low 5e-7 --schedule cosine --warmup-mode ramp"
LEGACY="--device mps --no-wandb --lr 3e-4 --lr-low 5e-7 --schedule log --warmup-mode legacy"

run () {  # tag, then trainer args...
  local tag=$1; shift
  if grep -q "done in" "tmp_suite/$tag.log" 2>/dev/null; then
    echo "skip $tag (done)"; return
  fi
  echo "=== $tag ($(date +%H:%M)) ==="
  $PY experiments/train_organism.py "$@" > "tmp_suite/$tag.log" 2>&1
  grep -E "done in" "tmp_suite/$tag.log" | tail -1
  grep -oE "'val/PerfectRecoPct_all': [0-9.]+" "tmp_suite/$tag.log" | tail -1
}

SIZES=("d20  --d-model 20  --blocks 2 --d-mlp 200"
       "d152 --d-model 152 --blocks 3 --d-mlp 400")
CONS=("none --entropy-weight 0"
      "ent"
      "bn1  --bottleneck 1 --entropy-weight 0"
      "both --bottleneck 1")

for size_spec in "${SIZES[@]}"; do
  set -- $size_spec; sz=$1; shift; szargs="$*"
  for con_spec in "${CONS[@]}"; do
    set -- $con_spec; cn=$1; shift; cnargs="$*"
    for seed in 0 1 2; do
      run "${sz}_${cn}_s${seed}" $RECIPE $szargs $cnargs --seed $seed
    done
  done
done
# legacy-recipe comparability runs (exact thesis-era schedule)
run "d152_both_s0_LEGACY" $LEGACY --d-model 152 --blocks 3 --d-mlp 400 --bottleneck 1 --seed 0
run "d20_ent_s0_LEGACY"   $LEGACY --d-model 20  --blocks 2 --d-mlp 200 --seed 0

echo "===== SUITE COMPLETE ====="
for f in tmp_suite/*.log; do
  v=$(grep -oE "'val/PerfectRecoPct_all': [0-9.]+" "$f" | tail -1)
  echo "$(basename $f .log): $v"
done

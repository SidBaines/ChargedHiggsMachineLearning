#!/usr/bin/env bash
set -euo pipefail

# Re-run of the remaining two-head joint schedules after the best_asimov_z
# O(N^2)->O(N log N) fix (single_softor + twohead_joint already completed with the
# correct-but-slow algorithm, so they are not redone). Pseudo-background (DSID 510124),
# so classification numbers are plumbing diagnostics only, not physics.

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
PSEUDO_BKG_DSID="${PSEUDO_BKG_DSID:-510124}"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="output/joint_first_set_${STAMP}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

COMMON=(
  --device mps
  --no-wandb
  --pseudo-bkg-dsid "${PSEUDO_BKG_DSID}"
  --d-model 20
  --blocks 2
  --num-heads 4
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
)

run_train() {
  local name="$1"; shift
  local log="${LOG_DIR}/${name}.log"
  echo "== $(date '+%F %T') starting ${name} ==" | tee -a "${RUN_ROOT}/manifest.log"
  echo ".venv/bin/python experiments/train_joint.py ${COMMON[*]} $*" | tee -a "${RUN_ROOT}/manifest.log"
  .venv/bin/python experiments/train_joint.py "${COMMON[@]}" "$@" 2>&1 | tee "${log}"
  echo "== $(date '+%F %T') finished ${name} ==" | tee -a "${RUN_ROOT}/manifest.log"
}

echo "run_root: ${RUN_ROOT}" | tee "${RUN_ROOT}/manifest.log"
echo "epochs: ${EPOCHS}" | tee -a "${RUN_ROOT}/manifest.log"
echo "batch_size: ${BATCH_SIZE}" | tee -a "${RUN_ROOT}/manifest.log"
echo "pseudo_bkg_dsid: ${PSEUDO_BKG_DSID}" | tee -a "${RUN_ROOT}/manifest.log"
echo "note: remaining-runs re-run after best_asimov_z vectorisation" | tee -a "${RUN_ROOT}/manifest.log"

run_train twohead_int      --mode twohead --schedule int
run_train twohead_intdetach --mode twohead --schedule intdetach
run_train twohead_seqfull   --mode twohead --schedule seqfull
run_train twohead_seqfrozen --mode twohead --schedule seqfrozen

echo "== $(date '+%F %T') all remaining runs complete ==" | tee -a "${RUN_ROOT}/manifest.log"

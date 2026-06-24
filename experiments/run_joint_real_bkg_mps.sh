#!/usr/bin/env bash
set -uo pipefail
# ---------------------------------------------------------------------------
# Overnight REAL-BACKGROUND joint reco+classifier queue (2026-06-24).
#
# First runs on REAL background memmaps (DSIDs: signal 510115-124 + the full
# background cocktail ttbar/W+Z+jets/diboson, ~20.5M bkg events) copied locally
# to tmp_data_20250321v1_all/. Fixed scale d20/b2/h4, 3-class event head, lambda=1.
#
# Covers the plan's Phase 2 (S vs T-INT + baselines) AND Phase 4 (all 5 schedules)
# AND baselines #1 (reco), #2 (standalone classifier), #3 (frozen probe) — a
# complete head-to-head at one scale, comparable to the pseudo-bkg diagnostic table
# and the reco anchor (d20 b2h4 = 0.847 @30ep / ~0.80 @10ep).
#
# CAVEATS (see plan R6): classification AUC/Z use abs(MC_Wts), NOT per-DSID
# luminosity-normalised yields -> AUC is a sound COMPARATIVE metric; absolute Z is
# uncalibrated. W&B off for unattended robustness (re-run with W&B for publishable).
#
# Runs are ORDERED by scientific priority (most informative first) so a short night
# still yields the key comparisons. Each run is independent (fresh process).
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
DATA_DIR="${DATA_DIR:-tmp_data_20250321v1_all}"
D_MODEL="${D_MODEL:-20}"
BLOCKS="${BLOCKS:-2}"
HEADS="${HEADS:-4}"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="output/joint_real_bkg_${STAMP}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"
MAN="${RUN_ROOT}/manifest.log"

COMMON=(
  --device mps
  --no-wandb
  --data-dir "${DATA_DIR}"
  --real-bkg
  --position-split
  --d-model "${D_MODEL}"
  --blocks "${BLOCKS}"
  --num-heads "${HEADS}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --event-classes 3
)

log() { echo "$@" | tee -a "${MAN}"; }

run_train() {
  local name="$1"; shift
  local log="${LOG_DIR}/${name}.log"
  log "== $(date '+%F %T') START ${name} =="
  log ".venv/bin/python -u experiments/train_joint.py ${COMMON[*]} $*"
  PYTHONUNBUFFERED=1 .venv/bin/python -u experiments/train_joint.py "${COMMON[@]}" "$@" 2>&1 | tee "${log}"
  local rc="${PIPESTATUS[0]}"
  log "== $(date '+%F %T') END ${name} (exit ${rc}) =="
}

log "run_root:   ${RUN_ROOT}"
log "data_dir:   ${DATA_DIR} (REAL background)"
log "scale:      d${D_MODEL} b${BLOCKS} h${HEADS}  epochs=${EPOCHS} batch=${BATCH_SIZE}"
log "started:    $(date '+%F %T')"
log "NOTE: classification AUC = comparative; absolute Z uncalibrated (R6). W&B off."

# Priority order: classification anchor -> headline joint -> the other structural
# approach (S) -> frozen probe -> interleaved -> reco anchor -> detach -> forgetting.
run_train standalone_classifier --mode twohead --schedule seqfull   --seq-split 0.0 --lambda-event 1.0
run_train twohead_joint         --mode twohead --schedule joint     --lambda-event 1.0
run_train single_softor         --mode single  --readout softor
run_train twohead_seqfrozen     --mode twohead --schedule seqfrozen --seq-split 0.5 --lambda-event 1.0
run_train twohead_int           --mode twohead --schedule int       --lambda-event 1.0
run_train pure_reco             --mode twohead --schedule joint      --lambda-event 0.0
run_train twohead_intdetach     --mode twohead --schedule intdetach --lambda-event 1.0
run_train twohead_seqfull       --mode twohead --schedule seqfull    --seq-split 0.5 --lambda-event 1.0

log "== $(date '+%F %T') ALL RUNS COMPLETE =="

#!/usr/bin/env bash
# run_ood_all.sh — full OOD evaluation for all methods × all priv_modes × all dr_scales
#
# Evaluates:
#   Baseline (1 config)
#   RMA2 Phase-2 FULL/INT/EXT × l8/l16/l32 (9 configs)
#   CTS  FULL/INT/EXT × l8/l16/l32  +  FULL l64/l128  (11 configs, cts_ext_l8 skipped)
#
# Each config tested at dr_scale 1.0 / 1.5 / 2.0 = 3 scales
# Total: (1 + 9 + 11) × 3 = 63 eval runs
#
# Results saved to results/ood_results_all.csv
# Summary table printed at the end.
#
# Usage:
#   bash scripts/one_leg/run_ood_all.sh

set -e
cd "$(dirname "$0")/../.."

RESULTS="results/ood_results_all.csv"
N_EPS=100
N_ENV=64
mkdir -p results

LOG="logs/one_leg/train_progress.log"
log() { echo "[OOD] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

log "===== OOD Evaluation START ====="

run_eval() {
    local METHOD=$1 PRIV=$2 CKPT=$3 SCALE=$4 P2=${5:-""}
    local P2_ARG=""
    [ -n "$P2" ] && P2_ARG="--phase2_checkpoint $P2"
    log "${METHOD} ${PRIV} scale=${SCALE}"
    python scripts/one_leg/eval_ood.py \
        --method "$METHOD" --priv_mode "$PRIV" \
        --checkpoint "$CKPT" $P2_ARG \
        --dr_scale "$SCALE" \
        --num_episodes $N_EPS --num_envs $N_ENV \
        --results_file "$RESULTS" --headless \
        && log "DONE ${METHOD} ${PRIV} scale=${SCALE}" \
        || log "FAIL ${METHOD} ${PRIV} scale=${SCALE}"
}

# ── Baseline ─────────────────────────────────────────────────────────────────
BASELINE_CKPT="logs/one_leg/baseline/2026-04-28_10-39-55/model_final.pt"
for S in 1.0 1.5 2.0; do
    run_eval baseline FULL "$BASELINE_CKPT" $S
done

# ── RMA Phase-2 (deployment) ──────────────────────────────────────────────────
declare -A RMA_P1 RMA_P2
RMA_P1[FULL_l8]="logs/one_leg/rma/2026-04-28_11-27-42_one_leg_rma_full_l8/model_final.pt"
RMA_P1[INT_l8]="logs/one_leg/rma/2026-04-28_12-16-18_one_leg_rma_int_l8/model_final.pt"
RMA_P1[EXT_l8]="logs/one_leg/rma/2026-04-28_13-05-11_one_leg_rma_ext_l8/model_final.pt"
RMA_P1[FULL_l16]="logs/one_leg/rma/2026-04-28_13-55-36_one_leg_rma_full_l16/model_final.pt"
RMA_P1[INT_l16]="logs/one_leg/rma/2026-04-28_14-45-37_one_leg_rma_int_l16/model_final.pt"
RMA_P1[EXT_l16]="logs/one_leg/rma/2026-04-28_15-35-54_one_leg_rma_ext_l16/model_final.pt"
RMA_P1[FULL_l32]="logs/one_leg/rma/2026-04-28_10-40-04_one_leg_rma_full_l32/model_final.pt"
RMA_P1[INT_l32]="logs/one_leg/rma/2026-04-28_11-28-35_one_leg_rma_int_l32/model_final.pt"
RMA_P1[EXT_l32]="logs/one_leg/rma/2026-04-28_12-16-43_one_leg_rma_ext_l32/model_final.pt"

RMA_P2[FULL_l8]="logs/one_leg/rma/2026-04-28_21-17-20_one_leg_rma_p2_full_l8/adaptation_module.pt"
RMA_P2[FULL_l16]="logs/one_leg/rma/2026-04-28_21-17-27_one_leg_rma_p2_full_l16/adaptation_module.pt"
RMA_P2[FULL_l32]="logs/one_leg/rma/2026-04-28_21-17-42_one_leg_rma_p2_full_l32/adaptation_module.pt"
RMA_P2[INT_l8]="logs/one_leg/rma/2026-04-28_21-17-51_one_leg_rma_p2_int_l8/adaptation_module.pt"
RMA_P2[INT_l16]="logs/one_leg/rma/2026-04-28_21-18-00_one_leg_rma_p2_int_l16/adaptation_module.pt"
RMA_P2[INT_l32]="logs/one_leg/rma/2026-04-28_21-18-17_one_leg_rma_p2_int_l32/adaptation_module.pt"
RMA_P2[EXT_l8]="logs/one_leg/rma/2026-04-28_21-18-26_one_leg_rma_p2_ext_l8/adaptation_module.pt"
RMA_P2[EXT_l16]="logs/one_leg/rma/2026-04-28_21-18-34_one_leg_rma_p2_ext_l16/adaptation_module.pt"
RMA_P2[EXT_l32]="logs/one_leg/rma/2026-04-28_21-18-50_one_leg_rma_p2_ext_l32/adaptation_module.pt"

for KEY in FULL_l8 INT_l8 EXT_l8 FULL_l16 INT_l16 EXT_l16 FULL_l32 INT_l32 EXT_l32; do
    PRIV="${KEY%%_*}"
    for S in 1.0 1.5 2.0; do
        run_eval rma2 "$PRIV" "${RMA_P1[$KEY]}" $S "${RMA_P2[$KEY]}"
    done
done

# ── CTS (student deployment) ──────────────────────────────────────────────────
declare -A CTS_CKPT
CTS_CKPT[FULL_l8]="logs/one_leg/cts/2026-04-28_16-26-13_one_leg_cts_full_l8/model_final.pt"
CTS_CKPT[INT_l8]="logs/one_leg/cts/2026-04-28_17-29-34_one_leg_cts_int_l8/model_final.pt"
# cts_ext_l8 skipped — model_final.pt missing
CTS_CKPT[FULL_l16]="logs/one_leg/cts/2026-04-28_14-01-17_one_leg_cts_full_l16/model_final.pt"
CTS_CKPT[INT_l16]="logs/one_leg/cts/2026-04-28_15-02-13_one_leg_cts_int_l16/model_final.pt"
CTS_CKPT[EXT_l16]="logs/one_leg/cts/2026-04-28_16-02-51_one_leg_cts_ext_l16/model_final.pt"
CTS_CKPT[FULL_l32]="logs/one_leg/cts/2026-04-28_16-41-37_one_leg_cts_full_l32/model_final.pt"
CTS_CKPT[INT_l32]="logs/one_leg/cts/2026-04-28_11-38-58_one_leg_cts_int_l32/model_final.pt"
CTS_CKPT[EXT_l32]="logs/one_leg/cts/2026-04-28_12-37-44_one_leg_cts_ext_l32/model_final.pt"
CTS_CKPT[FULL_l64]="logs/one_leg/cts/2026-04-28_17-05-24_one_leg_cts_full_l64/model_final.pt"
CTS_CKPT[FULL_l128]="logs/one_leg/cts/2026-04-28_15-39-51_one_leg_cts_full_l128/model_final.pt"

for KEY in FULL_l8 INT_l8 FULL_l16 INT_l16 EXT_l16 FULL_l32 INT_l32 EXT_l32 FULL_l64 FULL_l128; do
    PRIV="${KEY%%_*}"
    for S in 1.0 1.5 2.0; do
        run_eval cts "$PRIV" "${CTS_CKPT[$KEY]}" $S
    done
done

log "===== OOD Evaluation DONE — printing summary ====="
python scripts/one_leg/format_results.py "$RESULTS" --save_md

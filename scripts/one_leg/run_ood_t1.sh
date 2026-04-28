#!/usr/bin/env bash
# OOD Terminal 1 — Baseline + RMA2 l8 + RMA2 l16  (21 runs)
set -e
cd "$(dirname "$0")/../.."

N_EPS=100; N_ENV=64
RESULTS="results/ood_results_all.csv"
LOG="logs/one_leg/train_progress.log"
mkdir -p results logs/one_leg

log() { echo "[OOD-T1] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

run() {
    local METHOD=$1 PRIV=$2 CKPT=$3 SCALE=$4 P2=${5:-""}
    local P2_ARG=""; [ -n "$P2" ] && P2_ARG="--phase2_checkpoint $P2"
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

log "===== OOD T1 START ====="

# Baseline (3 runs) -----------------------------------------------------------
BASELINE="logs/one_leg/baseline/2026-04-28_10-39-55/model_final.pt"
for S in 1.0 1.5 2.0; do run baseline FULL "$BASELINE" $S; done

# RMA2 l=8 (9 runs) -----------------------------------------------------------
declare -A P1 P2
P1[FULL]="logs/one_leg/rma/2026-04-28_11-27-42_one_leg_rma_full_l8/model_final.pt"
P1[INT]="logs/one_leg/rma/2026-04-28_12-16-18_one_leg_rma_int_l8/model_final.pt"
P1[EXT]="logs/one_leg/rma/2026-04-28_13-05-11_one_leg_rma_ext_l8/model_final.pt"
P2[FULL]="logs/one_leg/rma/2026-04-28_21-17-20_one_leg_rma_p2_full_l8/adaptation_module.pt"
P2[INT]="logs/one_leg/rma/2026-04-28_21-17-51_one_leg_rma_p2_int_l8/adaptation_module.pt"
P2[EXT]="logs/one_leg/rma/2026-04-28_21-18-26_one_leg_rma_p2_ext_l8/adaptation_module.pt"
for PRIV in FULL INT EXT; do
    for S in 1.0 1.5 2.0; do run rma2 "$PRIV" "${P1[$PRIV]}" $S "${P2[$PRIV]}"; done
done

# RMA2 l=16 (9 runs) ----------------------------------------------------------
P1[FULL]="logs/one_leg/rma/2026-04-28_13-55-36_one_leg_rma_full_l16/model_final.pt"
P1[INT]="logs/one_leg/rma/2026-04-28_14-45-37_one_leg_rma_int_l16/model_final.pt"
P1[EXT]="logs/one_leg/rma/2026-04-28_15-35-54_one_leg_rma_ext_l16/model_final.pt"
P2[FULL]="logs/one_leg/rma/2026-04-28_21-17-27_one_leg_rma_p2_full_l16/adaptation_module.pt"
P2[INT]="logs/one_leg/rma/2026-04-28_21-18-00_one_leg_rma_p2_int_l16/adaptation_module.pt"
P2[EXT]="logs/one_leg/rma/2026-04-28_21-18-34_one_leg_rma_p2_ext_l16/adaptation_module.pt"
for PRIV in FULL INT EXT; do
    for S in 1.0 1.5 2.0; do run rma2 "$PRIV" "${P1[$PRIV]}" $S "${P2[$PRIV]}"; done
done

log "===== OOD T1 FINISHED ====="

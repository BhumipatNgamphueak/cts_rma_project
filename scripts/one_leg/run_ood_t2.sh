#!/usr/bin/env bash
# OOD Terminal 2 — RMA2 l32 + CTS FULL/INT l8 + CTS FULL/INT/EXT l16  (21 runs)
set -e
cd "$(dirname "$0")/../.."

N_EPS=100; N_ENV=64
RESULTS="results/ood_results_all.csv"
LOG="logs/one_leg/train_progress.log"
mkdir -p results logs/one_leg

log() { echo "[OOD-T2] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

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

log "===== OOD T2 START ====="

# RMA2 l=32 (9 runs) ----------------------------------------------------------
declare -A P1 P2
P1[FULL]="logs/one_leg/rma/2026-04-28_10-40-04_one_leg_rma_full_l32/model_final.pt"
P1[INT]="logs/one_leg/rma/2026-04-28_11-28-35_one_leg_rma_int_l32/model_final.pt"
P1[EXT]="logs/one_leg/rma/2026-04-28_12-16-43_one_leg_rma_ext_l32/model_final.pt"
P2[FULL]="logs/one_leg/rma/2026-04-28_21-17-42_one_leg_rma_p2_full_l32/adaptation_module.pt"
P2[INT]="logs/one_leg/rma/2026-04-28_21-18-17_one_leg_rma_p2_int_l32/adaptation_module.pt"
P2[EXT]="logs/one_leg/rma/2026-04-28_21-18-50_one_leg_rma_p2_ext_l32/adaptation_module.pt"
for PRIV in FULL INT EXT; do
    for S in 1.0 1.5 2.0; do run rma2 "$PRIV" "${P1[$PRIV]}" $S "${P2[$PRIV]}"; done
done

# CTS FULL l=8 + INT l=8 (6 runs, ext_l8 missing) ----------------------------
for S in 1.0 1.5 2.0; do
    run cts FULL "logs/one_leg/cts/2026-04-28_16-26-13_one_leg_cts_full_l8/model_final.pt" $S
done
for S in 1.0 1.5 2.0; do
    run cts INT  "logs/one_leg/cts/2026-04-28_17-29-34_one_leg_cts_int_l8/model_final.pt"  $S
done

# CTS l=16 (9 runs) -----------------------------------------------------------
declare -A CTS16
CTS16[FULL]="logs/one_leg/cts/2026-04-28_14-01-17_one_leg_cts_full_l16/model_final.pt"
CTS16[INT]="logs/one_leg/cts/2026-04-28_15-02-13_one_leg_cts_int_l16/model_final.pt"
CTS16[EXT]="logs/one_leg/cts/2026-04-28_16-02-51_one_leg_cts_ext_l16/model_final.pt"
for PRIV in FULL INT EXT; do
    for S in 1.0 1.5 2.0; do run cts "$PRIV" "${CTS16[$PRIV]}" $S; done
done

log "===== OOD T2 FINISHED ====="

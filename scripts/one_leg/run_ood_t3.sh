#!/usr/bin/env bash
# OOD Terminal 3 — CTS FULL/INT/EXT l32 + CTS FULL l64 + CTS FULL l128  (18 runs)
# Finishes: print summary table from shared results file
set -e
cd "$(dirname "$0")/../.."

N_EPS=100; N_ENV=64
RESULTS="results/ood_results_all.csv"
LOG="logs/one_leg/train_progress.log"
mkdir -p results logs/one_leg

log() { echo "[OOD-T3] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

run() {
    local METHOD=$1 PRIV=$2 CKPT=$3 SCALE=$4
    log "${METHOD} ${PRIV} scale=${SCALE}"
    python scripts/one_leg/eval_ood.py \
        --method "$METHOD" --priv_mode "$PRIV" \
        --checkpoint "$CKPT" \
        --dr_scale "$SCALE" \
        --num_episodes $N_EPS --num_envs $N_ENV \
        --results_file "$RESULTS" --headless \
        && log "DONE ${METHOD} ${PRIV} scale=${SCALE}" \
        || log "FAIL ${METHOD} ${PRIV} scale=${SCALE}"
}

log "===== OOD T3 START ====="

# CTS l=32 (9 runs) -----------------------------------------------------------
declare -A CTS32
CTS32[FULL]="logs/one_leg/cts/2026-04-28_16-41-37_one_leg_cts_full_l32/model_final.pt"
CTS32[INT]="logs/one_leg/cts/2026-04-28_11-38-58_one_leg_cts_int_l32/model_final.pt"
CTS32[EXT]="logs/one_leg/cts/2026-04-28_12-37-44_one_leg_cts_ext_l32/model_final.pt"
for PRIV in FULL INT EXT; do
    for S in 1.0 1.5 2.0; do run cts "$PRIV" "${CTS32[$PRIV]}" $S; done
done

# CTS FULL l=64 (3 runs) ------------------------------------------------------
for S in 1.0 1.5 2.0; do
    run cts FULL "logs/one_leg/cts/2026-04-28_17-05-24_one_leg_cts_full_l64/model_final.pt" $S
done

# CTS FULL l=128 (3 runs) -----------------------------------------------------
for S in 1.0 1.5 2.0; do
    run cts FULL "logs/one_leg/cts/2026-04-28_15-39-51_one_leg_cts_full_l128/model_final.pt" $S
done

log "===== OOD T3 FINISHED ====="

# Print full summary once all terminals are likely done -----------------------
echo ""
echo "Printing summary (all results collected so far):"
python scripts/one_leg/format_results.py "$RESULTS" --save_md

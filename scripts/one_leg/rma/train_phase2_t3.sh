#!/usr/bin/env bash
# Phase 2 Terminal 3 — FULL l32 / INT l32 / EXT l32  (3 runs ≈ 1.5h)
set -e
cd "$(dirname "$0")/../../.."

NUM_ENVS=1024
MAX_ITER=1000
LOG="logs/one_leg/train_progress.log"
mkdir -p logs/one_leg

log() { echo "[P2-T3] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

log "===== Phase2 T3 START ====="

declare -A CKPTS
CKPTS[FULL]="logs/one_leg/rma/2026-04-28_10-40-04_one_leg_rma_full_l32/model_final.pt"
CKPTS[INT]="logs/one_leg/rma/2026-04-28_11-28-35_one_leg_rma_int_l32/model_final.pt"
CKPTS[EXT]="logs/one_leg/rma/2026-04-28_12-16-43_one_leg_rma_ext_l32/model_final.pt"

for PRIV in FULL INT EXT; do
    NAME="rma_p2_${PRIV,,}_l32"
    log "$NAME"
    python scripts/one_leg/rma/train_phase2.py \
        --checkpoint "${CKPTS[$PRIV]}" \
        --priv_mode  "$PRIV" \
        --num_envs   $NUM_ENVS \
        --max_iterations $MAX_ITER \
        --headless \
        && log "DONE $NAME" || log "FAIL $NAME"
done

log "===== Phase2 T3 FINISHED ====="

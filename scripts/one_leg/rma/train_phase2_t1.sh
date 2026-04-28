#!/usr/bin/env bash
# Phase 2 Terminal 1 — FULL l8 / INT l8 / EXT l8  (3 runs ≈ 1.5h)
set -e
cd "$(dirname "$0")/../../.."

NUM_ENVS=1024
MAX_ITER=1000
LOG="logs/one_leg/train_progress.log"
mkdir -p logs/one_leg

log() { echo "[P2-T1] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

log "===== Phase2 T1 START ====="

declare -A CKPTS
CKPTS[FULL]="logs/one_leg/rma/2026-04-28_11-27-42_one_leg_rma_full_l8/model_final.pt"
CKPTS[INT]="logs/one_leg/rma/2026-04-28_12-16-18_one_leg_rma_int_l8/model_final.pt"
CKPTS[EXT]="logs/one_leg/rma/2026-04-28_13-05-11_one_leg_rma_ext_l8/model_final.pt"

for PRIV in FULL INT EXT; do
    NAME="rma_p2_${PRIV,,}_l8"
    log "$NAME"
    python scripts/one_leg/rma/train_phase2.py \
        --checkpoint "${CKPTS[$PRIV]}" \
        --priv_mode  "$PRIV" \
        --num_envs   $NUM_ENVS \
        --max_iterations $MAX_ITER \
        --headless \
        && log "DONE $NAME" || log "FAIL $NAME"
done

log "===== Phase2 T1 FINISHED ====="

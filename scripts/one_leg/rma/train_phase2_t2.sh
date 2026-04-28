#!/usr/bin/env bash
# Phase 2 Terminal 2 — FULL l16 / INT l16 / EXT l16  (3 runs ≈ 1.5h)
set -e
cd "$(dirname "$0")/../../.."

NUM_ENVS=1024
MAX_ITER=1000
LOG="logs/one_leg/train_progress.log"
mkdir -p logs/one_leg

log() { echo "[P2-T2] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

log "===== Phase2 T2 START ====="

declare -A CKPTS
CKPTS[FULL]="logs/one_leg/rma/2026-04-28_13-55-36_one_leg_rma_full_l16/model_final.pt"
CKPTS[INT]="logs/one_leg/rma/2026-04-28_14-45-37_one_leg_rma_int_l16/model_final.pt"
CKPTS[EXT]="logs/one_leg/rma/2026-04-28_15-35-54_one_leg_rma_ext_l16/model_final.pt"

for PRIV in FULL INT EXT; do
    NAME="rma_p2_${PRIV,,}_l16"
    log "$NAME"
    python scripts/one_leg/rma/train_phase2.py \
        --checkpoint "${CKPTS[$PRIV]}" \
        --priv_mode  "$PRIV" \
        --num_envs   $NUM_ENVS \
        --max_iterations $MAX_ITER \
        --headless \
        && log "DONE $NAME" || log "FAIL $NAME"
done

log "===== Phase2 T2 FINISHED ====="

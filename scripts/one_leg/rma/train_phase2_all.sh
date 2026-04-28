#!/usr/bin/env bash
# Run RMA Phase 2 for all 9 latest Phase-1 checkpoints (FULL/INT/EXT × l8/l16/l32)
set -e
cd "$(dirname "$0")/../../.."

NUM_ENVS=1024
MAX_ITER=1000
LOG="logs/one_leg/train_progress.log"
mkdir -p logs/one_leg

log() { echo "[P2] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

log "===== RMA Phase 2 ALL START ====="

# Latest Phase-1 checkpoints
declare -A CKPTS
CKPTS[FULL_l8]="logs/one_leg/rma/2026-04-28_11-27-42_one_leg_rma_full_l8/model_final.pt"
CKPTS[INT_l8]="logs/one_leg/rma/2026-04-28_12-16-18_one_leg_rma_int_l8/model_final.pt"
CKPTS[EXT_l8]="logs/one_leg/rma/2026-04-28_13-05-11_one_leg_rma_ext_l8/model_final.pt"
CKPTS[FULL_l16]="logs/one_leg/rma/2026-04-28_13-55-36_one_leg_rma_full_l16/model_final.pt"
CKPTS[INT_l16]="logs/one_leg/rma/2026-04-28_14-45-37_one_leg_rma_int_l16/model_final.pt"
CKPTS[EXT_l16]="logs/one_leg/rma/2026-04-28_15-35-54_one_leg_rma_ext_l16/model_final.pt"
CKPTS[FULL_l32]="logs/one_leg/rma/2026-04-28_10-40-04_one_leg_rma_full_l32/model_final.pt"
CKPTS[INT_l32]="logs/one_leg/rma/2026-04-28_11-28-35_one_leg_rma_int_l32/model_final.pt"
CKPTS[EXT_l32]="logs/one_leg/rma/2026-04-28_12-16-43_one_leg_rma_ext_l32/model_final.pt"

for KEY in FULL_l8 INT_l8 EXT_l8 FULL_l16 INT_l16 EXT_l16 FULL_l32 INT_l32 EXT_l32; do
    PRIV="${KEY%%_*}"        # FULL / INT / EXT
    CKPT="${CKPTS[$KEY]}"
    NAME="rma_p2_${KEY,,}"
    log "$NAME"
    python scripts/one_leg/rma/train_phase2.py \
        --checkpoint "$CKPT" \
        --priv_mode  "$PRIV" \
        --num_envs   $NUM_ENVS \
        --max_iterations $MAX_ITER \
        --headless \
        && log "DONE $NAME" || log "FAIL $NAME"
done

log "===== RMA Phase 2 ALL FINISHED ====="

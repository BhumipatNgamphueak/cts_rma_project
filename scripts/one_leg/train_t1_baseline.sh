#!/usr/bin/env bash
# Terminal 1 — Baseline + RMA l8 + RMA l16 + CTS FULL/INT l8  (9 runs ≈ 9.9h)
set -e
cd "$(dirname "$0")/../.."

NUM_ENVS=1024
MAX_ITER=2500
SEED=42
LOG="logs/one_leg/train_progress.log"
mkdir -p logs/one_leg

log() { echo "[T1] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

log "===== Terminal 1 START ====="

# Baseline (0.5h) -------------------------------------------------------------
log "baseline"
python scripts/one_leg/baseline/train.py \
    --num_envs $NUM_ENVS --max_iterations $MAX_ITER --seed $SEED --headless \
    && log "DONE baseline" || log "FAIL baseline"

# RMA l=8 (3 × 1h = 3h) -------------------------------------------------------
for PRIV in FULL INT EXT; do
    NAME="rma_${PRIV,,}_l8"
    log "$NAME"
    python scripts/one_leg/rma/train.py \
        --num_envs $NUM_ENVS --max_iterations $MAX_ITER \
        --priv_mode "$PRIV" --latent_dim 8 --seed $SEED --headless \
        && log "DONE $NAME" || log "FAIL $NAME"
done

# RMA l=16 (3 × 1h = 3h) ------------------------------------------------------
for PRIV in FULL INT EXT; do
    NAME="rma_${PRIV,,}_l16"
    log "$NAME"
    python scripts/one_leg/rma/train.py \
        --num_envs $NUM_ENVS --max_iterations $MAX_ITER \
        --priv_mode "$PRIV" --latent_dim 16 --seed $SEED --headless \
        && log "DONE $NAME" || log "FAIL $NAME"
done

# CTS FULL l=8 (1.2h) ---------------------------------------------------------
log "cts_full_l8"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations $MAX_ITER \
    --priv_mode FULL --latent_dim 8 --seed $SEED --headless \
    && log "DONE cts_full_l8" || log "FAIL cts_full_l8"

# CTS INT l=8 (1.2h) ----------------------------------------------------------
log "cts_int_l8"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations $MAX_ITER \
    --priv_mode INT --latent_dim 8 --seed $SEED --headless \
    && log "DONE cts_int_l8" || log "FAIL cts_int_l8"

log "===== Terminal 1 FINISHED ====="

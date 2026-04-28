#!/usr/bin/env bash
# Terminal 2 — RMA l32 + CTS EXT l8 + CTS l16 + CTS FULL l64  (8 runs ≈ 9.1h)
set -e
cd "$(dirname "$0")/../.."

NUM_ENVS=1024
MAX_ITER=2500
SEED=42
LOG="logs/one_leg/train_progress.log"
mkdir -p logs/one_leg

log() { echo "[T2] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

log "===== Terminal 2 START ====="

# RMA l=32 (3 × 1h = 3h) ------------------------------------------------------
for PRIV in FULL INT EXT; do
    NAME="rma_${PRIV,,}_l32"
    log "$NAME"
    python scripts/one_leg/rma/train.py \
        --num_envs $NUM_ENVS --max_iterations $MAX_ITER \
        --priv_mode "$PRIV" --latent_dim 32 --seed $SEED --headless \
        && log "DONE $NAME" || log "FAIL $NAME"
done

# CTS EXT l=8 (1.2h) ----------------------------------------------------------
log "cts_ext_l8"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations $MAX_ITER \
    --priv_mode EXT --latent_dim 8 --seed $SEED --headless \
    && log "DONE cts_ext_l8" || log "FAIL cts_ext_l8"

# CTS l=16: FULL / INT / EXT (3 × 1.2h = 3.6h) --------------------------------
for PRIV in FULL INT EXT; do
    NAME="cts_${PRIV,,}_l16"
    log "$NAME"
    python scripts/one_leg/cts/train.py \
        --num_envs $NUM_ENVS --max_iterations $MAX_ITER \
        --priv_mode "$PRIV" --latent_dim 16 --seed $SEED --headless \
        && log "DONE $NAME" || log "FAIL $NAME"
done

# [Hypothesis A] CTS FULL l=64 (1.3h) -----------------------------------------
log "cts_full_l64  [hyp-A: larger latent]"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations $MAX_ITER \
    --priv_mode FULL --latent_dim 64 --seed $SEED --headless \
    && log "DONE cts_full_l64" || log "FAIL cts_full_l64"

log "===== Terminal 2 FINISHED ====="

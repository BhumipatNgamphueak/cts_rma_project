#!/usr/bin/env bash
# Terminal 3 — CTS l32 + hypothesis B/C + CTS FULL l128  (7 runs ≈ 9.8h)
set -e
cd "$(dirname "$0")/../.."

NUM_ENVS=1024
SEED=42
LOG="logs/one_leg/train_progress.log"
mkdir -p logs/one_leg

log() { echo "[T3] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

log "===== Terminal 3 START ====="

# CTS l=32: FULL / INT / EXT (3 × 1.2h = 3.6h) --------------------------------
for PRIV in FULL INT EXT; do
    NAME="cts_${PRIV,,}_l32"
    log "$NAME"
    python scripts/one_leg/cts/train.py \
        --num_envs $NUM_ENVS --max_iterations 2500 \
        --priv_mode "$PRIV" --latent_dim 32 --seed $SEED --headless \
        && log "DONE $NAME" || log "FAIL $NAME"
done

# [Hypothesis B] Lower lambda_rec — less L_rec pressure on Et (2 × 1.2h = 2.4h)
# Reduces L_rec forcing on FULL so Et can freely use external signals
log "cts_full_l8_lam0.1  [hyp-B: lower lambda]"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations 2500 \
    --priv_mode FULL --latent_dim 8 --lambda_rec 0.1 --seed $SEED --headless \
    && log "DONE cts_full_l8_lam0.1" || log "FAIL cts_full_l8_lam0.1"

log "cts_full_l32_lam0.1  [hyp-B: lower lambda]"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations 2500 \
    --priv_mode FULL --latent_dim 32 --lambda_rec 0.1 --seed $SEED --headless \
    && log "DONE cts_full_l32_lam0.1" || log "FAIL cts_full_l32_lam0.1"

# [Hypothesis A] CTS FULL l=128 (1.4h) ----------------------------------------
log "cts_full_l128  [hyp-A: larger latent]"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations 2500 \
    --priv_mode FULL --latent_dim 128 --seed $SEED --headless \
    && log "DONE cts_full_l128" || log "FAIL cts_full_l128"

# [Hypothesis C] More iterations (2.4h) ----------------------------------------
# FULL may need longer to converge due to noisier gradients
log "cts_full_l32_5000iter  [hyp-C: more iters]"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations 5000 \
    --priv_mode FULL --latent_dim 32 --seed $SEED --headless \
    && log "DONE cts_full_l32_5000iter" || log "FAIL cts_full_l32_5000iter"

log "===== Terminal 3 FINISHED ====="

#!/usr/bin/env bash
# Terminal 4 — Hypothesis: CTS FULL > INT with right config  (6 runs ≈ 7.5h)
#
# Three angles testing why FULL < INT and when it flips:
#
# [A] Larger latent (l=64, l=128)
#     Hypothesis: at l=8/16/32 the encoder can't separate stable internal
#     from noisy external signals. Larger Z gives dedicated dimensions for each.
#     Expected: FULL catches up to INT at l=64 or l=128.
#
# [B] Lower lambda_rec for FULL (lambda=0.1 vs default 1.0)
#     Hypothesis: L_rec forces Et to produce student-recoverable zt.
#     External signals (contact, push) can't be predicted from obs history alone
#     → L_rec actively penalises Et for encoding external info → FULL hurt more.
#     Reducing lambda relaxes this, letting Et use external signals freely.
#     Expected: FULL l=32 lambda=0.1 > FULL l=32 lambda=1.0.
#
# [C] More training iterations for FULL (5000 vs 2500)
#     Hypothesis: noisy external signals make FULL gradients noisier → slower
#     convergence. FULL just needs more time, not a different architecture.
#     Expected: FULL l=32 at 5000 iters approaches INT l=32 at 2500 iters.
#
# Comparison baselines (re-use T3 results): CTS INT l=32, CTS EXT l=32

set -e
cd "$(dirname "$0")/../.."

NUM_ENVS=1024
SEED=42
LOG="logs/one_leg/train_progress.log"
mkdir -p logs/one_leg

log() { echo "[T4] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

log "===== Terminal 4 (Hypothesis) START ====="

# [A] Larger latent: l=64 and l=128 for FULL only  (2 runs ~2.4h) ──────────
log "[A] CTS FULL l=64"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations 2500 \
    --priv_mode FULL --latent_dim 64 --seed $SEED --headless \
    && log "DONE cts_full_l64" || log "FAIL cts_full_l64"

log "[A] CTS FULL l=128"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations 2500 \
    --priv_mode FULL --latent_dim 128 --seed $SEED --headless \
    && log "DONE cts_full_l128" || log "FAIL cts_full_l128"

# [B] Lower lambda_rec: FULL with lambda=0.1  (2 runs ~2.4h) ────────────────
log "[B] CTS FULL l=8 lambda_rec=0.1"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations 2500 \
    --priv_mode FULL --latent_dim 8 --lambda_rec 0.1 --seed $SEED --headless \
    && log "DONE cts_full_l8_lam0.1" || log "FAIL cts_full_l8_lam0.1"

log "[B] CTS FULL l=32 lambda_rec=0.1"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations 2500 \
    --priv_mode FULL --latent_dim 32 --lambda_rec 0.1 --seed $SEED --headless \
    && log "DONE cts_full_l32_lam0.1" || log "FAIL cts_full_l32_lam0.1"

# [C] More iterations: FULL l=32 trained to 5000 iters  (1 run ~2.4h) ───────
log "[C] CTS FULL l=32 5000 iters"
python scripts/one_leg/cts/train.py \
    --num_envs $NUM_ENVS --max_iterations 5000 \
    --priv_mode FULL --latent_dim 32 --seed $SEED --headless \
    && log "DONE cts_full_l32_5000iter" || log "FAIL cts_full_l32_5000iter"

log "===== Terminal 4 (Hypothesis) FINISHED ====="

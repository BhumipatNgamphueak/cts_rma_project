#!/usr/bin/env bash
# Diagnostic: which external component causes CTS FULL to underperform INT?
#
# Strategy: start from FULL (all 33D) and zero out ONE external component
# at a time. Compare each run's reward against:
#   - CTS FULL l=8  (baseline — all external included, from T1/T2)
#   - CTS INT  l=8  (target   — zero all external, from T1)
#
# If removing X causes reward to jump toward INT level → X is the culprit.
#
# External 13D layout:
#   [cf_vec(0:3)  cf_flag(3:4)  torques(4:7)  accels(7:10)  push_f(10:13)]
#
# Runs (4 × ~1.2h ≈ 5h total):
#   FULL_NO_CF    — remove contact force vec + contact flag
#   FULL_NO_TORQ  — remove joint torques
#   FULL_NO_ACCEL — remove joint accelerations
#   FULL_NO_PUSH  — remove push-force signal
#
# All trained at l=8 (same as baseline FULL/INT comparisons in T1/T2).

set -e
cd "$(dirname "$0")/../.."

NUM_ENVS=1024
MAX_ITER=2500
SEED=42
L=8
LOG="logs/one_leg/train_progress.log"
mkdir -p logs/one_leg

log() { echo "[DIAG] $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"; }

log "===== Diagnostic (EXT component ablation) START ====="

for MODE in FULL_NO_CF FULL_NO_TORQ FULL_NO_ACCEL FULL_NO_PUSH; do
    NAME="cts_${MODE,,}_l${L}"
    log "$NAME"
    python scripts/one_leg/cts/train.py \
        --num_envs $NUM_ENVS --max_iterations $MAX_ITER \
        --priv_mode "$MODE" --latent_dim $L \
        --seed $SEED --headless \
        && log "DONE $NAME" || log "FAIL $NAME"
done

log "===== Diagnostic FINISHED ====="
log "Compare rewards against:"
log "  CTS FULL l8  (all external) — logs/one_leg/cts/*cts_full_l8*"
log "  CTS INT  l8  (no external)  — logs/one_leg/cts/*cts_int_l8*"

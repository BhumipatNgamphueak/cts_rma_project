#!/usr/bin/env bash
# train_all.sh — run every config sequentially overnight
#
# Configs:
#   Baseline : 1 run
#   RMA      : FULL / INT / EXT  ×  l=8 / l=16 / l=32  =  9 runs
#   CTS      : FULL / INT / EXT  ×  l=8 / l=16 / l=32  =  9 runs
#   Total    : 19 runs
#
# Usage (run in a tmux/screen session):
#   bash scripts/one_leg/train_all.sh
#
# Logs per run go to logs/one_leg/{method}/<timestamp>_<name>/
# A summary of completed runs is written to logs/one_leg/train_all_progress.log

set -e
cd "$(dirname "$0")/../.."   # project root

NUM_ENVS=1024
MAX_ITER=2500
SEED=42
PROGRESS_LOG="logs/one_leg/train_all_progress.log"

mkdir -p logs/one_leg
echo "===== train_all.sh started: $(date) =====" | tee -a "$PROGRESS_LOG"

run_done() {
    echo "[DONE] $1  ($(date '+%H:%M:%S'))" | tee -a "$PROGRESS_LOG"
}

run_fail() {
    echo "[FAIL] $1  ($(date '+%H:%M:%S'))" | tee -a "$PROGRESS_LOG"
}

# ── Baseline ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS_LOG"
echo "--- Baseline ---" | tee -a "$PROGRESS_LOG"
python scripts/one_leg/baseline/train.py \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITER \
    --seed $SEED \
    --headless \
    && run_done "baseline" \
    || run_fail "baseline"

# ── RMA ───────────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS_LOG"
echo "--- RMA ---" | tee -a "$PROGRESS_LOG"
for PRIV in FULL INT EXT; do
    for L in 8 16 32; do
        NAME="rma_${PRIV,,}_l${L}"
        echo "  > $NAME" | tee -a "$PROGRESS_LOG"
        python scripts/one_leg/rma/train.py \
            --num_envs $NUM_ENVS \
            --max_iterations $MAX_ITER \
            --priv_mode "$PRIV" \
            --latent_dim $L \
            --seed $SEED \
            --headless \
            && run_done "$NAME" \
            || run_fail "$NAME"
    done
done

# ── CTS ───────────────────────────────────────────────────────────────────────
echo "" | tee -a "$PROGRESS_LOG"
echo "--- CTS ---" | tee -a "$PROGRESS_LOG"
for PRIV in FULL INT EXT; do
    for L in 8 16 32; do
        NAME="cts_${PRIV,,}_l${L}"
        echo "  > $NAME" | tee -a "$PROGRESS_LOG"
        python scripts/one_leg/cts/train.py \
            --num_envs $NUM_ENVS \
            --max_iterations $MAX_ITER \
            --priv_mode "$PRIV" \
            --latent_dim $L \
            --seed $SEED \
            --headless \
            && run_done "$NAME" \
            || run_fail "$NAME"
    done
done

echo "" | tee -a "$PROGRESS_LOG"
echo "===== train_all.sh finished: $(date) =====" | tee -a "$PROGRESS_LOG"
echo ""
echo "All done! Check progress: cat $PROGRESS_LOG"

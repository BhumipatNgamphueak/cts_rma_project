#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
#  scripts/run_one_leg_eval.sh
#
#  One-leg hexapod (Phase 1) — Isaac-Lab OOD evaluation, Isaac only.
#  Tests Baseline / RMA (Phase-1+Phase-2) / CTS in the fair config (FULL, l=8)
#  at DR×1 and DR×2.
#
#  Outputs (clearly prefixed so they can never be confused with Go2 results):
#    results/ood_one_leg.csv                   ← evaluation rows
#    results/figures/one_leg/fig_one_leg_*.png ← figures
#    results/one_leg_results_table.md          ← markdown summary
#
#  Usage (paste in a terminal, or run):
#    bash scripts/run_one_leg_eval.sh
# ════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/.."

ISAACLAB="${ISAACLAB:-/home/drl-68/IsaacLab/isaaclab.sh}"

# ── Fair-config checkpoints (FULL privilege, latent=8) ─────────────────
CKPT_BASELINE=logs/one_leg/baseline/2026-04-30_18-35-29/model_final.pt
CKPT_CTS_L8=logs/one_leg/cts/2026-05-01_08-46-33_one_leg_cts_full_l8/model_final.pt
CKPT_RMA_P1=logs/one_leg/rma/2026-04-30_19-01-54_rma_full_l8/model_final.pt
ADAPT_RMA=logs/one_leg/rma/2026-04-30_20-23-30_one_leg_rma_p2_full_l8/adaptation_module.pt

# ── Sanity-check that all checkpoints exist before launching anything ──
echo "── checkpoints ──"
for f in "$CKPT_BASELINE" "$CKPT_CTS_L8" "$CKPT_RMA_P1" "$ADAPT_RMA"; do
  if [ -f "$f" ]; then echo "[ok]      $f"; else echo "[MISSING] $f"; fi
done

# ── Clean the previous one-leg CSV (does NOT touch ood_go2.csv) ────────
rm -f results/ood_one_leg.csv
mkdir -p results/figures/one_leg
echo "[$(date +%H:%M)] cleared results/ood_one_leg.csv"

# ── Eval: 3 methods × {DR×1, DR×2} = 6 runs, all writing to ood_one_leg.csv ──
COMMON="--headless --num_episodes 100 --num_envs 64 --priv_mode FULL \
        --results_file results/ood_one_leg.csv"

for s in 1.0 2.0; do
  echo "──── One-leg DR=${s}×  Baseline ────"
  $ISAACLAB -p scripts/one_leg/eval_ood.py $COMMON \
      --method baseline --checkpoint "$CKPT_BASELINE" --dr_scale "$s"

  echo "──── One-leg DR=${s}×  CTS FULL l=8 ────"
  $ISAACLAB -p scripts/one_leg/eval_ood.py $COMMON \
      --method cts --checkpoint "$CKPT_CTS_L8" --dr_scale "$s" --history_len 50

  echo "──── One-leg DR=${s}×  RMA FULL l=8 (Phase-2 adapter) ────"
  $ISAACLAB -p scripts/one_leg/eval_ood.py $COMMON \
      --method rma --checkpoint "$CKPT_RMA_P1" \
      --phase2_checkpoint "$ADAPT_RMA" --dr_scale "$s"
done
echo "[$(date +%H:%M)] eval done — $(($(wc -l <results/ood_one_leg.csv)-1)) rows in ood_one_leg.csv"

# ── Plot the results (writes to results/figures/one_leg/) ──────────────
python scripts/plot_results_one_leg.py --ood results/ood_one_leg.csv

# ── Summary ────────────────────────────────────────────────────────────
echo
echo "── files written ──"
echo "  results/ood_one_leg.csv"
echo "  results/one_leg_results_table.md"
echo "  results/figures/one_leg/fig_one_leg_headline.{pdf,png}"
echo "  results/figures/one_leg/fig_one_leg_ood_retention.{pdf,png}"
echo
echo "── results/one_leg_results_table.md ───────────────────────────────"
cat results/one_leg_results_table.md 2>/dev/null

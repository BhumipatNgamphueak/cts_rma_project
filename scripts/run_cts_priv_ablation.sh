#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
#  scripts/run_cts_priv_ablation.sh
#
#  Go2 — CTS privileged-subset ablation (FULL vs INT vs EXT).
#  CTS FULL rows are already in the existing CSVs; this script adds
#  only the INT and EXT cells, then re-runs the plotter so the
#  figures + tables refresh automatically.
#
#  Test matrix (8 cells, 30 episodes each):
#      Isaac OOD     × {INT, EXT} × {DR x1, DR x2}     ← scripts/eval_ood_go2.py
#      MuJoCo sim2sim × {INT, EXT} × {DR x1, DR x2}    ← scripts/sim2sim/sim2sim_go2.py
#
#  Outputs:
#      results/ood_go2.csv         ← Isaac rows APPENDED (existing FULL rows kept)
#      results/sim2sim_go2.csv     ← MuJoCo rows APPENDED (existing FULL rows kept)
#      results/figures/fig_go2_cts_priv_ablation.{png,pdf}   (new)
#      results/go2_cts_priv_ablation_table.md                (new)
#      results/go2_results_table.md                          (refreshed)
#
#  Usage:
#      bash scripts/run_cts_priv_ablation.sh
# ════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/.."

ISAACLAB="${ISAACLAB:-/home/drl-68/IsaacLab/isaaclab.sh}"
MUJOCO_PY="${MUJOCO_PY:-conda run -n env_isaaclab python}"
DEVICE="${DEVICE:-cuda}"

# ── Checkpoints (CTS INT/EXT — finished 2026-05-11) ───────────────────
CKPT_CTS_INT=logs/cts/2026-05-11_12-12-02_cts_go2_v2_int_l8/model_final.pt
CKPT_CTS_EXT=logs/cts/2026-05-11_12-12-17_cts_go2_v2_ext_l8/model_final.pt

# ── Sanity-check checkpoints ──────────────────────────────────────────
echo "── checkpoints ──"
for f in "$CKPT_CTS_INT" "$CKPT_CTS_EXT"; do
  if [ -f "$f" ]; then echo "[ok]      $f"; else echo "[MISSING] $f"; exit 1; fi
done

# ── Shared eval args ──────────────────────────────────────────────────
EPS="${EPS:-30}"                              # episodes per cell
OOD_CSV="results/ood_go2.csv"
SIM_CSV="results/sim2sim_go2.csv"
COMMON_ISAAC="--headless --num_envs 64 --num_episodes $EPS \
              --method cts --latent_dim 8 --history_len 50 \
              --episode_length_s 10.0 --results_file $OOD_CSV"
COMMON_MUJOCO="--method cts --latent_dim 8 --history_len 50 \
               --episode_length_s 10.0 --num_episodes $EPS \
               --results_file $SIM_CSV"

# ── Isaac OOD: 4 cells (INT x {1,2}, EXT x {1,2}) ─────────────────────
echo
echo "════════════ Isaac OOD — CTS INT/EXT × DR{1,2} ════════════"
for priv in INT EXT; do
  case "$priv" in
    INT) ckpt="$CKPT_CTS_INT" ;;
    EXT) ckpt="$CKPT_CTS_EXT" ;;
  esac
  for s in 1.0 2.0; do
    echo
    echo "──── Isaac  CTS  priv=$priv  DR×${s} ────"
    $ISAACLAB -p scripts/eval_ood_go2.py $COMMON_ISAAC \
        --priv_mode "$priv" --checkpoint "$ckpt" --dr_scale "$s"
  done
done
echo "[$(date +%H:%M)] Isaac eval done — $(wc -l <$OOD_CSV) total rows in $OOD_CSV"

# ── MuJoCo sim2sim: 4 cells ───────────────────────────────────────────
echo
echo "════════════ MuJoCo sim2sim — CTS INT/EXT × DR{1,2} ═══════"
for priv in INT EXT; do
  case "$priv" in
    INT) ckpt="$CKPT_CTS_INT" ;;
    EXT) ckpt="$CKPT_CTS_EXT" ;;
  esac
  for s in 1.0 2.0; do
    echo
    echo "──── MuJoCo CTS  priv=$priv  DR×${s} ────"
    $MUJOCO_PY scripts/sim2sim/sim2sim_go2.py $COMMON_MUJOCO \
        --priv_mode "$priv" --checkpoint "$ckpt" --dr_scale "$s"
  done
done
echo "[$(date +%H:%M)] MuJoCo eval done — $(wc -l <$SIM_CSV) total rows in $SIM_CSV"

# ── Plot ──────────────────────────────────────────────────────────────
echo
echo "════════════ Re-running plotter ═══════════════════════════"
python scripts/plot_results_go2.py --ood "$OOD_CSV" --sim2sim "$SIM_CSV"

# ── Summary ───────────────────────────────────────────────────────────
echo
echo "── files written ──"
echo "  results/ood_go2.csv         (Isaac CTS-INT/EXT rows appended)"
echo "  results/sim2sim_go2.csv     (MuJoCo CTS-INT/EXT rows appended)"
echo "  results/figures/fig_go2_cts_priv_ablation.{pdf,png}   (new)"
echo "  results/go2_cts_priv_ablation_table.md                (new)"
echo "  results/go2_results_table.md                          (refreshed)"
echo
echo "── results/go2_cts_priv_ablation_table.md ────────────────"
cat results/go2_cts_priv_ablation_table.md 2>/dev/null

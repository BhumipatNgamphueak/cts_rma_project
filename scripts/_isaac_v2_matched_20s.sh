#!/usr/bin/env bash
# Matched-condition Isaac Lab eval: 20 s episodes, flat, no push, 30 episodes,
# v2 checkpoints — to align with MuJoCo eval_metrics.py (results/sim2sim_report_v2_matched.txt).
cd /home/drl-68/t_s_policy/cts_rma_project
export TERM=xterm
ISAACLAB="conda run -n env_isaaclab python"
CSV=results/isaac_v2_matched_20s.csv
COMMON="--headless --device cuda:0 --episode_length_s 20 --no_terrain --no_dist --num_episodes 30 --num_envs 64 --results_file $CSV"

BASE=logs/baseline/2026-05-10_17-07-38_baseline_go2_v2/model_final.pt
CTS=logs/cts/2026-05-10_17-07-53_cts_go2_v2_l8/model_final.pt
RMA=logs/rma/2026-05-10_17-07-44_rma_go2_v2_l8_l8/model_final.pt
ADAPT=logs/rma/phase2/2026-05-11_12-11-23/adapt_module_final.pt

rm -f "$CSV"
for S in 1.0 2.0; do
  echo "=== BASELINE dr=$S ==="
  $ISAACLAB scripts/eval_ood_go2.py $COMMON \
      --method baseline --checkpoint "$BASE" --dr_scale "$S"
  echo "=== CTS dr=$S ==="
  $ISAACLAB scripts/eval_ood_go2.py $COMMON \
      --method cts --checkpoint "$CTS" --priv_mode FULL --latent_dim 8 --dr_scale "$S"
  echo "=== RMA dr=$S (adapt_module) ==="
  $ISAACLAB scripts/eval_ood_go2.py $COMMON \
      --method rma --checkpoint "$RMA" --priv_mode FULL --latent_dim 8 \
      --adapt_module "$ADAPT" --dr_scale "$S"
done
echo "ALL DONE → $CSV"

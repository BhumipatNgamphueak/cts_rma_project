#!/usr/bin/env bash
# v2-TRAINING-FAITHFUL Isaac eval: flat ground + push_robot ON, NO impulses,
# 20 s episodes, 30 episodes, v2 checkpoints. Matches the exact distribution
# v2 was trained on (commit dd64881: GroundPlaneCfg + push_robot only).
cd /home/drl-68/t_s_policy/cts_rma_project
export TERM=xterm
PY="conda run -n env_isaaclab python"
CSV=results/isaac_v2_trainfaithful_20s.csv
COMMON="--headless --device cuda:0 --episode_length_s 20 --no_terrain --no_impulse --num_episodes 30 --num_envs 64 --results_file $CSV"

BASE=logs/baseline/2026-05-10_17-07-38_baseline_go2_v2/model_final.pt
CTS=logs/cts/2026-05-10_17-07-53_cts_go2_v2_l8/model_final.pt
RMA=logs/rma/2026-05-10_17-07-44_rma_go2_v2_l8_l8/model_final.pt
ADAPT=logs/rma/phase2/2026-05-11_12-11-23/adapt_module_final.pt

rm -f "$CSV"
for S in 1.0 2.0; do
  echo "=== BASELINE dr=$S ==="
  $PY scripts/eval_ood_go2.py $COMMON \
      --method baseline --checkpoint "$BASE" --dr_scale "$S"
  echo "=== CTS dr=$S ==="
  $PY scripts/eval_ood_go2.py $COMMON \
      --method cts --checkpoint "$CTS" --priv_mode FULL --latent_dim 8 --dr_scale "$S"
  echo "=== RMA dr=$S (adapt_module) ==="
  $PY scripts/eval_ood_go2.py $COMMON \
      --method rma --checkpoint "$RMA" --priv_mode FULL --latent_dim 8 \
      --adapt_module "$ADAPT" --dr_scale "$S"
done
echo "ALL DONE → $CSV"

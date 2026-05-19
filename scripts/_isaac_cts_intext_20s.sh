#!/usr/bin/env bash
# CTS INT/EXT privileged-subset ablation under the CORRECTED harness.
# Canonical condition = v2-training-faithful (flat + push, no impulse), 20 s,
# 30 ep, v2 INT/EXT checkpoints. FULL is already in isaac_v2_trainfaithful_20s.csv.
cd /home/drl-68/t_s_policy/cts_rma_project
export TERM=xterm
PY="conda run -n env_isaaclab python"
CSV=results/isaac_cts_intext_trainfaithful_20s.csv
COMMON="--headless --device cuda:0 --episode_length_s 20 --no_terrain --no_impulse --num_episodes 30 --num_envs 64 --method cts --latent_dim 8 --results_file $CSV"

INT=logs/cts/2026-05-11_12-12-02_cts_go2_v2_int_l8/model_final.pt
EXT=logs/cts/2026-05-11_12-12-17_cts_go2_v2_ext_l8/model_final.pt

rm -f "$CSV"
for S in 1.0 2.0; do
  echo "=== CTS INT dr=$S ==="
  $PY scripts/eval_ood_go2.py $COMMON --priv_mode INT --checkpoint "$INT" --dr_scale "$S"
  echo "=== CTS EXT dr=$S ==="
  $PY scripts/eval_ood_go2.py $COMMON --priv_mode EXT --checkpoint "$EXT" --dr_scale "$S"
done
echo "ALL DONE → $CSV"

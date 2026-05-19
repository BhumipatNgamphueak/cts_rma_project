#!/usr/bin/env bash
# Phase 2 of the corrected ablation data collection:
#   (A) Isaac NO-PUSH CTS INT/EXT (apples-to-apples sim2sim companion)
#   (B) MuJoCo CTS FULL/INT/EXT via the clean per-checkpoint sim2sim_go2.py
# Baseline/RMA/CTS-FULL MuJoCo already exist (sim2sim_report_v2_matched.json).
cd /home/drl-68/t_s_policy/cts_rma_project
export TERM=xterm
PY="conda run -n env_isaaclab python"

ISA_CSV=results/isaac_cts_intext_nopush_20s.csv
MJ_CSV=results/mujoco_cts_priv_20s.csv
ICOM="--headless --device cuda:0 --episode_length_s 20 --no_terrain --no_dist --num_episodes 30 --num_envs 64 --method cts --latent_dim 8 --results_file $ISA_CSV"

INT=logs/cts/2026-05-11_12-12-02_cts_go2_v2_int_l8/model_final.pt
EXT=logs/cts/2026-05-11_12-12-17_cts_go2_v2_ext_l8/model_final.pt
FULL=logs/cts/2026-05-10_17-07-53_cts_go2_v2_l8/model_final.pt

rm -f "$ISA_CSV" "$MJ_CSV"

# (A) Isaac no-push CTS INT/EXT
for S in 1.0 2.0; do
  echo "=== ISAAC no-push CTS INT dr=$S ==="
  $PY scripts/eval_ood_go2.py $ICOM --priv_mode INT --checkpoint "$INT" --dr_scale "$S"
  echo "=== ISAAC no-push CTS EXT dr=$S ==="
  $PY scripts/eval_ood_go2.py $ICOM --priv_mode EXT --checkpoint "$EXT" --dr_scale "$S"
done

# (B) MuJoCo CTS FULL/INT/EXT (sim2sim_go2.py is flat+no-push by construction)
for S in 1.0 2.0; do
  for PV in FULL:$FULL INT:$INT EXT:$EXT; do
    P="${PV%%:*}"; C="${PV##*:}"
    echo "=== MUJOCO CTS $P dr=$S ==="
    $PY scripts/sim2sim/sim2sim_go2.py --method cts --checkpoint "$C" \
        --priv_mode "$P" --latent_dim 8 --dr_scale "$S" \
        --num_episodes 30 --episode_length_s 20 --device cuda:0 \
        --results_file "$MJ_CSV"
  done
done
echo "ALL DONE → $ISA_CSV  +  $MJ_CSV"

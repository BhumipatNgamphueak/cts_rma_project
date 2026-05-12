#!/usr/bin/env bash
# =============================================================================
#  run_remaining_go2.sh  —  manifest of the remaining Go2 (Phase-2) jobs
#
#  This is the executable version of the "Go2 experiment matrix" in
#  project_answers.md (Section C). By default it is a DRY RUN: it only prints
#  the commands it would run. Flip DRY_RUN=0 (or pass --run) to actually launch.
#
#  Run a single stage:    ./scripts/run_remaining_go2.sh <stage>
#  Stages: train_ablation_cts | train_ablation_rma | eval_ood | eval_sim2sim
#          phase1_rerun | plot | all   (default: all)
#
#  IMPORTANT — order of operations:
#    1. Wait for the in-progress "v2" main runs to finish:
#         logs/baseline/2026-05-10_17-07-38_baseline_go2_v2/model_final.pt
#         logs/rma/2026-05-10_17-07-44_rma_go2_v2_l8_l8/model_final.pt   (Phase 1)
#         logs/cts/2026-05-10_17-07-53_cts_go2_v2_l8/model_final.pt
#       then set the CKPT_* variables below to the final checkpoints.
#    2. Smoke-test the new INT/EXT plumbing FIRST with --max_iterations 5
#       (see SMOKE block) before launching the full ablation runs.
#    3. NOTE: scripts/sim2sim/sim2sim_go2.py does NOT yet support --priv_mode or
#       the RMA Phase-2 adaptation module; INT/EXT and adapt-module sim2sim need
#       a small extension to that script (flagged TODO below).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."                    # repo root

# ── Config (edit me) ─────────────────────────────────────────────────────────
ISAACLAB="${ISAACLAB:-/home/drl-68/IsaacLab/isaaclab.sh}"   # Isaac Lab launcher
MUJOCO_PY="${MUJOCO_PY:-conda run -n env_isaaclab python}"  # python with mujoco+torch
NUM_ENVS="${NUM_ENVS:-4096}"
ITERS="${ITERS:-5000}"                     # PPO iterations for main/ablation training
P1_ITERS="${P1_ITERS:-15000}"              # RMA Phase-1 iterations
P2_ITERS="${P2_ITERS:-10000}"              # RMA Phase-2 iterations
OOD_EPS="${OOD_EPS:-100}"
SIM_EPS="${SIM_EPS:-30}"
OOD_CSV="results/ood_go2.csv"              # eval_ood_go2.py appends here
DR_SCALES=(1.0 2.0)                        # spec sheet uses only 1× and 2×
LATENTS=(16 32 64 128)                     # l=8 already trained on Go2
DEVICE="${DEVICE:-cuda:0}"

# Final main-run checkpoints — UPDATE these once the v2 runs finish.
CKPT_BASELINE="${CKPT_BASELINE:-logs/baseline/2026-05-10_17-07-38_baseline_go2_v2/model_final.pt}"
CKPT_CTS_L8="${CKPT_CTS_L8:-logs/cts/2026-05-10_17-07-53_cts_go2_v2_l8/model_final.pt}"
CKPT_RMA_P1_L8="${CKPT_RMA_P1_L8:-logs/rma/2026-05-10_17-07-44_rma_go2_v2_l8_l8/model_final.pt}"
ADAPT_RMA_L8="${ADAPT_RMA_L8:-logs/rma/phase2/2026-05-05_08-34-25/adapt_module_final.pt}"

# ── Dry-run plumbing ─────────────────────────────────────────────────────────
DRY_RUN="${DRY_RUN:-1}"
STAGE="all"
for a in "$@"; do
  case "$a" in
    --run) DRY_RUN=0 ;;
    --dry) DRY_RUN=1 ;;
    *)     STAGE="$a" ;;
  esac
done
run() {
  echo
  echo ">>> $*"
  if [[ "$DRY_RUN" == "0" ]]; then "$@"; else echo "    (dry-run — set DRY_RUN=0 or pass --run to execute)"; fi
}
hdr() { echo; echo "############################################################"; echo "# $*"; echo "############################################################"; }

# =============================================================================
# STAGE: smoke test the INT/EXT plumbing  (run this manually first!)
#   ./scripts/run_remaining_go2.sh smoke --run
# =============================================================================
do_smoke() {
  hdr "SMOKE TEST — verify INT/EXT shapes (5 iterations each, throw-away logs)"
  run $ISAACLAB -p scripts/cts/train.py --headless --device $DEVICE \
      --num_envs 256 --max_iterations 5 --latent_dim 8 --priv_mode INT --experiment smoke_cts_int
  run $ISAACLAB -p scripts/cts/train.py --headless --device $DEVICE \
      --num_envs 256 --max_iterations 5 --latent_dim 8 --priv_mode EXT --experiment smoke_cts_ext
  run $ISAACLAB -p scripts/rma/train_phase1.py --headless --device $DEVICE \
      --num_envs 256 --max_iterations 5 --latent_dim 8 --priv_mode INT --experiment smoke_rma_int
  run $ISAACLAB -p scripts/rma/train_phase1.py --headless --device $DEVICE \
      --num_envs 256 --max_iterations 5 --latent_dim 8 --priv_mode EXT --experiment smoke_rma_ext
  echo
  echo "  If those run without shape errors, delete logs/**/*smoke* and proceed."
}

# =============================================================================
# STAGE: CTS ablations  (priv subset INT/EXT @ l=8  +  latent sweep @ FULL)
# =============================================================================
do_train_ablation_cts() {
  hdr "CTS ablation — privileged subset (INT / EXT) at latent=8"
  for PRIV in INT EXT; do
    run $ISAACLAB -p scripts/cts/train.py --headless --device $DEVICE \
        --num_envs $NUM_ENVS --max_iterations $ITERS \
        --latent_dim 8 --priv_mode $PRIV \
        --experiment cts_go2_$(echo $PRIV | tr '[:upper:]' '[:lower:]')_l8
  done
  hdr "CTS ablation — latent-dimension sweep at priv=FULL"
  for L in "${LATENTS[@]}"; do
    run $ISAACLAB -p scripts/cts/train.py --headless --device $DEVICE \
        --num_envs $NUM_ENVS --max_iterations $ITERS \
        --latent_dim $L --priv_mode FULL \
        --experiment cts_go2_full_l${L}
  done
}

# =============================================================================
# STAGE: RMA ablations  (canonical two-stage: Phase 1 then Phase 2)
# =============================================================================
do_train_ablation_rma() {
  hdr "RMA ablation — Phase 1 (priv subset INT / EXT @ l=8  +  latent sweep @ FULL)"
  for PRIV in INT EXT; do
    run $ISAACLAB -p scripts/rma/train_phase1.py --device $DEVICE \
        --num_envs $NUM_ENVS --max_iterations $P1_ITERS \
        --latent_dim 8 --priv_mode $PRIV --experiment rma_go2
  done
  for L in "${LATENTS[@]}"; do
    run $ISAACLAB -p scripts/rma/train_phase1.py --device $DEVICE \
        --num_envs $NUM_ENVS --max_iterations $P1_ITERS \
        --latent_dim $L --priv_mode FULL --experiment rma_go2
  done
  hdr "RMA ablation — Phase 2 (point --checkpoint at each Phase-1 model_final.pt)"
  echo "  e.g.:"
  echo "  $ISAACLAB -p scripts/rma/train_phase2.py --device $DEVICE \\"
  echo "      --checkpoint logs/rma/<phase1_run>/model_final.pt \\"
  echo "      --num_envs $NUM_ENVS --num_iterations $P2_ITERS --latent_dim <L> --priv_mode <FULL|INT|EXT>"
  echo "  (priv_mode / latent_dim MUST match the Phase-1 checkpoint.)"
  # Phase-2 for the already-trained l=8 FULL teacher:
  run $ISAACLAB -p scripts/rma/train_phase2.py --device $DEVICE \
      --checkpoint "$CKPT_RMA_P1_L8" \
      --num_envs $NUM_ENVS --num_iterations $P2_ITERS --latent_dim 8 --priv_mode FULL
}

# =============================================================================
# STAGE: Isaac-Lab OOD evaluation  (all methods x DR scales -> results/ood_go2.csv)
# =============================================================================
_ood() {  # _ood <method> <ckpt> <priv_mode> <latent_dim> [adapt_module]
  local method="$1" ckpt="$2" priv="$3" lat="$4" adapt="${5:-}"
  [[ -f "$ckpt" ]] || { echo "  [skip] checkpoint not found: $ckpt"; return; }
  for S in "${DR_SCALES[@]}"; do
    local extra=""
    [[ -n "$adapt" ]] && extra="--adapt_module $adapt"
    run $ISAACLAB -p scripts/eval_ood_go2.py --headless --device $DEVICE \
        --method "$method" --checkpoint "$ckpt" --dr_scale "$S" \
        --priv_mode "$priv" --latent_dim "$lat" $extra \
        --num_episodes $OOD_EPS --num_envs 64 --results_file "$OOD_CSV"
  done
}
do_eval_ood() {
  hdr "Isaac-Lab OOD eval -> $OOD_CSV   (DR scales: ${DR_SCALES[*]})"
  echo "  NOTE: eval_ood_go2.py's DR table now mirrors shared_env_cfg.py."
  echo "  (If you want a clean CSV, 'rm $OOD_CSV' first.)"
  _ood baseline "$CKPT_BASELINE"   BASE N/A
  _ood cts      "$CKPT_CTS_L8"     FULL 8
  _ood rma      "$CKPT_RMA_P1_L8"  FULL 8  "$ADAPT_RMA_L8"
  echo
  echo "  ...then add one _ood line per ablation checkpoint you trained, e.g.:"
  echo '  _ood cts logs/cts/<run_cts_int_l8>/model_final.pt    INT 8'
  echo '  _ood cts logs/cts/<run_cts_ext_l8>/model_final.pt    EXT 8'
  echo '  _ood cts logs/cts/<run_cts_full_l16>/model_final.pt  FULL 16'
  echo '  _ood rma logs/rma/<run_rma_int_l8>/model_final.pt    INT 8  logs/rma/phase2/<run>/adapt_module_final.pt'
}

# =============================================================================
# STAGE: Isaac -> MuJoCo sim-to-sim evaluation
#   Writes a .txt report to <checkpoint_dir>/ood_eval/ ; the plot script reads
#   those via:  python scripts/plot_results_go2.py --scan-logs
#   TODO: scripts/sim2sim/sim2sim_go2.py needs --priv_mode + --adapt_module +
#         --results_file before INT/EXT and RMA-with-adapt sim2sim can be run.
# =============================================================================
_s2s() {  # _s2s <method> <ckpt> <latent_dim>
  local method="$1" ckpt="$2" lat="$3"
  [[ -f "$ckpt" ]] || { echo "  [skip] checkpoint not found: $ckpt"; return; }
  for S in "${DR_SCALES[@]}"; do
    run $MUJOCO_PY scripts/sim2sim/sim2sim_go2.py \
        --method "$method" --checkpoint "$ckpt" --dr_scale "$S" \
        --latent_dim "$lat" --num_episodes $SIM_EPS --device $DEVICE
  done
}
do_eval_sim2sim() {
  hdr "Isaac -> MuJoCo sim-to-sim eval   (DR scales: ${DR_SCALES[*]})"
  _s2s baseline "$CKPT_BASELINE"  8
  _s2s cts      "$CKPT_CTS_L8"    8
  _s2s rma      "$CKPT_RMA_P1_L8" 8
  echo
  echo "  (RMA sim2sim currently runs with z=0; add adapt-module support to"
  echo "   sim2sim_go2.py to evaluate the full Phase-2 RMA.)"
}

# =============================================================================
# STAGE: Phase-1 single-leg re-run  (one fair config: FULL, l=8, all 3 methods)
#   Re-trains + re-evaluates the single hexapod leg so the report has clean
#   Phase-1 numbers. See scripts/one_leg/*.sh for the existing launchers.
# =============================================================================
do_phase1_rerun() {
  hdr "Phase-1 single-leg re-run — one fair config (priv=FULL, latent=8)"
  run $ISAACLAB -p scripts/one_leg/baseline/train.py --headless --device $DEVICE \
      --num_envs 2048 --max_iterations 5000 --experiment one_leg_baseline_fair
  run $ISAACLAB -p scripts/one_leg/rma/train.py --headless --device $DEVICE \
      --num_envs 2048 --max_iterations 5000 --latent_dim 8 --priv_mode FULL --experiment one_leg_rma_fair
  echo "  ...then RMA Phase 2:"
  echo "  $ISAACLAB -p scripts/one_leg/rma/train_phase2.py --checkpoint logs/one_leg/rma/<run>/model_final.pt --latent_dim 8 ..."
  run $ISAACLAB -p scripts/one_leg/cts/train.py --headless --device $DEVICE \
      --num_envs 2048 --max_iterations 5000 --latent_dim 8 --priv_mode FULL --experiment one_leg_cts_fair
  echo
  echo "  Then evaluate (Isaac OOD): scripts/one_leg/eval_ood.py  (one fair config, 3 methods)"
  echo "  Then sim2sim (MuJoCo):     (single-leg MuJoCo script — confirm path; mirror sim2sim_go2.py)"
}

# =============================================================================
# STAGE: regenerate figures + tables from whatever results exist
# =============================================================================
do_plot() {
  hdr "Plot Go2 figures + tables"
  run python scripts/plot_results_go2.py --ood "$OOD_CSV" --sim2sim results/sim2sim_go2.csv --scan-logs
  echo "  -> results/figures/fig_go2_*.{pdf,png}, results/go2_results_table.{md,tex}"
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
echo "STAGE=$STAGE   DRY_RUN=$DRY_RUN   (ISAACLAB=$ISAACLAB)"
case "$STAGE" in
  smoke)               do_smoke ;;
  train_ablation_cts)  do_train_ablation_cts ;;
  train_ablation_rma)  do_train_ablation_rma ;;
  eval_ood)            do_eval_ood ;;
  eval_sim2sim)        do_eval_sim2sim ;;
  phase1_rerun)        do_phase1_rerun ;;
  plot)                do_plot ;;
  all)
    echo "  (full pipeline — smoke test first if you have not yet!)"
    do_train_ablation_cts
    do_train_ablation_rma
    do_eval_ood
    do_eval_sim2sim
    do_phase1_rerun
    do_plot
    ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
echo
echo "done (stage=$STAGE, dry_run=$DRY_RUN)."

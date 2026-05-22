# Comparative Study: Baseline / RMA / CTS Locomotion on Unitree GO2

FRA503 Deep Reinforcement Learning project comparing three locomotion policies for the Unitree GO2 quadruped in [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

## Methods

| Method | Description |
|--------|-------------|
| **Baseline** | PPO with proprioceptive observations only (37D) |
| **RMA** | Rapid Motor Adaptation — Phase 1 teacher encoder, Phase 2 adaptation module |
| **CTS** | Concurrent Teacher-Student — 75/25 env split with online distillation |

All three share the same scene, reward, and domain randomisation config ([`shared_env_cfg.py`](source/cts_rma_project/cts_rma_project/tasks/shared/shared_env_cfg.py)).

## Installation

```bash
# 1. Install Isaac Lab (tested with IsaacLab v2.x)
#    https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

# 2. Clone this repository
git clone <repo-url>
cd cts_rma_project

# 3. Install the package in editable mode
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e source/cts_rma_project

# 4. Clone MuJoCo Menagerie for sim-to-sim evaluation
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

## Training

### Baseline
```bash
/path/to/isaaclab.sh -p scripts/baseline/train.py --num_envs 4096 --max_iterations 5000
```

### RMA — Phase 1 (teacher)
```bash
/path/to/isaaclab.sh -p scripts/rma/train_phase1.py \
    --num_envs 4096 --max_iterations 5000 \
    --priv_mode FULL --latent_dim 8 --experiment rma_full_l8
```

### RMA — Phase 2 (adaptation module)
```bash
/path/to/isaaclab.sh -p scripts/rma/train_phase2.py \
    --checkpoint logs/rma/<run>/model_final.pt \
    --priv_mode FULL --latent_dim 8 --num_envs 4096
```

### CTS
```bash
/path/to/isaaclab.sh -p scripts/cts/train.py \
    --num_envs 4096 --max_iterations 5000 \
    --priv_mode FULL --latent_dim 8 --experiment cts_full_l8
```

Use `--priv_mode INT` or `--priv_mode EXT` for the privileged-info ablation.

## Evaluation

### Isaac Lab OOD test
```bash
/path/to/isaaclab.sh -p scripts/eval_ood_go2.py \
    --method baseline --checkpoint logs/baseline/<run>/model_final.pt \
    --dr_scales 1.0 2.0 --num_envs 512 --results_file results/eval_go2.csv
```

### MuJoCo sim-to-sim
```bash
python scripts/sim2sim/sim2sim_go2.py \
    --method baseline --checkpoint logs/baseline/<run>/model_final.pt \
    --dr_scales 1.0 2.0 --results_file results/eval_go2.csv
```

## Repository layout

```
scripts/
  baseline/      train + play scripts for Baseline
  rma/           train_phase1, train_phase2, play for RMA
  cts/           train + play scripts for CTS
  sim2sim/       MuJoCo sim-to-sim evaluation
  eval_ood_go2.py  Isaac Lab OOD evaluation (unified CSV output)
  plot_results_go2.py  generate comparison figures from CSV

source/cts_rma_project/cts_rma_project/tasks/
  baseline/      env config + PPO runner
  rma/           env config, network (RMAActorCritic), Phase 2 runner
  cts/           env config, network (CTSActorCritic), runner
  shared/        SharedEnvCfg, domain randomisation, reward / obs functions
  one_leg/       one-legged hopper experiment (auxiliary)
```

## License

BSD 3-Clause — see [LICENSE](LICENSE).

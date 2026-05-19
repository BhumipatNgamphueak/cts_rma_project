"""
RMA Phase 1 — teacher training with proper encoder.

Architecture:
  policy obs = [o_t(37) + x_t(26)] = 63D  (teacher sees full info)
  critic obs = [o_t(37) + x_t(26)] = 63D  (same)
  EnvFactorEncoder μ : x_t(26) → z(latent_dim)
  BasePolicy π       : [o_t, z] → a_t
  ValueFunction  V   : [o_t, z] → scalar

Usage:
    /home/drl-68/IsaacLab/isaaclab.sh -p scripts/rma/train_phase1.py \
        --num_envs 4096 --max_iterations 15000 \
        --latent_dim 8 --experiment rma_l8 --device cuda:0

    # latent ablations
    /home/drl-68/IsaacLab/isaaclab.sh -p scripts/rma/train_phase1.py \
        --num_envs 4096 --max_iterations 15000 \
        --latent_dim 16 --experiment rma_l16 --device cuda:0

    /home/drl-68/IsaacLab/isaaclab.sh -p scripts/rma/train_phase1.py \
        --num_envs 4096 --max_iterations 15000 \
        --latent_dim 32 --experiment rma_l32 --device cuda:0
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RMA Phase 1 teacher training")
parser.add_argument("--num_envs",       type=int,   default=4096)
parser.add_argument("--max_iterations", type=int,   default=15000)
parser.add_argument("--latent_dim",     type=int,   default=8,
                    help="Encoder bottleneck z dimension (ablation: 8, 16, 32, 64, 128)")
parser.add_argument("--priv_mode",      type=str,   default="FULL",
                    choices=["FULL", "INT", "EXT", "TERR", "FULL_T"],
                    help="Privileged knowledge fed to the encoder mu(x_t): "
                         "FULL=26D / INT=16D / EXT=10D / TERR=77D / FULL_T=103D")
parser.add_argument("--experiment",     type=str,   default="rma_teacher_go2")
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--checkpoint",     type=str,   default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports that require Isaac Sim to be running ──────────────────────────
import os
import gymnasium as gym
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore
from rsl_rl.runners import OnPolicyRunner           # type: ignore
import rsl_rl.runners.on_policy_runner as _rrl_runner  # type: ignore

import cts_rma_project.tasks  # noqa: F401  registers gym envs
from cts_rma_project.tasks.rma.rma_env_cfg            import RMATeacherEnvCfg
from cts_rma_project.tasks.rma.agents.rsl_rl_ppo_cfg  import RMATeacherPPORunnerCfg
from cts_rma_project.tasks.rma.rma_network             import RMAActorCritic
from cts_rma_project.tasks.shared.mdp                 import PRIV_DIMS

# Inject RMAActorCritic into RSL-RL's runner module namespace so that
# class_name="RMAActorCritic" in the policy cfg resolves correctly.
_rrl_runner.RMAActorCritic = RMAActorCritic


def main():
    device     = args_cli.device or "cuda"
    latent_dim = args_cli.latent_dim
    priv_mode  = args_cli.priv_mode.upper()
    priv_dim   = PRIV_DIMS[priv_mode]
    obs_dim    = 37 + priv_dim

    # ── environment ─────────────────────────────────────────────────────
    # RMATeacherEnvCfg: policy obs = [o_t(37)+x_t(priv_dim)]
    # so RSL-RL passes the full vector to act(), where RMAActorCritic splits and encodes x_t→z
    env_cfg = RMATeacherEnvCfg()
    env_cfg.scene.num_envs    = args_cli.num_envs
    env_cfg.sim.device        = device
    env_cfg.seed              = args_cli.seed
    env_cfg.priv_mode         = priv_mode
    # __post_init__ ran with the default FULL (63D); override now that priv_mode is set.
    env_cfg.observation_space = obs_dim
    env_cfg.state_space       = obs_dim

    env = gym.make("Template-RMA-Teacher-GO2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── runner config ───────────────────────────────────────────────────
    runner_cfg = RMATeacherPPORunnerCfg()
    runner_cfg.max_iterations  = args_cli.max_iterations
    runner_cfg.experiment_name = args_cli.experiment
    runner_cfg.seed            = args_cli.seed

    train_dict = runner_cfg.to_dict()
    # Pass latent_dim and env_factor_dim to RMAActorCritic constructor
    train_dict["policy"]["latent_dim"]     = latent_dim
    train_dict["policy"]["env_factor_dim"] = priv_dim

    log_dir = os.path.join(
        "logs", "rma",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + f"_{args_cli.experiment}_{priv_mode.lower()}_l{latent_dim}"
    )
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, train_dict, log_dir=log_dir, device=device)

    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)
        print(f"[INFO] Resumed from checkpoint: {args_cli.checkpoint}")

    print(f"[INFO] RMA Phase 1  priv={priv_mode}  latent_dim={latent_dim}  "
          f"obs={obs_dim}D  max_iters={args_cli.max_iterations}")

    # ── training ─────────────────────────────────────────────────────────
    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)

    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"[INFO] Phase 1 complete. Model saved to: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

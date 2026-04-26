"""
RMA Phase 1 training script.

Trains the base policy π and environment-factor encoder μ jointly using
PPO with ground-truth privileged observations e_t.

Usage:
    ./isaaclab.sh -p scripts/rma/train_phase1.py \
        --num_envs 4096 --max_iterations 5000 \
        --experiment rma_phase1 --device cuda:0
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RMA Phase 1 training")
parser.add_argument("--num_envs",       type=int,   default=4096)
parser.add_argument("--max_iterations", type=int,   default=5000)
parser.add_argument("--experiment",     type=str,   default="rma_phase1")
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--checkpoint",     type=str,   default=None,
                    help="Path to existing checkpoint to resume from")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports that require Isaac Sim to be running ──────────────────────────
import os
import torch
import gymnasium as gym
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore

import cts_rma_project.tasks  # noqa: F401  registers gym envs
from cts_rma_project.tasks.rma.rma_env_cfg    import RMAEnvCfg
from cts_rma_project.tasks.rma.rma_network    import RMAActorCritic
from cts_rma_project.tasks.rma.rma_runner     import RMAPhase1Runner
from cts_rma_project.tasks.rma.agents.rsl_rl_ppo_cfg import RMAPPORunnerCfg


def main():
    # ── environment ─────────────────────────────────────────────────────
    env_cfg = RMAEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = args_cli.device or "cuda"
    env_cfg.seed           = args_cli.seed

    env = gym.make("Template-RMA-GO2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── runner / agent config ────────────────────────────────────────────
    runner_cfg = RMAPPORunnerCfg()
    runner_cfg.max_iterations  = args_cli.max_iterations
    runner_cfg.experiment_name = args_cli.experiment
    runner_cfg.seed            = args_cli.seed

    log_dir = os.path.join(
        "logs", "rma", "phase1",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.experiment}"
    )
    os.makedirs(log_dir, exist_ok=True)

    runner = RMAPhase1Runner(env, runner_cfg.to_dict(), log_dir=log_dir,
                             device=env_cfg.sim.device)

    # Swap in the custom RMA actor-critic
    obs_dim = env.num_obs
    act_dim = env.num_actions
    runner.alg.actor_critic = RMAActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=act_dim,
        env_factor_dim=17,
        latent_dim=8,
    ).to(env_cfg.sim.device)

    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)
        print(f"[INFO] Resumed from checkpoint: {args_cli.checkpoint}")

    # ── training ─────────────────────────────────────────────────────────
    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)

    # Save final model
    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"[INFO] Phase 1 complete. Model saved to: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

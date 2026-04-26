"""
Baseline training script.

Trains a standard PPO policy on the 30-D proprioceptive observation with
Table 4 domain randomisation.  No teacher, no privileged information.
Uses the stock RSL-RL OnPolicyRunner — no custom runner required.

Usage:
    ./isaaclab.sh -p scripts/baseline/train.py \
        --num_envs 4096 --max_iterations 5000 \
        --experiment baseline_go2 --device cuda:0

    # Resume from checkpoint:
    ./isaaclab.sh -p scripts/baseline/train.py \
        --checkpoint logs/baseline/<run>/model_<iter>.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Baseline PPO training")
parser.add_argument("--num_envs",       type=int,  default=4096)
parser.add_argument("--max_iterations", type=int,  default=5000)
parser.add_argument("--experiment",     type=str,  default="baseline_go2")
parser.add_argument("--seed",           type=int,  default=42)
parser.add_argument("--checkpoint",     type=str,  default=None,
                    help="Path to checkpoint to resume from")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ───────────────────────────────────────────────────
import os
import gymnasium as gym
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner                # type: ignore
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper       # type: ignore

import cts_rma_project.tasks  # noqa: F401  — registers all gym envs
from cts_rma_project.tasks.baseline.baseline_env_cfg         import BaselineEnvCfg
from cts_rma_project.tasks.baseline.agents.rsl_rl_ppo_cfg    import BaselinePPORunnerCfg


def main():
    # ── environment ──────────────────────────────────────────────────────
    env_cfg = BaselineEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = args_cli.device or "cuda"
    env_cfg.seed           = args_cli.seed

    env = gym.make("Template-Baseline-GO2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── runner config ────────────────────────────────────────────────────
    runner_cfg = BaselinePPORunnerCfg()
    runner_cfg.max_iterations  = args_cli.max_iterations
    runner_cfg.experiment_name = args_cli.experiment
    runner_cfg.seed            = args_cli.seed

    log_dir = os.path.join(
        "logs", "baseline",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.experiment}",
    )
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir,
                            device=env_cfg.sim.device)

    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)
        print(f"[INFO] Resumed from: {args_cli.checkpoint}")

    # ── training ─────────────────────────────────────────────────────────
    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)

    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"[INFO] Training complete. Model saved to: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""
CTS (Concurrent Teacher-Student) training script.

Standard single-phase PPO — no curriculum.  The CTS contribution is the
concurrent teacher-student training mechanism, not command scheduling.
Command ranges are identical to Baseline and RMA (fair comparison).

Usage:
    python scripts/cts/train.py \
        --num_envs 4096 --max_iterations 5000 \
        --experiment cts_go2 --device cuda:0

    # Resume from checkpoint
    python scripts/cts/train.py \
        --checkpoint logs/cts/<run>/model_final.pt \
        --max_iterations 5000
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CTS training")
parser.add_argument("--num_envs",       type=int,   default=4096)
parser.add_argument("--max_iterations", type=int,   default=5000)
parser.add_argument("--experiment",     type=str,   default="cts_go2")
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--checkpoint",     type=str,   default=None,
                    help="Optional checkpoint to resume from")
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
from rsl_rl.runners import OnPolicyRunner           # type: ignore

import cts_rma_project.tasks  # noqa: F401
from cts_rma_project.tasks.cts.cts_env_cfg          import CTSEnvCfg
from cts_rma_project.tasks.cts.agents.rsl_rl_ppo_cfg import CTSPPORunnerCfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.deterministic    = False
torch.backends.cudnn.benchmark        = False


def main():
    device = args_cli.device or "cuda"

    # ── environment ─────────────────────────────────────────────────────
    env_cfg = CTSEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device
    env_cfg.seed           = args_cli.seed

    env = gym.make("Template-CTS-GO2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── runner / agent config ────────────────────────────────────────────
    runner_cfg = CTSPPORunnerCfg()
    runner_cfg.max_iterations  = args_cli.max_iterations
    runner_cfg.experiment_name = args_cli.experiment
    runner_cfg.seed            = args_cli.seed

    log_dir = os.path.join(
        "logs", "cts",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.experiment}"
    )
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=device)

    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)
        print(f"[INFO] Resumed from checkpoint: {args_cli.checkpoint}")

    # ── training ─────────────────────────────────────────────────────────
    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)

    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"[INFO] CTS training complete. Model saved to: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

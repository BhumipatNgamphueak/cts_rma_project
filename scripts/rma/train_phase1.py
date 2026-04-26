"""
RMA Phase 1 training script.

Trains with asymmetric actor-critic:
  actor  sees o_t (37D proprioceptive)  — same obs as Baseline at deployment
  critic sees o_t ⊕ e_t (61D)          — privileged DR params for better value estimates

Usage:
    python scripts/rma/train_phase1.py \
        --num_envs 4096 --max_iterations 5000 \
        --experiment rma_go2 --device cuda:0
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RMA training")
parser.add_argument("--num_envs",       type=int,   default=4096)
parser.add_argument("--max_iterations", type=int,   default=5000)
parser.add_argument("--experiment",     type=str,   default="rma_go2")
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--checkpoint",     type=str,   default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports that require Isaac Sim to be running ──────────────────────────
import os
import gymnasium as gym
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore
from rsl_rl.runners import OnPolicyRunner           # type: ignore

import cts_rma_project.tasks  # noqa: F401  registers gym envs
from cts_rma_project.tasks.rma.rma_env_cfg            import RMAEnvCfg
from cts_rma_project.tasks.rma.agents.rsl_rl_ppo_cfg  import RMAPPORunnerCfg


def main():
    device = args_cli.device or "cuda"

    # ── environment ─────────────────────────────────────────────────────
    env_cfg = RMAEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device
    env_cfg.seed           = args_cli.seed

    env = gym.make("Template-RMA-GO2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── runner / agent config ────────────────────────────────────────────
    runner_cfg = RMAPPORunnerCfg()
    runner_cfg.max_iterations  = args_cli.max_iterations
    runner_cfg.experiment_name = args_cli.experiment
    runner_cfg.seed            = args_cli.seed

    log_dir = os.path.join(
        "logs", "rma",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.experiment}"
    )
    os.makedirs(log_dir, exist_ok=True)

    # Standard OnPolicyRunner — the RMAEnvCfg "critic" obs group gives the
    # runner num_privileged_obs=61, so ActorCritic is built asymmetrically:
    #   actor  MLP: 37 → [512,256,128] → 12
    #   critic MLP: 61 → [512,256,128] → 1
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=device)

    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)
        print(f"[INFO] Resumed from checkpoint: {args_cli.checkpoint}")

    # ── training ─────────────────────────────────────────────────────────
    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)

    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"[INFO] RMA training complete. Model saved to: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

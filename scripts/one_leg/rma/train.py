"""
One-Leg RMA Phase-1 training script.

Actor  sees: 14-D proprioceptive obs
Critic sees: 22-D (14 prop + 8 privileged DR params)

Usage:
    python scripts/one_leg/rma/train.py --num_envs 1024 --headless
"""
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",       type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--experiment",     type=str, default="one_leg_rma")
parser.add_argument("--seed",           type=int, default=42)
parser.add_argument("--checkpoint",     type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, gymnasium as gym
from datetime import datetime
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper          # type: ignore
from rsl_rl.runners     import OnPolicyRunner               # type: ignore

import cts_rma_project.tasks  # noqa
from cts_rma_project.tasks.one_leg.rma.one_leg_rma_env_cfg    import OneLegRMAEnvCfg
from cts_rma_project.tasks.one_leg.rma.agents.rsl_rl_ppo_cfg  import OneLegRMAPPOCfg


def main():
    device = args_cli.device or "cuda"

    env_cfg = OneLegRMAEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device

    env = gym.make("OneLeg-RMA-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = OneLegRMAPPOCfg()
    runner_cfg.max_iterations  = args_cli.max_iterations
    runner_cfg.experiment_name = args_cli.experiment
    runner_cfg.seed            = args_cli.seed

    log_dir = os.path.join("logs", "one_leg", "rma",
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=device)
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)

    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)
    runner.save(os.path.join(log_dir, "model_final.pt"))
    print(f"[INFO] Saved to {log_dir}/model_final.pt")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

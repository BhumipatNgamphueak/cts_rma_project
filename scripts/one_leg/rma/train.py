"""
One-Leg RMA Phase-1 training script.

Actor sees [ot, xt] = 39D; RMAActorCritic internally encodes xt → zt(8)
then runs policy π([ot, zt]). Critic also sees 39D.

Usage:
    python scripts/one_leg/rma/train.py --num_envs 1024 --headless
"""
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",       type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--experiment",     type=str, default=None)
parser.add_argument("--seed",           type=int, default=42)
parser.add_argument("--checkpoint",     type=str, default=None)
parser.add_argument("--priv_mode",      type=str, default="FULL",
                    choices=["FULL", "INT", "EXT",
                             "FULL_NO_CF", "FULL_NO_TORQ", "FULL_NO_ACCEL", "FULL_NO_PUSH"],
                    help="Privileged knowledge ablation. FULL/INT/EXT=group; "
                         "FULL_NO_* removes one external component for diagnostic.")
parser.add_argument("--latent_dim",     type=int, default=8,
                    choices=[8, 16, 32],
                    help="Encoder bottleneck size Z (default 8)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, gymnasium as gym
from datetime import datetime
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper          # type: ignore
from rsl_rl.runners     import OnPolicyRunner               # type: ignore

# Inject RMAActorCritic into runner's namespace so eval("RMAActorCritic") works
from cts_rma_project.tasks.one_leg.rma.rma_policy import RMAActorCritic
import rsl_rl.runners.on_policy_runner as _runner_mod
_runner_mod.RMAActorCritic = RMAActorCritic

import cts_rma_project.tasks  # noqa
from cts_rma_project.tasks.one_leg.rma.one_leg_rma_env_cfg    import OneLegRMAEnvCfg
from cts_rma_project.tasks.one_leg.rma.agents.rsl_rl_ppo_cfg  import OneLegRMAPPOCfg


def main():
    device = args_cli.device or "cuda"

    priv_mode  = args_cli.priv_mode.upper()
    latent_dim = args_cli.latent_dim
    experiment = args_cli.experiment or f"one_leg_rma_{priv_mode.lower()}_l{latent_dim}"

    env_cfg = OneLegRMAEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device
    env_cfg.priv_mode      = priv_mode

    env = gym.make("OneLeg-RMA-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = OneLegRMAPPOCfg()
    runner_cfg.max_iterations  = args_cli.max_iterations
    runner_cfg.experiment_name = experiment
    runner_cfg.seed            = args_cli.seed

    log_dir = os.path.join(
        "logs", "one_leg", "rma",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{experiment}"
    )
    os.makedirs(log_dir, exist_ok=True)

    train_dict = runner_cfg.to_dict()
    train_dict["policy"]["latent_dim"] = latent_dim

    runner = OnPolicyRunner(env, train_dict, log_dir=log_dir, device=device)
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)

    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)

    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"[INFO] RMA Phase-1 complete. Saved: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

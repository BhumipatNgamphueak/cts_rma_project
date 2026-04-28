"""
One-Leg CTS training script (concurrent teacher-student, paper Section 3.2).

Single-stage training:
  - 75 % teacher envs: obs = [ot,xt,pad36,flag=1], critic = [ot,xt]
  - 25 % student envs: obs = [history(75D), flag=0], critic = [ot,xt]
  - L_rec = MSE(Es(history), Et([ot,xt])) over student envs (before PPO update)
  - PPO update with CTSActorCritic routing teacher/student via obs[:,-1] flag

Usage:
    python scripts/one_leg/cts/train.py --num_envs 1024 --headless
"""
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",       type=int,   default=2048)
parser.add_argument("--max_iterations", type=int,   default=5000)
parser.add_argument("--experiment",     type=str,   default=None)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--lambda_rec",     type=float, default=1.0)
parser.add_argument("--checkpoint",     type=str,   default=None)
parser.add_argument("--priv_mode",      type=str,   default="FULL",
                    choices=["FULL", "INT", "EXT",
                             "FULL_NO_CF", "FULL_NO_TORQ", "FULL_NO_ACCEL", "FULL_NO_PUSH"],
                    help="Privileged knowledge ablation. FULL/INT/EXT=group; "
                         "FULL_NO_* removes one external component for diagnostic.")
parser.add_argument("--latent_dim",     type=int,   default=8,
                    choices=[8, 16, 32, 64, 128],
                    help="Encoder bottleneck size Z (default 8; use 16/32/64/128 for ablation)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, gymnasium as gym
from datetime import datetime
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper              # type: ignore

# Inject CTSActorCritic into runner namespace for eval("CTSActorCritic")
from cts_rma_project.tasks.one_leg.cts.cts_policy import CTSActorCritic
import rsl_rl.runners.on_policy_runner as _runner_mod
_runner_mod.CTSActorCritic = CTSActorCritic

from cts_rma_project.tasks.one_leg.cts.cts_runner import CTSRunner

import cts_rma_project.tasks  # noqa
from cts_rma_project.tasks.one_leg.cts.one_leg_cts_env_cfg    import OneLegCTSEnvCfg
from cts_rma_project.tasks.one_leg.cts.agents.rsl_rl_ppo_cfg  import OneLegCTSPPOCfg


def main():
    device = args_cli.device or "cuda"

    priv_mode  = args_cli.priv_mode.upper()
    latent_dim = args_cli.latent_dim
    experiment = args_cli.experiment or f"one_leg_cts_{priv_mode.lower()}_l{latent_dim}"

    env_cfg = OneLegCTSEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device
    env_cfg.priv_mode      = priv_mode

    env = gym.make("OneLeg-CTS-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = OneLegCTSPPOCfg()
    runner_cfg.max_iterations  = args_cli.max_iterations
    runner_cfg.experiment_name = experiment
    runner_cfg.seed            = args_cli.seed

    log_dir = os.path.join(
        "logs", "one_leg", "cts",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{experiment}"
    )
    os.makedirs(log_dir, exist_ok=True)

    train_dict = runner_cfg.to_dict()
    train_dict["cts_lambda_rec"]          = args_cli.lambda_rec   # consumed by CTSRunner
    train_dict["policy"]["latent_dim"]    = latent_dim            # passed to CTSActorCritic

    runner = CTSRunner(env, train_dict, log_dir=log_dir, device=device)
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)

    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)
    print(f"[INFO] CTS training complete. Saved: {log_dir}/model_final.pt")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

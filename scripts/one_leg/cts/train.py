"""
One-Leg CTS training script (concurrent teacher-student, paper Section 3.2).

Single-stage training:
  - 90 % teacher envs: obs = [ot,xt,pad702,flag=1], critic = [ot,xt]
  - 10 % student envs: obs = [history(750D H=50), flag=0], critic = [ot,xt]
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
parser.add_argument("--warm_start",     type=str,   default=None,
                    help="Path to Baseline model_final.pt — copies actor weights "
                         "so CTS starts at Baseline performance and only learns encoder")
parser.add_argument("--priv_mode",      type=str,   default="FULL",
                    choices=["FULL", "INT", "EXT",
                             "FULL_NO_CF", "FULL_NO_TORQ", "FULL_NO_ACCEL", "FULL_NO_PUSH"],
                    help="Privileged knowledge ablation. FULL/INT/EXT=group; "
                         "FULL_NO_* removes one external component for diagnostic.")
parser.add_argument("--latent_dim",     type=int,   default=8,
                    choices=[8, 16, 32, 64, 128],
                    help="Encoder bottleneck size Z (default 8; use 16/32/64/128 for ablation)")
parser.add_argument("--history_len",    type=int,   default=50,
                    help="Student obs history length H in steps (default 50 = 500ms at 100Hz)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, torch, gymnasium as gym
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


def _warm_start_actor(policy, baseline_ckpt_path: str, device: str):
    """Copy Baseline actor weights into a CTS policy.

    First actor layer: Baseline (256,15) → CTS (256,15+Z).
    Copy the ot columns (0:15) exactly; zero-init the z columns so the actor
    starts as Baseline behaviour and only learns to exploit z over time.
    All subsequent hidden layers are the same shape and copied directly.
    """
    ckpt    = torch.load(baseline_ckpt_path, map_location=device)
    base_sd = ckpt.get("model_state_dict", ckpt)
    pol_sd  = policy.state_dict()

    for k in [k for k in base_sd if k.startswith("actor.")]:
        if k not in pol_sd:
            continue
        src, dst = base_sd[k], pol_sd[k]
        if src.shape == dst.shape:
            pol_sd[k] = src.clone()
        elif src.dim() == 2 and dst.shape[0] == src.shape[0] and dst.shape[1] > src.shape[1]:
            new_w = torch.zeros_like(dst)
            new_w[:, :src.shape[1]] = src.clone()
            pol_sd[k] = new_w

    if "std" in base_sd and "std" in pol_sd:
        pol_sd["std"] = base_sd["std"].clone()

    policy.load_state_dict(pol_sd, strict=False)
    print(f"[warm_start] actor loaded from {baseline_ckpt_path}")


def main():
    device = args_cli.device or "cuda"

    priv_mode   = args_cli.priv_mode.upper()
    latent_dim  = args_cli.latent_dim
    history_len = args_cli.history_len
    experiment  = args_cli.experiment or f"one_leg_cts_{priv_mode.lower()}_l{latent_dim}_h{history_len}"

    env_cfg = OneLegCTSEnvCfg()
    env_cfg.scene.num_envs    = args_cli.num_envs
    env_cfg.sim.device        = device
    env_cfg.priv_mode         = priv_mode
    env_cfg.history_len       = history_len
    env_cfg.observation_space = history_len * 15 + 1   # H×15 unified + 1 flag

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
    train_dict["cts_lambda_rec"]          = args_cli.lambda_rec
    train_dict["cts_rec_warmup"]          = max(50, args_cli.max_iterations // 5)
    train_dict["policy"]["latent_dim"]    = latent_dim
    train_dict["policy"]["history_len"]   = history_len

    runner = CTSRunner(env, train_dict, log_dir=log_dir, device=device)
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)
    if args_cli.warm_start:
        _warm_start_actor(runner.alg.policy, args_cli.warm_start, device)

    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)
    print(f"[INFO] CTS training complete. Saved: {log_dir}/model_final.pt")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

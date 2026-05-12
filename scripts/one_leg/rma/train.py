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
parser.add_argument("--warm_start",     type=str, default=None,
                    help="Path to Baseline model_final.pt — copies actor weights "
                         "so RMA starts at Baseline performance and only learns encoder")
parser.add_argument("--priv_mode",      type=str, default="FULL",
                    choices=["FULL", "INT", "EXT",
                             "FULL_NO_CF", "FULL_NO_TORQ", "FULL_NO_ACCEL", "FULL_NO_PUSH"],
                    help="Privileged knowledge ablation. FULL/INT/EXT=group; "
                         "FULL_NO_* removes one external component for diagnostic.")
parser.add_argument("--latent_dim",     type=int, default=8,
                    choices=[8, 16, 32, 64, 128],
                    help="Encoder bottleneck size Z (default 8)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, torch, gymnasium as gym
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


def _warm_start_actor(policy, baseline_ckpt_path: str, device: str):
    """Copy Baseline actor weights into an RMA policy.

    First actor layer: Baseline (256,15) → RMA (256,15+Z).
    Copy the ot columns (0:15) exactly; zero-init the z columns so the actor
    starts as if z=0 (= Baseline behaviour) and only learns to exploit z.
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
            # First layer: copy ot columns, leave z columns near-zero
            new_w = torch.zeros_like(dst)
            new_w[:, :src.shape[1]] = src.clone()
            pol_sd[k] = new_w

    if "std" in base_sd and "std" in pol_sd:
        pol_sd["std"] = base_sd["std"].clone()

    policy.load_state_dict(pol_sd, strict=False)
    print(f"[warm_start] actor loaded from {baseline_ckpt_path}")


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
    if args_cli.warm_start:
        _warm_start_actor(runner.alg.policy, args_cli.warm_start, device)

    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)

    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"[INFO] RMA Phase-1 complete. Saved: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""
GO2 CTS (Concurrent Teacher-Student) training script.

Single-stage training:
  75% teacher envs: obs = [ot(37), xt(26), zeros(H*37-63), flag=1]
  25% student envs: obs = [flat_history(H×37), flag=0]
  L_rec = MSE(E^s(history), detach(E^t(xt))) — shapes E^s after each PPO update

Usage:
    python scripts/cts/train.py \\
        --num_envs 4096 --max_iterations 5000 \\
        --latent_dim 8 --history_len 50 \\
        --experiment cts_go2_full_l8 --device cuda:0 --headless

    # Warm-start actor from a trained Baseline checkpoint
    python scripts/cts/train.py \\
        --num_envs 4096 --max_iterations 5000 \\
        --warm_start logs/baseline/<run>/model_final.pt \\
        --experiment cts_go2_ws_l8 --headless

    # Resume from checkpoint
    python scripts/cts/train.py \\
        --checkpoint logs/cts/<run>/model_500.pt \\
        --max_iterations 5000 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GO2 CTS training")
parser.add_argument("--num_envs",       type=int,   default=4096)
parser.add_argument("--max_iterations", type=int,   default=5000)
parser.add_argument("--experiment",     type=str,   default=None)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--latent_dim",     type=int,   default=8,
                    choices=[8, 16, 32, 64, 128],
                    help="Encoder bottleneck Z (default 8)")
parser.add_argument("--priv_mode",      type=str,   default="FULL",
                    choices=["FULL", "INT", "EXT", "TERR", "FULL_T"],
                    help="Privileged knowledge fed to the teacher encoder / critic "
                         "(FULL=26D, INT=16D, EXT=10D, TERR=77D terrain, FULL_T=103D)")
parser.add_argument("--history_len",    type=int,   default=50,
                    help="Student obs history H in steps (default 50 = 1000ms at 50Hz)")
parser.add_argument("--lambda_rec",     type=float, default=5.0,
                    help="L_rec loss weight (default 5.0 — sim2sim fix #4)")
parser.add_argument("--rec_warmup",     type=int,   default=None,
                    help="L_rec warmup iters (default: max_iterations//5)")
parser.add_argument("--checkpoint",     type=str,   default=None,
                    help="Checkpoint to resume from")
parser.add_argument("--warm_start",     type=str,   default=None,
                    help="Baseline model_final.pt — copies actor weights "
                         "into CTS so training starts at Baseline performance")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports that require Isaac Sim to be running ─────────────────────────────
import os
import torch
import gymnasium as gym
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore

# Inject CTSActorCritic into runner namespace so eval("CTSActorCritic") resolves
from cts_rma_project.tasks.cts.cts_network import CTSActorCritic
import rsl_rl.runners.on_policy_runner as _runner_mod
_runner_mod.CTSActorCritic = CTSActorCritic

from cts_rma_project.tasks.cts.cts_runner              import CTSRunner
import cts_rma_project.tasks  # noqa: F401  (registers gym envs)
from cts_rma_project.tasks.cts.cts_env_cfg             import CTSEnvCfg
from cts_rma_project.tasks.cts.agents.rsl_rl_ppo_cfg   import CTSPPORunnerCfg
from cts_rma_project.tasks.shared.mdp                  import PRIV_DIMS

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.deterministic    = False
torch.backends.cudnn.benchmark        = False


def _warm_start_actor(policy, baseline_ckpt_path: str, device: str):
    """Copy Baseline actor weights into a CTS policy.

    First actor layer: Baseline (H, 37) → CTS (H, 37+Z).
    Copy the ot columns (0:37) exactly; zero-init the z columns so the actor
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
    device      = args_cli.device or "cuda"
    latent_dim  = args_cli.latent_dim
    history_len = args_cli.history_len
    priv_mode   = args_cli.priv_mode.upper()
    priv_dim    = PRIV_DIMS[priv_mode]
    experiment  = args_cli.experiment or f"cts_go2_{priv_mode.lower()}_l{latent_dim}_h{history_len}"

    # ── Environment ──────────────────────────────────────────────────────────
    env_cfg = CTSEnvCfg()
    env_cfg.scene.num_envs    = args_cli.num_envs
    env_cfg.sim.device        = device
    env_cfg.seed              = args_cli.seed
    env_cfg.history_len       = history_len
    env_cfg.priv_mode         = priv_mode
    env_cfg.observation_space = history_len * 37 + 1   # H×37+1 unified + flag
    env_cfg.state_space       = 37 + priv_dim          # [ot(37), xt(priv_dim)] critic

    env = gym.make("Go2-CTS-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── Runner / agent config ─────────────────────────────────────────────────
    runner_cfg = CTSPPORunnerCfg()
    runner_cfg.max_iterations  = args_cli.max_iterations
    runner_cfg.experiment_name = experiment
    runner_cfg.seed            = args_cli.seed

    log_dir = os.path.join(
        "logs", "cts",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{experiment}"
    )
    os.makedirs(log_dir, exist_ok=True)

    train_dict = runner_cfg.to_dict()
    train_dict["cts_lambda_rec"]        = args_cli.lambda_rec
    train_dict["cts_rec_warmup"]        = (args_cli.rec_warmup
                                            if args_cli.rec_warmup is not None
                                            else max(50, args_cli.max_iterations // 5))
    train_dict["policy"]["latent_dim"]  = latent_dim
    train_dict["policy"]["history_len"] = history_len
    train_dict["policy"]["priv_dim"]    = priv_dim

    runner = CTSRunner(env, train_dict, log_dir=log_dir, device=device)

    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)
        print(f"[INFO] Resumed from: {args_cli.checkpoint}")

    if args_cli.warm_start:
        _warm_start_actor(runner.alg.policy, args_cli.warm_start, device)

    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)
    print(f"[INFO] CTS training complete. Saved: {log_dir}/model_final.pt")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

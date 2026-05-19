"""
RMA Phase 2 training script.

Trains the adaptation module φ via supervised learning.  φ learns to map
a 0.5 s history of (state, action) pairs to the latent code ẑ_t that
approximates z_t = μ(e_t) produced by the frozen Phase 1 encoder.

A Phase 1 checkpoint must be supplied via --checkpoint.

Usage:
    ./isaaclab.sh -p scripts/rma/train_phase2.py \
        --checkpoint logs/rma/phase1/<run>/model_final.pt \
        --num_envs 4096 --num_iterations 1000 --device cuda:0
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RMA Phase 2 training")
parser.add_argument("--checkpoint",     type=str,   required=True,
                    help="Path to the Phase 1 model checkpoint (.pt)")
parser.add_argument("--num_envs",       type=int,   default=4096)
parser.add_argument("--num_iterations", type=int,   default=1000)
parser.add_argument("--history_len",    type=int,   default=50,
                    help="Number of (state, action) timesteps fed to φ")
parser.add_argument("--batch_size",     type=int,   default=80000)
parser.add_argument("--lr",             type=float, default=5e-4)
parser.add_argument("--latent_dim",     type=int,   default=8,
                    help="Must match the Phase 1 checkpoint latent_dim (8/16/32/64/128)")
parser.add_argument("--priv_mode",      type=str,   default="FULL",
                    choices=["FULL", "INT", "EXT", "TERR", "FULL_T"],
                    help="Must match the Phase 1 checkpoint priv_mode (FULL/INT/EXT/FULL_T)")
parser.add_argument("--seed",           type=int,   default=42)
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

import cts_rma_project.tasks  # noqa: F401
from cts_rma_project.tasks.rma.rma_env_cfg import RMAEnvCfg
from cts_rma_project.tasks.rma.rma_network import RMAActorCritic
from cts_rma_project.tasks.rma.rma_runner  import RMAPhase2Runner
from cts_rma_project.tasks.shared.mdp      import PRIV_DIMS


def main():
    device    = args_cli.device or "cuda"
    priv_mode = args_cli.priv_mode.upper()
    priv_dim  = PRIV_DIMS[priv_mode]

    # ── environment ─────────────────────────────────────────────────────
    env_cfg  = RMAEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device
    env_cfg.seed           = args_cli.seed
    env_cfg.priv_mode      = priv_mode
    # Height scanner is only needed when x_t includes terrain heights.
    # For FULL/INT/EXT modes, removing it avoids a silent warp-mesh init failure
    # that leaves RayCaster.drift unset and crashes on the first env.reset().
    if priv_mode not in ("TERR", "FULL_T"):
        env_cfg.scene.height_scanner = None

    env = gym.make("Template-RMA-GO2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── load Phase 1 model ───────────────────────────────────────────────
    obs_dim  = env.num_obs
    crit_dim = env.num_privileged_obs or obs_dim
    act_dim  = env.num_actions

    rma_model = RMAActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=crit_dim,
        num_actions=act_dim,
        env_factor_dim=priv_dim,
        latent_dim=args_cli.latent_dim,
    ).to(device)

    checkpoint = torch.load(args_cli.checkpoint, map_location=device)
    # RSL-RL saves actor_critic state inside the checkpoint dict
    state_dict = checkpoint.get("model_state_dict",
                  checkpoint.get("actor_critic", checkpoint))
    rma_model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded Phase 1 weights from: {args_cli.checkpoint}")

    # ── Phase 2 runner ───────────────────────────────────────────────────
    log_dir = os.path.join(
        "logs", "rma", "phase2",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    phase2_runner = RMAPhase2Runner(
        env=env.unwrapped,
        rma_model=rma_model,
        history_len=args_cli.history_len,
        num_iterations=args_cli.num_iterations,
        batch_size=args_cli.batch_size,
        learning_rate=args_cli.lr,
        log_dir=log_dir,
        device=device,
        state_dim=obs_dim,
        latent_dim=args_cli.latent_dim,
        priv_dim=priv_dim,
    )

    phase2_runner.collect_and_train()
    print(f"[INFO] Phase 2 complete. Adaptation module saved to: {log_dir}/")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

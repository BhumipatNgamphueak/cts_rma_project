"""
Baseline play / evaluation script.

Loads a trained Baseline checkpoint and runs the GO2 robot in Isaac Sim
using a deterministic (noise-free) policy.

Usage:
    ./isaaclab.sh -p scripts/baseline/play.py \
        --checkpoint logs/baseline/<run>/model_final.pt \
        --num_envs 32

    # Record a video:
    ./isaaclab.sh -p scripts/baseline/play.py \
        --checkpoint logs/baseline/<run>/model_final.pt \
        --video --video_length 300
"""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Baseline play")
parser.add_argument("--checkpoint",   type=str, required=True)
parser.add_argument("--num_envs",     type=int, default=32)
parser.add_argument("--video",        action="store_true")
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--real_time",    action="store_true")
parser.add_argument("--no_push",       action="store_true",
                    help="Disable push_robot and impulse disturbance events during play")
parser.add_argument("--no_reset_rand", action="store_true",
                    help="Disable reset randomisation (default pose + zero velocity every episode)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ───────────────────────────────────────────────────
import os
import torch
import gymnasium as gym

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore

import cts_rma_project.tasks  # noqa: F401
from cts_rma_project.tasks.baseline.baseline_env_cfg import BaselineEnvCfg_PLAY


def main():
    device = args_cli.device or "cuda"

    # ── environment ──────────────────────────────────────────────────────
    env_cfg = BaselineEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device

    if args_cli.no_push:
        env_cfg.events.push_robot       = None
        env_cfg.events.impulse_interval = None
        env_cfg.events.impulse_reset    = None
        print("[play] Disturbance events DISABLED")

    if args_cli.no_reset_rand:
        # Disable reset randomisation — clean start every episode (default pose, zero vel).
        # If robot still walks → policy genuinely learned locomotion.
        # If robot stands still → policy only learned to RECOVER from perturbations.
        env_cfg.events.reset_base   = None
        env_cfg.events.reset_joints = None
        print("[play] Reset randomisation DISABLED — clean start (default pose, zero velocity)")

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("Template-Baseline-GO2-Play-v0", cfg=env_cfg,
                   render_mode=render_mode)

    if args_cli.video:
        log_dir = os.path.dirname(args_cli.checkpoint)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos"),
            step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env)

    # ── load policy weights ──────────────────────────────────────────────
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    # RSL-RL saves actor_critic state dict under "model_state_dict" or flat
    actor_critic_state = ckpt.get("model_state_dict", ckpt.get("actor_critic", ckpt))

    # Build a minimal actor using RSL-RL's ActorCritic so we can call act_inference
    from rsl_rl.modules import ActorCritic  # type: ignore
    obs_dim = env.num_obs
    act_dim = env.num_actions
    actor_critic = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=act_dim,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    ).to(device)
    actor_critic.load_state_dict(actor_critic_state, strict=False)
    actor_critic.eval()
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

    # ── rollout ──────────────────────────────────────────────────────────
    dt = env.unwrapped.physics_dt
    obs, _ = env.get_observations()
    timestep = 0

    with torch.inference_mode():
        while simulation_app.is_running():
            t0 = time.time()

            actions = actor_critic.act_inference(obs)
            obs, _, dones, _ = env.step(actions)
            timestep += 1

            if args_cli.video and timestep >= args_cli.video_length:
                break

            if args_cli.real_time:
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

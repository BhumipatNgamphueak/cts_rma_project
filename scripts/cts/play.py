"""
CTS play/evaluation script.

Loads a trained CTS checkpoint and runs the GO2 robot in Isaac Sim
using the deterministic (mean) action from CTSActorCritic.

Usage:
    ./isaaclab.sh -p scripts/cts/play.py \
        --checkpoint logs/cts/<run>/model_final.pt \
        --num_envs 32

    # Record a video
    ./isaaclab.sh -p scripts/cts/play.py \
        --checkpoint logs/cts/<run>/model_final.pt \
        --video --video_length 300
"""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CTS play")
parser.add_argument("--checkpoint",   type=str, required=True)
parser.add_argument("--num_envs",     type=int, default=32)
parser.add_argument("--video",        action="store_true")
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--real_time",    action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports that require Isaac Sim ────────────────────────────────────────
import os
import torch
import gymnasium as gym

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore

import cts_rma_project.tasks  # noqa: F401
from cts_rma_project.tasks.cts.cts_env_cfg  import CTSEnvCfg_PLAY
from cts_rma_project.tasks.cts.cts_network  import CTSActorCritic


def main():
    device = args_cli.device or "cuda"

    # ── environment ─────────────────────────────────────────────────────
    env_cfg = CTSEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("Template-CTS-GO2-Play-v0", cfg=env_cfg, render_mode=render_mode)

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

    # ── load model ───────────────────────────────────────────────────────
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = CTSActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=act_dim,
    ).to(device)

    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("actor_critic", ckpt))
    policy.load_state_dict(state, strict=False)
    policy.eval()
    print(f"[INFO] Loaded CTS model from: {args_cli.checkpoint}")

    # ── rollout ──────────────────────────────────────────────────────────
    dt = env.unwrapped.physics_dt
    obs, _ = env.get_observations()
    timestep = 0

    with torch.inference_mode():
        while simulation_app.is_running():
            t0 = time.time()
            actions = policy.act_inference(obs)
            obs, _, _, _ = env.step(actions)
            timestep += 1

            if args_cli.video and timestep >= args_cli.video_length:
                break

            if args_cli.real_time:
                sleep = dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

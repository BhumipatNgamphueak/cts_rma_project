"""
RMA play/evaluation script.

Loads a Phase 1 checkpoint (and optionally a Phase 2 adaptation module)
and runs the GO2 robot in Isaac Sim.

If --adapt_checkpoint is supplied, the adaptation module φ is used to
estimate the latent code ẑ_t from a rolling 0.5 s observation history,
matching the full deployment setup described in the RMA paper.

Usage:
    # Phase 1 policy only (ẑ = 0)
    ./isaaclab.sh -p scripts/rma/play.py \
        --checkpoint logs/rma/phase1/<run>/model_final.pt

    # Phase 1 + Phase 2 adaptation
    ./isaaclab.sh -p scripts/rma/play.py \
        --checkpoint     logs/rma/phase1/<run>/model_final.pt \
        --adapt_checkpoint logs/rma/phase2/<run>/adapt_module_final.pt
"""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RMA play")
parser.add_argument("--checkpoint",       type=str, required=True)
parser.add_argument("--adapt_checkpoint", type=str, default=None,
                    help="Path to Phase 2 adaptation module .pt")
parser.add_argument("--num_envs",         type=int, default=32)
parser.add_argument("--history_len",      type=int, default=50)
parser.add_argument("--video",            action="store_true")
parser.add_argument("--video_length",     type=int, default=200)
parser.add_argument("--real_time",        action="store_true")
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
from cts_rma_project.tasks.rma.rma_env_cfg  import RMAEnvCfg_PLAY
from cts_rma_project.tasks.rma.rma_network  import RMAActorCritic, AdaptationModule


def main():
    device = args_cli.device or "cuda"

    # ── environment ─────────────────────────────────────────────────────
    env_cfg = RMAEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("Template-RMA-GO2-Play-v0", cfg=env_cfg, render_mode=render_mode)

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

    # ── load Phase 1 model ───────────────────────────────────────────────
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    rma_model = RMAActorCritic(
        num_actor_obs=obs_dim, num_critic_obs=obs_dim,
        num_actions=act_dim, env_factor_dim=17, latent_dim=8,
    ).to(device)

    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("actor_critic", ckpt))
    rma_model.load_state_dict(state, strict=False)
    rma_model.eval()
    print(f"[INFO] Loaded Phase 1 from: {args_cli.checkpoint}")

    # ── optionally load adaptation module ────────────────────────────────
    adapt_module = None
    if args_cli.adapt_checkpoint:
        adapt_module = AdaptationModule(
            state_dim=obs_dim, action_dim=act_dim,
            embed_dim=32, latent_dim=8, history_len=args_cli.history_len,
        ).to(device)
        adapt_module.load_state_dict(
            torch.load(args_cli.adapt_checkpoint, map_location=device)
        )
        adapt_module.eval()
        print(f"[INFO] Loaded adaptation module from: {args_cli.adapt_checkpoint}")

    # ── history buffers for adaptation module ────────────────────────────
    N = args_cli.num_envs
    H = args_cli.history_len
    state_hist  = torch.zeros(N, H, obs_dim, device=device)
    action_hist = torch.zeros(N, H, act_dim, device=device)

    def roll_history(obs, act):
        state_hist.roll_(-1, dims=1)
        action_hist.roll_(-1, dims=1)
        state_hist[:, -1, :]  = obs
        action_hist[:, -1, :] = act

    # ── rollout ──────────────────────────────────────────────────────────
    dt = env.unwrapped.physics_dt
    obs, _ = env.get_observations()
    timestep = 0

    with torch.inference_mode():
        while simulation_app.is_running():
            t0 = time.time()

            if adapt_module is not None:
                z_hat = adapt_module(state_hist, action_hist)
                actions = rma_model.act_inference(obs, z_override=z_hat)
            else:
                actions = rma_model.act_inference(obs)

            obs_next, _, dones, _ = env.step(actions)
            roll_history(obs, actions)

            if dones.any():
                state_hist[dones]  = 0.0
                action_hist[dones] = 0.0

            obs = obs_next
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

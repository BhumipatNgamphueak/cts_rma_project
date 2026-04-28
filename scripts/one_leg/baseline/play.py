"""
One-Leg Baseline play script.

Auto-detects obs dimension from the checkpoint — works with both the
old 14-D and the new 15-D observation spaces.

Usage:
    python scripts/one_leg/baseline/play.py \
        --checkpoint logs/one_leg/baseline/<run>/model_final.pt \
        --num_envs 16

    # With video capture:
    python scripts/one_leg/baseline/play.py \
        --checkpoint logs/one_leg/baseline/<run>/model_final.pt \
        --num_envs 4 --video --video_length 500
"""
import argparse, sys, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",   type=str, required=True)
parser.add_argument("--num_envs",     type=int, default=16)
parser.add_argument("--video",        action="store_true")
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--real_time",    action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, torch, gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper          # type: ignore
from rsl_rl.modules     import ActorCritic                  # type: ignore

import cts_rma_project.tasks  # noqa
from cts_rma_project.tasks.one_leg.baseline.one_leg_env_cfg import OneLegEnvCfg_PLAY


def main():
    device = args_cli.device or "cuda"

    # ── Detect obs size from checkpoint ──────────────────────────────────
    ckpt        = torch.load(args_cli.checkpoint, map_location=device)
    state_dict  = ckpt.get("model_state_dict", ckpt)
    ckpt_obs_dim = state_dict["actor.0.weight"].shape[1]  # input width of first layer
    ckpt_iter   = ckpt.get("iter", "?")
    print(f"[play] checkpoint iteration: {ckpt_iter}, actor obs dim: {ckpt_obs_dim}")

    # ── Environment ───────────────────────────────────────────────────────
    env_cfg = OneLegEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device
    # Match env obs_space to checkpoint if they differ
    env_cfg.observation_space = ckpt_obs_dim

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("OneLeg-Baseline-Play-v0", cfg=env_cfg, render_mode=render_mode)

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

    # ── Reconstruct actor (hidden dims inferred from state_dict) ──────────
    # Detect hidden dims from weight keys: actor.0, actor.2, actor.4 ... actor.N
    actor_keys = sorted([k for k in state_dict if k.startswith("actor.") and "weight" in k])
    hidden_dims = [state_dict[k].shape[0] for k in actor_keys[:-1]]  # exclude last (output)
    num_actions = state_dict[actor_keys[-1]].shape[0]
    print(f"[play] actor hidden dims: {hidden_dims}, actions: {num_actions}")

    actor_critic = ActorCritic(
        num_actor_obs  = ckpt_obs_dim,
        num_critic_obs = ckpt_obs_dim,
        num_actions    = num_actions,
        actor_hidden_dims  = hidden_dims,
        critic_hidden_dims = hidden_dims,
        activation     = "elu",
    ).to(device)
    actor_critic.load_state_dict(state_dict, strict=False)
    actor_critic.eval()
    print(f"[play] Loaded: {args_cli.checkpoint}")

    # ── Rollout ───────────────────────────────────────────────────────────
    step_dt = env.unwrapped.step_dt
    obs, _  = env.get_observations()
    step    = 0

    print("[play] Running — press Ctrl+C to stop.")
    with torch.inference_mode():
        while simulation_app.is_running():
            t0 = time.time()

            # Slice obs to match checkpoint if env returns more dims (e.g. 15→14)
            obs_in = obs[:, :ckpt_obs_dim]
            actions = actor_critic.act_inference(obs_in)
            obs, _, dones, _ = env.step(actions)
            step += 1

            if args_cli.video and step >= args_cli.video_length:
                print(f"[play] Video done ({step} steps).")
                break

            if args_cli.real_time:
                elapsed = time.time() - t0
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

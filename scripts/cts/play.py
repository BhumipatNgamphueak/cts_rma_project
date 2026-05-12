"""
CTS play script — Isaac Lab viewer.

Adapted from OpenTopic/unitree_rl_lab/scripts/rsl_rl/play.py (same owner).
Uses CTSRunner.load() + get_inference_policy() instead of manual policy loading.
CTSActorCritic is injected into the runner namespace the same way train.py does.

Usage:
    cd /home/drl-68/t_s_policy/cts_rma_project
    /home/drl-68/IsaacLab/isaaclab.sh -p scripts/cts/play.py \\
        --checkpoint logs/cts/2026-05-04_10-28-44_cts_go2_l8/model_16800.pt \\
        --latent_dim 8 --history_len 50 --num_envs 32

    # Teacher mode (bypasses student encoder — use to confirm policy quality)
    /home/drl-68/IsaacLab/isaaclab.sh -p scripts/cts/play.py \\
        --checkpoint logs/cts/2026-05-04_10-28-44_cts_go2_l8/model_16800.pt \\
        --teacher_mode

    # Record video
    /home/drl-68/IsaacLab/isaaclab.sh -p scripts/cts/play.py \\
        --checkpoint logs/cts/2026-05-04_10-28-44_cts_go2_l8/model_16800.pt \\
        --video --video_length 300
"""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CTS play — Isaac Lab")
parser.add_argument("--checkpoint",   type=str, required=True)
parser.add_argument("--num_envs",     type=int, default=32)
parser.add_argument("--latent_dim",   type=int, default=8)
parser.add_argument("--history_len",  type=int, default=50)
parser.add_argument("--teacher_mode", action="store_true",
                    help="Run all envs in teacher mode (flag=1). "
                         "Uses teacher encoder — bypasses student to confirm policy quality.")
parser.add_argument("--video",        action="store_true")
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--real_time",    action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports that require Isaac Sim ─────────────────────────────────────────────
import os
import time
import torch
import gymnasium as gym

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore

# Inject CTSActorCritic into the rsl_rl runner namespace so eval("CTSActorCritic")
# resolves correctly inside OnPolicyRunner — identical to what train.py does.
from cts_rma_project.tasks.cts.cts_network import CTSActorCritic
import rsl_rl.runners.on_policy_runner as _runner_mod
_runner_mod.CTSActorCritic = CTSActorCritic

from cts_rma_project.tasks.cts.cts_runner    import CTSRunner
from cts_rma_project.tasks.cts.cts_env_cfg   import CTSEnvCfg_PLAY
from cts_rma_project.tasks.cts.agents.rsl_rl_ppo_cfg import CTSPPORunnerCfg
import cts_rma_project.tasks  # noqa: F401  (registers gym envs)


def main():
    device = args_cli.device or "cuda"

    # ── Environment ────────────────────────────────────────────────────────────
    env_cfg = CTSEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device
    env_cfg.history_len    = args_cli.history_len

    # teacher_mode: force all envs to use teacher encoder (flag=1)
    # Use this to verify the actor is producing correct outputs independent of the student encoder.
    if args_cli.teacher_mode:
        env_cfg.teacher_ratio = 1.0
        print("[INFO] Teacher mode: all envs use teacher encoder (flag=1).")
    else:
        env_cfg.teacher_ratio = 0.0   # full student deployment

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

    # ── Runner — mirrors train.py exactly ─────────────────────────────────────
    runner_cfg = CTSPPORunnerCfg()
    train_dict = runner_cfg.to_dict()
    train_dict["policy"]["latent_dim"]  = args_cli.latent_dim
    train_dict["policy"]["history_len"] = args_cli.history_len

    runner = CTSRunner(env, train_dict, log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

    # get_inference_policy() returns a callable: obs → actions (mean, no noise)
    policy = runner.get_inference_policy(device=device)

    # extract the neural network module for reset() on episode boundaries
    try:
        policy_nn = runner.alg.policy        # rsl_rl 2.3+
    except AttributeError:
        policy_nn = runner.alg.actor_critic  # rsl_rl 2.2

    # ── Rollout ────────────────────────────────────────────────────────────────
    obs, _ = env.get_observations()
    timestep = 0

    print(f"[INFO] obs shape={obs.shape}  actions={env.num_actions}  "
          f"mode={'teacher' if args_cli.teacher_mode else 'student'}")
    print(f"[INFO] Running... (Ctrl-C to stop)\n")

    while simulation_app.is_running():
        t0 = time.time()

        with torch.inference_mode():
            actions             = policy(obs)
            obs, _, dones, _    = env.step(actions)
            # reset() is a no-op for CTSActorCritic but kept for safety
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        timestep += 1

        if args_cli.video and timestep >= args_cli.video_length:
            break

        if args_cli.real_time:
            sleep = env.unwrapped.step_dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

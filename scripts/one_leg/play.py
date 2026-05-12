"""
Real-time policy playback in the Isaac Lab viewer.

Runs a single environment at policy frequency (100 Hz) with a sleep between
steps so motion is visible.  Loops continuously until the viewer is closed.

Usage examples:
  # CTS best policy
  python scripts/one_leg/play.py \\
      --method cts --priv_mode FULL \\
      --checkpoint logs/one_leg/cts/2026-04-28_15-39-51_one_leg_cts_full_l128/model_final.pt

  # Baseline
  python scripts/one_leg/play.py \\
      --method baseline \\
      --checkpoint logs/one_leg/baseline/<run>/model_final.pt

  # RMA2
  python scripts/one_leg/play.py \\
      --method rma2 --priv_mode FULL \\
      --checkpoint logs/one_leg/rma/<p1_run>/model_final.pt \\
      --phase2_checkpoint logs/one_leg/rma/<p2_run>/adaptation_module.pt
"""
import argparse, sys, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Real-time policy playback")
parser.add_argument("--method",            type=str, required=True,
                    choices=["baseline", "rma", "rma2", "cts"])
parser.add_argument("--checkpoint",        type=str, required=True)
parser.add_argument("--phase2_checkpoint", type=str, default=None)
parser.add_argument("--priv_mode",         type=str, default="FULL",
                    choices=["FULL", "INT", "EXT"])
parser.add_argument("--dr_scale",          type=float, default=1.0,
                    help="DR randomisation scale (default 1.0 = nominal)")
parser.add_argument("--history_len",       type=int,   default=50,
                    help="CTS student history length H (must match training; default 50)")
parser.add_argument("--speed",             type=float, default=1.0,
                    help="Playback speed multiplier; <1 slows down (default 1.0)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Never headless in play mode
args_cli.headless = False
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore

import cts_rma_project.tasks  # noqa


# ── Adaptation module (identical to eval_ood.py) ──────────────────────────────
class AdaptationModule(nn.Module):
    def __init__(self, obs_dim=15, latent_dim=8, history_len=50):
        super().__init__()
        self.history_len = history_len
        self.cnn = nn.Sequential(
            nn.Conv1d(obs_dim, 32,  kernel_size=8, stride=4), nn.ELU(),
            nn.Conv1d(32,      64,  kernel_size=5, stride=2), nn.ELU(),
            nn.Conv1d(64,      128, kernel_size=3, stride=1), nn.ELU(),
        )
        dummy = torch.zeros(1, obs_dim, history_len)
        flat  = int(torch.prod(torch.tensor(self.cnn(dummy).shape[1:])))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.ELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, h):
        return self.fc(self.cnn(h.transpose(1, 2)))


def _detect_rma_latent(sd):
    keys = sorted([k for k in sd if k.startswith("encoder.") and "weight" in k])
    return sd[keys[-1]].shape[0]


def main():
    device    = args_cli.device or "cuda"
    method    = args_cli.method
    priv_mode = args_cli.priv_mode.upper()
    dr_scale  = args_cli.dr_scale

    if method == "rma2" and not args_cli.phase2_checkpoint:
        raise ValueError("--phase2_checkpoint is required for --method rma2")

    # ── Load policy ───────────────────────────────────────────────────────────
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    sd   = ckpt.get("model_state_dict", ckpt)

    if method == "baseline":
        from rsl_rl.modules import ActorCritic
        actor_keys  = sorted([k for k in sd if k.startswith("actor.") and "weight" in k])
        hidden_dims = [sd[k].shape[0] for k in actor_keys[:-1]]
        policy = ActorCritic(
            num_actor_obs=sd[actor_keys[0]].shape[1],
            num_critic_obs=sd[actor_keys[0]].shape[1],
            num_actions=sd[actor_keys[-1]].shape[0],
            actor_hidden_dims=hidden_dims, critic_hidden_dims=hidden_dims,
            activation="elu",
        ).to(device)
        policy.load_state_dict(sd, strict=False)
        phi, latent_dim = None, "N/A"

    elif method in ("rma", "rma2"):
        from cts_rma_project.tasks.one_leg.rma.rma_policy import RMAActorCritic
        latent_dim = _detect_rma_latent(sd)
        policy = RMAActorCritic(num_actor_obs=48, num_critic_obs=48,
                                num_actions=3, latent_dim=latent_dim).to(device)
        policy.load_state_dict(sd, strict=False)
        if method == "rma2":
            for p in policy.parameters():
                p.requires_grad_(False)
            phi = AdaptationModule(obs_dim=15, latent_dim=latent_dim, history_len=50).to(device)
            phi.load_state_dict(torch.load(args_cli.phase2_checkpoint, map_location=device))
            phi.eval()
        else:
            phi = None

    else:  # cts
        from cts_rma_project.tasks.one_leg.cts.cts_policy import CTSActorCritic
        latent_dim  = sd["student_fc.weight"].shape[0]
        history_len = args_cli.history_len
        policy = CTSActorCritic(num_actor_obs=history_len * 15 + 1, num_critic_obs=48,
                                num_actions=3, latent_dim=latent_dim,
                                history_len=history_len).to(device)
        policy.load_state_dict(sd, strict=False)
        phi = None

    policy.eval()
    print(f"[play] {method.upper()} priv={priv_mode} latent={latent_dim} "
          f"dr={dr_scale:.1f}x  speed={args_cli.speed:.2f}x")

    # ── Environment (1 env, no headless) ─────────────────────────────────────
    if method == "baseline":
        from cts_rma_project.tasks.one_leg.baseline.one_leg_env_cfg import OneLegEnvCfg
        cfg = OneLegEnvCfg(); gym_id = "OneLeg-Baseline-v0"
    elif method in ("rma", "rma2"):
        from cts_rma_project.tasks.one_leg.rma.one_leg_rma_env_cfg import OneLegRMAEnvCfg
        cfg = OneLegRMAEnvCfg(); cfg.priv_mode = priv_mode; gym_id = "OneLeg-RMA-v0"
    else:
        from cts_rma_project.tasks.one_leg.cts.one_leg_cts_env_cfg import OneLegCTSEnvCfg
        cfg = OneLegCTSEnvCfg()
        cfg.priv_mode         = priv_mode
        cfg.teacher_ratio     = 0.0
        cfg.history_len       = args_cli.history_len
        cfg.observation_space = args_cli.history_len * 15 + 1
        gym_id = "OneLeg-CTS-v0"

    cfg.scene.num_envs = 1
    cfg.sim.device     = device
    cfg.dr_scale       = dr_scale
    env = RslRlVecEnvWrapper(gym.make(gym_id, cfg=cfg))

    # Policy dt in real-world seconds; sleep this long between steps
    policy_dt = 0.005 * 2  # physics_dt * decimation = 0.01 s = 100 Hz
    sleep_dt  = policy_dt / args_cli.speed

    # ── RMA2 history ──────────────────────────────────────────────────────────
    if method == "rma2":
        obs_history = torch.zeros(1, phi.history_len, 15, device=device)
    else:
        obs_history = None

    obs, _      = env.get_observations()
    obs         = obs.to(device)
    step_count  = 0
    ep_count    = 0
    ep_reward   = 0.0
    t_last      = time.perf_counter()

    # Access the underlying Isaac Lab env for diagnostics
    raw_env = env.unwrapped

    print("[play] viewer open — close the window or Ctrl-C to stop")
    print(f"{'step':>6}  {'C_frc':>6}  {'C_vel':>6}  {'contact':>7}  "
          f"{'foot_h':>7}  {'action[0]':>9}  {'reward':>7}")
    with torch.inference_mode():
        while simulation_app.is_running():
            if method == "rma2":
                ot = obs[:, :15]
                obs_history = torch.roll(obs_history, -1, dims=1)
                obs_history[:, -1, :] = ot
                z_hat   = phi(obs_history)
                actions = policy.actor(torch.cat([ot, z_hat], dim=-1))
            else:
                actions = policy.act_inference(obs)

            obs, rewards, dones, _ = env.step(actions)
            obs        = obs.to(device)
            ep_reward += rewards[0].item()
            step_count += 1

            # ── Diagnostics every 10 steps ────────────────────────────────
            if step_count % 10 == 0 and hasattr(raw_env, "C_frc"):
                c_frc    = raw_env.C_frc[0].item()
                c_vel    = raw_env.C_vel[0].item()
                contact  = raw_env.is_foot_in_contact[0].item()
                foot_h   = raw_env.foot_height[0].item()
                act0     = actions[0, 0].item()
                rew      = rewards[0].item()
                print(f"{step_count:>6}  {c_frc:>6.3f}  {c_vel:>6.3f}  "
                      f"{'YES' if contact else 'no':>7}  "
                      f"{foot_h:>7.4f}  {act0:>9.4f}  {rew:>7.4f}")

            if dones[0]:
                ep_count += 1
                print(f"[play] ep {ep_count:3d} | steps={step_count:4d} "
                      f"reward={ep_reward:+.1f}")
                step_count = 0
                ep_reward  = 0.0
                if obs_history is not None:
                    obs_history[:] = 0.0

            # Real-time pacing
            t_now   = time.perf_counter()
            elapsed = t_now - t_last
            leftover = sleep_dt - elapsed
            if leftover > 0:
                time.sleep(leftover)
            t_last = time.perf_counter()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

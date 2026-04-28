"""
OOD evaluation for one-leg hopper policies.

Methods:
  baseline : ActorCritic on ot(15D)
  rma      : RMAActorCritic Phase-1 — oracle xt (upper bound, NOT deployable)
  rma2     : RMAActorCritic Phase-1 + adaptation module ϕ — obs history only (fair deployment)
  cts      : CTSActorCritic, teacher_ratio=0.0 — student encoder only (fair deployment)

Usage:
    python scripts/one_leg/eval_ood.py \\
        --method rma2 --priv_mode FULL \\
        --checkpoint logs/one_leg/rma/<p1_run>/model_final.pt \\
        --phase2_checkpoint logs/one_leg/rma/<p2_run>/adaptation_module.pt \\
        --dr_scale 1.5 --num_episodes 100 --num_envs 64 --headless
"""
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="One-leg hopper OOD evaluation")
parser.add_argument("--method",            type=str, required=True,
                    choices=["baseline", "rma", "rma2", "cts"])
parser.add_argument("--checkpoint",        type=str, required=True,
                    help="Phase-1 model_final.pt (all methods)")
parser.add_argument("--phase2_checkpoint", type=str, default=None,
                    help="adaptation_module.pt — required for --method rma2")
parser.add_argument("--dr_scale",          type=float, default=1.0)
parser.add_argument("--num_episodes",      type=int,   default=100)
parser.add_argument("--num_envs",          type=int,   default=64)
parser.add_argument("--priv_mode",         type=str,   default="FULL",
                    choices=["FULL", "INT", "EXT"])
parser.add_argument("--results_file",      type=str,   default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, csv, statistics
import torch
import torch.nn as nn
import gymnasium as gym
from datetime import datetime
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore

import cts_rma_project.tasks  # noqa


# ── Adaptation module (mirrors train_phase2.py) ───────────────────────────────
class AdaptationModule(nn.Module):
    def __init__(self, obs_dim: int = 15, latent_dim: int = 8, history_len: int = 50):
        super().__init__()
        self.history_len = history_len
        self.cnn = nn.Sequential(
            nn.Conv1d(obs_dim, 32,  kernel_size=8, stride=4), nn.ELU(),
            nn.Conv1d(32,      64,  kernel_size=5, stride=2), nn.ELU(),
            nn.Conv1d(64,      128, kernel_size=3, stride=1), nn.ELU(),
        )
        dummy   = torch.zeros(1, obs_dim, history_len)
        flat    = int(torch.prod(torch.tensor(self.cnn(dummy).shape[1:])))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.ELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(history.transpose(1, 2)))


# ── Loaders ───────────────────────────────────────────────────────────────────
def _detect_rma_latent(sd: dict) -> int:
    keys = sorted([k for k in sd if k.startswith("encoder.") and "weight" in k])
    return sd[keys[-1]].shape[0]


def _load_baseline(ckpt: dict, device: str):
    from rsl_rl.modules import ActorCritic
    sd          = ckpt.get("model_state_dict", ckpt)
    actor_keys  = sorted([k for k in sd if k.startswith("actor.") and "weight" in k])
    hidden_dims = [sd[k].shape[0] for k in actor_keys[:-1]]
    num_actions = sd[actor_keys[-1]].shape[0]
    obs_dim     = sd[actor_keys[0]].shape[1]
    policy = ActorCritic(
        num_actor_obs=obs_dim, num_critic_obs=obs_dim, num_actions=num_actions,
        actor_hidden_dims=hidden_dims, critic_hidden_dims=hidden_dims, activation="elu",
    ).to(device)
    policy.load_state_dict(sd, strict=False)
    return policy, None, "N/A"


def _load_rma(ckpt: dict, device: str):
    from cts_rma_project.tasks.one_leg.rma.rma_policy import RMAActorCritic
    sd         = ckpt.get("model_state_dict", ckpt)
    latent_dim = _detect_rma_latent(sd)
    print(f"[eval] RMA latent_dim={latent_dim}")
    policy = RMAActorCritic(num_actor_obs=48, num_critic_obs=48,
                            num_actions=3, latent_dim=latent_dim).to(device)
    policy.load_state_dict(sd, strict=False)
    return policy, None, latent_dim


def _load_rma2(ckpt: dict, phase2_path: str, device: str):
    from cts_rma_project.tasks.one_leg.rma.rma_policy import RMAActorCritic
    sd         = ckpt.get("model_state_dict", ckpt)
    latent_dim = _detect_rma_latent(sd)
    print(f"[eval] RMA2 latent_dim={latent_dim}")

    ac = RMAActorCritic(num_actor_obs=48, num_critic_obs=48,
                        num_actions=3, latent_dim=latent_dim).to(device)
    ac.load_state_dict(sd, strict=False)
    for p in ac.parameters():
        p.requires_grad_(False)
    ac.eval()

    phi = AdaptationModule(obs_dim=15, latent_dim=latent_dim, history_len=50).to(device)
    phi.load_state_dict(torch.load(phase2_path, map_location=device))
    phi.eval()
    return ac, phi, latent_dim


def _load_cts(ckpt: dict, device: str):
    from cts_rma_project.tasks.one_leg.cts.cts_policy import CTSActorCritic
    sd         = ckpt.get("model_state_dict", ckpt)
    latent_dim = sd["student_fc.weight"].shape[0]
    print(f"[eval] CTS latent_dim={latent_dim}")
    policy = CTSActorCritic(num_actor_obs=76, num_critic_obs=48,
                            num_actions=3, latent_dim=latent_dim).to(device)
    policy.load_state_dict(sd, strict=False)
    return policy, None, latent_dim


def _make_env(method: str, num_envs: int, dr_scale: float, priv_mode: str, device: str):
    if method == "baseline":
        from cts_rma_project.tasks.one_leg.baseline.one_leg_env_cfg import OneLegEnvCfg
        cfg    = OneLegEnvCfg()
        gym_id = "OneLeg-Baseline-v0"
    elif method in ("rma", "rma2"):
        from cts_rma_project.tasks.one_leg.rma.one_leg_rma_env_cfg import OneLegRMAEnvCfg
        cfg           = OneLegRMAEnvCfg()
        cfg.priv_mode = priv_mode
        gym_id        = "OneLeg-RMA-v0"
    else:  # cts
        from cts_rma_project.tasks.one_leg.cts.one_leg_cts_env_cfg import OneLegCTSEnvCfg
        cfg               = OneLegCTSEnvCfg()
        cfg.priv_mode     = priv_mode
        cfg.teacher_ratio = 0.0
        gym_id            = "OneLeg-CTS-v0"

    cfg.scene.num_envs = num_envs
    cfg.sim.device     = device
    cfg.dr_scale       = dr_scale
    return RslRlVecEnvWrapper(gym.make(gym_id, cfg=cfg))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    device    = args_cli.device or "cuda"
    method    = args_cli.method
    dr_scale  = args_cli.dr_scale
    num_eps   = args_cli.num_episodes
    num_envs  = args_cli.num_envs
    priv_mode = args_cli.priv_mode.upper()

    if method == "rma2" and not args_cli.phase2_checkpoint:
        raise ValueError("--phase2_checkpoint is required for --method rma2")

    # ── Load policy ───────────────────────────────────────────────────────
    ckpt      = torch.load(args_cli.checkpoint, map_location=device)
    ckpt_iter = ckpt.get("iter", "?")
    print(f"[eval] checkpoint iter={ckpt_iter}")

    if method == "baseline":
        policy, phi, latent_dim = _load_baseline(ckpt, device)
    elif method == "rma":
        policy, phi, latent_dim = _load_rma(ckpt, device)
    elif method == "rma2":
        policy, phi, latent_dim = _load_rma2(ckpt, args_cli.phase2_checkpoint, device)
    else:
        policy, phi, latent_dim = _load_cts(ckpt, device)

    policy.eval()

    # ── Environment ───────────────────────────────────────────────────────
    env = _make_env(method, num_envs, dr_scale, priv_mode, device)
    max_steps      = int(env.unwrapped.max_episode_length)
    success_thresh = int(0.8 * max_steps)
    print(f"[eval] method={method} priv={priv_mode} dr={dr_scale:.1f}x "
          f"max_steps={max_steps} success>{success_thresh}")

    # ── History buffer for rma2 ───────────────────────────────────────────
    H           = phi.history_len if phi is not None else 0
    obs_history = torch.zeros(num_envs, H, 15, device=device) if method == "rma2" else None

    # ── Rollout ───────────────────────────────────────────────────────────
    obs, _     = env.get_observations()
    obs        = obs.to(device)
    ep_rewards: list[float] = []
    ep_lengths: list[float] = []
    cur_reward = torch.zeros(num_envs, device=device)
    cur_length = torch.zeros(num_envs, device=device)

    print(f"[eval] collecting {num_eps} episodes ...")
    with torch.inference_mode():
        while len(ep_rewards) < num_eps and simulation_app.is_running():

            if method == "rma2":
                ot = obs[:, :15]
                obs_history = torch.roll(obs_history, -1, dims=1)
                obs_history[:, -1, :] = ot
                z_hat   = phi(obs_history)
                actions = policy.actor(torch.cat([ot, z_hat], dim=-1))
            else:
                actions = policy.act_inference(obs)

            obs, rewards, dones, _ = env.step(actions)
            obs     = obs.to(device)
            rewards = rewards.to(device)
            dones   = dones.to(device)

            cur_reward += rewards
            cur_length += 1

            done_ids = (dones > 0).nonzero(as_tuple=False)[:, 0]
            for idx in done_ids.tolist():
                ep_rewards.append(cur_reward[idx].item())
                ep_lengths.append(cur_length[idx].item())
                cur_reward[idx] = 0.0
                cur_length[idx] = 0.0
                if method == "rma2":
                    obs_history[idx] = 0.0   # clear history on episode reset
                if len(ep_rewards) % 10 == 0:
                    print(f"  episodes: {len(ep_rewards)}/{num_eps}", end="\r")

    env.close()
    print()

    # ── Stats ─────────────────────────────────────────────────────────────
    rewards_used = ep_rewards[:num_eps]
    lengths_used = ep_lengths[:num_eps]
    mean_rew = statistics.mean(rewards_used)
    std_rew  = statistics.stdev(rewards_used) if len(rewards_used) > 1 else 0.0
    mean_len = statistics.mean(lengths_used)
    std_len  = statistics.stdev(lengths_used) if len(lengths_used) > 1 else 0.0
    success  = sum(l >= success_thresh for l in lengths_used) / len(lengths_used) * 100.0

    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  OOD Evaluation Results")
    print(f"{sep}")
    print(f"  Method      : {method.upper()}  priv={priv_mode}  l={latent_dim}")
    print(f"  DR scale    : {dr_scale:.1f}x")
    print(f"  Episodes    : {len(rewards_used)}")
    print(f"{sep}")
    print(f"  Mean reward : {mean_rew:+.2f} ± {std_rew:.2f}")
    print(f"  Mean length : {mean_len:.1f} ± {std_len:.1f} steps")
    print(f"  Success rate: {success:.1f}%  (> {success_thresh} steps)")
    print(f"{sep}\n")

    # ── Save to CSV ───────────────────────────────────────────────────────
    if args_cli.results_file:
        rpath      = args_cli.results_file
        os.makedirs(os.path.dirname(os.path.abspath(rpath)), exist_ok=True)
        new_file   = not os.path.exists(rpath)
        with open(rpath, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["method", "priv_mode", "latent_dim", "dr_scale",
                            "mean_reward", "std_reward", "mean_length", "std_length",
                            "success_rate", "episodes", "checkpoint", "timestamp"])
            w.writerow([method.upper(), priv_mode, latent_dim, f"{dr_scale:.1f}",
                        f"{mean_rew:.4f}", f"{std_rew:.4f}",
                        f"{mean_len:.1f}", f"{std_len:.1f}",
                        f"{success:.1f}", len(rewards_used),
                        os.path.basename(args_cli.checkpoint),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        print(f"[eval] saved → {rpath}")


if __name__ == "__main__":
    main()
    simulation_app.close()

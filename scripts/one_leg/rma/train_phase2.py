"""
One-Leg RMA Phase-2 training script (paper Section 3.1).

Trains adaptation module ϕ (1D-CNN) via supervised learning:
  ϕ(ot-H:t) → ẑt  such that  ||ẑt - µ(xt)||² is minimised.

The Phase-1 RMAActorCritic checkpoint is loaded; encoder µ is FROZEN.
ϕ collects rollouts and regresses its output against µ(xt) on-policy.

Usage:
    python scripts/one_leg/rma/train_phase2.py \
        --checkpoint logs/one_leg/rma/<run>/model_final.pt \
        --num_envs 1024 --max_iterations 1000 --headless
"""
from __future__ import annotations
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",     type=str,   required=True)
parser.add_argument("--num_envs",       type=int,   default=1024)
parser.add_argument("--max_iterations", type=int,   default=1000)
parser.add_argument("--history_len",    type=int,   default=50,
                    help="Obs-history steps fed to ϕ (paper uses 50 × 10ms = 0.5s)")
parser.add_argument("--lr",             type=float, default=5e-4)
parser.add_argument("--batch_size",     type=int,   default=4096)
parser.add_argument("--experiment",     type=str,   default=None)
parser.add_argument("--priv_mode",      type=str,   default="FULL",
                    choices=["FULL", "INT", "EXT"],
                    help="Must match the priv_mode used in Phase 1")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, torch, gymnasium as gym
from datetime import datetime
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper              # type: ignore

# Inject RMAActorCritic so checkpoint loads correctly
from cts_rma_project.tasks.one_leg.rma.rma_policy import RMAActorCritic
import rsl_rl.runners.on_policy_runner as _runner_mod
_runner_mod.RMAActorCritic = RMAActorCritic

import cts_rma_project.tasks  # noqa
from cts_rma_project.tasks.one_leg.rma.one_leg_rma_env_cfg import OneLegRMAEnvCfg


# ── 1D-CNN Adaptation Module ϕ ────────────────────────────────────────────────
class AdaptationModule(torch.nn.Module):
    """ϕ: history of ot (T×15) → ẑ (8) — matches µ output."""

    def __init__(self, obs_dim: int = 15, latent_dim: int = 8, history_len: int = 50):
        super().__init__()
        self.history_len = history_len
        # 1D-CNN: input (batch, obs_dim, history_len)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(obs_dim, 32,  kernel_size=8, stride=4),  torch.nn.ELU(),
            torch.nn.Conv1d(32,      64,  kernel_size=5, stride=2),  torch.nn.ELU(),
            torch.nn.Conv1d(64,      128, kernel_size=3, stride=1),  torch.nn.ELU(),
        )
        # Compute flattened size
        dummy = torch.zeros(1, obs_dim, history_len)
        cnn_out = self.cnn(dummy).shape[1:]
        flat_dim = 1
        for d in cnn_out:
            flat_dim *= d
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(flat_dim, 256), torch.nn.ELU(),
            torch.nn.Linear(256, latent_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: (N, T, obs_dim) → ẑ: (N, latent_dim)."""
        x = history.transpose(1, 2)   # (N, obs_dim, T)
        x = self.cnn(x)
        return self.fc(x)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device    = args_cli.device or "cuda"
    H         = args_cli.history_len
    priv_mode = args_cli.priv_mode.upper()

    # ── Auto-detect latent_dim from checkpoint ────────────────────────────
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    sd   = ckpt.get("model_state_dict", ckpt)
    enc_w_keys = sorted([k for k in sd if k.startswith("encoder.") and "weight" in k])
    latent_dim = sd[enc_w_keys[-1]].shape[0]
    print(f"[RMA Phase 2] Detected latent_dim={latent_dim}, priv_mode={priv_mode}")

    experiment = args_cli.experiment or f"one_leg_rma_p2_{priv_mode.lower()}_l{latent_dim}"

    # ── Environment ──────────────────────────────────────────────────────
    env_cfg = OneLegRMAEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device
    env_cfg.priv_mode      = priv_mode
    env = gym.make("OneLeg-RMA-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── Load frozen Phase-1 actor (encoder µ) ────────────────────────────
    ac = RMAActorCritic(
        num_actor_obs=48, num_critic_obs=48, num_actions=3,
        latent_dim=latent_dim,
    ).to(device)
    ac.load_state_dict(sd)
    ac.eval()
    for p in ac.parameters():
        p.requires_grad_(False)

    # ── Adaptation module ϕ ───────────────────────────────────────────────
    phi   = AdaptationModule(obs_dim=15, latent_dim=latent_dim, history_len=H).to(device)
    optim = torch.optim.Adam(phi.parameters(), lr=args_cli.lr)

    log_dir = os.path.join(
        "logs", "one_leg", "rma",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{experiment}"
    )
    os.makedirs(log_dir, exist_ok=True)

    N        = env.num_envs
    obs_buf  = torch.zeros(N, H, 15, device=device)  # rolling history

    obs, extras = env.get_observations()
    obs = obs.to(device)

    print(f"[RMA Phase 2] Training ϕ for {args_cli.max_iterations} iterations ...")

    for it in range(args_cli.max_iterations):
        # ── Collect one step with Phase-1 actor ──────────────────────────
        with torch.no_grad():
            actions = ac.act_inference(obs)
        obs_new, _, dones, extras = env.step(actions)
        obs_new = obs_new.to(device)

        # Update obs history: ot = obs[:, :15]
        ot = obs[:, :15]
        obs_buf = torch.roll(obs_buf, -1, dims=1)
        obs_buf[:, -1, :] = ot

        # ── Supervised regression: ẑ vs µ(xt) ───────────────────────────
        if it >= H:   # wait until history is populated
            with torch.no_grad():
                zt_target = ac.encode_latent(obs)    # µ(xt)  (N, 8)

            z_hat = phi(obs_buf)                     # ϕ(history)  (N, 8)
            loss  = torch.nn.functional.mse_loss(z_hat, zt_target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if it % 50 == 0:
                print(f"  [phase2] iter={it:5d}  L_rec={loss.item():.6f}")

        obs = obs_new

    # ── Save adaptation module ────────────────────────────────────────────
    save_path = os.path.join(log_dir, "adaptation_module.pt")
    torch.save(phi.state_dict(), save_path)
    print(f"[INFO] Phase-2 complete. ϕ saved: {save_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

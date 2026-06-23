# tasks/rma/rma_runner.py
"""
RMA Phase 2 runner — supervised adaptation module training.

Phase 1 uses standard OnPolicyRunner (see scripts/rma/train_phase1.py).
Phase 2 trains AdaptationModule φ to regress z_t = μ(e_t) from obs/action history.
"""

import csv
import os
import torch
import torch.nn as nn
from torch.optim import Adam

from .rma_network import RMAActorCritic, AdaptationModule


class RMAPhase2Runner:
    """
    Phase 2: Train adaptation module φ via supervised learning.
    Collects on-policy data using current ẑ (or random ẑ at start)
    and trains φ to regress z_t = μ(e_t).
    """

    def __init__(self, env, rma_model: RMAActorCritic,
                 history_len: int = 50,
                 num_iterations: int = 1000,
                 batch_size: int = 80000,
                 learning_rate: float = 5e-4,
                 log_dir: str = "logs/rma/phase2",
                 device: str = "cuda",
                 state_dim: int = 37,
                 latent_dim: int = 8,
                 priv_dim: int = 26):
        self.env          = env
        self.rma          = rma_model.to(device)
        self.device       = device
        self.history_len  = history_len
        self.num_iters    = num_iterations
        self.batch_size   = batch_size
        self.log_dir      = log_dir
        self.state_dim    = state_dim
        self.latent_dim   = latent_dim
        self.priv_dim     = priv_dim   # x_t size in the critic group: FULL=26/INT=16/EXT=10

        self.adapt_module = AdaptationModule(
            state_dim=state_dim, action_dim=12, embed_dim=32, latent_dim=latent_dim,
            history_len=history_len
        ).to(device)

        self.optimizer = Adam(self.adapt_module.parameters(), lr=learning_rate)
        self.loss_fn   = nn.MSELoss()

        # First `stat_warmup` iters collect teacher latents WITHOUT updating φ,
        # to estimate per-dim mean/std of z_true. φ is then trained to regress
        # the *standardised* target (well-conditioned MSE) and de-normalises at
        # deployment via the z_mean/z_std buffers. Without this the unregularised
        # env_encoder output (|z|~48) makes MSE collapse to a constant.
        self.stat_warmup = 10
        self._z_accum    = []

        N = env.num_envs
        self.state_hist  = torch.zeros(N, history_len, state_dim, device=device)
        self.action_hist = torch.zeros(N, history_len, 12, device=device)

        os.makedirs(log_dir, exist_ok=True)

    def _roll_history(self, new_state: torch.Tensor, new_action: torch.Tensor):
        """Shift history buffer and insert new observation."""
        self.state_hist  = torch.roll(self.state_hist,  -1, dims=1)
        self.action_hist = torch.roll(self.action_hist, -1, dims=1)
        self.state_hist[:, -1, :]  = new_state
        self.action_hist[:, -1, :] = new_action

    def collect_and_train(self):
        """Main Phase 2 loop."""
        self.rma.eval()  # freeze base policy and env encoder
        for param in self.rma.parameters():
            param.requires_grad_(False)

        csv_path = os.path.join(self.log_dir, "phase2_loss.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["iteration", "mse_loss"])

        steps_per_iter = self.batch_size // self.env.num_envs

        # One-time env reset + history warmup. φ sees a `history_len`-step buffer
        # of (obs, action). With batch_size/num_envs (e.g. 80000/4096 ≈ 19) <
        # history_len (50), if we reset envs every iteration and only collect 19
        # fresh steps, every training sample has ≥ 31 zero-padded entries.
        # φ then learns the trivial constant ẑ ≈ E[z]. Fix: reset once, prime
        # the buffer with `history_len` no-collect steps, and let the env run
        # continuously across iterations so non-terminating envs accumulate
        # full histories (matching the deployment distribution).
        obs_dict, _  = self.env.reset()
        x_t          = obs_dict["policy"].to(self.device)
        # critic group: [ot(37), xt(priv_dim)] — extract xt for teacher encoder
        xt           = obs_dict["critic"].to(self.device)[:, 37:37 + self.priv_dim]

        print(f"[Phase 2] Warming up history buffer ({self.history_len} steps)...")
        for _ in range(self.history_len):
            with torch.inference_mode():
                z_hat  = self.adapt_module(self.state_hist, self.action_hist)
                action = self.rma.policy(x_t, z_hat)
            obs_dict_next, _, terminated, truncated, _ = self.env.step(action)
            dones  = terminated | truncated
            next_x = obs_dict_next["policy"].to(self.device)
            xt     = obs_dict_next["critic"].to(self.device)[:, 37:37 + self.priv_dim]
            self._roll_history(x_t, action)
            if dones.any():
                self.state_hist[dones]  = 0.0
                self.action_hist[dones] = 0.0
            x_t = next_x

        for iteration in range(self.num_iters):
            # === Collect on-policy data using current ẑ ===
            collected_states  = []
            collected_actions = []
            collected_z_true  = []

            for _ in range(steps_per_iter):
                with torch.inference_mode():
                    z_hat  = self.adapt_module(self.state_hist, self.action_hist)
                    action = self.rma.policy(x_t, z_hat)   # BasePolicy: (ot, z) → a
                    z_true = self.rma.env_encoder(xt)

                collected_states.append(self.state_hist.clone())
                collected_actions.append(self.action_hist.clone())
                collected_z_true.append(z_true)

                obs_dict_next, _, terminated, truncated, _ = self.env.step(action)
                dones  = terminated | truncated
                next_x = obs_dict_next["policy"].to(self.device)
                xt     = obs_dict_next["critic"].to(self.device)[:, 37:37 + self.priv_dim]

                self._roll_history(x_t, action)

                if dones.any():
                    self.state_hist[dones]  = 0.0
                    self.action_hist[dones] = 0.0

                x_t = next_x

            # === Supervised update ===
            states  = torch.stack(collected_states,  dim=1).flatten(0, 1)   # [steps*N, H, 37]
            actions = torch.stack(collected_actions, dim=1).flatten(0, 1)   # [steps*N, H, 12]
            z_truth = torch.cat(collected_z_true, dim=0)                    # [steps*N, 8]

            # === Stat-warmup: estimate teacher-latent mean/std, no φ update ===
            if iteration < self.stat_warmup:
                self._z_accum.append(z_truth.detach())
                if iteration == self.stat_warmup - 1:
                    allz = torch.cat(self._z_accum, dim=0)
                    self.adapt_module.z_mean.copy_(allz.mean(0))
                    self.adapt_module.z_std.copy_(allz.std(0).clamp_min(1e-6))
                    self._z_accum = []
                    print(f"[Phase 2] z-stats set after {self.stat_warmup} iters | "
                          f"|mean|={self.adapt_module.z_mean.norm():.2f} "
                          f"|std|={self.adapt_module.z_std.norm():.2f}")
                    # Log pre-training loss (untrained φ) as iteration 0
                    z_tgt_init = (z_truth - self.adapt_module.z_mean) / self.adapt_module.z_std
                    idx0   = torch.randperm(states.shape[0], device=self.device)
                    mb0    = states.shape[0] // 4
                    init_losses = []
                    with torch.no_grad():
                        for i in range(4):
                            mb_idx = idx0[i*mb0:(i+1)*mb0]
                            z_pred = self.adapt_module.core(states[mb_idx], actions[mb_idx])
                            init_losses.append(self.loss_fn(z_pred, z_tgt_init[mb_idx]).item())
                    csv_writer.writerow([0, f"{sum(init_losses)/len(init_losses):.6f}"])
                    csv_file.flush()
                continue

            # Standardised regression target (de-normalised at deployment)
            z_tgt = (z_truth - self.adapt_module.z_mean) / self.adapt_module.z_std

            # Mini-batch update
            idx      = torch.randperm(states.shape[0], device=self.device)
            mb_size  = states.shape[0] // 4
            mb_losses = []

            for i in range(4):
                mb_idx = idx[i*mb_size:(i+1)*mb_size]
                z_pred = self.adapt_module.core(states[mb_idx], actions[mb_idx])
                loss   = self.loss_fn(z_pred, z_tgt[mb_idx])

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.adapt_module.parameters(), 1.0)
                self.optimizer.step()
                mb_losses.append(loss.item())

            mean_loss = sum(mb_losses) / len(mb_losses)
            csv_writer.writerow([iteration, f"{mean_loss:.6f}"])
            csv_file.flush()

            if iteration % 50 == 0:
                print(f"[Phase 2 Iter {iteration:4d}] MSE loss = {mean_loss:.6f}")

            if iteration % 200 == 0:
                torch.save(self.adapt_module.state_dict(),
                           os.path.join(self.log_dir, f"adapt_module_{iteration}.pt"))

        print("[Phase 2] Training complete.")
        csv_file.close()
        torch.save(self.adapt_module.state_dict(),
                   os.path.join(self.log_dir, "adapt_module_final.pt"))
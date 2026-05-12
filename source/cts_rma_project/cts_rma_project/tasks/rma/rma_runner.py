# tasks/rma/rma_runner.py
"""
RMA Phase 2 runner — supervised adaptation module training.

Phase 1 uses standard OnPolicyRunner (see scripts/rma/train_phase1.py).
Phase 2 trains AdaptationModule φ to regress z_t = μ(e_t) from obs/action history.
"""

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

        for iteration in range(self.num_iters):
            # === Collect on-policy data using current ẑ ===
            collected_states  = []
            collected_actions = []
            collected_z_true  = []

            obs_dict, _  = self.env.reset()
            x_t          = obs_dict["policy"].to(self.device)
            # critic group: [ot(37), xt(priv_dim)] — extract xt for teacher encoder
            xt           = obs_dict["critic"].to(self.device)[:, 37:37 + self.priv_dim]

            steps_per_iter = self.batch_size // self.env.num_envs

            for step in range(steps_per_iter):
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
            states  = torch.stack(collected_states,  dim=1).flatten(0, 1)   # [steps*N, H, 30]
            actions = torch.stack(collected_actions, dim=1).flatten(0, 1)   # [steps*N, H, 12]
            z_truth = torch.cat(collected_z_true, dim=0)                    # [steps*N, 8]

            # Mini-batch update
            idx      = torch.randperm(states.shape[0], device=self.device)
            mb_size  = states.shape[0] // 4

            for i in range(4):
                mb_idx = idx[i*mb_size:(i+1)*mb_size]
                z_pred = self.adapt_module(states[mb_idx], actions[mb_idx])
                loss   = self.loss_fn(z_pred, z_truth[mb_idx])

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.adapt_module.parameters(), 1.0)
                self.optimizer.step()

            if iteration % 50 == 0:
                print(f"[Phase 2 Iter {iteration:4d}] MSE loss = {loss.item():.6f}")

            if iteration % 200 == 0:
                torch.save(self.adapt_module.state_dict(),
                           os.path.join(self.log_dir, f"adapt_module_{iteration}.pt"))

        print("[Phase 2] Training complete.")
        torch.save(self.adapt_module.state_dict(),
                   os.path.join(self.log_dir, "adapt_module_final.pt"))
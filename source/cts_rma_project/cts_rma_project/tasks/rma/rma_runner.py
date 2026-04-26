# tasks/rma/rma_runner.py
"""
Two-phase RMA training runner.

Phase 1: PPO with ground truth extrinsics z_t = μ(e_t)
Phase 2: Supervised learning for adaptation module φ
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner  # type: ignore

from .rma_network import RMAActorCritic, AdaptationModule


class RMAPhase1Runner(RslRlOnPolicyRunner):
    """
    Phase 1: extend RSL-RL runner to pass privileged_obs to actor.
    The env must provide obs_dict["privileged"] = e_t.
    """

    def _collect_rollout(self):
        """Override to pass privileged obs to actor."""
        for _ in range(self.num_steps_per_env):
            obs_dict = self.env.get_observations()
            obs        = obs_dict["policy"]       # x_t  [N, 30]
            priv_obs   = obs_dict["privileged"]   # e_t  [N, 17]

            with torch.inference_mode():
                actions = self.alg.actor_critic.act(obs, privileged_obs=priv_obs)

            obs_dict_next, rewards, dones, infos = self.env.step(actions)
            self.alg.process_env_step(rewards, dones, infos)
            obs = obs_dict_next["policy"]

        last_obs      = obs
        last_priv_obs = self.env.get_observations()["privileged"]
        last_values   = self.alg.actor_critic.evaluate(last_obs, last_priv_obs)[2]
        self.alg.compute_returns(last_values)


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
                 device: str = "cuda"):
        self.env          = env
        self.rma          = rma_model.to(device)
        self.device       = device
        self.history_len  = history_len
        self.num_iters    = num_iterations
        self.batch_size   = batch_size
        self.log_dir      = log_dir

        # Build adaptation module
        self.adapt_module = AdaptationModule(
            state_dim=30, action_dim=12, embed_dim=32, latent_dim=8,
            history_len=history_len
        ).to(device)

        self.optimizer = Adam(self.adapt_module.parameters(), lr=learning_rate)
        self.loss_fn   = nn.MSELoss()

        # Ring buffers for history  [N, history_len, dim]
        N = env.num_envs
        self.state_hist  = torch.zeros(N, history_len, 30, device=device)
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

            obs_dict   = self.env.reset()
            x_t        = obs_dict["policy"]
            priv_obs   = obs_dict["privileged"]
            prev_action = torch.zeros(self.env.num_envs, 12, device=self.device)

            steps_per_iter = self.batch_size // self.env.num_envs

            for step in range(steps_per_iter):
                # Estimate ẑ using current adaptation module
                with torch.inference_mode():
                    z_hat  = self.adapt_module(self.state_hist, self.action_hist)
                    action = self.rma.policy(x_t, prev_action, z_hat)

                # Ground truth z from privileged encoder
                with torch.inference_mode():
                    z_true = self.rma.env_encoder(priv_obs)

                collected_states.append(self.state_hist.clone())
                collected_actions.append(self.action_hist.clone())
                collected_z_true.append(z_true)

                obs_dict_next, _, dones, _ = self.env.step(action)
                next_x = obs_dict_next["policy"]

                # Update history
                self._roll_history(x_t, action)

                # Reset history on episode end
                if dones.any():
                    self.state_hist[dones]  = 0.0
                    self.action_hist[dones] = 0.0

                x_t        = next_x
                priv_obs   = obs_dict_next["privileged"]
                prev_action = action.detach()

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
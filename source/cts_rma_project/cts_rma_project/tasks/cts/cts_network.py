# tasks/cts/cts_network.py
"""
CTS Actor-Critic Network.

Architecture:
  Actor:  MLP  37 → 256 → 128 → 64 → 12  (ELU)
  Critic: MLP  37 → 256 → 128 → 64 → 1   (ELU)
  log_std: learnable parameter, shape (12,)

No privileged encoder and no adaptation module — the curriculum does the
heavy lifting instead of architecture complexity.

RSL-RL OnPolicyRunner interface:
  act(obs)              → actions (sampled)
  act_inference(obs)    → actions (deterministic mean)
  evaluate(critic_obs)  → (actions_mean, log_std, value, entropy)
  get_actions_log_prob(actions) → log probs
  is_recurrent = False
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(dims: list[int], activation: type[nn.Module] = nn.ELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class CTSActorCritic(nn.Module):
    """Standard actor-critic for CTS. Compatible with RSL-RL OnPolicyRunner."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: tuple[int, ...] = (256, 128, 64),
        critic_hidden_dims: tuple[int, ...] = (256, 128, 64),
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = num_actions

        actor_dims  = [num_actor_obs,  *actor_hidden_dims,  num_actions]
        critic_dims = [num_critic_obs, *critic_hidden_dims, 1]

        self.actor  = _mlp(actor_dims)
        self.critic = _mlp(critic_dims)
        self.log_std = nn.Parameter(
            torch.full((num_actions,), fill_value=init_noise_std).log()
        )

        # buffers written by act() and read by the runner / algorithm
        self.actions_mean: torch.Tensor | None     = None
        self.actions_log_prob: torch.Tensor | None = None
        self.value: torch.Tensor | None            = None
        self._dist: Normal | None                  = None

    def reset(self, dones=None):
        pass

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------
    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample action from the stochastic policy and cache statistics."""
        self.actions_mean = self.actor(observations)
        std = self.log_std.exp().expand_as(self.actions_mean)
        self._dist = Normal(self.actions_mean, std)
        actions = self._dist.sample()
        self.actions_log_prob = self._dist.log_prob(actions).sum(-1)
        self.value = self.critic(observations).squeeze(-1)
        return actions

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(actions).sum(-1)

    def evaluate(
        self,
        critic_observations: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        value   = self.critic(critic_observations).squeeze(-1)
        entropy = self._dist.entropy().sum(-1)
        return self.actions_mean, self.log_std, value, entropy

    # ------------------------------------------------------------------
    # Inference interface
    # ------------------------------------------------------------------
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic (mean) action for deployment / evaluation."""
        return self.actor(observations)

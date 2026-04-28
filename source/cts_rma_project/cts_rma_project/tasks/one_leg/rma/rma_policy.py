# tasks/one_leg/rma/rma_policy.py
"""
RMAActorCritic — Phase-1 network with explicit environment encoder µ.

Architecture (paper Section 3.1):
  Actor:
    encoder µ:  xt(24) → [256 → 128] → zt(8)
    policy  π:  [ot(15), zt(8)] = 23D → [256,128,64] → actions(3)
  Critic:
    V:          [ot, xt] = 39D → [256,128,64] → value(1)

Usage with RSL-RL:
    In the training script, inject into the runner's namespace before
    instantiating OnPolicyRunner:

        import rsl_rl.runners.on_policy_runner as _runner_module
        from cts_rma_project.tasks.one_leg.rma.rma_policy import RMAActorCritic
        _runner_module.RMAActorCritic = RMAActorCritic

    Then set in the agents PPO config:
        policy = RslRlPpoActorCriticCfg(class_name="RMAActorCritic", ...)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class RMAActorCritic(nn.Module):
    """Actor-critic with explicit environment encoder µ (paper Section 3.1).

    Phase 1: train with oracle xt — encoder is supervised by PPO gradient.
    Phase 2: freeze encoder; train 1D-CNN adaptation module ϕ to match µ(xt).
    """

    is_recurrent = False

    OBS_DIM    = 15   # proprioceptive obs ot
    PRIV_DIM   = 33   # privileged obs xt = [External(13), Internal(20)]
    LATENT_DIM = 8    # default encoder output zt (overridable via latent_dim kwarg)

    def __init__(
        self,
        num_actor_obs: int,           # 48 (OBS_DIM + PRIV_DIM)
        num_critic_obs: int,          # 48
        num_actions: int,             # 3
        actor_hidden_dims: list = None,
        critic_hidden_dims: list = None,
        encoder_hidden_dims: list = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        latent_dim: int = None,       # overrides class LATENT_DIM; use 8/16/32
        **kwargs,
    ):
        if kwargs:
            print(f"RMAActorCritic: ignoring unexpected kwargs {list(kwargs)}")
        super().__init__()

        lat = latent_dim if latent_dim is not None else self.LATENT_DIM
        self._latent_dim = lat

        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128, 64]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 128, 64]
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [256, 128]

        act_fn = resolve_nn_activation(activation)

        # ── Encoder µ: xt(33) → zt(lat) ──────────────────────────────────
        enc_layers, in_d = [], self.PRIV_DIM
        for h in encoder_hidden_dims:
            enc_layers += [nn.Linear(in_d, h), act_fn]
            in_d = h
        enc_last = nn.Linear(in_d, lat)
        nn.init.orthogonal_(enc_last.weight, gain=0.01)
        nn.init.constant_(enc_last.bias, 0.0)
        enc_layers.append(enc_last)
        self.encoder = nn.Sequential(*enc_layers)

        # ── Policy π: [ot, zt](15+lat) → actions ─────────────────────────
        pol_layers, in_d = [], self.OBS_DIM + lat
        for h in actor_hidden_dims:
            pol_layers += [nn.Linear(in_d, h), act_fn]
            in_d = h
        pol_layers.append(nn.Linear(in_d, num_actions))
        self.actor = nn.Sequential(*pol_layers)

        # ── Critic V: [ot, xt](39) → value ───────────────────────────────
        crit_layers, in_d = [], num_critic_obs
        for h in critic_hidden_dims:
            crit_layers += [nn.Linear(in_d, h), act_fn]
            in_d = h
        crit_layers.append(nn.Linear(in_d, 1))
        self.critic = nn.Sequential(*crit_layers)

        # ── Action noise ──────────────────────────────────────────────────
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

    # ── Core forward helpers ──────────────────────────────────────────────
    def encode_latent(self, observations: torch.Tensor) -> torch.Tensor:
        """Return zt = µ(xt); used externally for Phase-2 supervision."""
        return self.encoder(observations[:, self.OBS_DIM:])

    def _mean_actions(self, observations: torch.Tensor) -> torch.Tensor:
        ot = observations[:, :self.OBS_DIM]
        zt = self.encoder(observations[:, self.OBS_DIM:])
        return self.actor(torch.cat([ot, zt], dim=-1))

    # ── RSL-RL interface ──────────────────────────────────────────────────
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def update_distribution(self, observations: torch.Tensor):
        mean = self._mean_actions(observations)
        std  = self.std.clamp(min=1e-6)
        self.distribution = Normal(mean, std.expand_as(mean))

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self._mean_actions(observations)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

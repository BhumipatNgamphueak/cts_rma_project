# tasks/rma/rma_network.py
"""
RMA Network for GO2 (Phase 1 + Phase 2).

Phase 1 — teacher training
  env sees  policy obs = [o_t(37) + x_t(26)] = 63-D  (teacher gets full info)
  EnvFactorEncoder μ : x_t(26)  → z(latent_dim)
  BasePolicy π       : [o_t(37) + z] → a_t(12)
  ValueFunction  V   : [o_t(37) + z] → scalar

Phase 2 — adaptation module
  AdaptationModule φ : history of (o_t, a_t) → ẑ
  Actor at deployment: [o_t(37) + ẑ] → a_t   (same π, no x_t)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

_OT_DIM = 37   # proprioceptive obs for GO2


class EnvFactorEncoder(nn.Module):
    """μ: x_t(env_dim) → z(latent_dim)"""
    def __init__(self, env_dim: int = 26, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env_dim, 256), nn.ELU(),
            nn.Linear(256, 128),     nn.ELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


class BasePolicy(nn.Module):
    """π: (o_t, z) → a_t  — no prev_action to stay compatible with PPO mini-batch updates."""
    def __init__(self, action_dim: int = 12, latent_dim: int = 8):
        super().__init__()
        in_dim = _OT_DIM + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ELU(),
            nn.Linear(256, 128),    nn.ELU(),
            nn.Linear(128, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, ot: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ot, z], dim=-1))

    def get_dist(self, ot: torch.Tensor, z: torch.Tensor) -> Normal:
        mean = self.forward(ot, z)
        std  = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)


class ValueFunction(nn.Module):
    """V: (o_t, z) → scalar"""
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_OT_DIM + latent_dim, 256), nn.ELU(),
            nn.Linear(256, 128),                  nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, ot: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ot, z], dim=-1))


class AdaptationModule(nn.Module):
    """φ: history of (o_t, a_t) → ẑ"""
    def __init__(self, state_dim: int = 37, action_dim: int = 12,
                 embed_dim: int = 32, latent_dim: int = 8, history_len: int = 50):
        super().__init__()
        self.history_len = history_len
        self.embed = nn.Sequential(
            nn.Linear(state_dim + action_dim, embed_dim), nn.ELU(),
            nn.Linear(embed_dim, embed_dim),               nn.ELU(),
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, 32, kernel_size=8, stride=4), nn.ELU(),
            nn.Conv1d(32,        32, kernel_size=5, stride=1), nn.ELU(),
            nn.Conv1d(32,        32, kernel_size=5, stride=1), nn.ELU(),
        )
        dummy        = torch.zeros(1, embed_dim, history_len)
        cnn_out_size = self.cnn(dummy).flatten(1).shape[1]
        self.proj    = nn.Linear(cnn_out_size, latent_dim)

    def forward(self, x_hist: torch.Tensor, a_hist: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_hist.shape
        inp     = torch.cat([x_hist, a_hist], dim=-1).view(B * T, -1)
        emb     = self.embed(inp).view(B, T, -1).permute(0, 2, 1)
        return self.proj(self.cnn(emb).flatten(1))


class RMAActorCritic(nn.Module):
    """
    RMA Phase 1 actor-critic — RSL-RL compatible interface.

    Teacher mode  (num_actor_obs == 63):
      act(obs_63d) splits into o_t(37) + x_t(26),
      encodes x_t → z via EnvFactorEncoder.
      Used during Phase 1 training.

    Deployment mode  (num_actor_obs == 37):
      act_inference(obs_37d) uses z = 0 (or z_override from adaptation module).
    """
    is_recurrent = False

    def __init__(self, num_actor_obs: int, num_critic_obs: int,
                 num_actions: int, env_factor_dim: int = 26,
                 latent_dim: int = 8, **kwargs):
        super().__init__()
        self.latent_dim      = latent_dim
        self.action_dim      = num_actions
        self._env_factor_dim = env_factor_dim
        # teacher mode when policy obs = [ot + xt]
        self._teacher_mode   = (num_actor_obs == _OT_DIM + env_factor_dim)

        self.env_encoder = EnvFactorEncoder(env_factor_dim, latent_dim)
        self.policy      = BasePolicy(num_actions, latent_dim)
        self.value_fn    = ValueFunction(latent_dim)

        # RSL-RL interface attributes (set by act())
        self.action_mean = None
        self.action_std  = None
        self.entropy     = None
        self._dist       = None

    def reset(self, dones=None):
        pass  # no recurrent state to reset

    def _split_obs(self, obs: torch.Tensor):
        """Return (ot, xt_or_None) depending on obs dimension."""
        if self._teacher_mode or obs.shape[-1] == _OT_DIM + self._env_factor_dim:
            return obs[:, :_OT_DIM], obs[:, _OT_DIM:_OT_DIM + self._env_factor_dim]
        return obs[:, :_OT_DIM], None

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        ot, xt = self._split_obs(observations)
        B, device = ot.shape[0], ot.device
        z = self.env_encoder(xt) if xt is not None else \
            torch.zeros(B, self.latent_dim, device=device)

        self._dist       = self.policy.get_dist(ot, z)
        actions          = self._dist.sample()
        self.action_mean = self._dist.loc
        self.action_std  = self._dist.scale
        self.entropy     = self._dist.entropy().sum(-1)
        return actions

    def act_inference(self, observations: torch.Tensor,
                      z_override: torch.Tensor = None) -> torch.Tensor:
        ot = observations[:, :_OT_DIM]
        B, device = ot.shape[0], ot.device
        z = z_override if z_override is not None else \
            torch.zeros(B, self.latent_dim, device=device)
        return self.policy(ot, z)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(actions).sum(-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns value tensor [N,1] — RSL-RL PPO stores values with shape [T,N,1]."""
        ot = critic_observations[:, :_OT_DIM]
        xt = critic_observations[:, _OT_DIM:_OT_DIM + self._env_factor_dim]
        z  = self.env_encoder(xt)
        return self.value_fn(ot, z)

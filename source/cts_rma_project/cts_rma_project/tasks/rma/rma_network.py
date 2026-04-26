# tasks/rma/rma_network.py
"""
RMA Network Architecture (from paper Section IV-B):

Base Policy π:
  MLP 3-layer, hidden=128
  Input: x_t(30) + a_{t-1}(12) + z_t(8) = 50
  Output: a_t (12)

Env Factor Encoder μ:
  MLP 3-layer, hidden=[256, 128]
  Input:  e_t (17)
  Output: z_t (8)

Adaptation Module φ:
  Embed(state+action → 32) then 1D-CNN over 50 timesteps
  CNN layers: [32,32,8,4], [32,32,5,1], [32,32,5,1]
  Output: ẑ_t (8)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class EnvFactorEncoder(nn.Module):
    """μ: e_t(17) → z_t(8)"""
    def __init__(self, env_dim: int = 17, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env_dim, 256), nn.ELU(),
            nn.Linear(256, 128),    nn.ELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


class BasePolicy(nn.Module):
    """π: (x_t, a_{t-1}, z_t) → a_t"""
    def __init__(self, state_dim: int = 30, action_dim: int = 12, latent_dim: int = 8):
        super().__init__()
        in_dim = state_dim + action_dim + latent_dim  # 50
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ELU(),
            nn.Linear(128, 128),    nn.ELU(),
            nn.Linear(128, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor, a_prev: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, a_prev, z], dim=-1)
        return self.net(inp)

    def get_dist(self, x, a_prev, z):
        mean = self.forward(x, a_prev, z)
        std  = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)


class ValueFunction(nn.Module):
    """V: (x_t, z_t) → scalar"""
    def __init__(self, state_dim: int = 30, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z], dim=-1))


class AdaptationModule(nn.Module):
    """
    φ: history of (x, a) → ẑ_t
    Uses 2-layer MLP embed then 1D-CNN across time.
    History length k=50 (0.5s at 100Hz)
    """
    def __init__(self, state_dim: int = 30, action_dim: int = 12,
                 embed_dim: int = 32, latent_dim: int = 8, history_len: int = 50):
        super().__init__()
        self.history_len = history_len

        # Embed each (state, action) pair
        self.embed = nn.Sequential(
            nn.Linear(state_dim + action_dim, embed_dim), nn.ELU(),
            nn.Linear(embed_dim, embed_dim),              nn.ELU(),
        )

        # 1D-CNN across time dimension
        # Input: [batch, embed_dim=32, time=50]
        self.cnn = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=8, stride=4), nn.ELU(),  # → [B, 32, 11]
            nn.Conv1d(32, 32, kernel_size=5, stride=1), nn.ELU(),  # → [B, 32, 7]
            nn.Conv1d(32, 32, kernel_size=5, stride=1), nn.ELU(),  # → [B, 32, 3]
        )

        # Project CNN output to latent
        cnn_out_size = self._get_cnn_out_size(embed_dim, history_len)
        self.proj = nn.Linear(cnn_out_size, latent_dim)

    def _get_cnn_out_size(self, embed_dim, history_len):
        dummy = torch.zeros(1, embed_dim, history_len)
        return self.cnn(dummy).flatten(1).shape[1]

    def forward(self, x_hist: torch.Tensor, a_hist: torch.Tensor) -> torch.Tensor:
        """
        x_hist: [B, history_len, state_dim]
        a_hist: [B, history_len, action_dim]
        returns: [B, latent_dim]
        """
        B, T, _ = x_hist.shape
        # Embed each timestep: [B*T, state+action] → [B*T, embed_dim]
        inp = torch.cat([x_hist, a_hist], dim=-1).view(B * T, -1)
        emb = self.embed(inp).view(B, T, -1)        # [B, T, embed_dim]
        emb = emb.permute(0, 2, 1)                  # [B, embed_dim, T]  for Conv1d
        cnn_out = self.cnn(emb).flatten(1)           # [B, cnn_out_size]
        return self.proj(cnn_out)                    # [B, latent_dim]


class RMAActorCritic(nn.Module):
    """
    Full RMA Phase 1 module.
    Compatible with RSL-RL ActorCritic interface.
    """
    is_recurrent = False

    def __init__(self, num_actor_obs: int, num_critic_obs: int,
                 num_actions: int, env_factor_dim: int = 17,
                 latent_dim: int = 8, **kwargs):
        super().__init__()
        self.latent_dim  = latent_dim
        self.action_dim  = num_actions
        self.state_dim   = num_actor_obs       # 30

        self.env_encoder = EnvFactorEncoder(env_factor_dim, latent_dim)
        self.policy      = BasePolicy(self.state_dim, num_actions, latent_dim)
        self.value_fn    = ValueFunction(self.state_dim, latent_dim)

        # Buffers filled by act()
        self.actions_mean     = None
        self.actions_log_prob = None
        self.value            = None
        self._z               = None
        self._dist            = None
        self._prev_action     = None

    def reset(self, dones=None):
        pass

    def act(self, observations: torch.Tensor, privileged_obs: torch.Tensor = None, **kwargs):
        """
        observations:    x_t  [B, 30]
        privileged_obs:  e_t  [B, 17]  (only available during Phase 1 training)
        """
        B = observations.shape[0]
        device = observations.device

        if self._prev_action is None:
            self._prev_action = torch.zeros(B, self.action_dim, device=device)

        if privileged_obs is not None:
            z = self.env_encoder(privileged_obs)
        else:
            z = torch.zeros(B, self.latent_dim, device=device)

        self._z    = z
        self._dist = self.policy.get_dist(observations, self._prev_action, z)
        actions    = self._dist.sample()

        self.actions_mean     = self._dist.loc
        self.actions_log_prob = self._dist.log_prob(actions).sum(-1)
        self.value            = self.value_fn(observations, z)
        self._prev_action     = actions.detach()

        return actions

    def act_inference(self, observations: torch.Tensor,
                      z_override: torch.Tensor = None) -> torch.Tensor:
        """Deterministic (mean) action. Used at deployment with ẑ from adaptation module."""
        B = observations.shape[0]
        device = observations.device
        if self._prev_action is None:
            self._prev_action = torch.zeros(B, self.action_dim, device=device)
        z = z_override if z_override is not None else torch.zeros(B, self.latent_dim, device=device)
        return self.policy(observations, self._prev_action, z)

    def get_actions_log_prob(self, actions):
        return self._dist.log_prob(actions).sum(-1)

    def evaluate(self, critic_observations, privileged_obs=None, **kwargs):
        if privileged_obs is not None:
            z = self.env_encoder(privileged_obs)
        else:
            z = self._z if self._z is not None else torch.zeros(
                critic_observations.shape[0], self.latent_dim, device=critic_observations.device)
        value   = self.value_fn(critic_observations, z)
        entropy = self._dist.entropy().sum(-1)
        return self.actions_mean, self.policy.log_std, value, entropy
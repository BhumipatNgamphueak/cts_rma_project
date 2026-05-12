# tasks/cts/cts_network.py
"""
CTSActorCritic — concurrent teacher-student network for GO2 walking.

Architecture:
  Teacher encoder E^t: xt(26) → Z        (MLP; updated by PPO + L_rec)
  Student encoder E^s: H×ot(37) → Z      (1D-CNN; updated by L_rec only)
  Shared actor  π:     [ot(37), z(Z)] → 12 actions
  Critic V:            [ot(37), zt(Z)] → value  (critic sees teacher-encoded xt)

Obs routing via obs[:, -1] flag:
  is_teacher=1 → E^t(xt) for z;  ot = obs[:, :37]
  is_teacher=0 → E^s(history) for z;  ot = obs[:, (H-1)*37 : H*37]

L_rec = MSE(E^s(history), detach(E^t(xt_student)))
  Gradient flows through E^s only (E^t is detached as L_rec target).
  CTSRunner runs this after every PPO update.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

_OBS_DIM    = 37   # proprioceptive obs for GO2 (ot)
_PRIV_DIM   = 26   # privileged obs xt = x_int(16) + x_ext(10)
_TEACHER_IN = 63   # [ot(37), xt(26)] — teacher input in unified obs
_H          = 50   # default history length (1000 ms at 50 Hz policy)
_LATENT_DIM = 8    # default encoder bottleneck Z


class CTSActorCritic(nn.Module):
    """Concurrent teacher-student actor-critic for GO2.

    latent_dim  — encoder bottleneck Z (default 8; ablation: 16/32/64/128)
    history_len — student obs history H (default 50 = 500 ms at 50 Hz)
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list = None,
        critic_hidden_dims: list = None,
        teacher_enc_hidden: list = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        latent_dim: int = _LATENT_DIM,
        history_len: int = _H,
        priv_dim: int = _PRIV_DIM,     # privileged x_t size: FULL=26, INT=16, EXT=10
        **kwargs,
    ):
        if kwargs:
            print(f"CTSActorCritic: ignoring unexpected kwargs {list(kwargs)}")
        super().__init__()

        self._latent_dim  = latent_dim
        self._history_len = history_len
        self._priv_dim    = priv_dim
        self._teacher_in  = _OBS_DIM + priv_dim      # [ot(37), xt(priv_dim)] in unified
        self._unified_dim = history_len * _OBS_DIM   # H*37
        actor_in          = _OBS_DIM + latent_dim     # 37+Z

        if actor_hidden_dims  is None: actor_hidden_dims  = [512, 256, 128]
        if critic_hidden_dims is None: critic_hidden_dims = [512, 256, 128]
        if teacher_enc_hidden is None: teacher_enc_hidden = [256, 128]

        act_fn = resolve_nn_activation(activation)

        # ── Teacher encoder E^t: xt(priv_dim) → zt(Z) ───────────────────
        te_layers, in_d = [], priv_dim
        for h in teacher_enc_hidden:
            te_layers += [nn.Linear(in_d, h), act_fn]
            in_d = h
        te_last = nn.Linear(in_d, latent_dim)
        nn.init.orthogonal_(te_last.weight, gain=0.01)
        nn.init.constant_(te_last.bias, 0.0)
        te_layers.append(te_last)
        self.teacher_encoder = nn.Sequential(*te_layers)

        # ── Student encoder E^s: history(H×37) → zs(Z) — 1D-CNN ─────────
        # Conv1d operates on (batch, channels=37, length=H=50):
        #   after Conv1(k=8,s=4): floor((50-8)/4+1) = 11
        #   after Conv2(k=5,s=2): floor((11-5)/2+1) = 4
        #   after Conv3(k=3,s=1): floor((4-3)/1+1)  = 2   → flat=128*2=256
        self.student_conv = nn.Sequential(
            nn.Conv1d(_OBS_DIM, 32,  kernel_size=8, stride=4), nn.ELU(),
            nn.Conv1d(32,       64,  kernel_size=5, stride=2), nn.ELU(),
            nn.Conv1d(64,       128, kernel_size=3, stride=1), nn.ELU(),
        )
        dummy    = torch.zeros(1, _OBS_DIM, history_len)
        flat_dim = int(torch.prod(torch.tensor(self.student_conv(dummy).shape[1:])))
        self.student_fc = nn.Linear(flat_dim, latent_dim)
        nn.init.orthogonal_(self.student_fc.weight, gain=0.01)
        nn.init.constant_(self.student_fc.bias, 0.0)

        # ── Shared policy π: [ot(37), z(Z)] → actions(12) ───────────────
        pol_layers, in_d = [], actor_in
        for h in actor_hidden_dims:
            pol_layers += [nn.Linear(in_d, h), act_fn]
            in_d = h
        pol_layers.append(nn.Linear(in_d, num_actions))
        self.actor = nn.Sequential(*pol_layers)

        # ── Critic V: [ot(37), zt(Z)] → value(1) ────────────────────────
        crit_layers, in_d = [], actor_in
        for h in critic_hidden_dims:
            crit_layers += [nn.Linear(in_d, h), act_fn]
            in_d = h
        crit_layers.append(nn.Linear(in_d, 1))
        self.critic = nn.Sequential(*crit_layers)

        # ── Action noise ─────────────────────────────────────────────────
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

    # ── Encoder helpers (exposed for CTSRunner L_rec) ─────────────────────
    def encode_teacher(self, xt: torch.Tensor) -> torch.Tensor:
        """E^t(xt) → zt.  Input: xt(priv_dim) privileged state (FULL=26/INT=16/EXT=10)."""
        return self.teacher_encoder(xt)

    def encode_student(self, unified_obs: torch.Tensor) -> torch.Tensor:
        """E^s(history) → zs.  Input: flattened history (H×37D)."""
        N = unified_obs.shape[0]
        x = unified_obs.reshape(N, self._history_len, _OBS_DIM).transpose(1, 2)  # (N,37,H)
        x = self.student_conv(x)   # (N,128,2)
        x = x.flatten(1)           # (N,256)
        return self.student_fc(x)  # (N, Z)

    # ── Routing: teacher/student via obs[:, -1] flag ──────────────────────
    def _encode_from_obs(self, observations: torch.Tensor):
        """Route obs to teacher/student encoders. Returns (ot, z).

        Teacher: ot = obs[:, :37],  xt = obs[:, 37 : 37+priv_dim]  → E^t → zt
        Student: history = obs[:, :H*37] → E^s → zs;
                 ot = obs[:, (H-1)*37 : H*37]  (most recent frame)
        """
        is_teacher = observations[:, -1] > 0.5
        unified    = observations[:, :self._unified_dim]   # (N, H*37)
        N          = observations.shape[0]
        z  = torch.empty(N, self._latent_dim, device=observations.device)
        ot = torch.empty(N, _OBS_DIM,         device=observations.device)

        if is_teacher.any():
            ot[is_teacher] = unified[is_teacher, :_OBS_DIM]
            xt             = unified[is_teacher, _OBS_DIM:self._teacher_in]  # (Nt, priv_dim)
            z[is_teacher]  = self.encode_teacher(xt)

        if (~is_teacher).any():
            z[~is_teacher]  = self.encode_student(unified[~is_teacher])
            ot[~is_teacher] = unified[~is_teacher, (self._history_len - 1) * _OBS_DIM:]

        return ot, z

    # ── RSL-RL interface ─────────────────────────────────────────────────
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def update_distribution(self, observations: torch.Tensor, **kwargs):
        ot, z = self._encode_from_obs(observations)
        mean  = self.actor(torch.cat([ot, z], dim=-1))
        std   = self.std.clamp(min=1e-6)
        self.distribution = Normal(mean, std.expand_as(mean))

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        ot, z = self._encode_from_obs(observations)
        return self.actor(torch.cat([ot, z], dim=-1))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """critic_observations: (N, 37+priv_dim) = [ot(37), xt(priv_dim)]."""
        ot = critic_observations[:, :_OBS_DIM]                 # (N, 37)
        xt = critic_observations[:, _OBS_DIM:_OBS_DIM + self._priv_dim]   # (N, priv_dim)
        zt = self.encode_teacher(xt)
        return self.critic(torch.cat([ot, zt], dim=-1))

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

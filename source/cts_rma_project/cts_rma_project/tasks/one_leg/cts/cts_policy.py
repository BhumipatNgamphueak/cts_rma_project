# tasks/one_leg/cts/cts_policy.py
"""
CTSActorCritic — paper Section 4.3 concurrent teacher-student network.

Encoder design: both RMA and CTS compress only x(t) (unobservable privileged
state). o(t) is observable from sensors and flows directly to the actor.
This is the consistent design — the encoder's job is to handle what the
student cannot observe, not what it already has.

  Teacher encoder E^t: xt(33) → zt ∈ R^Z   (same structure as RMA µ; Z=latent_dim)
    Updated by PPO gradient AND L^rec gradient (both flow through E^t).
  Student encoder E^s: o_{t-H:t} ∈ R^{5×15} → zs ∈ R^Z
    Learns to approximate E^t(xt) from ot history alone.
  Shared policy π: [ot(15), z(Z)] = (15+Z)D → [256,128] → actions(3)
  Critic V: [ot(15), E^t(xt)(Z)] = (15+Z)D → [256,128] → value(1)

L^rec = ||E^s(o_{t-H:t}) - E^t(xt_student)||^2
  Gradient flows through BOTH encoders — shapes E^t to produce latents
  recoverable from proprioceptive history.

RMA vs CTS — structurally identical, only training method differs:
  RMA:  µ(xt) → zt(Z),  actor([ot, zt]) = (15+Z)D  — sequential 2-stage
  CTS: Et(xt) → zt(Z),  actor([ot, zt]) = (15+Z)D  — concurrent, L^rec co-shapes Et

latent_dim (Z) is configurable via --latent_dim CLI arg (default 8).
Pass latent_dim=16 or 32 to compare bottleneck sizes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

_OBS_DIM    = 15
_PRIV_DIM   = 33   # xt = [External(13), Internal(20)] — full DR coverage
_TEACHER_IN = 48   # [ot(15), xt(33)] — used for critic and obs splitting
_H          = 50   # default obs history length (500 ms at 100 Hz)
_LATENT_DIM = 8    # default; overridden per-run via latent_dim kwarg


class CTSActorCritic(nn.Module):
    """Concurrent teacher-student actor-critic (paper Section 4.3).

    latent_dim controls the encoder bottleneck Z (default 8).
    history_len controls how many past obs steps the student encoder sees (default 50).
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
        student_enc_channels: list = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        latent_dim: int = _LATENT_DIM,
        history_len: int = _H,          # configurable memory length H
        **kwargs,
    ):
        if kwargs:
            print(f"CTSActorCritic: ignoring unexpected kwargs {list(kwargs)}")
        super().__init__()

        self._latent_dim  = latent_dim
        self._history_len = history_len
        self._unified_dim = history_len * _OBS_DIM
        actor_in          = _OBS_DIM + latent_dim   # (15+Z)D

        if actor_hidden_dims    is None: actor_hidden_dims    = [256, 128]
        if critic_hidden_dims   is None: critic_hidden_dims   = [256, 128]
        if teacher_enc_hidden   is None: teacher_enc_hidden   = [512, 256]
        if student_enc_channels is None: student_enc_channels = [32, 64, 128, 256]

        act_fn = resolve_nn_activation(activation)

        # ── Teacher encoder E^t: xt(24) → zt(Z) ─────────────────────────
        # Encodes unobservable xt only — same design as RMA µ.
        # Updated by both PPO and L^rec gradients.
        te_layers, in_d = [], _PRIV_DIM
        for h in teacher_enc_hidden:
            te_layers += [nn.Linear(in_d, h), act_fn]
            in_d = h
        te_last = nn.Linear(in_d, latent_dim)
        nn.init.orthogonal_(te_last.weight, gain=0.01)
        nn.init.constant_(te_last.bias, 0.0)
        te_layers.append(te_last)
        self.teacher_encoder = nn.Sequential(*te_layers)

        # ── Student encoder E^s: history(H×15=750) → zs(Z) — 1D-CNN ─────
        # Matches Phase-2 adaptation module architecture (H=50 → 11 → 4 → 2).
        # Conv1d operates on (batch, channels=15, length=H=50)
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

        # ── Shared policy π: [ot(15), z(Z)] = (15+Z)D → actions ─────────
        # ot fed directly (Figure 3); z from teacher or student encoder.
        pol_layers, in_d = [], actor_in
        for h in actor_hidden_dims:
            pol_layers += [nn.Linear(in_d, h), act_fn]
            in_d = h
        pol_layers.append(nn.Linear(in_d, num_actions))
        self.actor = nn.Sequential(*pol_layers)

        # ── Critic V: [ot(15), E^t(xt)(Z)] = (15+Z)D → value ────────────
        # Critic sees ot directly + encoded xt — mirrors actor input.
        crit_layers, in_d = [], actor_in
        for h in critic_hidden_dims:
            crit_layers += [nn.Linear(in_d, h), act_fn]
            in_d = h
        crit_layers.append(nn.Linear(in_d, 1))
        self.critic = nn.Sequential(*crit_layers)

        # ── Action noise ──────────────────────────────────────────────────
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

    # ── Encoder helpers (exposed for CTSRunner L_rec computation) ─────────
    def encode_teacher(self, xt: torch.Tensor) -> torch.Tensor:
        """E^t(xt) → zt ∈ R^Z.  Input is xt(24D) — privileged state only."""
        return self.teacher_encoder(xt)

    def encode_student(self, unified_obs: torch.Tensor) -> torch.Tensor:
        """E^s(o_{t-H:t}) → zs ∈ R^Z.  Input is flattened history (H×15D)."""
        N = unified_obs.shape[0]
        x = unified_obs.reshape(N, self._history_len, _OBS_DIM).transpose(1, 2)  # (N,15,H)
        x = self.student_conv(x)    # (N,128,2)
        x = x.flatten(1)            # (N,256)
        return self.student_fc(x)   # (N, Z)

    # ── Routing: encode based on is_teacher flag (obs[:, -1]) ─────────────
    def _encode_from_obs(self, observations: torch.Tensor):
        """Route teacher/student envs to their encoders. Returns (ot, z).

        Teacher: ot = obs[:, :15],  xt = obs[:, 15:48]  → E^t → zt
        Student: history = obs[:, :H*15] → E^s → zs; ot = obs[:, (H-1)*15 : H*15]
        """
        is_teacher = observations[:, -1] > 0.5
        unified    = observations[:, :self._unified_dim]   # (N, H*15)
        N          = observations.shape[0]
        z  = torch.empty(N, self._latent_dim, device=observations.device)
        ot = torch.empty(N, _OBS_DIM,         device=observations.device)

        if is_teacher.any():
            ot[is_teacher] = unified[is_teacher, :_OBS_DIM]
            xt             = unified[is_teacher, _OBS_DIM:_TEACHER_IN]  # (Nt, 33)
            z[is_teacher]  = self.encode_teacher(xt)

        if (~is_teacher).any():
            z[~is_teacher]  = self.encode_student(unified[~is_teacher])
            ot[~is_teacher] = unified[~is_teacher, (self._history_len - 1) * _OBS_DIM:]

        return ot, z

    # ── RSL-RL interface ──────────────────────────────────────────────────
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
        """critic_observations: (N, 48) = [ot(15), xt(33)] — encode xt, concat with ot, get V."""
        ot = critic_observations[:, :_OBS_DIM]   # (N, 15)
        xt = critic_observations[:, _OBS_DIM:]   # (N, 33)
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

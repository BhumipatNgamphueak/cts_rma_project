# tasks/one_leg/cts/one_leg_cts_env.py
"""
One-legged hopper — CTS environment (paper Section 3.2).

Concurrent Teacher-Student (1-stage, 75 % teacher / 25 % student).

obs["policy"]  = 751-D  (750D unified obs + 1D is_teacher flag)
    Teacher envs (is_teacher=1): [ot(15), xt(33), zeros(702), 1.0]
    Student envs (is_teacher=0): [obs_history(H=50×15=750), 0.0]

obs["critic"]  = 48-D  [ot, xt]  for ALL envs — privileged critic.
    (Used both for value estimation and as L_rec supervision target.)

The env inherits all DR, reward, scene, and reset logic from OneLegRMAEnv.
"""
from __future__ import annotations
import torch

from ..rma.one_leg_rma_env import OneLegRMAEnv
from .one_leg_cts_env_cfg   import OneLegCTSEnvCfg

_OBS_DIM    = 15    # ot dimension
_PRIV_DIM   = 33    # xt dimension = External(13) + Internal(20)
_TEACHER_IN = _OBS_DIM + _PRIV_DIM   # 48


class OneLegCTSEnv(OneLegRMAEnv):
    """Concurrent Teacher-Student env — 75 % teacher / 25 % student envs."""

    cfg: OneLegCTSEnvCfg

    def __init__(self, cfg: OneLegCTSEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        N = self.num_envs
        self._H           = cfg.history_len
        self._unified_dim = cfg.history_len * _OBS_DIM

        # ── Teacher/student split (fixed for the training run) ────────────
        n_teacher = int(N * cfg.teacher_ratio)
        is_t      = torch.zeros(N, dtype=torch.bool, device=self.device)
        is_t[:n_teacher] = True
        self.is_teacher: torch.Tensor = is_t            # (N,) bool

        # ── Observation history buffer for student encoder ────────────────
        self.obs_history = torch.zeros(N, self._H, _OBS_DIM, device=self.device)

    # ══════════════════════════════════════════════════════════════════════
    # Observations
    # ══════════════════════════════════════════════════════════════════════
    def _get_observations(self) -> dict:
        # 1. RMA env gives [ot, xt] = 39D for all envs
        base = super()._get_observations()          # {"policy": (N, 39)}
        teacher_input = base["policy"]              # (N, 48): [ot, xt]
        ot = teacher_input[:, :_OBS_DIM]            # (N, 15)

        # 2. Roll history: drop oldest, append latest ot
        self.obs_history = torch.roll(self.obs_history, -1, dims=1)
        self.obs_history[:, -1, :] = ot

        # 3. Build unified obs (H×15)D
        unified = torch.zeros(self.num_envs, self._unified_dim, device=self.device)
        # Teacher envs: first 48D = [ot, xt], rest = zeros (already zero)
        unified[self.is_teacher, :_TEACHER_IN] = teacher_input[self.is_teacher]
        # Student envs: full H×15D = flattened history
        unified[~self.is_teacher] = self.obs_history[~self.is_teacher].reshape(-1, self._unified_dim)

        # 4. Append is_teacher flag as final dim → 76D policy obs
        flag      = self.is_teacher.float().unsqueeze(-1)   # (N, 1)
        policy_76 = torch.cat([unified, flag], dim=-1)      # (N, 76)

        return {
            "policy": policy_76,      # (N, 751) — routed by CTSActorCritic via obs[:, -1]
            "critic": teacher_input,  # (N, 48)  — privileged for critic + L_rec target
        }

    # ══════════════════════════════════════════════════════════════════════
    # Reset: clear history for reset envs
    # ══════════════════════════════════════════════════════════════════════
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is not None:
            self.obs_history[env_ids] = 0.0

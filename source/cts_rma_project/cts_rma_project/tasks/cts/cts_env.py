# tasks/cts/cts_env.py
"""
Go2 CTS environment — extends ManagerBasedRLEnv with teacher/student state.

Adds is_teacher (N, bool) and obs_history (N, H, 37) after super().__init__()
completes.  The observation term cts_teacher_student_obs (registered in the
"policy" obs group) reads these attributes every step.

Design note: ManagerBasedRLEnv.step() calls observation_manager.compute()
directly, bypassing any get_observations() override.  The teacher-student
routing must therefore live inside the obs term function, not here.
"""
from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv
from .cts_env_cfg import CTSEnvCfg

_OBS_DIM = 37   # proprioceptive obs dimension for GO2


class Go2CTSEnv(ManagerBasedRLEnv):
    cfg: CTSEnvCfg

    def __init__(self, cfg: CTSEnvCfg, render_mode: str | None = None, **kwargs):
        # super().__init__ calls load_managers → ObservationManager._prepare_terms
        # which calls cts_teacher_student_obs once for shape detection.
        # At that point is_teacher / obs_history don't exist yet — the obs function
        # handles this by returning a dummy zeros tensor (see mdp/observations.py).
        super().__init__(cfg, render_mode, **kwargs)

        # Now num_envs and device are available — set up teacher/student state
        N = self.num_envs
        self._H = cfg.history_len
        n_teacher = int(N * cfg.teacher_ratio)
        is_t = torch.zeros(N, dtype=torch.bool, device=self.device)
        is_t[:n_teacher] = True
        self.is_teacher = is_t

        # Observation history buffer for student encoder: (N, H, 37)
        self.obs_history = torch.zeros(N, self._H, _OBS_DIM, device=self.device)

    # ── Reset: clear history for terminated envs ─────────────────────────────
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is not None and len(env_ids) > 0:
            self.obs_history[env_ids] = 0.0

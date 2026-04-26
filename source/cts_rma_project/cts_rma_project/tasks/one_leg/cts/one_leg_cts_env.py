# tasks/one_leg/cts/one_leg_cts_env.py
"""
One-legged hopper — CTS environments.

CTS = Concurrent Teacher-Student (sequential implementation):

  Phase 1 — Teacher training
    OneLegCTSTeacherEnv: policy sees 22-D (14 prop + 8 privileged).
    Trained with standard PPO. Saved as the "teacher checkpoint".

  Phase 2 — Student distillation
    OneLegCTSStudentEnv: policy sees 14-D (same as Baseline).
    A frozen teacher is loaded at init. At each step the teacher's
    action is computed and added as a "teacher imitation reward":
        r_imitate = -alpha * ||student_action - teacher_action||²
    alpha decays from 1.0 → 0.1 over the first 1000 iterations so
    the student transitions from pure imitation to RL-guided behavior.
"""
from __future__ import annotations
import torch

from ..baseline.one_leg_env import OneLegBaselineEnv
from ..rma.one_leg_rma_env  import OneLegRMAEnv
from .one_leg_cts_env_cfg   import OneLegCTSTeacherEnvCfg, OneLegCTSStudentEnvCfg


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Teacher  (22-D obs, full privileged info)
# ══════════════════════════════════════════════════════════════════════════════

class OneLegCTSTeacherEnv(OneLegRMAEnv):
    """
    CTS teacher: actor AND critic see the full 22-D privileged obs.

    Difference from RMA:
    - RMA   → actor=14D, critic=22D  (asymmetric AC)
    - CTS teacher → actor=22D, critic=22D  (both see everything)

    At deployment the teacher policy cannot run on real hardware (needs
    privileged info).  It is frozen and used only to guide the student.
    """

    cfg: OneLegCTSTeacherEnvCfg

    def _get_observations(self) -> dict:
        # Build base obs (also updates all state buffers)
        base_dict  = OneLegBaselineEnv._get_observations(self)  # → {"policy": 14D}
        privileged = self._get_privileged_obs()                  # (N, 8)
        teacher_obs = torch.cat([base_dict["policy"], privileged], dim=-1)  # (N, 22)

        # Both actor and critic see the full 22-D obs
        return {"policy": teacher_obs}


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Student  (14-D obs + teacher imitation reward)
# ══════════════════════════════════════════════════════════════════════════════

class OneLegCTSStudentEnv(OneLegRMAEnv):
    """
    CTS student: actor sees 14-D obs (deployable on real hardware).

    Loads a frozen teacher checkpoint at init. Each step computes the
    teacher's action and adds a behavioral-cloning reward that decays
    over training so the student ultimately relies on RL signal.
    """

    cfg: OneLegCTSStudentEnvCfg

    def __init__(self, cfg: OneLegCTSStudentEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._teacher_policy  = None   # set via load_teacher()
        self._teacher_actions = torch.zeros(self.num_envs, self.num_actuated, device=self.device)
        self._imitation_alpha = cfg.imitation_alpha_start
        self._iteration       = 0

    # ── Public API ────────────────────────────────────────────────────────
    def load_teacher(self, teacher_policy):
        """Call this once after creating the env to supply the frozen teacher."""
        self._teacher_policy = teacher_policy
        for p in self._teacher_policy.parameters():
            p.requires_grad_(False)
        self._teacher_policy.eval()

    def step_iteration(self):
        """Call once per training iteration to decay the imitation weight."""
        self._iteration += 1
        frac = min(self._iteration / self.cfg.imitation_decay_iters, 1.0)
        self._imitation_alpha = (
            self.cfg.imitation_alpha_start
            + frac * (self.cfg.imitation_alpha_end - self.cfg.imitation_alpha_start)
        )

    # ── Override obs — student sees only 14-D ────────────────────────────
    def _get_observations(self) -> dict:
        base_dict  = OneLegBaselineEnv._get_observations(self)  # {"policy": 14D}
        privileged = self._get_privileged_obs()                  # (N, 8) — for teacher only

        # Compute teacher action with frozen teacher (no grad)
        if self._teacher_policy is not None:
            teacher_obs = torch.cat([base_dict["policy"], privileged], dim=-1)
            with torch.no_grad():
                self._teacher_actions = self._teacher_policy(teacher_obs)

        return base_dict   # student sees 14-D only

    # ── Override reward — add imitation term ─────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        rl_reward = super()._get_rewards()

        # Imitation reward: negative MSE between student and teacher actions
        # Computed on self.actions which holds the student policy output
        imitation = -torch.sum(
            (self.actions - self._teacher_actions.detach()) ** 2, dim=1
        )
        total = rl_reward + self._imitation_alpha * imitation

        self.extras["log"]["imitation_alpha"] = self._imitation_alpha
        self.extras["log"]["imitation_loss"]  = float(-imitation.mean())
        return total

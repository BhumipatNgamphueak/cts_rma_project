# tasks/one_leg/rma/one_leg_rma_env.py
"""
One-legged hopper — RMA Phase-1 environment (paper Section 4.2).

Extends OneLegBaselineEnv (which already applies full Table 4 DR).
Adds privileged obs xt ∈ R^33 for actor + critic:

    xt = [External(13), Internal(20)]          ← paper Eq (2) ordering

    External (13): fcontact(3) | cbin(1) | τt(3) | q̈t(3) | fpush(3)
    Internal (20): µ(1) | erest(1) | Δm(4) | mpay(1) | ΔKp(3) | σms(3) | dact(1)
                   ΔKd(3) | Δcom(3)

Full actor+critic input: [ot(15), xt(33)] = 48D.
"""
from __future__ import annotations
import torch

from ..baseline.one_leg_env import OneLegBaselineEnv
from .one_leg_rma_env_cfg import OneLegRMAEnvCfg


class OneLegRMAEnv(OneLegBaselineEnv):
    """RMA Phase-1: actor+critic see full [ot, xt] = 39D."""

    cfg: OneLegRMAEnvCfg

    # ══════════════════════════════════════════════════════════════════════
    # Observations: prepend xt to baseline's 15D ot
    # ══════════════════════════════════════════════════════════════════════
    def _get_observations(self) -> dict:
        obs_dict = super()._get_observations()          # {"policy": (N, 15)}
        xt       = self._build_privileged_obs()         # (N, 24)
        full_obs = torch.cat([obs_dict["policy"], xt], dim=-1)   # (N, 39)
        return {"policy": full_obs}

    def _build_privileged_obs(self) -> torch.Tensor:
        """Build xt ∈ R^33 = [External(13), Internal(20)]  — full DR coverage."""

        # ── External 13D (timestep-varying signals) ───────────────────────
        cf_vec     = self.scene["contact_sensor"].data.net_forces_w[:, 0, :] / 100.0  # (N, 3)
        cf_flag    = self.is_foot_in_contact.float().unsqueeze(-1)                    # (N, 1)
        torques    = self.robot.data.applied_torque[:, self.actuated_dof_indices] / 50.0  # (N, 3)
        accels     = self.robot.data.joint_acc[:, self.actuated_dof_indices] / 50.0   # (N, 3)
        push_f     = self.dr_push_force / max(self.cfg.push_force_max, 1e-6)          # (N, 3) ∈ [-1,1]
        external   = torch.cat([cf_vec, cf_flag, torques, accels, push_f], dim=-1)   # (N, 13)

        # ── Internal 20D (episode-constant body parameters) ───────────────
        fric       = self.dr_friction.unsqueeze(-1)                        # (N, 1)
        rest       = self.dr_restitution.unsqueeze(-1)                     # (N, 1)
        m_del      = self.dr_mass_scale - 1.0                              # (N, 4)
        pyld       = self.dr_payload.unsqueeze(-1)                         # (N, 1)
        kp_del     = self.dr_kp_scale - 1.0                                # (N, 3)
        ms_del     = self.dr_motor_str - 1.0                               # (N, 3)
        delay      = (self.dr_action_delay.float() / 2.0).unsqueeze(-1)    # (N, 1) ∈ [0,1]
        kd_del     = self.dr_kd_scale - 1.0                                # (N, 3)
        com_off    = self.dr_com_offset                                     # (N, 3) metres
        internal   = torch.cat(
            [fric, rest, m_del, pyld, kp_del, ms_del, delay, kd_del, com_off], dim=-1)  # (N, 20)

        # Ablation masking — Exp. 2 (group) and diagnostic (per-component)
        # xt layout: external = [cf_vec(0:3), cf_flag(3:4), torques(4:7), accels(7:10), push_f(10:13)]
        mode = getattr(self.cfg, "priv_mode", "FULL").upper()
        if mode == "INT":
            external = torch.zeros_like(external)
        elif mode == "EXT":
            internal = torch.zeros_like(internal)
        # Diagnostic: FULL minus one external component at a time
        elif mode == "FULL_NO_CF":      # remove contact force vec + flag
            external = external.clone(); external[:, 0:4] = 0.0
        elif mode == "FULL_NO_TORQ":    # remove joint torques
            external = external.clone(); external[:, 4:7] = 0.0
        elif mode == "FULL_NO_ACCEL":   # remove joint accelerations
            external = external.clone(); external[:, 7:10] = 0.0
        elif mode == "FULL_NO_PUSH":    # remove push-force signal
            external = external.clone(); external[:, 10:13] = 0.0

        return torch.cat([external, internal], dim=-1)   # (N, 33) — Ext first, Int second

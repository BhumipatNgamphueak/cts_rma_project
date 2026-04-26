# tasks/one_leg/rma/one_leg_rma_env.py
"""
One-legged hopper — RMA environment (Phase 1: asymmetric actor-critic).

Actor  sees: 14-D proprioceptive obs (identical to Baseline at deployment)
Critic sees: 14-D + 8-D privileged = 22-D (sim-only training signal)

Privileged obs (8-D):
    stiffness_scale(1)  — actuator Kp scale drawn at episode reset
    damping_scale(1)    — actuator Kd scale
    exact_contact(1)    — unsmoothed contact force magnitude
    foot_vel_x(1)       — end-effector x-velocity (world frame)
    foot_vel_z(1)       — end-effector z-velocity (lift quality)
    body_vel_x(1)       — prismatic joint velocity (rail motion)
    cycle_time_norm(1)  — normalised cycle time in [0,1] ((T-0.4)/0.6)
    linear_joint_vel(1) — velocity of linear_left_right joint

RSL-RL auto-builds an asymmetric ActorCritic when "critic" key is returned
from _get_observations() with a different size than "policy".
"""
from __future__ import annotations
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg

from ..baseline.one_leg_env import OneLegBaselineEnv
from .one_leg_rma_env_cfg import OneLegRMAEnvCfg

_NOMINAL_STIFFNESS = 20.0
_NOMINAL_DAMPING   = 1.0
_STIFFNESS_RANGE   = (0.8, 1.2)   # scale
_DAMPING_RANGE     = (0.8, 1.2)


class OneLegRMAEnv(OneLegBaselineEnv):
    """RMA Phase-1: asymmetric actor-critic with privileged critic info."""

    cfg: OneLegRMAEnvCfg

    # ── Override: add privileged critic channel ───────────────────────────
    def _get_observations(self) -> dict:
        obs_dict   = super()._get_observations()
        privileged = self._get_privileged_obs()
        obs_dict["critic"] = torch.cat(
            [obs_dict["policy"], privileged], dim=-1
        )   # (N, 22)
        return obs_dict

    def _get_privileged_obs(self) -> torch.Tensor:
        """8-D privileged obs — only available in simulation."""
        # 1-2: actuator DR params (episode-constant)
        stiffness_norm = (self.stiffness_scale - 1.0).unsqueeze(-1)   # (N,1)
        damping_norm   = (self.damping_scale   - 1.0).unsqueeze(-1)   # (N,1)

        # 3: exact (unsmoothed) contact force
        net_forces   = self.scene["contact_sensor"].data.net_forces_w
        exact_force  = torch.norm(net_forces, dim=-1).squeeze(-1).unsqueeze(-1)  # (N,1)

        # 4-5: end-effector velocity (x, z)
        ee_vel   = self.robot.data.body_lin_vel_w[:, self._ee_body_idx, :]  # (N,3)
        foot_vx  = ee_vel[:, 0:1]   # x
        foot_vz  = ee_vel[:, 2:3]   # z

        # 6: body velocity along rail (x-velocity of prismatic body)
        body_vx = self.robot.data.body_lin_vel_w[:, self._linear_body_idx, 0:1]  # (N,1)

        # 7: normalised cycle time  [0.4s,1.0s] → [0,1]
        cycle_norm = ((self.cycle_time - 0.4) / 0.6).unsqueeze(-1)   # (N,1)

        # 8: linear joint velocity (velocity of prismatic joint)
        lin_vel = self.robot.data.joint_vel[:, self._linear_joint_idx]   # (N,1) or (N,?)
        if lin_vel.dim() == 1:
            lin_vel = lin_vel.unsqueeze(-1)

        return torch.cat(
            [stiffness_norm, damping_norm, exact_force,
             foot_vx, foot_vz, body_vx, cycle_norm, lin_vel],
            dim=-1,
        )   # (N, 8)

    # ── DR: randomise actuator gains at each episode reset ────────────────
    def _reset_dr(self, env_ids: torch.Tensor):
        n  = len(env_ids)
        ks = torch.empty(n, device=self.device).uniform_(*_STIFFNESS_RANGE)
        kd = torch.empty(n, device=self.device).uniform_(*_DAMPING_RANGE)

        self.stiffness_scale[env_ids] = ks
        self.damping_scale[env_ids]   = kd

        mdp.randomize_actuator_gains(
            self,
            env_ids,
            asset_cfg=SceneEntityCfg("robot"),
            stiffness_distribution_params=(_NOMINAL_STIFFNESS * _STIFFNESS_RANGE[0],
                                           _NOMINAL_STIFFNESS * _STIFFNESS_RANGE[1]),
            damping_distribution_params  =(_NOMINAL_DAMPING   * _DAMPING_RANGE[0],
                                           _NOMINAL_DAMPING   * _DAMPING_RANGE[1]),
            operation="abs",
        )

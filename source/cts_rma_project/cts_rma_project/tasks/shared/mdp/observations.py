# tasks/shared/mdp/observations.py
"""
Observation functions shared across Baseline, RMA, and CTS on GO2.

Proprioceptive  o_t  ∈ R^30  (runtime — available on real hardware)
Privileged      x_t  ∈ R^24  (sim-only — 14 internal + 10 external)

Privileged decomposition (Table 2 of project proposal, adapted for GO2):

  Internal  x^int (14-D, episode-constant body parameters):
    µ           (1)  ground friction
    e_rest      (1)  restitution
    ∆m_leg      (4)  per-leg link mass scale (deviation from 1.0)
    m_pay       (1)  payload added to base [kg]
    ∆Kp         (3)  Kp scale per joint type (hip/thigh/calf), deviation from 1.0
    σ_ms        (3)  motor-strength scale per joint type, deviation from 1.0
    d_act       (1)  action delay [ms], normalised to [0, 1] (20 ms → 1.0)

  External  x^ext (10-D, timestep-varying interaction signals):
    f_contact   (3)  net contact force summed over all four feet [N] (world frame)
    c_bin       (4)  per-foot binary contact flag (raw sensor, >1 N threshold)
    τ_avg       (3)  mean applied torque per joint type (hip/thigh/calf) [N·m]
"""
from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# ── Proprioceptive ────────────────────────────────────────────────────────────

def proprioceptive_obs_go2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    o_t ∈ R^37 — runtime-only proprioceptive state for GO2.

    joint_pos_rel(12) + joint_vel(12) + ang_vel_b(3) + gravity_b(3)
    + vel_cmd(3) + foot_contact(4) = 37

    sensor_cfg should have body_names=".*_foot" so body_ids resolves to the 4 feet.
    """
    asset = env.scene[asset_cfg.name]
    contact = env.scene[sensor_cfg.name]

    joint_pos    = asset.data.joint_pos - asset.data.default_joint_pos   # [N, 12]
    joint_vel    = asset.data.joint_vel                                   # [N, 12]
    ang_vel_b    = asset.data.root_ang_vel_b                              # [N, 3]
    gravity_b    = asset.data.projected_gravity_b                         # [N, 3]
    vel_cmd      = env.command_manager.get_command("base_velocity")[:, :3]  # [N, 3]
    foot_contact = (
        contact.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2] > 1.0
    ).float()                                                             # [N, 4]

    return torch.cat([joint_pos, joint_vel, ang_vel_b, gravity_b, vel_cmd, foot_contact], dim=-1)


# ── Privileged — Internal ─────────────────────────────────────────────────────

def privileged_internal_go2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    x^int ∈ R^14 — episode-constant body parameters.

    Reads from env.extras["dr"] which is populated by the custom DR events
    in tasks/shared/mdp/events.py at each episode reset.
    Falls back to neutral values if the buffer has not been initialised yet.
    """
    N, D = env.num_envs, env.device
    dr = env.extras.get("dr", {})

    friction       = dr.get("friction",       torch.full((N, 1), 0.95, device=D))
    restitution    = dr.get("restitution",    torch.full((N, 1), 0.50, device=D))
    # leg mass scale stored as raw scale; send (scale - 1) so 0 = nominal
    leg_mass_scale = dr.get("leg_mass_scale", torch.ones(N, 4, device=D)) - 1.0
    payload        = dr.get("payload",        torch.zeros(N, 1, device=D))
    # Kp/Kd/σ_ms stored as raw scale; send (scale - 1) so 0 = nominal
    kp_scale       = dr.get("kp_scale",       torch.ones(N, 3, device=D)) - 1.0
    motor_strength = dr.get("motor_strength", torch.ones(N, 3, device=D)) - 1.0
    # action delay: normalise [0, 20 ms] → [0, 1]
    action_delay   = dr.get("action_delay_ms", torch.zeros(N, 1, device=D)) / 20.0

    return torch.cat(
        [friction, restitution, leg_mass_scale, payload,
         kp_scale, motor_strength, action_delay],
        dim=-1,
    )   # [N, 14]


# ── Privileged — External ─────────────────────────────────────────────────────

def privileged_external_go2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    x^ext ∈ R^10 — timestep-varying interaction signals.

    f_contact_sum (3) + c_bin (4) + τ_avg_per_type (3) = 10

    sensor_cfg should have body_names=".*_foot" so body_ids resolves to the 4 feet.
    """
    asset = env.scene[asset_cfg.name]
    contact = env.scene[sensor_cfg.name]

    # Sum of net contact force vectors across all four feet [N, 4, 3] → [N, 3]
    f_contact_sum = contact.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :].sum(dim=1)  # [N, 3]

    # Per-foot binary contact flag (more precise than proprioceptive obs)
    c_bin = (
        contact.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2] > 1.0
    ).float()                                                                   # [N, 4]

    # Average applied torque per joint type — GO2 ordering: hips 0-3, thighs 4-7, calfs 8-11
    torques = asset.data.applied_torque                                          # [N, 12]
    tau_hip   = torques[:, 0:4].abs().mean(dim=1, keepdim=True)
    tau_thigh = torques[:, 4:8].abs().mean(dim=1, keepdim=True)
    tau_calf  = torques[:, 8:12].abs().mean(dim=1, keepdim=True)
    tau_avg   = torch.cat([tau_hip, tau_thigh, tau_calf], dim=-1)               # [N, 3]

    return torch.cat([f_contact_sum, c_bin, tau_avg], dim=-1)   # [N, 10]


# ── RMA asymmetric critic: o_t ⊕ x_t ────────────────────────────────────────

def combined_obs_rma(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    RMA asymmetric critic input: o_t (37) ⊕ x_t (24) = 61D.
    The actor sees only o_t; the critic sees full state for better value estimates.
    sensor_cfg should have body_names=".*_foot".
    """
    return torch.cat([
        proprioceptive_obs_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg),
        privileged_full_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg),
    ], dim=-1)


# ── Privileged — Full (Internal ∪ External) ───────────────────────────────────

def privileged_full_go2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    x_t ∈ R^24 = x^int (14) ⊕ x^ext (10).
    Used by RMA Phase 1 teacher and CTS teacher group.
    """
    return torch.cat(
        [
            privileged_internal_go2(env),
            privileged_external_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg),
        ],
        dim=-1,
    )

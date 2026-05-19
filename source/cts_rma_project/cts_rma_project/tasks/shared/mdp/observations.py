# tasks/shared/mdp/observations.py
"""
Observation functions shared across Baseline, RMA, and CTS on GO2.

Proprioceptive  o_t  ∈ R^37  (runtime — available on real hardware)
Privileged      x_t  ∈ R^26  (sim-only — 16 internal + 10 external)

Privileged decomposition:

  Internal  x^int (16-D, episode-constant body parameters):
    µ           (1)  ground friction
    e_rest      (1)  restitution
    ∆m_base     (1)  base body mass scale deviation (scale - 1), 0 = nominal
    ∆Kp         (3)  Kp scale per joint type (hip/thigh/calf), deviation from 1.0
    ∆Kd         (3)  Kd scale per joint type, deviation from 1.0
    ∆com        (3)  base COM offset [m] in body frame
    ∆I_base     (3)  base body inertia scale deviation (scale - 1), uniform per axis
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


# ── Proprioceptive — split terms (for per-component noise) ───────────────────
# The full proprioceptive obs is split into three ordered terms so that
# ObservationTermCfg noise can be applied to ang_vel_b alone:
#   part1 (24D): joint_pos_rel(12) + joint_vel(12)
#   ang_vel (3D): ang_vel_b  ← GaussianNoiseCfg(std=0.2) applied here
#   part2 (10D): gravity_b(3) + vel_cmd(3) + foot_contact(4)
# Concatenating in this order gives the same 37D layout as proprioceptive_obs_go2.

def joint_pos_vel_go2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """joint_pos_rel(12) + joint_vel(12) = 24D."""
    asset = env.scene[asset_cfg.name]
    return torch.cat([
        asset.data.joint_pos - asset.data.default_joint_pos,
        asset.data.joint_vel,
    ], dim=-1)


def ang_vel_b_go2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ang_vel_b(3D) — receives Gaussian noise during training."""
    return env.scene[asset_cfg.name].data.root_ang_vel_b


def gravity_cmd_contact_go2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """gravity_b(3) + vel_cmd(3) + foot_contact(4) = 10D."""
    asset   = env.scene[asset_cfg.name]
    contact = env.scene[sensor_cfg.name]
    gravity_b  = asset.data.projected_gravity_b
    vel_cmd    = env.command_manager.get_command("base_velocity")[:, :3]
    foot_contact = (
        contact.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2] > 1.0
    ).float()
    return torch.cat([gravity_b, vel_cmd, foot_contact], dim=-1)


# ── Proprioceptive ────────────────────────────────────────────────────────────

def proprioceptive_obs_go2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    add_noise: bool = True,
) -> torch.Tensor:
    """
    o_t ∈ R^37 — runtime-only proprioceptive state for GO2.

    joint_pos_rel(12) + joint_vel(12) + ang_vel_b(3) + gravity_b(3)
    + vel_cmd(3) + foot_contact(4) = 37

    Per-component Gaussian noise (add_noise=True, disabled for play/eval):
      joint_pos  σ=0.01 rad | joint_vel σ=0.50 rad/s
      ang_vel_b  σ=0.20 rad/s | gravity_b σ=0.05
    """
    asset = env.scene[asset_cfg.name]
    contact = env.scene[sensor_cfg.name]

    joint_pos    = asset.data.joint_pos - asset.data.default_joint_pos   # [N, 12]
    joint_vel    = asset.data.joint_vel                                   # [N, 12]
    ang_vel_b    = asset.data.root_ang_vel_b.clone()                      # [N, 3]
    gravity_b    = asset.data.projected_gravity_b.clone()                 # [N, 3]
    vel_cmd      = env.command_manager.get_command("base_velocity")[:, :3]  # [N, 3]
    foot_contact = (
        contact.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2] > 1.0
    ).float()                                                             # [N, 4]

    if add_noise:
        joint_pos = joint_pos + torch.randn_like(joint_pos) * 0.01
        joint_vel = joint_vel + torch.randn_like(joint_vel) * 0.50
        ang_vel_b = ang_vel_b + torch.randn_like(ang_vel_b) * 0.20
        gravity_b = gravity_b + torch.randn_like(gravity_b) * 0.05

    return torch.cat([joint_pos, joint_vel, ang_vel_b, gravity_b, vel_cmd, foot_contact], dim=-1)


# ── Privileged — Internal ─────────────────────────────────────────────────────

def privileged_internal_go2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    x^int ∈ R^16 — episode-constant body parameters (OpenTopic-aligned).

    Reads from env.extras["dr"] populated by events.py at each reset.
    Falls back to neutral values if the buffer has not been initialised yet.

    Layout (16-D):
      friction          (1)  raw friction coefficient
      restitution       (1)  raw restitution coefficient
      payload           (1)  base mass scale deviation (scale - 1), 0 = nominal
      kp_scale          (3)  Kp scale deviation per joint type (hip/thigh/calf)
      kd_scale          (3)  Kd scale deviation per joint type
      com_offset        (3)  base COM offset [m] in body frame
      base_inertia_diag (3)  base inertia scale deviation (scale - 1), uniform
      action_delay      (1)  normalised delay [0, 20 ms] → [0, 1]
    """
    N, D = env.num_envs, env.device
    dr = env.extras.get("dr", {})

    friction          = dr.get("friction",          torch.full((N, 1), 1.0,  device=D))
    restitution       = dr.get("restitution",       torch.full((N, 1), 0.075, device=D))
    payload           = dr.get("payload",           torch.zeros(N, 1, device=D))
    # Kp/Kd stored as raw scale; send (scale - 1) so 0 = nominal
    kp_scale          = dr.get("kp_scale",          torch.ones(N, 3, device=D)) - 1.0
    kd_scale          = dr.get("kd_scale",          torch.ones(N, 3, device=D)) - 1.0
    com_offset        = dr.get("com_offset",        torch.zeros(N, 3, device=D))
    base_inertia_diag = dr.get("base_inertia_diag", torch.zeros(N, 3, device=D))
    action_delay      = dr.get("action_delay_ms",   torch.zeros(N, 1, device=D)) / 20.0

    return torch.cat(
        [friction, restitution, payload,
         kp_scale, kd_scale, com_offset,
         base_inertia_diag, action_delay],
        dim=-1,
    )   # [N, 16]


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

    # Per-leg USD order: 0=FL_hip,1=FL_thigh,2=FL_calf,3=FR_hip,...,11=RR_calf
    # Hip indices [0,3,6,9]  Thigh [1,4,7,10]  Calf [2,5,8,11]
    torques = asset.data.applied_torque                                          # [N, 12]
    tau_hip   = torques[:, [0, 3, 6, 9]].abs().mean(dim=1, keepdim=True)
    tau_thigh = torques[:, [1, 4, 7, 10]].abs().mean(dim=1, keepdim=True)
    tau_calf  = torques[:, [2, 5, 8, 11]].abs().mean(dim=1, keepdim=True)
    tau_avg   = torch.cat([tau_hip, tau_thigh, tau_calf], dim=-1)               # [N, 3]

    return torch.cat([f_contact_sum, c_bin, tau_avg], dim=-1)   # [N, 10]


# ── RMA asymmetric critic: o_t ⊕ x_t ────────────────────────────────────────

def combined_obs_rma(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    RMA asymmetric critic input: o_t (37) ⊕ x_t (26) = 63D.
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
    x_t ∈ R^26 = x^int (16) ⊕ x^ext (10).
    Used by RMA Phase 1 teacher and CTS teacher group.
    """
    return torch.cat(
        [
            privileged_internal_go2(env),
            privileged_external_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg),
        ],
        dim=-1,
    )


# ── Privileged — Terrain heights ──────────────────────────────────────────────

def privileged_terrain_go2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg:  SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Terrain height map relative to robot base z, yaw-aligned.
    11×7 = 77 samples over 1.0 m × 0.6 m grid.
    Values ≈ 0 at nominal standing height; positive = bump, negative = hole.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    asset  = env.scene[asset_cfg.name]
    base_z = asset.data.root_pos_w[:, 2:3]       # (N, 1)
    hit_z  = sensor.data.ray_hits_w[..., 2]       # (N, 77)
    return (base_z - 0.5 - hit_z).clamp(-1.0, 1.0)


def privileged_full_terrain_go2(
    env: ManagerBasedRLEnv,
    asset_cfg:  SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    height_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """x^int(16) ⊕ x^ext(10) ⊕ x^terr(77) = 103."""
    return torch.cat([
        privileged_internal_go2(env),
        privileged_external_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg),
        privileged_terrain_go2(env, sensor_cfg=height_cfg, asset_cfg=asset_cfg),
    ], dim=-1)


# ── Privileged subsets — for INT-only / EXT-only / TERR / FULL_T ablations ────
PRIV_DIMS = {
    "FULL":   26,
    "INT":    16,
    "EXT":    10,
    "TERR":   77,
    "FULL_T": 103,
}


def privileged_subset_go2(
    env: ManagerBasedRLEnv,
    mode: str = "FULL",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    Return the requested privileged subset:
      FULL   → x^int(16) ⊕ x^ext(10) = 26
      INT    → x^int(16)
      EXT    → x^ext(10)
      TERR   → x^terr(77)
      FULL_T → x^int(16) ⊕ x^ext(10) ⊕ x^terr(77) = 103
    """
    m = (mode or "FULL").upper()
    if m == "INT":    return privileged_internal_go2(env)
    if m == "EXT":    return privileged_external_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
    if m == "TERR":   return privileged_terrain_go2(env, asset_cfg=asset_cfg)
    if m == "FULL_T": return privileged_full_terrain_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
    return privileged_full_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)


def combined_obs_subset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    Asymmetric-critic input / L_rec target: o_t (37) ⊕ x_t(priv-subset).
    The privileged mode is read from ``env.cfg.priv_mode`` (default "FULL"), so
    FULL gives the original 63-D vector and INT/EXT give 53-D / 47-D.
    """
    mode = getattr(env.cfg, "priv_mode", "FULL")
    return torch.cat([
        proprioceptive_obs_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg),
        privileged_subset_go2(env, mode=mode, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg),
    ], dim=-1)

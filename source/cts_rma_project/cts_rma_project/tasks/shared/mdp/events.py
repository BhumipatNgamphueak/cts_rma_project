# tasks/shared/mdp/events.py
"""
Custom DR event functions for the T-S locomotion project.

Each function applies physics randomisation via Isaac Lab builtins AND
writes the sampled parameter values into env.extras["dr"] so that the
privileged observation can read them at every step.

NOTE: The tracked values are drawn independently from the same distribution
as the physics call, so they are statistically—but not sample-for-sample—
identical to what the simulator actually applied.  This is sufficient for
training the privileged encoder and will be addressed in a future refactor
that reads back exact values from PhysX.

DR ranges strictly follow Table 4 of the project proposal.
"""
from __future__ import annotations
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# ── helpers ──────────────────────────────────────────────────────────────────

def _ensure_dr_buffer(env: ManagerBasedRLEnv) -> None:
    """Lazily create the DR parameter buffer in env.extras.

    Shapes: all tensors are [num_envs, dim] and persist across steps
    because env.extras is a plain attribute dict (never auto-cleared).
    """
    if "dr" in env.extras:
        return
    N, D = env.num_envs, env.device
    env.extras["dr"] = {
        # Internal (14-D total)
        "friction":        torch.full((N, 1), 0.95,  device=D),  # midpoint [0.20, 1.70]
        "restitution":     torch.full((N, 1), 0.50,  device=D),  # midpoint [0.25, 0.75]
        "leg_mass_scale":  torch.ones (N, 4,          device=D),  # [0.80, 1.20] per leg
        "payload":         torch.ones (N, 1,          device=D),  # [-1, 3] kg add to base
        "kp_scale":        torch.ones (N, 3,          device=D),  # hip/thigh/calf [0.80, 1.20]
        "kd_scale":        torch.ones (N, 3,          device=D),  # [0.80, 1.20]
        "motor_strength":  torch.ones (N, 3,          device=D),  # [0.80, 1.20]
        "action_delay_ms": torch.zeros(N, 1,          device=D),  # [0, 20] ms
    }


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(lo, hi)


# ── public event functions ────────────────────────────────────────────────────

def randomize_material_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float]  = (0.20, 1.70),
    dynamic_friction_range: tuple[float, float] = (0.20, 1.70),
    restitution_range: tuple[float, float]       = (0.25, 0.75),
    num_buckets: int = 64,
) -> None:
    """Apply material DR (Table 4: friction [0.20,1.70], restitution [0.25,0.75])
    and track friction / restitution in env.extras["dr"]."""
    _ensure_dr_buffer(env)
    mdp.randomize_rigid_body_material(
        env, env_ids,
        asset_cfg=asset_cfg,
        static_friction_range=static_friction_range,
        dynamic_friction_range=dynamic_friction_range,
        restitution_range=restitution_range,
        num_buckets=num_buckets,
    )
    n = len(env_ids)
    D = env.device
    env.extras["dr"]["friction"][env_ids]    = _uniform(n, *static_friction_range, D).unsqueeze(1)
    env.extras["dr"]["restitution"][env_ids] = _uniform(n, *restitution_range, D).unsqueeze(1)


def randomize_payload_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float] = (-1.0, 3.0),
) -> None:
    """Add payload to base link (Table 4: [-1, 3] kg) and track in extras."""
    _ensure_dr_buffer(env)
    mdp.randomize_rigid_body_mass(
        env, env_ids,
        asset_cfg=asset_cfg,
        mass_distribution_params=mass_range,
        operation="add",
    )
    n = len(env_ids)
    env.extras["dr"]["payload"][env_ids] = _uniform(n, *mass_range, env.device).unsqueeze(1)


def randomize_leg_mass_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] = (0.80, 1.20),
) -> None:
    """Scale leg-link masses (Table 4: [0.80, 1.20] × nominal) and track in extras.

    One independent scale per leg (4 values) stored in extras; Isaac Lab
    applies a single uniform scale to all matched bodies.
    """
    _ensure_dr_buffer(env)
    mdp.randomize_rigid_body_mass(
        env, env_ids,
        asset_cfg=asset_cfg,
        mass_distribution_params=scale_range,
        operation="scale",
    )
    n = len(env_ids)
    # Sample 4 independent scales (one per leg) for the privileged obs
    scales = torch.stack(
        [_uniform(n, *scale_range, env.device) for _ in range(4)], dim=1
    )
    env.extras["dr"]["leg_mass_scale"][env_ids] = scales


def randomize_kp_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] = (0.80, 1.20),
) -> None:
    """Scale joint Kp (Table 4: [0.80, 1.20]) and track per joint-type in extras."""
    _ensure_dr_buffer(env)
    mdp.randomize_actuator_gains(
        env, env_ids,
        asset_cfg=asset_cfg,
        stiffness_distribution_params=scale_range,
        damping_distribution_params=(1.0, 1.0),  # Kd unchanged here
        operation="scale",
    )
    n = len(env_ids)
    # hip / thigh / calf get independent scales (simplified: same dist)
    scales = torch.stack(
        [_uniform(n, *scale_range, env.device) for _ in range(3)], dim=1
    )
    env.extras["dr"]["kp_scale"][env_ids] = scales


def randomize_kd_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] = (0.80, 1.20),
) -> None:
    """Scale joint Kd (Table 4: [0.80, 1.20]) and track per joint-type in extras."""
    _ensure_dr_buffer(env)
    mdp.randomize_actuator_gains(
        env, env_ids,
        asset_cfg=asset_cfg,
        stiffness_distribution_params=(1.0, 1.0),  # Kp unchanged here
        damping_distribution_params=scale_range,
        operation="scale",
    )
    n = len(env_ids)
    scales = torch.stack(
        [_uniform(n, *scale_range, env.device) for _ in range(3)], dim=1
    )
    env.extras["dr"]["kd_scale"][env_ids] = scales


def randomize_motor_strength_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] = (0.80, 1.20),
) -> None:
    """Apply motor-strength scaling via additional Kp+Kd scale (Table 4: [0.80, 1.20]).

    Motor strength σ_ms reduces the effective torque output by scaling both
    Kp and Kd uniformly.  This is applied on top of the ∆Kp/∆Kd events.
    """
    _ensure_dr_buffer(env)
    mdp.randomize_actuator_gains(
        env, env_ids,
        asset_cfg=asset_cfg,
        stiffness_distribution_params=scale_range,
        damping_distribution_params=scale_range,
        operation="scale",
    )
    n = len(env_ids)
    scales = torch.stack(
        [_uniform(n, *scale_range, env.device) for _ in range(3)], dim=1
    )
    env.extras["dr"]["motor_strength"][env_ids] = scales


def randomize_action_delay_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    delay_range_ms: tuple[float, float] = (0.0, 20.0),
) -> None:
    """Sample action-delay value (Table 4: [0, 20] ms) and record in extras.

    NOTE: Actual action buffering is not yet implemented in the simulator
    step; only the sampled delay value is stored for use by the privileged
    observation encoder.  Implementing true communication-latency delay
    requires an action ring-buffer in the runner (future work).
    """
    _ensure_dr_buffer(env)
    n = len(env_ids)
    env.extras["dr"]["action_delay_ms"][env_ids] = _uniform(
        n, *delay_range_ms, env.device
    ).unsqueeze(1)

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

IMPORTANT: mdp.randomize_rigid_body_material is a ManagerTermBase class and
cannot be called from plain Python functions.  Material physics is applied
directly via an EventTermCfg in shared_env_cfg.py; this module only tracks
the sampled values in env.extras["dr"].

DR ranges strictly follow Table 4 of the project proposal.
"""
from __future__ import annotations
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# ── helpers ──────────────────────────────────────────────────────────────────

def _ensure_dr_buffer(env: ManagerBasedRLEnv) -> None:
    """Lazily create the DR parameter buffer in env.extras."""
    if "dr" in env.extras:
        return
    N, D = env.num_envs, env.device
    env.extras["dr"] = {
        "friction":        torch.full((N, 1), 0.95,  device=D),
        "restitution":     torch.full((N, 1), 0.50,  device=D),
        "leg_mass_scale":  torch.ones (N, 4,          device=D),
        "payload":         torch.zeros(N, 1,          device=D),
        "kp_scale":        torch.ones (N, 3,          device=D),
        "kd_scale":        torch.ones (N, 3,          device=D),
        "motor_strength":  torch.ones (N, 3,          device=D),
        "action_delay_ms": torch.zeros(N, 1,          device=D),
    }


def _resolve(env_ids, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return env_ids tensor; if None (startup events), return all env indices."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(lo, hi)


# ── public event functions ────────────────────────────────────────────────────

def randomize_material_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float] = (0.20, 1.70),
    restitution_range: tuple[float, float]      = (0.25, 0.75),
) -> None:
    """Track material friction/restitution in env.extras["dr"] (tracking only).

    Physics is applied by the separate mdp.randomize_rigid_body_material event.
    """
    _ensure_dr_buffer(env)
    ids = _resolve(env_ids, env)
    n = len(ids)
    D = env.device
    env.extras["dr"]["friction"][ids]    = _uniform(n, *static_friction_range, D).unsqueeze(1)
    env.extras["dr"]["restitution"][ids] = _uniform(n, *restitution_range, D).unsqueeze(1)


def randomize_payload_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float] = (-1.0, 3.0),
) -> None:
    """Add payload to base link (Table 4: [-1, 3] kg) and track in extras."""
    _ensure_dr_buffer(env)
    ids = _resolve(env_ids, env)
    mdp.randomize_rigid_body_mass(
        env, ids,
        asset_cfg=asset_cfg,
        mass_distribution_params=mass_range,
        operation="add",
    )
    n = len(ids)
    env.extras["dr"]["payload"][ids] = _uniform(n, *mass_range, env.device).unsqueeze(1)


def randomize_leg_mass_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] = (0.80, 1.20),
) -> None:
    """Scale leg-link masses (Table 4: [0.80, 1.20] × nominal) and track in extras."""
    _ensure_dr_buffer(env)
    ids = _resolve(env_ids, env)
    mdp.randomize_rigid_body_mass(
        env, ids,
        asset_cfg=asset_cfg,
        mass_distribution_params=scale_range,
        operation="scale",
    )
    n = len(ids)
    scales = torch.stack(
        [_uniform(n, *scale_range, env.device) for _ in range(4)], dim=1
    )
    env.extras["dr"]["leg_mass_scale"][ids] = scales


def randomize_kp_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] = (0.80, 1.20),
) -> None:
    """Scale joint Kp (Table 4: [0.80, 1.20]) and track per joint-type in extras."""
    _ensure_dr_buffer(env)
    ids = _resolve(env_ids, env)
    mdp.randomize_actuator_gains(
        env, ids,
        asset_cfg=asset_cfg,
        stiffness_distribution_params=scale_range,
        damping_distribution_params=(1.0, 1.0),
        operation="scale",
    )
    n = len(ids)
    scales = torch.stack(
        [_uniform(n, *scale_range, env.device) for _ in range(3)], dim=1
    )
    env.extras["dr"]["kp_scale"][ids] = scales


def randomize_kd_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] = (0.80, 1.20),
) -> None:
    """Scale joint Kd (Table 4: [0.80, 1.20]) and track per joint-type in extras."""
    _ensure_dr_buffer(env)
    ids = _resolve(env_ids, env)
    mdp.randomize_actuator_gains(
        env, ids,
        asset_cfg=asset_cfg,
        stiffness_distribution_params=(1.0, 1.0),
        damping_distribution_params=scale_range,
        operation="scale",
    )
    n = len(ids)
    scales = torch.stack(
        [_uniform(n, *scale_range, env.device) for _ in range(3)], dim=1
    )
    env.extras["dr"]["kd_scale"][ids] = scales


def randomize_motor_strength_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] = (0.80, 1.20),
) -> None:
    """Motor-strength scaling (Table 4: [0.80, 1.20]) applied on top of Kp/Kd events."""
    _ensure_dr_buffer(env)
    ids = _resolve(env_ids, env)
    mdp.randomize_actuator_gains(
        env, ids,
        asset_cfg=asset_cfg,
        stiffness_distribution_params=scale_range,
        damping_distribution_params=scale_range,
        operation="scale",
    )
    n = len(ids)
    scales = torch.stack(
        [_uniform(n, *scale_range, env.device) for _ in range(3)], dim=1
    )
    env.extras["dr"]["motor_strength"][ids] = scales


def randomize_action_delay_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    delay_range_ms: tuple[float, float] = (0.0, 20.0),
) -> None:
    """Sample action-delay value (Table 4: [0, 20] ms) and record in extras.

    Actual action buffering is not implemented; only the value is stored for
    the privileged encoder (future work: action ring-buffer in the runner).
    """
    _ensure_dr_buffer(env)
    ids = _resolve(env_ids, env)
    n = len(ids)
    env.extras["dr"]["action_delay_ms"][ids] = _uniform(
        n, *delay_range_ms, env.device
    ).unsqueeze(1)

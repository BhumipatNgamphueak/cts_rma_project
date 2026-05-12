# tasks/shared/mdp/events.py
"""
Custom DR event functions — T-S locomotion project.

DR structure matches OpenTopic (unitree_rl_lab):
  - friction / restitution: re-randomised every episode (mode=reset)
  - base mass scale [0.9, 1.1]: reset, recompute_inertia=False (decoupled from inertia)
  - base inertia scale [0.8, 1.2]: independent of mass, reset
  - Kp scale [0.85, 1.15] / Kd scale [0.80, 1.20]: combined in one event, reset
  - COM offset ±0.05 m per axis: reset (cached to avoid compounding)
  - action delay [0, 20] ms: reset
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
        "friction":          torch.full((N, 1), 1.0,   device=D),  # mid of (0.5,1.5)
        "restitution":       torch.full((N, 1), 0.075, device=D),  # mid of (0.0,0.15)
        "payload":           torch.zeros(N, 1,          device=D),  # base mass (scale-1)
        "kp_scale":          torch.ones (N, 3,          device=D),  # raw scale, obs sends (s-1)
        "kd_scale":          torch.ones (N, 3,          device=D),  # raw scale, obs sends (s-1)
        "com_offset":        torch.zeros(N, 3,          device=D),  # metres
        "base_inertia_diag": torch.zeros(N, 3,          device=D),  # (scale-1) replicated
        "action_delay_ms":   torch.zeros(N, 1,          device=D),
    }


def _resolve(env_ids, env: ManagerBasedRLEnv) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(lo, hi)


# ── Material (friction / restitution) ────────────────────────────────────────

def track_material_from_physx(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Read actual material properties from PhysX after randomize_material applies them.

    Must be ordered AFTER mdp.randomize_rigid_body_material in SharedEventCfg.
    get_material_properties() → CPU tensor [N, num_shapes, 3]:
      col 0 = static_friction, col 1 = dynamic_friction, col 2 = restitution
    """
    _ensure_dr_buffer(env)
    ids   = _resolve(env_ids, env)
    asset = env.scene[asset_cfg.name]

    props = asset.root_physx_view.get_material_properties().to(env.device)  # [N, S, 3]
    env.extras["dr"]["friction"][ids]    = props[ids, :, 0].mean(dim=1, keepdim=True)
    env.extras["dr"]["restitution"][ids] = props[ids, :, 2].mean(dim=1, keepdim=True)


# ── Mass ─────────────────────────────────────────────────────────────────────

def randomize_payload_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_scale_range: tuple[float, float] = (0.9, 1.1),
) -> None:
    """Scale base-link mass [0.9, 1.1] × nominal. recompute_inertia=False so
    inertia DR (randomize_inertia_and_track) stays independent.

    Stores (scale - 1) in dr["payload"] so 0 = nominal.
    """
    _ensure_dr_buffer(env)
    ids = _resolve(env_ids, env)
    asset = env.scene[asset_cfg.name]

    mdp.randomize_rigid_body_mass(
        env, ids,
        asset_cfg=asset_cfg,
        mass_distribution_params=mass_scale_range,
        operation="scale",
        recompute_inertia=False,
    )

    actual_mass = asset.root_physx_view.get_masses().to(env.device)   # [N, B]
    body_ids    = asset_cfg.body_ids
    nom_mass    = asset.data.default_mass.to(env.device)              # [N, B]
    scale_actual = (
        actual_mass[ids][:, body_ids] /
        nom_mass[ids][:, body_ids].clamp(min=1e-8)
    ).mean(dim=1, keepdim=True)
    env.extras["dr"]["payload"][ids] = scale_actual - 1.0


# ── Inertia (independent of mass) ────────────────────────────────────────────

def randomize_inertia_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    scale_range: tuple[float, float] = (0.8, 1.2),
) -> None:
    """Scale base body inertia independently of mass [0.8, 1.2] × default.

    Mirrors OpenTopic's randomize_rigid_body_inertia_scale:
      - uses asset.data.default_inertia as the base each call (no compounding)
      - samples one scalar scale per env and applies to all 9 inertia components
    Stores (scale - 1) replicated to [N, 3] in dr["base_inertia_diag"].
    """
    _ensure_dr_buffer(env)
    ids   = _resolve(env_ids, env)
    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        ids_cpu = torch.arange(env.num_envs, device="cpu")
    else:
        ids_cpu = env_ids.cpu()

    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids_t = torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
    elif isinstance(body_ids, (list, tuple)):
        body_ids_t = torch.tensor(list(body_ids), dtype=torch.long, device="cpu")
    else:
        body_ids_t = torch.tensor([body_ids], dtype=torch.long, device="cpu")

    n_envs  = ids_cpu.shape[0]
    n_bodies = body_ids_t.shape[0]
    scales  = torch.empty(n_envs, n_bodies).uniform_(scale_range[0], scale_range[1])

    inertias = asset.root_physx_view.get_inertias()           # [N, B, 9] on CPU
    default  = asset.data.default_inertia.cpu()               # [N, B, 9]

    env_idx  = ids_cpu.view(-1, 1).expand(n_envs, n_bodies)
    body_idx = body_ids_t.view(1, -1).expand(n_envs, n_bodies)
    inertias[env_idx, body_idx] = default[env_idx, body_idx] * scales.unsqueeze(-1)
    asset.root_physx_view.set_inertias(inertias, ids_cpu)

    # Store (scale - 1) replicated over 3 axes; uses first body's scale if multiple
    scale_dev = (scales[:, 0] - 1.0).to(env.device)          # [n]
    env.extras["dr"]["base_inertia_diag"][ids] = scale_dev.unsqueeze(1).expand(-1, 3).clone()


# ── Actuator gains (Kp, Kd) ──────────────────────────────────────────────────

def randomize_gains_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    kp_scale_range: tuple[float, float] = (0.85, 1.15),
    kd_scale_range: tuple[float, float] = (0.80, 1.20),
) -> None:
    """Apply Kp and Kd scaling in ONE event (matches OpenTopic randomize_actuator_gains).

    Kp_final = nominal_Kp * kp_scale
    Kd_final = nominal_Kd * kd_scale

    Writes directly to actuator.stiffness / actuator.damping.
    Tracked as 3-element vectors (hip/thigh/calf) in dr["kp_scale"] / dr["kd_scale"]
    storing raw scale values; observations send (scale - 1) so 0 = nominal.
    """
    _ensure_dr_buffer(env)
    ids = _resolve(env_ids, env)
    n, D = len(ids), env.device
    asset = env.scene[asset_cfg.name]

    kp_s = _uniform(n, *kp_scale_range, D)  # [n]
    kd_s = _uniform(n, *kd_scale_range, D)  # [n]

    kp_eff = kp_s.unsqueeze(1)   # [n, 1] → broadcasts over joints
    kd_eff = kd_s.unsqueeze(1)

    nom_kp = asset.data.default_joint_stiffness.to(D)  # [N, n_joints]
    nom_kd = asset.data.default_joint_damping.to(D)    # [N, n_joints]

    for actuator in asset.actuators.values():
        if isinstance(actuator.joint_indices, slice):
            kp_act = nom_kp[ids] * kp_eff
            kd_act = nom_kd[ids] * kd_eff
        else:
            j = list(actuator.joint_indices)
            kp_act = nom_kp[ids][:, j] * kp_eff
            kd_act = nom_kd[ids][:, j] * kd_eff

        actuator.stiffness[ids] = kp_act
        actuator.damping[ids]   = kd_act

    env.extras["dr"]["kp_scale"][ids] = kp_s.unsqueeze(1).expand(-1, 3).clone()
    env.extras["dr"]["kd_scale"][ids] = kd_s.unsqueeze(1).expand(-1, 3).clone()


# ── COM offset ────────────────────────────────────────────────────────────────

def randomize_com_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    com_range: tuple[float, float] = (-0.05, 0.05),
) -> None:
    """Randomize base body COM offset (±0.05 m per axis) and track in extras.

    Caches default COMs on first call so reset-mode calls don't compound offsets
    (mirrors OpenTopic's randomize_body_com_offset caching pattern).
    """
    _ensure_dr_buffer(env)
    ids = _resolve(env_ids, env)
    n, D = len(ids), env.device
    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        ids_cpu = torch.arange(env.num_envs, device="cpu")
    else:
        ids_cpu = env_ids.cpu()

    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids_t = torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
    elif isinstance(body_ids, (list, tuple)):
        body_ids_t = torch.tensor(list(body_ids), dtype=torch.long, device="cpu")
    else:
        body_ids_t = torch.tensor([body_ids], dtype=torch.long, device="cpu")

    # Cache default COMs once so we always apply relative to default (no compounding)
    cache_key = (asset_cfg.name, "com_default")
    if not hasattr(env, "_dr_cache"):
        env._dr_cache = {}
    if cache_key not in env._dr_cache:
        env._dr_cache[cache_key] = asset.root_physx_view.get_coms().clone()
    default_coms = env._dr_cache[cache_key]

    com_offsets = torch.stack(
        [_uniform(n, *com_range, D) for _ in range(3)], dim=1
    )  # [n, 3]

    coms = asset.root_physx_view.get_coms()
    n_envs   = ids_cpu.shape[0]
    n_bodies = body_ids_t.shape[0]
    env_idx  = ids_cpu.view(-1, 1).expand(n_envs, n_bodies)
    body_idx = body_ids_t.view(1, -1).expand(n_envs, n_bodies)

    for ax in range(3):
        coms[env_idx, body_idx, ax] = (
            default_coms[env_idx, body_idx, ax] + com_offsets[:, ax].cpu().unsqueeze(1)
        )
    asset.root_physx_view.set_coms(coms, ids_cpu)

    env.extras["dr"]["com_offset"][ids] = com_offsets


# ── Action delay ─────────────────────────────────────────────────────────────

def randomize_action_delay_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    delay_range_ms: tuple[float, float] = (0.0, 20.0),
) -> None:
    """Sample action-delay [0, 20] ms and record in extras.

    Actual delay is not applied in the runner; the encoder learns to predict
    behaviour as if the delay were present.
    """
    _ensure_dr_buffer(env)
    ids = _resolve(env_ids, env)
    env.extras["dr"]["action_delay_ms"][ids] = _uniform(
        len(ids), *delay_range_ms, env.device
    ).unsqueeze(1)

# tasks/cts/mdp/observations.py
"""
CTS policy observation term for GO2.

cts_teacher_student_obs returns (N, H*37+1) — the unified teacher-student obs:
  Teacher envs (is_teacher=1): [ot(37), xt(26), zeros(H*37-63), flag=1]
  Student envs (is_teacher=0): [flat_history(H*37), flag=0]

This function is registered as the single term in the "policy" obs group so
observation_manager.compute() returns the full (N, 1851) obs directly, and
group_obs_dim["policy"] = (1851,) → num_obs = 1851 for the runner.

During ObservationManager._prepare_terms() the function is called once to
detect output shape. At that point Go2CTSEnv.is_teacher / obs_history are
not yet set (we're still inside super().__init__), so we return a zero dummy
of the correct shape. Real computation happens once training begins.
"""
from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ...shared.mdp import proprioceptive_obs_go2, privileged_subset_go2

_OBS_DIM = 37   # ot — proprioceptive obs for GO2


def cts_teacher_student_obs(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Unified teacher-student policy obs: (N, H*37+1).

    Called once per observation_manager.compute() call (every env step).
    Returns a dummy zeros tensor during ObservationManager initialisation
    (before Go2CTSEnv finishes __init__) so the obs shape is detected correctly.
    """
    H = env.cfg.history_len
    N = env.num_envs

    # Shape-detection call during _prepare_terms — is_teacher not set yet
    if not hasattr(env, "is_teacher"):
        return torch.zeros(N, H * _OBS_DIM + 1, device=env.device)

    priv_mode = getattr(env.cfg, "priv_mode", "FULL")
    ot = proprioceptive_obs_go2(env, sensor_cfg=sensor_cfg)            # (N, 37)
    xt = privileged_subset_go2(env, mode=priv_mode, sensor_cfg=sensor_cfg)  # (N, priv_dim)
    teacher_in = _OBS_DIM + xt.shape[-1]   # 63 (FULL) / 53 (INT) / 47 (EXT)

    # Gaussian noise σ=0.2 rad/s on ang_vel_b (indices 24:27) — matches
    # Baseline/RMA GaussianNoiseCfg on ang_vel term; applied here because
    # enable_corruption=False on the CTS policy group (flag must not be noised).
    ot = ot.clone()
    ot[:, 24:27] += torch.randn_like(ot[:, 24:27]) * 0.2

    # Roll history: drop oldest, append current ot
    env.obs_history = torch.roll(env.obs_history, -1, dims=1)
    env.obs_history[:, -1, :] = ot

    unified = torch.zeros(N, H * _OBS_DIM, device=env.device)

    # Teacher envs: [ot(37), xt(priv_dim)] in the first `teacher_in` dims; rest stays zero
    teacher_input = torch.cat([ot, xt], dim=-1)  # (N, teacher_in)
    if env.is_teacher.any():
        unified[env.is_teacher, :teacher_in] = teacher_input[env.is_teacher]

    # Student envs: full flattened H×37 history
    if (~env.is_teacher).any():
        unified[~env.is_teacher] = env.obs_history[~env.is_teacher].reshape(
            -1, H * _OBS_DIM)

    # Append is_teacher flag — never corrupted by obs noise (flag drives routing)
    flag = env.is_teacher.float().unsqueeze(-1)   # (N, 1)
    return torch.cat([unified, flag], dim=-1)      # (N, H*37+1 = 1851)

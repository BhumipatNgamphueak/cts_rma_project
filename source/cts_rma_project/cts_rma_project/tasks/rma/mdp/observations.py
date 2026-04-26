# tasks/rma/mdp/observations.py
from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def base_state_rma(env: ManagerBasedRLEnv,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                   sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")) -> torch.Tensor:
    """
    RMA state x_t: 30-dimensional
    joint_pos(12) + joint_vel(12) + roll_pitch(2) + foot_contact(4) = 30
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]

    joint_pos = asset.data.joint_pos          # [N, 12]
    joint_vel = asset.data.joint_vel          # [N, 12]

    # roll, pitch from IMU (gravity projection approximation)
    gravity_b = asset.data.projected_gravity_b  # [N, 3]
    roll  = torch.atan2(gravity_b[:, 1], gravity_b[:, 2]).unsqueeze(1)  # [N, 1]
    pitch = torch.atan2(-gravity_b[:, 0],
                         torch.sqrt(gravity_b[:, 1]**2 + gravity_b[:, 2]**2)).unsqueeze(1)  # [N, 1]

    # Binary foot contact (4 feet)
    foot_contact = (contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2] > 1.0).float()  # [N, 4]

    return torch.cat([joint_pos, joint_vel, roll, pitch, foot_contact], dim=-1)  # [N, 30]


def privileged_env_factors(env: ManagerBasedRLEnv,
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    RMA environment factors e_t: 17-dimensional
    Matches paper exactly:
      base_mass_offset(1) + com_offset(3) + motor_strength(12) + friction(1) = 17

    These are randomized each episode and stored in env extras by the EventCfg.
    We read them from env.extras set by domain randomization events.
    """
    # These are populated by EventCfg randomization events (see rma_env_cfg.py)
    # Fall back to zeros if not set yet (first step before reset)
    N = env.num_envs
    device = env.device

    mass_offset    = env.extras.get("mass_offset",    torch.zeros(N, 1,  device=device))
    com_offset     = env.extras.get("com_offset",     torch.zeros(N, 3,  device=device))
    motor_strength = env.extras.get("motor_strength", torch.ones (N, 12, device=device))
    friction       = env.extras.get("friction",       torch.ones (N, 1,  device=device) * 0.8)

    return torch.cat([mass_offset, com_offset, motor_strength, friction], dim=-1)  # [N, 17]
# tasks/cts/mdp/observations.py
from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def base_state_cts(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    CTS proprioceptive state: 37D
      joint_pos_rel(12) + joint_vel(12) + ang_vel_b(3) + gravity_b(3)
      + vel_cmd(3) + foot_contact(4) = 37

    Unlike RMA, the velocity command is embedded directly in the observation
    so the policy can close the loop without a separate encoder.
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene["contact_forces"]

    # Joint positions relative to default standing pose
    joint_pos = asset.data.joint_pos - asset.data.default_joint_pos  # [N, 12]
    joint_vel = asset.data.joint_vel                                  # [N, 12]

    # IMU-equivalent signals
    ang_vel_b = asset.data.root_ang_vel_b                             # [N, 3]
    gravity_b = asset.data.projected_gravity_b                        # [N, 3]

    # Velocity command from the command manager [vx, vy, wz]
    vel_cmd = env.command_manager.get_command("base_velocity")[:, :3]  # [N, 3]

    # Binary foot contact (4 feet); threshold = 1 N
    foot_contact = (
        contact_sensor.data.net_forces_w_history[:, 0, :, 2] > 1.0
    ).float()  # [N, 4]

    return torch.cat(
        [joint_pos, joint_vel, ang_vel_b, gravity_b, vel_cmd, foot_contact], dim=-1
    )  # [N, 37]

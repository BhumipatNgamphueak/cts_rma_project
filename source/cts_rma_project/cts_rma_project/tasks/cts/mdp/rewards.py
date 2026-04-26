# tasks/cts/mdp/rewards.py
# CTS uses curriculum-friendly rewards that scale well across velocity levels.
from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for tracking 2D linear velocity command (vx, vy)."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command("base_velocity")[:, :2]   # [N, 2]
    vel_b = asset.data.root_lin_vel_b[:, :2]                        # [N, 2]
    error = torch.sum(torch.square(cmd - vel_b), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for tracking angular velocity command around z-axis."""
    asset = env.scene[asset_cfg.name]
    cmd_wz = env.command_manager.get_command("base_velocity")[:, 2]  # [N]
    ang_vel_z = asset.data.root_ang_vel_b[:, 2]                      # [N]
    return torch.exp(-torch.square(cmd_wz - ang_vel_z) / std**2)


def penalize_foot_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot velocity when the foot is in ground contact (slip)."""
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]

    # Binary contact flag per foot
    in_contact = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2] > 1.0  # [N, 4]
    foot_speed = asset.data.joint_vel[:, [2, 5, 8, 11]].abs()  # [N, 4] per-foot calf speed
    return torch.sum(in_contact.float() * foot_speed, dim=1)


def penalize_joint_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joints that exceed soft position limits."""
    asset = env.scene[asset_cfg.name]
    lower_violation = -(
        asset.data.joint_pos - asset.data.soft_joint_pos_limits[:, :, 0]
    ).clip(max=0.0)
    upper_violation = (
        asset.data.joint_pos - asset.data.soft_joint_pos_limits[:, :, 1]
    ).clip(min=0.0)
    return torch.sum(lower_violation + upper_violation, dim=1)

# tasks/shared/mdp/rewards.py
"""
Shared reward functions used by ALL three methods (Baseline, RMA, CTS).

GO2 velocity-tracking adaptation of the paper's reward:
  R = r_track_lin + r_track_ang + r_smooth + r_torque + r_stable

The transfer mechanism (Baseline / RMA / CTS) is the sole experimental
variable; reward and DR are identical across methods.
"""
from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-||v_cmd_xy − v_xy||² / std²) — tracks forward and lateral velocity."""
    asset = env.scene[asset_cfg.name]
    cmd   = env.command_manager.get_command("base_velocity")[:, :2]
    err   = torch.sum(torch.square(cmd - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-err / std ** 2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-||ω_cmd_z − ω_z||² / std²) — tracks yaw rate."""
    asset  = env.scene[asset_cfg.name]
    cmd_wz = env.command_manager.get_command("base_velocity")[:, 2]
    err    = torch.square(cmd_wz - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-err / std ** 2)


def penalize_ang_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """−||ω_xy||² — penalise roll and pitch angular rate (stability)."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def penalize_lin_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """−v_z² — penalise vertical body velocity (stability)."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def penalize_foot_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg  = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """−Σ contact_i × |calf_vel_i| — penalise foot sliding during stance."""
    asset   = env.scene[asset_cfg.name]
    contact = env.scene[sensor_cfg.name]
    in_contact = contact.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2] > 1.0  # [N, 4]
    calf_speed = asset.data.joint_vel[:, [2, 5, 8, 11]].abs()           # [N, 4]
    return torch.sum(in_contact.float() * calf_speed, dim=1)

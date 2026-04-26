# tasks/rma/mdp/rewards.py
# RMA uses bioenergetics-inspired rewards (from paper Section III-A)

from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def track_lin_vel_x_exp(env: ManagerBasedRLEnv, std: float = 0.25,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward forward velocity tracking (RMA uses max(v_x, 0.35) but we use exp-tracking)."""
    asset = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command("base_velocity")[:, :2]
                     - asset.data.root_lin_vel_b[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def penalize_lateral_motion(env: ManagerBasedRLEnv,
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize lateral and yaw motion."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, 1:2]), dim=1)


def penalize_work(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize mechanical work: |tau^T * delta_q| (bioenergetics reward from RMA paper)."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1
    )


def penalize_ground_impact(env: ManagerBasedRLEnv,
                            sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")) -> torch.Tensor:
    """Penalize large changes in ground reaction forces."""
    contact_sensor = env.scene[sensor_cfg.name]
    # difference in net contact forces between steps
    return torch.sum(
        torch.norm(contact_sensor.data.net_forces_w_history[:, 0] -
                   contact_sensor.data.net_forces_w_history[:, 1], dim=-1), dim=1
    )


def penalize_torque_smoothness(env: ManagerBasedRLEnv,
                                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize torque jerk (smoothness term from RMA paper)."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(
        asset.data.applied_torque - env.reward_manager._episode_sums.get("prev_torque",
        torch.zeros_like(asset.data.applied_torque))
    ), dim=1)


def penalize_foot_slip(env: ManagerBasedRLEnv,
                        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize foot velocity when foot is in contact (foot slip)."""
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]
    # Binary foot contact: True if contact force > threshold
    contact = contact_sensor.data.net_forces_w_history[:, 0, :, 2] > 1.0  # [N, 4]
    # Foot velocities (approximate from body velocity — proper impl needs foot frames)
    # Using joint vel proxy for now; replace with actual foot frame velocities if available
    foot_vel = torch.norm(asset.data.joint_vel[:, [2, 5, 8, 11]], dim=-1, keepdim=True)  # calf joints
    return torch.sum(contact.float() * foot_vel.squeeze(-1), dim=1)
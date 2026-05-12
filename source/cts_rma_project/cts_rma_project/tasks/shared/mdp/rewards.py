# tasks/shared/mdp/rewards.py
"""
Shared reward functions used by ALL three methods (Baseline, RMA, CTS).
Reward structure matches the TXL locomotion task.
"""
from __future__ import annotations
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize energy: |joint_vel| * |applied_torque| summed over joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def joint_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize deviation from default joint positions; larger penalty when standing still."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
    return torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        reward,
        stand_still_scale * reward,
    )


def air_time_variance_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize variance in per-foot air time and contact time — encourages symmetric gait."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if not contact_sensor.cfg.track_air_time:
        raise RuntimeError("ContactSensor must have track_air_time=True")
    last_air_time     = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return (
        torch.var(torch.clip(last_air_time,     max=0.5), dim=1)
        + torch.var(torch.clip(last_contact_time, max=0.5), dim=1)
    )


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding on the ground (contact * foot_xy_velocity)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1).max(dim=1)[0] > 1.0
    )
    asset: RigidObject = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

# tasks/shared/mdp/rewards.py
"""
Shared reward functions used by ALL three methods (Baseline, RMA, CTS).

Tuned to match Isaac Lab's official UnitreeGo2FlatEnv reference config.
"""
from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward feet that spend time in the air — encourages lifting feet and stepping.

    Ported from isaaclab_tasks locomotion velocity mdp (GO2 flat env weight = 0.25).
    Zero reward when velocity command is near zero (standing still is fine).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

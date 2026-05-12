# tasks/shared/mdp/curriculums.py
"""
Velocity command curriculum for flat-ground training.

Expands command ranges by 0.1 m/s toward limits each time the mean
per-second tracking reward exceeds 80% of the reward weight.
Fires at every episode boundary (common_step_counter % max_episode_length == 0).
"""
from __future__ import annotations
from collections.abc import Sequence
import torch
from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    limit_vel_x: tuple[float, float] = (-1.0, 1.0),
    limit_vel_y: tuple[float, float] = (-1.0, 1.0),
    limit_ang_z: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Expand linear + angular velocity command ranges based on tracking performance.

    At every episode boundary: if mean episode tracking reward / episode_length_s
    > 80% of the reward weight, widen lin_vel_x, lin_vel_y, and ang_vel_z by
    0.1 m/s (or rad/s) toward their limits.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges

    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward_per_step = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length   # divide by steps, not seconds
    )
    threshold = reward_cfg.weight * 0.8

    if env.common_step_counter % env.max_episode_length == 0 and reward_per_step > threshold:
        delta = torch.tensor([-0.1, 0.1], device=env.device)

        ranges.lin_vel_x = torch.clamp(
            torch.tensor(list(ranges.lin_vel_x), device=env.device) + delta,
            min=limit_vel_x[0], max=limit_vel_x[1],
        ).tolist()

        ranges.lin_vel_y = torch.clamp(
            torch.tensor(list(ranges.lin_vel_y), device=env.device) + delta,
            min=limit_vel_y[0], max=limit_vel_y[1],
        ).tolist()

        ranges.ang_vel_z = torch.clamp(
            torch.tensor(list(ranges.ang_vel_z), device=env.device) + delta,
            min=limit_ang_z[0], max=limit_ang_z[1],
        ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)

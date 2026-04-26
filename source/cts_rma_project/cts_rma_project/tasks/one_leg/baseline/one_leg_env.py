# tasks/one_leg/baseline/one_leg_env.py
"""
One-legged hopper — Baseline environment.

Proprioceptive-only policy (14-D). Trains on point-to-point goal reaching
along a linear rail. Serves as the control condition against RMA and CTS.

Obs (14-D):
    dof_pos(3) + dof_vel(3) + prev_actions(3) + robot_pos_x(1)
    + C_frc(1) + C_vel(1) + foot_contact(1) + cmd_distance(1)
"""
from __future__ import annotations
import math
import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor

from .one_leg_env_cfg import OneLegEnvCfg


def _sigmoid(x: torch.Tensor, kappa: float) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-kappa * x))


class OneLegBaselineEnv(DirectRLEnv):
    """One-legged hopper baseline — proprioceptive policy only."""

    cfg: OneLegEnvCfg

    def __init__(self, cfg: OneLegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── Actuated joint indices ────────────────────────────────────────
        self.actuated_dof_indices = sorted([
            self.robot.joint_names.index(n) for n in cfg.actuated_joint_names
        ])
        self.num_actuated = len(self.actuated_dof_indices)

        # ── Body / joint lookup ───────────────────────────────────────────
        self._linear_joint_idx = self.robot.find_joints("linear_left_right")[0]
        self._linear_body_idx  = self.robot.find_bodies("linear_left_right_Link")[0][0]
        self._ee_body_idx      = self.robot.find_bodies("end_effector")[0][0]

        # ── Normalization limits ──────────────────────────────────────────
        self.dof_pos_limits = torch.full(
            (self.num_envs, self.num_actuated), math.pi, device=self.device)
        self.dof_vel_limits = torch.full(
            (self.num_envs, self.num_actuated), math.pi, device=self.device)

        # ── Action buffers ────────────────────────────────────────────────
        self.actions           = torch.zeros(self.num_envs, self.num_actuated, device=self.device)
        self.prev_actions      = torch.zeros_like(self.actions)
        self.prev_prev_actions = torch.zeros_like(self.actions)

        # ── Contact state ─────────────────────────────────────────────────
        self.foot_contact_force  = torch.zeros(self.num_envs, device=self.device)
        self.is_foot_in_contact  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ── Phase clock ───────────────────────────────────────────────────
        self.cycle_time  = torch.full((self.num_envs,), 0.75, device=self.device)
        self.phase_time  = torch.zeros(self.num_envs, device=self.device)
        self.C_frc       = torch.zeros(self.num_envs, device=self.device)
        self.C_vel       = torch.zeros(self.num_envs, device=self.device)
        self.phase_value = torch.zeros(self.num_envs, device=self.device)
        self.stance_ratio = 0.4

        # ── Robot state ───────────────────────────────────────────────────
        self.default_joint     = self.robot.data.default_joint_pos.clone()
        self.robot_pos_x       = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_robot_pos_x  = torch.zeros_like(self.robot_pos_x)

        # ── Goal / command ────────────────────────────────────────────────
        self.target_x         = torch.zeros(self.num_envs, 1, device=self.device)
        self.commands         = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_obs      = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_command_obs = torch.zeros(self.num_envs, 1, device=self.device)
        self.reached_goal     = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.foot_height      = torch.zeros(self.num_envs, device=self.device)

        # ── Curriculum ───────────────────────────────────────────────────
        self.curriculum_level  = 0
        self.curriculum_goals  = [1.0, 2.0, 3.0, 5.0]
        self.success_rate_buf  = torch.zeros(100, device=self.device)
        self.success_idx       = 0

        # ── DR state (overridden by RMA/CTS subclasses) ───────────────────
        self.stiffness_scale = torch.ones(self.num_envs, device=self.device)
        self.damping_scale   = torch.ones(self.num_envs, device=self.device)

        # ── Goal visualizer ───────────────────────────────────────────────
        marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
        marker_cfg.markers["arrow"].scale = (0.2, 0.2, 0.2)
        self.goal_visualizer = VisualizationMarkers(marker_cfg)
        self.goal_visualizer.set_visibility(True)

        # ── Episode reward logging ────────────────────────────────────────
        self._ep_rew = {k: torch.zeros(self.num_envs, device=self.device)
                        for k in ["progress", "phase", "distance", "stand", "reg", "total"]}

    # ══════════════════════════════════════════════════════════════════════
    # Scene
    # ══════════════════════════════════════════════════════════════════════
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.contact_sensor = ContactSensor(self.cfg.contact_force)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
        )

    # ══════════════════════════════════════════════════════════════════════
    # Physics step
    # ══════════════════════════════════════════════════════════════════════
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_prev_actions = self.prev_actions.clone()
        self.prev_actions      = self.actions.clone()
        self.actions           = actions.clone()
        self.phase_time       += self.step_dt

    def _apply_action(self):
        self.robot.set_joint_position_target(
            self.actions, joint_ids=self.actuated_dof_indices)

    # ══════════════════════════════════════════════════════════════════════
    # Phase clock
    # ══════════════════════════════════════════════════════════════════════
    def _compute_phase_signals(self):
        kappa = 50.0
        r     = self.stance_ratio
        phase = (self.phase_time % self.cycle_time) / self.cycle_time

        sp = _sigmoid(phase,           kappa)
        ep = _sigmoid(phase - r,       kappa)
        wa = _sigmoid(phase - 1.0,     kappa) - _sigmoid(phase - 1.0 - r, kappa)

        self.C_vel       = -torch.clamp(sp - ep + wa, 0.0, 1.0)
        self.C_frc       = (1.0 + self.C_vel) * -1.0
        self.phase_value = phase

    # ══════════════════════════════════════════════════════════════════════
    # Contact
    # ══════════════════════════════════════════════════════════════════════
    def _update_contact(self):
        net_forces = self.scene["contact_sensor"].data.net_forces_w
        force_mag  = torch.norm(net_forces, dim=-1).squeeze(-1)
        self.foot_contact_force = 0.8 * force_mag + 0.2 * self.foot_contact_force
        self.is_foot_in_contact = self.foot_contact_force > 1.0

    # ══════════════════════════════════════════════════════════════════════
    # Observations
    # ══════════════════════════════════════════════════════════════════════
    def _get_observations(self) -> dict:
        self._update_contact()
        self._compute_phase_signals()

        dof_pos = self.robot.data.joint_pos[:, self.actuated_dof_indices]
        dof_vel = self.robot.data.joint_vel[:, self.actuated_dof_indices]

        self.prev_robot_pos_x  = self.robot_pos_x.clone()
        self.robot_pos_x       = self.robot.data.joint_pos[:, self._linear_joint_idx]
        self.prev_command_obs  = self.command_obs.clone()
        body_x                 = self.robot.data.body_link_pos_w[:, self._linear_body_idx, 0:1]
        self.command_obs       = self.commands[:, 0:1] - body_x

        dist_to_goal     = torch.abs(self.command_obs).squeeze(-1)
        self.reached_goal = dist_to_goal < 0.10
        self.foot_height  = self.robot.data.body_link_pos_w[:, self._ee_body_idx, 2]

        # Zero phase signals when at goal
        self.C_frc = torch.where(self.reached_goal, torch.zeros_like(self.C_frc), self.C_frc)
        self.C_vel = torch.where(self.reached_goal, torch.zeros_like(self.C_vel), self.C_vel)

        policy_obs = torch.cat([
            dof_pos,                                       # (N, 3)
            dof_vel,                                       # (N, 3)
            self.prev_actions,                             # (N, 3)
            self.robot_pos_x,                              # (N, 1)
            self.C_frc.unsqueeze(-1),                      # (N, 1)
            self.C_vel.unsqueeze(-1),                      # (N, 1)
            self.is_foot_in_contact.float().unsqueeze(-1), # (N, 1)
            self.command_obs,                              # (N, 1)
        ], dim=-1)  # → (N, 14)

        return {"policy": policy_obs}

    # ══════════════════════════════════════════════════════════════════════
    # Rewards
    # ══════════════════════════════════════════════════════════════════════
    def _get_rewards(self) -> torch.Tensor:
        # 1. Progress
        r_progress = torch.clamp(
            (torch.abs(self.prev_command_obs) - torch.abs(self.command_obs)).squeeze(-1) * 100.0,
            -1.0, 2.0,
        )

        # 2. Phase: stance contact + swing clearance
        contact_reward = 1.0 - torch.exp(-0.01 * self.foot_contact_force)
        foot_clearance = torch.clamp(self.foot_height - 0.02, min=0.0)
        swing_quality  = 1.0 - torch.exp(-20.0 * foot_clearance)
        tip_vel_x      = self.robot.data.body_lin_vel_w[:, self._ee_body_idx, 0]
        goal_direction = torch.sign(self.command_obs).squeeze(-1)
        fwd_vel        = torch.clamp(tip_vel_x * goal_direction * 0.5, -0.2, 0.5)
        swing_reward   = 0.7 * swing_quality + 0.3 * torch.clamp(fwd_vel, min=0.0)
        r_phase        = (-self.C_frc) * contact_reward + (-self.C_vel) * swing_reward

        # 3. Distance
        r_distance = torch.exp(-2.0 * torch.abs(self.command_obs).squeeze(-1))

        # 4. Stand at goal
        joint_err = torch.sum(
            (self.robot.data.joint_pos[:, self.actuated_dof_indices]
             - self.default_joint[:, self.actuated_dof_indices]) ** 2, dim=1)
        r_stand = torch.where(self.reached_goal, torch.exp(-5.0 * joint_err),
                               torch.zeros_like(joint_err))

        # 5. Regularization
        torques_sq  = torch.sum(self.robot.data.applied_torque[:, self.actuated_dof_indices] ** 2, dim=1)
        accels      = torch.sum(self.robot.data.joint_acc[:, self.actuated_dof_indices] ** 2, dim=1)
        action_rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=1)
        action_jerk = torch.sum(
            (self.actions - 2.0 * self.prev_actions + self.prev_prev_actions) ** 2, dim=1)
        torque      = self.robot.data.applied_torque[:, self.actuated_dof_indices]
        joint_vel   = self.robot.data.joint_vel[:, self.actuated_dof_indices]
        energy      = torch.sum(torch.abs(torque * joint_vel), dim=1)

        r_reg = (-0.05 * action_rate - 0.02 * action_jerk
                 - 2.5e-5 * torques_sq - 6.0e-7 * accels - 2.5e-5 * energy)

        # Combine
        loco_scale = torch.where(self.reached_goal,
                                  torch.tensor(0.1, device=self.device),
                                  torch.tensor(1.0, device=self.device))
        reward = (2.0 * r_progress * loco_scale
                  + 1.5 * r_phase   * loco_scale
                  + 1.0 * r_distance
                  + 1.0 * r_stand
                  + 1.0 * r_reg)

        for k, v in zip(["progress", "phase", "distance", "stand", "reg", "total"],
                         [r_progress, r_phase, r_distance, r_stand, r_reg, reward]):
            self._ep_rew[k] += v
        return reward

    # ══════════════════════════════════════════════════════════════════════
    # Terminations
    # ══════════════════════════════════════════════════════════════════════
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        died     = torch.zeros_like(time_out)

        done_envs = time_out | died
        if done_envs.any():
            reached = self.reached_goal[done_envs].float().mean()
            self.success_rate_buf[self.success_idx % 100] = reached
            self.success_idx += 1
            if self.success_idx >= 100:
                avg = self.success_rate_buf.mean()
                if avg > 0.7 and self.curriculum_level < len(self.curriculum_goals) - 1:
                    self.curriculum_level += 1

        log = {k: self._ep_rew[k].mean() for k in self._ep_rew}
        log["curriculum_level"] = float(self.curriculum_level)
        self.extras["log"] = log
        for v in self._ep_rew.values():
            v.zero_()

        return died, time_out

    # ══════════════════════════════════════════════════════════════════════
    # Reset
    # ══════════════════════════════════════════════════════════════════════
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        n = len(env_ids)

        self.actions[env_ids]           = 0.0
        self.prev_actions[env_ids]      = 0.0
        self.prev_prev_actions[env_ids] = 0.0

        # Perturbed joint positions
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        noise     = torch.empty(n, self.robot.num_joints, device=self.device).uniform_(-0.05, 0.05)
        joint_pos = torch.clamp(
            joint_pos + noise,
            self.robot.data.joint_pos_limits[env_ids, :, 0],
            self.robot.data.joint_pos_limits[env_ids, :, 1],
        )
        joint_vel = torch.zeros(n, self.robot.num_joints, device=self.device)

        default_root = self.robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Phase clock
        self.cycle_time[env_ids] = 0.6
        self.phase_time[env_ids] = 0.0

        # Actuator DR (Baseline: nominal; RMA/CTS override this)
        self._reset_dr(env_ids)

        # Goal
        raw = torch.empty(n, 1, device=self.device).uniform_(-5.0, 5.0)
        sgn = torch.sign(raw)
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        raw = torch.where(torch.abs(raw) < 0.3, sgn * 0.3, raw)
        self.target_x[env_ids] = raw

        self.commands[env_ids, 0] = (self.scene.env_origins[env_ids, 0]
                                     + self.target_x[env_ids].squeeze(-1) - 0.07444)
        self.commands[env_ids, 1] = self.scene.env_origins[env_ids, 1] - 0.43475
        self.commands[env_ids, 2] = 0.6

        body_x = self.robot.data.body_link_pos_w[env_ids, self._linear_body_idx, 0:1]
        self.command_obs[env_ids]      = self.commands[env_ids, 0:1] - body_x
        self.prev_command_obs[env_ids] = self.command_obs[env_ids].clone()
        self.robot_pos_x[env_ids]      = self.robot.data.joint_pos[env_ids][:, self._linear_joint_idx]

        self.goal_visualizer.visualize(self.commands)

    def _reset_dr(self, env_ids: torch.Tensor):
        """Domain randomisation hook — override in RMA/CTS subclasses."""
        pass

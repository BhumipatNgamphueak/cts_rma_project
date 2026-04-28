# tasks/one_leg/baseline/one_leg_env.py
"""
One-legged hopper — Baseline environment (paper Section 4.1 / Table 4).

Obs (15-D):
    qt(3) + q̇t(3) + at-1(3) + p_ref_foot(3) + ct(1) + sinφt(1) + cosφt(1)

Full Table 4 DR applied identically across Baseline / RMA / CTS (paper Section 6.2).
Baseline sees only ot; RMA/CTS additionally receive privileged xt.
"""
from __future__ import annotations
import math
import torch

_NOM_KP = 20.0
_NOM_KD = 1.0

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

        # ── DR state buffers (Table 4 — shared by Baseline / RMA / CTS) ────
        N = self.num_envs
        self.dr_friction     = torch.ones(N, device=self.device)
        self.dr_restitution  = torch.full((N,), 0.5, device=self.device)
        self.dr_mass_scale   = torch.ones(N, 4, device=self.device)
        self.dr_payload      = torch.zeros(N, device=self.device)
        self.dr_kp_scale     = torch.ones(N, 3, device=self.device)
        self.dr_kd_scale     = torch.ones(N, 3, device=self.device)
        self.dr_motor_str    = torch.ones(N, 3, device=self.device)
        self.dr_action_delay = torch.zeros(N, dtype=torch.long, device=self.device)
        self.dr_com_offset   = torch.zeros(N, 3, device=self.device)

        # Action delay ring buffer: 3 slots for 0–2 step delays at 100 Hz
        self._action_buf = torch.zeros(N, 3, self.num_actuated, device=self.device)
        self._buf_slot   = 0

        # ── Push-force disturbance buffers ────────────────────────────────
        self.dr_push_force   = torch.zeros(N, 3, device=self.device)   # current (N,3)
        self.push_countdown  = torch.zeros(N, dtype=torch.long, device=self.device)

        # Body indices for leg-link mass / CoM DR
        self._leg_body_ids = [
            self.robot.find_bodies(name)[0][0] for name in cfg.leg_link_names
        ]

        # ── Goal visualizer ───────────────────────────────────────────────
        marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
        marker_cfg.markers["arrow"].scale = (0.2, 0.2, 0.2)
        self.goal_visualizer = VisualizationMarkers(marker_cfg)
        self.goal_visualizer.set_visibility(True)

        # ── Episode reward logging ────────────────────────────────────────
        self._ep_rew = {k: torch.zeros(self.num_envs, device=self.device)
                        for k in ["track", "contact", "clear",
                                  "progress", "phase", "stand", "reg", "total"]}

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

        # ── Resample push force when countdown expires ────────────────────
        expired = self.push_countdown <= 0
        if expired.any():
            n_exp = int(expired.sum().item())
            angle = torch.empty(n_exp, device=self.device).uniform_(0.0, 2.0 * math.pi)
            mag   = torch.empty(n_exp, device=self.device).uniform_(
                0.0, self.cfg.push_force_max)
            self.dr_push_force[expired, 0] = mag * torch.cos(angle)
            self.dr_push_force[expired, 1] = mag * torch.sin(angle)
            self.dr_push_force[expired, 2] = 0.0
            self.push_countdown[expired] = torch.randint(
                self.cfg.push_interval_min, self.cfg.push_interval_max + 1,
                (n_exp,), device=self.device)
        self.push_countdown -= 1

    def _apply_action(self):
        N      = self.num_envs
        arange = torch.arange(N, device=self.device)
        slot   = self._buf_slot % 3

        self._action_buf[arange, slot] = self.actions * self.dr_motor_str

        read_slot = (self._buf_slot - self.dr_action_delay) % 3
        delayed   = self._action_buf[arange, read_slot]   # (N, 3)

        self._buf_slot = (self._buf_slot + 1) % 3
        self.robot.set_joint_position_target(delayed, joint_ids=self.actuated_dof_indices)

        # Apply push-force disturbance to the main body every step
        forces  = self.dr_push_force.unsqueeze(1)          # (N, 1, 3)
        torques = torch.zeros_like(forces)
        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            body_ids=[self._linear_body_idx],
        )

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

        dist_to_goal      = torch.abs(self.command_obs).squeeze(-1)
        self.reached_goal = dist_to_goal < 0.10
        self.foot_height  = self.robot.data.body_link_pos_w[:, self._ee_body_idx, 2]

        # Zero phase signals when at goal
        self.C_frc = torch.where(self.reached_goal, torch.zeros_like(self.C_frc), self.C_frc)
        self.C_vel = torch.where(self.reached_goal, torch.zeros_like(self.C_vel), self.C_vel)

        # p_ref_foot: goal foot position in local (env-origin-relative) frame — Table 1
        p_ref_foot = self.commands[:, :3] - self.scene.env_origins   # (N, 3)

        # Phase encoding: sinφ, cosφ  (φ normalised to [0,1))
        phase     = (self.phase_time % self.cycle_time) / self.cycle_time
        sin_phase = torch.sin(2.0 * math.pi * phase)                 # (N,)
        cos_phase = torch.cos(2.0 * math.pi * phase)                 # (N,)

        policy_obs = torch.cat([
            dof_pos,                                        # (N, 3)  qt
            dof_vel,                                        # (N, 3)  q̇t
            self.prev_actions,                              # (N, 3)  at-1
            p_ref_foot,                                     # (N, 3)  p_ref_foot
            self.is_foot_in_contact.float().unsqueeze(-1),  # (N, 1)  ct
            sin_phase.unsqueeze(-1),                        # (N, 1)  sinφt
            cos_phase.unsqueeze(-1),                        # (N, 1)  cosφt
        ], dim=-1)  # → (N, 15)

        return {"policy": policy_obs}

    # ══════════════════════════════════════════════════════════════════════
    # Rewards
    # ══════════════════════════════════════════════════════════════════════
    def _get_rewards(self) -> torch.Tensor:
        # ── Paper terms (Section 2.2) ─────────────────────────────────────

        # r_track = exp(-α‖p_ref_foot(φt) - p_foot,t‖²),  α=2
        p_foot    = self.robot.data.body_link_pos_w[:, self._ee_body_idx, :]  # (N, 3)
        track_err = torch.sum((self.commands[:, :3] - p_foot) ** 2, dim=-1)   # (N,)
        r_track   = torch.exp(-2.0 * track_err)

        # r_contact — binary: foot in contact during stance (φ ∈ [0, 0.5))
        is_stance = self.phase_value < 0.5
        r_contact = (is_stance & self.is_foot_in_contact).float()

        # r_clear — binary: foot height ≥ h_clear during swing (φ ∈ [0.5, 1.0))
        h_clear  = 0.05   # 5 cm
        r_clear  = (~is_stance & (self.foot_height >= h_clear)).float()

        # r_smooth and r_torque are covered by r_reg below

        # ── Current terms (kept as-is) ────────────────────────────────────

        # Progress: 1D goal approach
        r_progress = torch.clamp(
            (torch.abs(self.prev_command_obs) - torch.abs(self.command_obs)).squeeze(-1) * 100.0,
            -1.0, 2.0,
        )

        # Phase: continuous stance contact + swing clearance (smooth gating)
        contact_reward = 1.0 - torch.exp(-0.01 * self.foot_contact_force)
        foot_clearance = torch.clamp(self.foot_height - 0.02, min=0.0)
        swing_quality  = 1.0 - torch.exp(-20.0 * foot_clearance)
        tip_vel_x      = self.robot.data.body_lin_vel_w[:, self._ee_body_idx, 0]
        goal_direction = torch.sign(self.command_obs).squeeze(-1)
        fwd_vel        = torch.clamp(tip_vel_x * goal_direction * 0.5, -0.2, 0.5)
        swing_reward   = 0.7 * swing_quality + 0.3 * torch.clamp(fwd_vel, min=0.0)
        r_phase        = (-self.C_frc) * contact_reward + (-self.C_vel) * swing_reward

        # Stand at goal
        joint_err = torch.sum(
            (self.robot.data.joint_pos[:, self.actuated_dof_indices]
             - self.default_joint[:, self.actuated_dof_indices]) ** 2, dim=1)
        r_stand = torch.where(self.reached_goal, torch.exp(-5.0 * joint_err),
                               torch.zeros_like(joint_err))

        # Regularization (covers paper's r_smooth + r_torque)
        torques_sq  = torch.sum(self.robot.data.applied_torque[:, self.actuated_dof_indices] ** 2, dim=1)
        accels      = torch.sum(self.robot.data.joint_acc[:, self.actuated_dof_indices] ** 2, dim=1)
        action_rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=1)
        action_jerk = torch.sum(
            (self.actions - 2.0 * self.prev_actions + self.prev_prev_actions) ** 2, dim=1)
        torque    = self.robot.data.applied_torque[:, self.actuated_dof_indices]
        joint_vel = self.robot.data.joint_vel[:, self.actuated_dof_indices]
        energy    = torch.sum(torch.abs(torque * joint_vel), dim=1)

        r_reg = (-0.05 * action_rate - 0.02 * action_jerk
                 - 2.5e-5 * torques_sq - 6.0e-7 * accels - 2.5e-5 * energy)

        # ── Combine ───────────────────────────────────────────────────────
        # Smooth loco_scale: 1.0 when ≥0.5m away, decays to 0.2 at goal.
        # Avoids the sudden 10× cliff that caused reward to crash on first goal reach.
        dist       = torch.abs(self.command_obs).squeeze(-1)   # (N,) x-distance to goal
        loco_scale = torch.clamp(dist / 0.5, min=0.2, max=1.0)
        reward = (1.0 * r_track                      # paper: 3D foot tracking
                  + 0.5 * r_contact                  # paper: binary stance contact
                  + 0.5 * r_clear                    # paper: binary swing clearance
                  + 2.0 * r_progress * loco_scale    # current: 1D goal progress
                  + 1.5 * r_phase    * loco_scale    # current: continuous phase gating
                  + 1.0 * r_stand                    # current: stand at goal
                  + 1.0 * r_reg)                     # current: r_smooth + r_torque

        for k, v in zip(
            ["track", "contact", "clear", "progress", "phase", "stand", "reg", "total"],
            [r_track, r_contact, r_clear, r_progress, r_phase, r_stand, r_reg, reward],
        ):
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
        self.dr_push_force[env_ids]     = 0.0
        self.push_countdown[env_ids]    = 0

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
        """Full Table 4 DR scaled by cfg.dr_scale (1.0=training, 1.5/2.0=OOD).

        Each parameter is scaled by expanding its half-range from its nominal:
            scaled_range = [nominal - s*half_range, nominal + s*half_range]
        where s = cfg.dr_scale.
        """
        n = len(env_ids)
        s = float(getattr(self.cfg, "dr_scale", 1.0))

        # 1. Action delay: nominal=1 step, half=1 → [0, round(2s)] steps
        max_delay = max(1, round(2 * s))
        self.dr_action_delay[env_ids] = torch.randint(0, max_delay + 1, (n,), device=self.device)
        self._action_buf[env_ids] = 0.0

        # 2. Motor strength: nom=1.0, half=0.20 → [1-0.2s, 1+0.2s]
        lo, hi = max(0.1, 1.0 - 0.2 * s), min(2.0, 1.0 + 0.2 * s)
        self.dr_motor_str[env_ids] = torch.empty(n, 3, device=self.device).uniform_(lo, hi)

        # 3. Joint Kp/Kd: nom=1.0, half=0.20
        kp = torch.empty(n, 3, device=self.device).uniform_(lo, hi)
        kd = torch.empty(n, 3, device=self.device).uniform_(lo, hi)
        self.dr_kp_scale[env_ids] = kp
        self.dr_kd_scale[env_ids] = kd
        self.robot.write_joint_stiffness_to_sim(
            kp * _NOM_KP, joint_ids=self.actuated_dof_indices, env_ids=env_ids)
        self.robot.write_joint_damping_to_sim(
            kd * _NOM_KD, joint_ids=self.actuated_dof_indices, env_ids=env_ids)

        # 4. Link mass: nom=1.0, half=0.20
        mass_scales = torch.empty(n, 4, device=self.device).uniform_(lo, hi)
        self.dr_mass_scale[env_ids] = mass_scales
        masses = self.robot.root_physx_view.get_masses()
        for i, body_idx in enumerate(self._leg_body_ids):
            masses[env_ids.cpu(), body_idx] = (
                self.robot.data.default_mass[env_ids.cpu(), body_idx]
                * mass_scales[:, i].cpu()
            )

        # 5. Payload: nom=1.0 kg, half=2.0 → [1-2s, 1+2s] kg
        pyld_lo, pyld_hi = 1.0 - 2.0 * s, 1.0 + 2.0 * s
        payload = torch.empty(n, device=self.device).uniform_(pyld_lo, pyld_hi)
        self.dr_payload[env_ids] = payload
        masses[env_ids.cpu(), self._leg_body_ids[0]] += payload.cpu()
        masses[env_ids.cpu(), self._leg_body_ids[0]] = (
            masses[env_ids.cpu(), self._leg_body_ids[0]].clamp(min=0.01))
        self.robot.root_physx_view.set_masses(masses, env_ids.cpu())

        # 6–7. Friction: nom=0.95, half=0.75 → [0.95-0.75s, 0.95+0.75s] clamped [0.05, ∞)
        #       Restitution: nom=0.50, half=0.25 → [0.5-0.25s, 0.5+0.25s] clamped [0, 1]
        friction    = torch.empty(n, device=self.device).uniform_(
            max(0.05, 0.95 - 0.75 * s), 0.95 + 0.75 * s)
        restitution = torch.empty(n, device=self.device).uniform_(
            max(0.0, 0.50 - 0.25 * s), min(1.0, 0.50 + 0.25 * s))
        self.dr_friction[env_ids]    = friction
        self.dr_restitution[env_ids] = restitution
        try:
            mats = self.robot.root_physx_view.get_material_properties()
            for i, env_idx in enumerate(env_ids.cpu().tolist()):
                mats[env_idx, :, 0] = friction[i].item()
                mats[env_idx, :, 1] = friction[i].item()
                mats[env_idx, :, 2] = restitution[i].item()
            self.robot.root_physx_view.set_material_properties(mats, env_ids.cpu())
        except Exception:
            pass

        # 8. CoM offset: nom=0, half=[0.075, 0.05] scaled by s
        com_x  = torch.empty(n, device=self.device).uniform_(-0.075 * s,  0.075 * s)
        com_yz = torch.empty(n, 2, device=self.device).uniform_(-0.05  * s, 0.05  * s)
        self.dr_com_offset[env_ids, 0]  = com_x
        self.dr_com_offset[env_ids, 1:] = com_yz
        try:
            coms = self.robot.root_physx_view.get_coms()
            for i, env_idx in enumerate(env_ids.cpu().tolist()):
                coms[env_idx, self._leg_body_ids[0], 0] = com_x[i].item()
                coms[env_idx, self._leg_body_ids[0], 1] = com_yz[i, 0].item()
                coms[env_idx, self._leg_body_ids[0], 2] = com_yz[i, 1].item()
            self.robot.root_physx_view.set_coms(coms, env_ids.cpu())
        except Exception:
            pass

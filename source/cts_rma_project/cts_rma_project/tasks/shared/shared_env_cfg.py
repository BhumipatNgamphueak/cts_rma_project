# tasks/shared/shared_env_cfg.py
"""
SharedEnvCfg — base environment configuration for the T-S locomotion study.

Contains everything that is IDENTICAL across Baseline, RMA, and CTS:
  • Scene (GO2 + contact sensors)
  • Actions (joint position targets)
  • Commands (uniform velocity)
  • Rewards (velocity tracking + stability, Table 4 weights)
  • Terminations
  • Events / Domain Randomisation (Table 4 ranges, all three DR groups)

Observation groups are NOT defined here; each method subclass provides its
own observations (proprioceptive-only for Baseline, +privileged for RMA/CTS).

DR groups (Table 4 of project proposal):
  Mass / inertia : link mass scale [0.80, 1.20], payload [-1, 3] kg
  Actuator       : Kp [0.80, 1.20], Kd [0.80, 1.20], motor strength [0.80, 1.20], delay [0, 20] ms
  Contact surface: friction [0.20, 1.70], restitution [0.25, 0.75]
"""
from __future__ import annotations
import math
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # type: ignore

from . import mdp as shared_mdp


###############################################################################
# Scene
###############################################################################
@configclass
class SharedSceneCfg(InteractiveSceneCfg):
    terrain = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


###############################################################################
# Actions
###############################################################################
@configclass
class SharedActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


###############################################################################
# Commands
###############################################################################
@configclass
class SharedCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


###############################################################################
# Rewards  (shared across all three methods)
###############################################################################
@configclass
class SharedRewardsCfg:
    # ── Primary: command tracking ───────────────────────────────────────────
    track_lin_vel = RewardTermCfg(
        func=shared_mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"std": 0.25},
    )
    track_ang_vel = RewardTermCfg(
        func=shared_mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"std": 0.25},
    )
    # ── Stability ───────────────────────────────────────────────────────────
    penalize_ang_vel_xy = RewardTermCfg(
        func=shared_mdp.penalize_ang_vel_xy, weight=-0.05
    )
    penalize_z_vel = RewardTermCfg(
        func=shared_mdp.penalize_lin_vel_z, weight=-2.0
    )
    # ── Smoothness  (r_smooth in paper) ────────────────────────────────────
    penalize_action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    # ── Torque cost  (r_torque in paper) ───────────────────────────────────
    penalize_joint_torque = RewardTermCfg(func=mdp.joint_torques_l2, weight=-2e-4)
    # ── Contact quality ─────────────────────────────────────────────────────
    penalize_foot_slip = RewardTermCfg(
        func=shared_mdp.penalize_foot_slip,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )


###############################################################################
# Terminations
###############################################################################
@configclass
class SharedTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )
    base_height = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.28},
    )


###############################################################################
# Events — Domain Randomisation  (Table 4, all three DR groups)
###############################################################################
@configclass
class SharedEventCfg:
    # ── Reset base pose ─────────────────────────────────────────────────────
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # ── Contact surface (startup — once per training run per env) ───────────
    # Physics: mdp.randomize_rigid_body_material is a ManagerTermBase class;
    # it must be registered as an EventTermCfg and cannot be called from plain Python.
    randomize_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range":  (0.20, 1.70),
            "dynamic_friction_range": (0.20, 1.70),
            "restitution_range":      (0.25, 0.75),
            "num_buckets": 64,
        },
    )
    # Tracking: independent draw from the same distribution → extras["dr"]
    randomize_material_track = EventTermCfg(
        func=shared_mdp.randomize_material_and_track,
        mode="startup",
        params={
            "static_friction_range": (0.20, 1.70),
            "restitution_range":     (0.25, 0.75),
        },
    )

    # ── Mass / inertia (reset — per episode) ───────────────────────────────
    # Payload [-1, 3] kg added to base
    randomize_payload = EventTermCfg(
        func=shared_mdp.randomize_payload_and_track,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_range": (-1.0, 3.0),
        },
    )
    # Leg link mass scale [0.80, 1.20] × nominal
    randomize_leg_mass = EventTermCfg(
        func=shared_mdp.randomize_leg_mass_and_track,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_hip|.*_thigh|.*_calf"),
            "scale_range": (0.80, 1.20),
        },
    )

    # ── Actuator (reset — per episode) ─────────────────────────────────────
    # Kp scale [0.80, 1.20]
    randomize_kp = EventTermCfg(
        func=shared_mdp.randomize_kp_and_track,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale_range": (0.80, 1.20),
        },
    )
    # Kd scale [0.80, 1.20]
    randomize_kd = EventTermCfg(
        func=shared_mdp.randomize_kd_and_track,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale_range": (0.80, 1.20),
        },
    )
    # Motor strength [0.80, 1.20]
    randomize_motor_strength = EventTermCfg(
        func=shared_mdp.randomize_motor_strength_and_track,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale_range": (0.80, 1.20),
        },
    )
    # Action delay [0, 20] ms
    randomize_action_delay = EventTermCfg(
        func=shared_mdp.randomize_action_delay_and_track,
        mode="reset",
        params={"delay_range_ms": (0.0, 20.0)},
    )

    # ── Random pushes (interval — disturbance robustness) ──────────────────
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.3, 0.3),
                "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3),
            }
        },
    )


###############################################################################
# Abstract base env  (subclasses add their observation group)
###############################################################################
@configclass
class SharedEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration shared by Baseline, RMA, and CTS on GO2.

    Subclasses must define:
        observations: <MethodObsCfg>  — at minimum a 'policy' group
    """
    scene:        SharedSceneCfg        = SharedSceneCfg(num_envs=4096, env_spacing=2.5)
    actions:      SharedActionsCfg      = SharedActionsCfg()
    commands:     SharedCommandsCfg     = SharedCommandsCfg()
    rewards:      SharedRewardsCfg      = SharedRewardsCfg()
    terminations: SharedTerminationsCfg = SharedTerminationsCfg()
    events:       SharedEventCfg        = SharedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt      = 0.005   # 200 Hz physics
        self.decimation  = 2       # 100 Hz policy (matches paper §4.1)
        self.episode_length_s = 20.0
        self.action_space = 12

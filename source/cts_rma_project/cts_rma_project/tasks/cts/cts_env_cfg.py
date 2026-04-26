# tasks/cts/cts_env_cfg.py
"""
CTS (Curriculum Training System) environment configuration.

Strategy:
  - Single-phase PPO with no privileged information.
  - A 37-D proprioceptive observation: joint pos/vel, IMU, velocity command, foot contact.
  - Curriculum managed externally by CTSRunner: velocity command ranges start small
    and expand as the policy learns to track at the current difficulty.
  - Domain randomization is applied at moderate intensity throughout training
    (no progressive DR — the curriculum is purely over command difficulty).
"""
from __future__ import annotations
import math
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import GO2_CFG  # type: ignore

from . import mdp as cts_mdp


###############################################################################
# Scene
###############################################################################
@configclass
class CTSSceneCfg(InteractiveSceneCfg):
    terrain = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )
    robot: ArticulationCfg = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*foot",
        history_length=3,
        track_air_time=True,
    )


###############################################################################
# Observations  (single group — no privileged channel in CTS)
###############################################################################
@configclass
class CTSObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """x_t = joint_pos_rel(12) + joint_vel(12) + ang_vel_b(3)
                + gravity_b(3) + vel_cmd(3) + foot_contact(4) = 37D"""
        state = ObservationTermCfg(func=cts_mdp.base_state_cts)
        enable_corruption = True
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


###############################################################################
# Actions
###############################################################################
@configclass
class CTSActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


###############################################################################
# Commands  — start with reduced range; CTSRunner expands it via curriculum
###############################################################################
@configclass
class CTSCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.5,
        heading_command=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),        # curriculum start — runner expands to ±1.5
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


###############################################################################
# Rewards
###############################################################################
@configclass
class CTSRewardsCfg:
    # Primary: command tracking
    track_lin_vel = RewardTermCfg(
        func=cts_mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"std": 0.25},
    )
    track_ang_vel = RewardTermCfg(
        func=cts_mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"std": 0.25},
    )
    # Stability penalties
    penalize_ang_vel_xy = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    penalize_z_vel = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
    # Smoothness
    penalize_action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    penalize_joint_vel = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001)
    # Contact quality
    penalize_foot_slip = RewardTermCfg(
        func=cts_mdp.penalize_foot_slip,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )
    # Joint limit safety
    penalize_joint_limits = RewardTermCfg(
        func=cts_mdp.penalize_joint_limits,
        weight=-1.0,
    )


###############################################################################
# Terminations
###############################################################################
@configclass
class CTSTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )
    base_height = TerminationTermCfg(
        func=mdp.base_height_below,
        params={"minimum_height": 0.28},
    )


###############################################################################
# Events — moderate domain randomization throughout curriculum
###############################################################################
@configclass
class CTSEventCfg:
    randomize_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.0, 2.0),
            "operation": "add",
        },
    )
    randomize_friction = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 2.0),
            "dynamic_friction_range": (0.2, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    randomize_pd_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )
    # Random velocity pushes test robustness — CTS-specific, absent from RMA config
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
# Main Config
###############################################################################
@configclass
class CTSEnvCfg(ManagerBasedRLEnvCfg):
    scene:        CTSSceneCfg        = CTSSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: CTSObsCfg         = CTSObsCfg()
    actions:      CTSActionsCfg     = CTSActionsCfg()
    commands:     CTSCommandsCfg    = CTSCommandsCfg()
    rewards:      CTSRewardsCfg     = CTSRewardsCfg()
    terminations: CTSTerminationsCfg = CTSTerminationsCfg()
    events:       CTSEventCfg       = CTSEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt      = 0.005   # 200 Hz physics
        self.decimation  = 4       # 50 Hz policy
        self.episode_length_s = 20.0
        self.observation_space = 37
        self.action_space      = 12


@configclass
class CTSEnvCfg_PLAY(CTSEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None  # no pushes during visualization
        # Fixed forward velocity for play
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

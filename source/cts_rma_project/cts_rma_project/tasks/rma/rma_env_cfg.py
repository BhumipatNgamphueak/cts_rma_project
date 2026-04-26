# tasks/rma/rma_env_cfg.py
from __future__ import annotations
import math
import torch
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg, ObservationGroupCfg, ObservationTermCfg,
    RewardTermCfg, SceneEntityCfg, TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.actuators import IdealPDActuatorCfg
import isaaclab.envs.mdp as mdp
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # type: ignore
from . import mdp as rma_mdp


###############################################################################
# Scene
###############################################################################
@configclass
class RMASceneCfg(InteractiveSceneCfg):
    terrain = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9))
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3, track_air_time=True
    )


###############################################################################
# Observations
# RMA has TWO observation groups:
#   "policy"     → x_t (30D) fed to base policy
#   "privileged" → e_t (17D) fed to env factor encoder
###############################################################################
@configclass
class RMAObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """x_t = joint_pos(12) + joint_vel(12) + roll_pitch(2) + contact(4) = 30D"""
        state = ObservationTermCfg(
            func=rma_mdp.base_state_rma,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        enable_corruption = True
        concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObservationGroupCfg):
        """e_t = mass(1) + com(3) + motor_strength(12) + friction(1) = 17D"""
        env_factors = ObservationTermCfg(func=rma_mdp.privileged_env_factors)
        enable_corruption = False
        concatenate_terms = True

    policy:     PolicyCfg     = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


###############################################################################
# Actions
###############################################################################
@configclass
class RMAActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


###############################################################################
# Commands
###############################################################################
@configclass
class RMACommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi),
        ),
    )


###############################################################################
# Rewards  (bioenergetics from RMA paper)
###############################################################################
@configclass
class RMARewardsCfg:
    # Forward locomotion
    track_lin_vel = RewardTermCfg(
        func=rma_mdp.track_lin_vel_x_exp,
        weight=1.0, params={"std": 0.25}
    )
    # Penalize lateral and yaw (weight 21 in paper = 2.0 here)
    penalize_lateral = RewardTermCfg(
        func=rma_mdp.penalize_lateral_motion, weight=-2.0
    )
    # Bioenergetics: work
    penalize_work = RewardTermCfg(
        func=rma_mdp.penalize_work, weight=-0.002
    )
    # Ground impact
    penalize_impact = RewardTermCfg(
        func=rma_mdp.penalize_ground_impact,
        weight=-0.02,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")}
    )
    # Smoothness
    penalize_smoothness = RewardTermCfg(
        func=mdp.action_rate_l2, weight=-0.001
    )
    # Joint speed
    penalize_joint_speed = RewardTermCfg(
        func=mdp.joint_vel_l2, weight=-0.002
    )
    # Orientation
    penalize_orientation = RewardTermCfg(
        func=mdp.ang_vel_xy_l2, weight=-1.5
    )
    # Z acceleration
    penalize_z_accel = RewardTermCfg(
        func=mdp.lin_vel_z_l2, weight=-2.0
    )
    # Foot slip
    penalize_foot_slip = RewardTermCfg(
        func=rma_mdp.penalize_foot_slip,
        weight=-0.8,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")}
    )


###############################################################################
# Terminations
###############################################################################
@configclass
class RMATerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0}
    )
    # Early termination if height < 0.28m or |roll| > 0.4 or |pitch| > 0.2 (from RMA paper)
    base_height = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.28}
    )


###############################################################################
# Events — Domain Randomization
# These write into env.extras so observation functions can read them
###############################################################################
@configclass
class RMAEventCfg:
    # Randomize mass
    randomize_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.0, 3.0),  # add 0 to 3 kg payload
            "operation": "add",
        }
    )
    # Randomize friction
    randomize_friction = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.05, 4.5),
            "dynamic_friction_range": (0.05, 4.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        }
    )
    # Randomize joint PD gains (motor strength)
    randomize_pd_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),   # ×nominal
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        }
    )
    # Reset robot position
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            }
        }
    )


###############################################################################
# Main Config
###############################################################################
@configclass
class RMAEnvCfg(ManagerBasedRLEnvCfg):
    scene:        RMASceneCfg        = RMASceneCfg(num_envs=4096, env_spacing=2.5)
    observations: RMAObsCfg         = RMAObsCfg()
    actions:      RMAActionsCfg     = RMAActionsCfg()
    commands:     RMACommandsCfg    = RMACommandsCfg()
    rewards:      RMARewardsCfg     = RMARewardsCfg()
    terminations: RMATerminationsCfg = RMATerminationsCfg()
    events:       RMAEventCfg       = RMAEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt          = 0.01    # 100 Hz (RMA paper)
        self.decimation      = 1       # policy also at 100 Hz
        self.episode_length_s = 10.0  # 1000 steps max
        # Observation spaces
        self.observation_space = 30    # x_t
        self.action_space      = 12    # joint targets


@configclass
class RMAEnvCfg_PLAY(RMAEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
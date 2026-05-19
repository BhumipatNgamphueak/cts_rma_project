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

DR groups (OpenTopic-aligned, mode=reset):
  Mass / inertia : base mass scale [0.90, 1.10], base inertia scale [0.80, 1.20] (independent)
  Actuator       : Kp scale [0.85, 1.15], Kd scale [0.80, 1.20], delay [0, 20] ms
  Contact surface: friction [0.50, 1.50], restitution [0.00, 0.15]
"""
from __future__ import annotations
import math
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # type: ignore

from . import mdp as shared_mdp


# Flat plane only for now; add rougher sub-terrains later as needed
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)


###############################################################################
# Scene
###############################################################################
@configclass
class SharedSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
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
    # 11×7 = 77 height samples over 1.0 m × 0.6 m grid, yaw-aligned
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
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
    # Start with small commands; curriculum expands toward limit_vel_x/y/ang_z
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.10,
        rel_heading_envs=1.0,
        heading_command=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )


###############################################################################
# Rewards  (matches TXL locomotion task reward structure)
###############################################################################
@configclass
class SharedRewardsCfg:
    # ── Task: velocity tracking ──────────────────────────────────────────────
    track_lin_vel_xy = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"std": math.sqrt(0.25), "command_name": "base_velocity"},
    )
    track_ang_vel_z = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"std": math.sqrt(0.25), "command_name": "base_velocity"},
    )
    # ── Base penalties ───────────────────────────────────────────────────────
    base_linear_velocity  = RewardTermCfg(func=mdp.lin_vel_z_l2,      weight=-2.0)
    base_angular_velocity = RewardTermCfg(func=mdp.ang_vel_xy_l2,     weight=-0.05)
    joint_vel             = RewardTermCfg(func=mdp.joint_vel_l2,       weight=-0.001)
    joint_acc             = RewardTermCfg(func=mdp.joint_acc_l2,       weight=-2.5e-7)
    joint_torques         = RewardTermCfg(func=mdp.joint_torques_l2,   weight=-2e-4)
    action_rate           = RewardTermCfg(func=mdp.action_rate_l2,     weight=-0.1)
    dof_pos_limits        = RewardTermCfg(func=mdp.joint_pos_limits,   weight=-10.0)
    energy                = RewardTermCfg(func=shared_mdp.energy,      weight=-2e-5)
    # ── Robot posture ────────────────────────────────────────────────────────
    flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-2.5)
    joint_pos = RewardTermCfg(
        func=shared_mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    # ── Feet / gait ──────────────────────────────────────────────────────────
    feet_air_time = RewardTermCfg(
        func=shared_mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewardTermCfg(
        func=shared_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_slide = RewardTermCfg(
        func=shared_mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    # ── Survival ─────────────────────────────────────────────────────────────
    alive               = RewardTermCfg(func=mdp.is_alive,      weight=1.0)
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=0.0)
    # ── Contacts ─────────────────────────────────────────────────────────────
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
        },
    )


###############################################################################
# Terminations
###############################################################################
@configclass
class SharedTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    # 100 N threshold (OpenTopic): avoids killing episodes on minor base touches.
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 100.0,
        },
    )
    # ~69° tilt limit (OpenTopic): gives the robot room to recover from tilts.
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.2},
    )


###############################################################################
# Events — Domain Randomisation  (per DR table)
###############################################################################
@configclass
class SharedEventCfg:
    # ── Reset base pose ─────────────────────────────────────────────────────
    # SIM2SIM FIX (#1): velocity_range zeroed.
    # Previously the robot reset with a random ±0.5 m/s linear and ±1.0 rad/s
    # angular velocity push. The CTS student encoder learned that "robot
    # starts moving" is the only valid initial state, producing OOD latents
    # at MuJoCo deployment (which starts at rest) → actor outputs cautious
    # "stand still" actions and the robot never bootstraps walking.
    # Resetting at zero velocity teaches all three methods (Baseline / RMA /
    # CTS) to start walking from rest — the natural deployment scenario.
    # This change is APPLIED IDENTICALLY to all three methods; comparison
    # remains fair.
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5),
                "z": (0.0, 0.3),       # height 0–0.3 m
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    # Joint position ±60° = ±1.047 rad offset from default; velocity ±1.0 rad/s
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-1.047, 1.047),
            "velocity_range": (-1.0,   1.0),
        },
    )

    # ── Physics material (reset — re-randomised every episode) ─────────────
    # SIM2SIM FIX (#2): wider friction range to span Isaac↔MuJoCo contact
    # model gap. Even at the same friction coefficient, PhysX and MuJoCo
    # produce slightly different ground reaction forces; wider DR teaches
    # all methods to be robust to this. Restitution kept narrow (gait
    # quality is highly sensitive to bounce).
    randomize_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range":  (0.3, 1.7),    # was (0.5, 1.5)
            "dynamic_friction_range": (0.3, 1.7),    # was (0.5, 1.5)
            "restitution_range":      (0.0, 0.15),
            "num_buckets": 64,
        },
    )
    # Step 2: read ACTUAL values back from PhysX (must follow randomize_material)
    track_material = EventTermCfg(
        func=shared_mdp.track_material_from_physx,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── Mass / inertia (reset — re-randomised every episode) ────────────────
    # SIM2SIM FIX (#2): mass kept narrow (gait sensitive). Inertia widened
    # to span more dynamics variation.
    randomize_payload = EventTermCfg(
        func=shared_mdp.randomize_payload_and_track,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_scale_range": (0.85, 1.15),    # was (0.9, 1.1)
        },
    )
    # Base inertia widened ±30% (was ±20%).
    randomize_base_inertia = EventTermCfg(
        func=shared_mdp.randomize_inertia_and_track,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "scale_range": (0.7, 1.3),    # was (0.8, 1.2)
        },
    )
    # ── Actuator gains (reset — re-randomised every episode) ────────────────
    # SIM2SIM FIX (#2): wider Kp/Kd ranges so the learned actuator-response
    # mapping covers MuJoCo's slightly different actuator dynamics.
    randomize_gains = EventTermCfg(
        func=shared_mdp.randomize_gains_and_track,
        mode="reset",
        params={
            "asset_cfg":      SceneEntityCfg("robot"),
            "kp_scale_range": (0.7, 1.3),    # was (0.85, 1.15)
            "kd_scale_range": (0.65, 1.35),  # was (0.80, 1.20)
        },
    )
    # SIM2SIM FIX (#2): wider COM offset (±0.08 m, was ±0.05).
    randomize_com = EventTermCfg(
        func=shared_mdp.randomize_com_and_track,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": (-0.08, 0.08),    # was (-0.05, 0.05)
        },
    )
    # SIM2SIM FIX (#2): wider action delay range (covers MuJoCo's longer
    # decimation latency).
    randomize_action_delay = EventTermCfg(
        func=shared_mdp.randomize_action_delay_and_track,
        mode="reset",
        params={"delay_range_ms": (0.0, 30.0)},    # was (0.0, 20.0)
    )

    # ── Disturbances ────────────────────────────────────────────────────────
    # Velocity push ±1.0 m/s linear + ±1.0 rad/s angular every 5–10 s
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0), "y": (-1.0, 1.0),
                "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0),
            }
        },
    )
    # Force/torque impulse at reset: ±5 N, ±2 Nm
    impulse_reset = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range":  (-5.0, 5.0),
            "torque_range": (-2.0, 2.0),
        },
    )
    # Force/torque impulse every 3–8 s: ±10 N, ±3 Nm
    impulse_interval = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(3.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range":  (-10.0, 10.0),
            "torque_range": (-3.0,   3.0),
        },
    )


###############################################################################
# Curriculum — velocity command expansion
###############################################################################
@configclass
class SharedCurriculumCfg:
    # Expand lin_vel and ang_vel ranges when tracking reward > 80% of weight.
    # Starts at (-0.3,0.3) / (-0.2,0.2) / (-0.5,0.5) → grows to limits below.
    lin_vel_cmd = CurriculumTermCfg(
        func=shared_mdp.lin_vel_cmd_curriculum,
        params={
            "reward_term_name": "track_lin_vel_xy",
            "limit_vel_x": (-1.0, 1.0),
            "limit_vel_y": (-1.0, 1.0),
            "limit_ang_z": (-1.0, 1.0),
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
    curriculum:   SharedCurriculumCfg   = SharedCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt      = 0.005   # 200 Hz physics
        self.decimation  = 4       # 50 Hz policy (matches OpenTopic)
        self.episode_length_s = 20.0
        self.action_space = 12
        # Tick height scanner once per policy step
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

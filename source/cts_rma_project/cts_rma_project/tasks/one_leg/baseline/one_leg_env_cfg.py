# tasks/one_leg/baseline/one_leg_env_cfg.py
"""Configuration for the one-legged hopper point-to-point task."""
from __future__ import annotations
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..assets import ONE_LEG_CFG


@configclass
class OneLegEnvCfg(DirectRLEnvCfg):
    # ── Simulation ───────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4)
    decimation: int = 4          # 50 Hz policy (0.005 × 4 = 0.02 s)

    # ── Episode ──────────────────────────────────────────────────────────────
    episode_length_s: float = 10.0

    # ── Observation / action spaces ──────────────────────────────────────────
    # policy obs: dof_pos(3)+dof_vel(3)+prev_actions(3)+robot_pos_x(1)+
    #             C_frc(1)+C_vel(1)+contact(1)+cmd(1) = 14
    observation_space: int = 14
    action_space: int = 3        # hip, knee, ankle position targets
    state_space: int = 0

    # ── Scene ────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=3.0
    )

    # ── Assets ───────────────────────────────────────────────────────────────
    robot = ONE_LEG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_force: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/end_effector",
        history_length=3,
        track_air_time=True,
    )
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ── Task-specific ────────────────────────────────────────────────────────
    # Joints controlled by the policy (must match joint names in USD)
    actuated_joint_names: list = None

    def __post_init__(self):
        if self.actuated_joint_names is None:
            self.actuated_joint_names = ["hip_joint", "knee_joint", "ankle_joint"]
        self.sim.render_interval = self.decimation


@configclass
class OneLegEnvCfg_PLAY(OneLegEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 20.0

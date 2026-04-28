# tasks/one_leg/baseline/one_leg_env_cfg.py
"""Configuration for the one-legged hopper — matches paper Table 1 / Section 4."""
from __future__ import annotations

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
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=2)
    decimation: int = 2          # 100 Hz policy (0.005 × 2 = 0.01 s) — paper Section 4.1

    # ── Episode ──────────────────────────────────────────────────────────────
    episode_length_s: float = 10.0

    # ── Observation / action spaces ──────────────────────────────────────────
    # Table 1: qt(3)+q̇t(3)+at-1(3)+p_ref_foot(3)+ct(1)+sinφ(1)+cosφ(1) = 15
    observation_space: int = 15
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
    actuated_joint_names: list = None
    # Body names of the 4 leg links whose mass is randomised (Table 4)
    leg_link_names: list = None
    # Privileged knowledge ablation: "FULL" | "INT" | "EXT"  (Exp. 2, Table 6)
    priv_mode: str = "FULL"

    # ── Push-force disturbance DR ─────────────────────────────────────────
    push_force_max:    float = 15.0   # N  — max horizontal push magnitude
    push_interval_min: int   = 100    # steps — min steps between pushes (~1 s at 100 Hz)
    push_interval_max: int   = 400    # steps — max steps between pushes (~4 s at 100 Hz)

    # ── OOD scale factor (1.0 = training range, 1.5 / 2.0 = OOD test) ───
    # Multiplies the half-range of every DR parameter from its nominal value.
    dr_scale: float = 1.0

    def __post_init__(self):
        if self.actuated_joint_names is None:
            self.actuated_joint_names = ["hip_joint", "knee_joint", "ankle_joint"]
        if self.leg_link_names is None:
            self.leg_link_names = [
                "linear_up_down_link", "hip_link", "knee_link", "ankle_link"
            ]
        self.sim.render_interval = self.decimation


@configclass
class OneLegEnvCfg_PLAY(OneLegEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 20.0

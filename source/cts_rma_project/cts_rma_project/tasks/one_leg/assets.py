# tasks/one_leg/assets.py
"""ArticulationCfg for the one-legged hopper on a linear rail."""
from __future__ import annotations
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Resolve path relative to project root (6 levels up from this file).
# Place one_leggy8.usd at the repository root before running one-leg tasks.
_USD_PATH = str(Path(__file__).parents[5] / "one_leggy8.usd")

ONE_LEG_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.01),
        rot=(0.707, 0.0, 0.0, 0.707),
        joint_pos={"linear_left_right": 0.1},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint"],
            stiffness=20.0,
            damping=1.0,
        ),
    },
)
ONE_LEG_CFG.base_link_names = ["base_link"]

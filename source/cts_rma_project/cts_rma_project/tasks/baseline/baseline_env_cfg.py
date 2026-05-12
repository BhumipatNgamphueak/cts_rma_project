# tasks/baseline/baseline_env_cfg.py
"""
Baseline environment configuration — pure domain randomisation, no teacher.

Inherits ALL scene, actions, commands, rewards, terminations, and events from
SharedEnvCfg (Table 4 DR ranges, Table 4 reward weights).

Adds only a 'policy' observation group with the 37-D proprioceptive state:
    joint_pos_rel (12) + joint_vel (12) + ang_vel_b (3) + gravity_b (3)
    + vel_cmd (3) + foot_contact (4) = 37

This serves as the experimental control against RMA and CTS.
"""
from __future__ import annotations
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ..shared.shared_env_cfg import SharedEnvCfg
from ..shared import mdp as shared_mdp


###############################################################################
# Observations  (proprioceptive only — no privileged info)
###############################################################################
@configclass
class BaselineObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """o_t ∈ R^37 — runtime-only proprioceptive state.

        Split into three ordered terms so Gaussian noise (σ=0.2 rad/s) can be
        applied to ang_vel_b alone.  Concatenation order matches the original
        37D layout: joint_pos_rel(12)+joint_vel(12)+ang_vel_b(3)+gravity_b(3)
        +vel_cmd(3)+foot_contact(4).
        """
        state = ObservationTermCfg(
            func=shared_mdp.proprioceptive_obs_go2,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "add_noise": True,
            },
        )
        enable_corruption = False
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


###############################################################################
# Baseline env  (SharedEnvCfg + obs)
###############################################################################
@configclass
class BaselineEnvCfg(SharedEnvCfg):
    """Baseline: standard PPO on proprioceptive obs with Table 4 DR."""
    observations: BaselineObsCfg = BaselineObsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = 37   # proprioceptive + ang_vel + gravity + vel_cmd


@configclass
class BaselineEnvCfg_PLAY(BaselineEnvCfg):
    """Evaluation variant: fewer envs, no observation noise, fixed command."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.state.params["add_noise"] = False
        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

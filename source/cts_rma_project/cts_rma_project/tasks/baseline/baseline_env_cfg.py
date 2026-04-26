# tasks/baseline/baseline_env_cfg.py
"""
Baseline environment configuration — pure domain randomisation, no teacher.

Inherits ALL scene, actions, commands, rewards, terminations, and events from
SharedEnvCfg (Table 4 DR ranges, Table 4 reward weights).

Adds only a 'policy' observation group with the 30-D proprioceptive state:
    joint_pos_rel (12) + joint_vel (12) + roll (1) + pitch (1) + foot_contact (4)

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
        """o_t ∈ R^30 — runtime-only proprioceptive state."""
        state = ObservationTermCfg(
            func=shared_mdp.proprioceptive_obs_go2,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        enable_corruption = True
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
        self.observation_space = 30   # proprioceptive only


@configclass
class BaselineEnvCfg_PLAY(BaselineEnvCfg):
    """Evaluation variant: fewer envs, no observation noise, fixed command."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

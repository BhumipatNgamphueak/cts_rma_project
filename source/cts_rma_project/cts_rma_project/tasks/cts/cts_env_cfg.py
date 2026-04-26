# tasks/cts/cts_env_cfg.py
"""
CTS (Concurrent Teacher-Student) environment configuration.

Inherits ALL scene, actions, commands, rewards, events from SharedEnvCfg
(identical DR, rewards, physics, velocity range to Baseline and RMA).

No curriculum override — curriculum is not the CTS contribution.
The CTS novelty is concurrent teacher-student training, not command scheduling.
"""
from __future__ import annotations
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ..shared.shared_env_cfg import SharedEnvCfg
from ..shared import mdp as shared_mdp


###############################################################################
# Observations  (single group — no privileged channel in CTS)
###############################################################################
@configclass
class CTSObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """o_t ∈ R^37 — proprioceptive (identical to Baseline and RMA actor obs)."""
        state = ObservationTermCfg(
            func=shared_mdp.proprioceptive_obs_go2,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        enable_corruption = True
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


###############################################################################
# CTS env  (SharedEnvCfg — no curriculum overrides)
###############################################################################
@configclass
class CTSEnvCfg(SharedEnvCfg):
    """CTS: concurrent teacher-student PPO.  Scene/DR/rewards/commands == Baseline."""
    observations: CTSObsCfg = CTSObsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = 37
        self.action_space      = 12


@configclass
class CTSEnvCfg_PLAY(CTSEnvCfg):
    """Evaluation variant: fewer envs, no obs noise, fixed forward command."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

# tasks/rma/rma_env_cfg.py
"""
RMA environment configuration — asymmetric actor-critic with privileged info.

Inherits ALL scene, actions, commands, rewards, events from SharedEnvCfg
(identical DR, rewards, physics to Baseline and CTS for fair comparison).

Observation groups:
  policy  (37D) — same proprioceptive obs as Baseline; actor sees this at train AND test time
  critic  (61D) — o_t(37) ⊕ e_t(24); only used during training for better value estimates

This implements the "asymmetric actor-critic" variant of privileged-info training:
the actor policy is identical to Baseline at deployment, but the training signal is
improved because the critic has access to ground-truth environment parameters.
"""
from __future__ import annotations
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ..shared.shared_env_cfg import SharedEnvCfg
from ..shared import mdp as shared_mdp


###############################################################################
# Observations
###############################################################################
@configclass
class RMAObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """o_t ∈ R^37 — proprioceptive (identical to Baseline; runs on real robot)."""
        state = ObservationTermCfg(
            func=shared_mdp.proprioceptive_obs_go2,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        enable_corruption = True
        concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """o_t ⊕ e_t ∈ R^61 — asymmetric critic with privileged env info (sim-only)."""
        combined = ObservationTermCfg(
            func=shared_mdp.combined_obs_rma,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        enable_corruption = False
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


###############################################################################
# RMA env  (SharedEnvCfg + asymmetric obs)
###############################################################################
@configclass
class RMAEnvCfg(SharedEnvCfg):
    """RMA: standard PPO with privileged critic.  Scene/DR/rewards == Baseline."""
    observations: RMAObsCfg = RMAObsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = 37   # actor obs (policy group)
        self.action_space      = 12


@configclass
class RMAEnvCfg_PLAY(RMAEnvCfg):
    """Evaluation variant: fewer envs, no obs noise, fixed forward command."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

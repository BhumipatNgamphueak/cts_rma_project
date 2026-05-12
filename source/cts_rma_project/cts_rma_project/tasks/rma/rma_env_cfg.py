# tasks/rma/rma_env_cfg.py
"""
RMA environment configuration — asymmetric actor-critic with privileged info.

Inherits ALL scene, actions, commands, rewards, events from SharedEnvCfg
(identical DR, rewards, physics to Baseline and CTS for fair comparison).

Observation groups:
  policy  (37D) — same proprioceptive obs as Baseline; actor sees this at train AND test time
  critic  (63D) — o_t(37) ⊕ x_t(26); only used during training for better value estimates

This implements the "asymmetric actor-critic" variant of privileged-info training:
the actor policy is identical to Baseline at deployment, but the training signal is
improved because the critic has access to ground-truth environment parameters.
"""
from __future__ import annotations
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ..shared.shared_env_cfg import SharedEnvCfg
from ..shared import mdp as shared_mdp
from ..shared.mdp import PRIV_DIMS


###############################################################################
# Observations
###############################################################################
@configclass
class RMAObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """o_t ∈ R^37 — proprioceptive (identical to Baseline; runs on real robot).

        Split into three ordered terms so Gaussian noise (σ=0.2 rad/s) is
        applied to ang_vel_b alone, matching baseline training conditions.
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

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """o_t ⊕ x_t — asymmetric critic with privileged env info (sim-only).
        Dim = 37 + priv_dim (priv_dim from env.cfg.priv_mode: 26/16/10)."""
        combined = ObservationTermCfg(
            func=shared_mdp.combined_obs_subset,
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
    """RMA: standard PPO with privileged critic.  Scene/DR/rewards == Baseline.

    Used as the Phase-2 env (actor sees 37D proprioception; critic group carries
    [o_t, x_t(priv subset)] for the Phase-2 history→latent target extraction).
    """
    observations: RMAObsCfg = RMAObsCfg()
    priv_mode: str = "FULL"   # privileged subset in the critic group: FULL/INT/EXT

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = 37   # actor obs (policy group)
        self.action_space      = 12


@configclass
class RMATeacherObsCfg:
    """Policy AND critic both return [o_t(37) + x_t(priv subset)] for Phase 1 teacher.
    Dim = 37 + priv_dim (priv_dim from env.cfg.priv_mode: FULL=26 / INT=16 / EXT=10)."""
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        combined = ObservationTermCfg(
            func=shared_mdp.combined_obs_subset,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        combined = ObservationTermCfg(
            func=shared_mdp.combined_obs_subset,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        enable_corruption = False
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RMATeacherEnvCfg(SharedEnvCfg):
    """Phase 1 teacher: actor receives [o_t + x_t] so the encoder gets x_t during training.

    priv_mode selects which privileged subset feeds the encoder:
      FULL → x_t = 26D (internal 16 + external 10)   obs/state = 63D
      INT  → x_t = 16D                                obs/state = 53D
      EXT  → x_t = 10D                                obs/state = 47D
    """
    observations: RMATeacherObsCfg = RMATeacherObsCfg()
    priv_mode: str = "FULL"

    def __post_init__(self):
        super().__post_init__()
        priv_dim = PRIV_DIMS.get(str(self.priv_mode).upper(), 26)
        self.observation_space = 37 + priv_dim   # [o_t(37) + x_t(priv_dim)]
        self.state_space       = 37 + priv_dim
        self.action_space      = 12


@configclass
class RMAEnvCfg_PLAY(RMAEnvCfg):
    """Evaluation variant: fewer envs, no obs noise, fixed forward command."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.state.params["add_noise"] = False
        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

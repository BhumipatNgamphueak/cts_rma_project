# tasks/cts/cts_env_cfg.py
"""
CTS (Concurrent Teacher-Student) environment configuration for GO2 walking.

Observation groups (processed by observation_manager.compute() every step):
  policy  (H*37+1 D) — cts_teacher_student_obs returns the unified teacher-student
                        obs directly; group_obs_dim["policy"] = (1851,) at H=50
  critic  (63D)       — [ot(37), xt(26)] for asymmetric critic + L_rec target

enable_corruption=False on the policy group because selective noise on only the
ot portion of the 1851D obs is not supported by the obs manager, and corrupting
the is_teacher flag (last dim) would break teacher/student routing.
"""
from __future__ import annotations
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ..shared.shared_env_cfg import SharedEnvCfg
from ..shared import mdp as shared_mdp
from ..shared.mdp import PRIV_DIMS
from .mdp import observations as cts_mdp


@configclass
class CTSObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Unified teacher-student obs (H*37+1)D — built by cts_teacher_student_obs."""
        state = ObservationTermCfg(
            func=cts_mdp.cts_teacher_student_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        enable_corruption = False   # flag must not be corrupted; ot noise applied manually
        concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """[ot(37), xt(priv_dim)] — privileged critic + L_rec target.
        priv_dim = 26 (FULL) / 16 (INT) / 10 (EXT); read from env.cfg.priv_mode."""
        combined = ObservationTermCfg(
            func=shared_mdp.combined_obs_subset,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        enable_corruption = False
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CTSEnvCfg(SharedEnvCfg):
    """CTS: concurrent teacher-student PPO on GO2 walking."""
    observations: CTSObsCfg = CTSObsCfg()

    history_len:   int   = 50      # student obs history H (steps at 50 Hz = 1000 ms)
    teacher_ratio: float = 0.75    # 3:1 teacher-to-student ratio
    priv_mode:     str   = "FULL"  # privileged knowledge fed to E^t / critic: FULL/INT/EXT

    def __post_init__(self):
        super().__post_init__()
        priv_dim = PRIV_DIMS.get(str(self.priv_mode).upper(), 26)   # FULL=26 / INT=16 / EXT=10
        # observation_space is informational; actual num_obs comes from
        # observation_manager.group_obs_dim["policy"][0] = H*37+1
        self.observation_space = self.history_len * 37 + 1  # 1851 at H=50
        self.state_space       = 37 + priv_dim              # [ot(37), xt(priv_dim)] critic
        self.action_space      = 12


@configclass
class CTSEnvCfg_PLAY(CTSEnvCfg):
    """Evaluation variant: fewer envs, no noise, fixed forward command."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs                          = 32
        self.teacher_ratio                           = 0.0   # deploy student encoder only
        self.commands.base_velocity.debug_vis        = True
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

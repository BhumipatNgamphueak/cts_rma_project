# tasks/one_leg/cts/one_leg_cts_env_cfg.py
from isaaclab.utils import configclass
from ..baseline.one_leg_env_cfg import OneLegEnvCfg, OneLegEnvCfg_PLAY


@configclass
class OneLegCTSTeacherEnvCfg(OneLegEnvCfg):
    """CTS teacher — policy sees 22-D (14 prop + 8 privileged)."""
    observation_space: int = 22


@configclass
class OneLegCTSTeacherEnvCfg_PLAY(OneLegEnvCfg_PLAY):
    observation_space: int = 22


@configclass
class OneLegCTSStudentEnvCfg(OneLegEnvCfg):
    """CTS student — policy sees 14-D; teacher imitation reward decays."""
    observation_space: int = 14

    # Imitation reward weight: decays from alpha_start → alpha_end
    imitation_alpha_start: float = 1.0
    imitation_alpha_end:   float = 0.1
    imitation_decay_iters: int   = 1000   # iterations over which alpha decays


@configclass
class OneLegCTSStudentEnvCfg_PLAY(OneLegEnvCfg_PLAY):
    observation_space: int = 14
    imitation_alpha_start: float = 0.0
    imitation_alpha_end:   float = 0.0
    imitation_decay_iters: int   = 1

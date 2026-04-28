# tasks/one_leg/cts/one_leg_cts_env_cfg.py
from isaaclab.utils import configclass
from ..baseline.one_leg_env_cfg import OneLegEnvCfg, OneLegEnvCfg_PLAY


@configclass
class OneLegCTSEnvCfg(OneLegEnvCfg):
    """CTS concurrent env.

    policy obs  = 76D: [unified(75), is_teacher_flag(1)]
    critic obs  = 39D: [ot, xt] for all envs (privileged critic + L_rec target)
    """
    observation_space: int = 76   # 75D unified + 1D flag
    state_space:       int = 48   # [ot(15), xt(33)] for asymmetric critic

    teacher_ratio: float = 0.75


@configclass
class OneLegCTSEnvCfg_PLAY(OneLegEnvCfg_PLAY):
    observation_space: int = 76
    state_space:       int = 39
    teacher_ratio: float   = 0.0   # deploy student encoder only

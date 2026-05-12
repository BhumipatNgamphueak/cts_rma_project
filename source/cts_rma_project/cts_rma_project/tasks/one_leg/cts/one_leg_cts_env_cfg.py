# tasks/one_leg/cts/one_leg_cts_env_cfg.py
from isaaclab.utils import configclass
from ..baseline.one_leg_env_cfg import OneLegEnvCfg, OneLegEnvCfg_PLAY


@configclass
class OneLegCTSEnvCfg(OneLegEnvCfg):
    """CTS concurrent env.

    policy obs  = (H×15+1)D: [unified(H×15), is_teacher_flag(1)]
    critic obs  = 48D: [ot, xt] for all envs (privileged critic + L_rec target)
    """
    history_len:       int   = 50    # student obs history H (steps at 100 Hz)
    observation_space: int   = 751   # history_len*15 + 1; updated by train.py
    state_space:       int   = 48    # [ot(15), xt(33)] for asymmetric critic

    teacher_ratio: float = 0.90


@configclass
class OneLegCTSEnvCfg_PLAY(OneLegEnvCfg_PLAY):
    observation_space: int = 751
    state_space:       int = 48
    teacher_ratio: float   = 0.0   # deploy student encoder only

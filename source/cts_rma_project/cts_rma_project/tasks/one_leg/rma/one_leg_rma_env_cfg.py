# tasks/one_leg/rma/one_leg_rma_env_cfg.py
from isaaclab.utils import configclass
from ..baseline.one_leg_env_cfg import OneLegEnvCfg, OneLegEnvCfg_PLAY


@configclass
class OneLegRMAEnvCfg(OneLegEnvCfg):
    """RMA Phase-1: actor+critic both see [ot, xt] = 48D (symmetric AC).

    The RMAActorCritic in rma_policy.py internally splits obs into
    ot(15) and xt(33), encodes xt → zt(Z), then runs π([ot, zt]).
    """
    observation_space: int = 48   # 15 prop + 33 privileged


@configclass
class OneLegRMAEnvCfg_PLAY(OneLegEnvCfg_PLAY):
    observation_space: int = 48

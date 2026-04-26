# tasks/one_leg/rma/one_leg_rma_env_cfg.py
from isaaclab.utils import configclass
from ..baseline.one_leg_env_cfg import OneLegEnvCfg, OneLegEnvCfg_PLAY


@configclass
class OneLegRMAEnvCfg(OneLegEnvCfg):
    """RMA variant: actor 14-D, critic 22-D (14+8 privileged)."""
    # num_observations stays 14 (actor size); RSL-RL reads critic size
    # from the returned obs_dict["critic"] shape automatically.
    pass


@configclass
class OneLegRMAEnvCfg_PLAY(OneLegEnvCfg_PLAY):
    pass

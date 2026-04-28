import gymnasium as gym
from . import agents
from .one_leg_cts_env_cfg import OneLegCTSEnvCfg, OneLegCTSEnvCfg_PLAY

gym.register(
    id="OneLeg-CTS-v0",
    entry_point="cts_rma_project.tasks.one_leg.cts.one_leg_cts_env:OneLegCTSEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OneLegCTSEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegCTSPPOCfg",
    },
)
gym.register(
    id="OneLeg-CTS-Play-v0",
    entry_point="cts_rma_project.tasks.one_leg.cts.one_leg_cts_env:OneLegCTSEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OneLegCTSEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegCTSPPOCfg",
    },
)

import gymnasium as gym
from . import agents
from .one_leg_rma_env_cfg import OneLegRMAEnvCfg, OneLegRMAEnvCfg_PLAY

gym.register(
    id="OneLeg-RMA-v0",
    entry_point="cts_rma_project.tasks.one_leg.rma.one_leg_rma_env:OneLegRMAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OneLegRMAEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegRMAPPOCfg",
    },
)
gym.register(
    id="OneLeg-RMA-Play-v0",
    entry_point="cts_rma_project.tasks.one_leg.rma.one_leg_rma_env:OneLegRMAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OneLegRMAEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegRMAPPOCfg",
    },
)

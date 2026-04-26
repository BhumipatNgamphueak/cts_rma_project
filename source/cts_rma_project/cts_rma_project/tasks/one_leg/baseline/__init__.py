import gymnasium as gym
from . import agents
from .one_leg_env_cfg import OneLegEnvCfg, OneLegEnvCfg_PLAY

gym.register(
    id="OneLeg-Baseline-v0",
    entry_point="cts_rma_project.tasks.one_leg.baseline.one_leg_env:OneLegBaselineEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OneLegEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegBaselinePPOCfg",
    },
)
gym.register(
    id="OneLeg-Baseline-Play-v0",
    entry_point="cts_rma_project.tasks.one_leg.baseline.one_leg_env:OneLegBaselineEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OneLegEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegBaselinePPOCfg",
    },
)

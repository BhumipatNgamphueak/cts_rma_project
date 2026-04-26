import gymnasium as gym
from . import agents
from .one_leg_cts_env_cfg import (
    OneLegCTSTeacherEnvCfg, OneLegCTSTeacherEnvCfg_PLAY,
    OneLegCTSStudentEnvCfg, OneLegCTSStudentEnvCfg_PLAY,
)

gym.register(
    id="OneLeg-CTS-Teacher-v0",
    entry_point="cts_rma_project.tasks.one_leg.cts.one_leg_cts_env:OneLegCTSTeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OneLegCTSTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegCTSTeacherPPOCfg",
    },
)
gym.register(
    id="OneLeg-CTS-Student-v0",
    entry_point="cts_rma_project.tasks.one_leg.cts.one_leg_cts_env:OneLegCTSStudentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OneLegCTSStudentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegCTSStudentPPOCfg",
    },
)
gym.register(
    id="OneLeg-CTS-Teacher-Play-v0",
    entry_point="cts_rma_project.tasks.one_leg.cts.one_leg_cts_env:OneLegCTSTeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OneLegCTSTeacherEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegCTSTeacherPPOCfg",
    },
)

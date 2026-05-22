# tasks/cts/__init__.py
import gymnasium as gym
from . import agents
from . import cts_env_cfg

gym.register(
    id="Go2-CTS-v0",
    entry_point="cts_rma_project.tasks.cts.cts_env:Go2CTSEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cts_env_cfg.CTSEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CTSPPORunnerCfg",
    },
)

gym.register(
    id="Go2-CTS-Play-v0",
    entry_point="cts_rma_project.tasks.cts.cts_env:Go2CTSEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cts_env_cfg.CTSEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CTSPPORunnerCfg",
    },
)

gym.register(
    id="Template-CTS-GO2-v0",
    entry_point="cts_rma_project.tasks.cts.cts_env:Go2CTSEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cts_env_cfg.CTSEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CTSPPORunnerCfg",
    },
)

gym.register(
    id="Template-CTS-GO2-Play-v0",
    entry_point="cts_rma_project.tasks.cts.cts_env:Go2CTSEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cts_env_cfg.CTSEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CTSPPORunnerCfg",
    },
)

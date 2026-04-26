# tasks/baseline/__init__.py
import gymnasium as gym
from . import agents, baseline_env_cfg

gym.register(
    id="Template-Baseline-GO2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baseline_env_cfg.BaselineEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselinePPORunnerCfg",
    },
)

gym.register(
    id="Template-Baseline-GO2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baseline_env_cfg.BaselineEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselinePPORunnerCfg",
    },
)

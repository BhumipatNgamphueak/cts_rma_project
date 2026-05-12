# tasks/rma/__init__.py
import gymnasium as gym
from . import agents, rma_env_cfg

gym.register(
    id="Template-RMA-GO2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rma_env_cfg.RMAEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RMAPPORunnerCfg",
    },
)

gym.register(
    id="Template-RMA-GO2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rma_env_cfg.RMAEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RMAPPORunnerCfg",
    },
)

gym.register(
    id="Template-RMA-Teacher-GO2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rma_env_cfg.RMATeacherEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RMATeacherPPORunnerCfg",
    },
)
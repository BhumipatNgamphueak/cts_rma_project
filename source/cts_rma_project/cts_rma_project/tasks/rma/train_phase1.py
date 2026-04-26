# tasks/rma/train_phase1.py
"""
Convenience helpers for constructing the Phase 1 training objects.

The actual launch script lives in scripts/rma/train_phase1.py.
This module can be imported by other tools (e.g. sweep scripts) to
build a Phase 1 runner without duplicating boilerplate.
"""
from __future__ import annotations

import os
import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore

from .rma_env_cfg              import RMAEnvCfg
from .rma_network              import RMAActorCritic
from .rma_runner               import RMAPhase1Runner
from .agents.rsl_rl_ppo_cfg    import RMAPPORunnerCfg


def build_phase1_runner(
    num_envs: int = 4096,
    device: str = "cuda",
    log_dir: str = "logs/rma/phase1",
    seed: int = 42,
    max_iterations: int = 5000,
    checkpoint: str | None = None,
) -> RMAPhase1Runner:
    """Build and return a ready-to-run Phase 1 runner."""
    env_cfg = RMAEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device     = device
    env_cfg.seed           = seed

    env = gym.make("Template-RMA-GO2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = RMAPPORunnerCfg()
    runner_cfg.max_iterations = max_iterations
    runner_cfg.seed           = seed

    os.makedirs(log_dir, exist_ok=True)
    runner = RMAPhase1Runner(env, runner_cfg.to_dict(),
                             log_dir=log_dir, device=device)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    runner.alg.actor_critic = RMAActorCritic(
        num_actor_obs=obs_dim, num_critic_obs=obs_dim,
        num_actions=act_dim, env_factor_dim=17, latent_dim=8,
    ).to(device)

    if checkpoint:
        runner.load(checkpoint)

    return runner

# tasks/rma/agents/rsl_rl_ppo_cfg.py
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # type: ignore
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RMAPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for RMA Phase 1 training.

    Phase 1: trains base policy π and environment-factor encoder μ
    using ground-truth privileged observations e_t.
    """
    num_steps_per_env   = 16
    max_iterations      = 5000
    save_interval       = 200
    experiment_name     = "rma_phase1"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

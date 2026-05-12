# tasks/rma/agents/rsl_rl_ppo_cfg.py
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # type: ignore
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RMAPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Asymmetric actor-critic (actor=37D, critic=63D). No encoder in actor."""
    num_steps_per_env        = 24
    max_iterations           = 5000
    save_interval            = 200
    experiment_name          = "rma_go2"
    empirical_normalization  = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
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


@configclass
class RMATeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1 teacher: RMAActorCritic gets [o_t+x_t]=63D, encodes x_t→z."""
    num_steps_per_env        = 24
    max_iterations           = 5000
    save_interval            = 200
    experiment_name          = "rma_teacher_go2"
    empirical_normalization  = False

    policy = RslRlPpoActorCriticCfg(
        class_name="RMAActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
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

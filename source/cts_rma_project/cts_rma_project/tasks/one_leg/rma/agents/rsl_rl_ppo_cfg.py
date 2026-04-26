# tasks/one_leg/rma/agents/rsl_rl_ppo_cfg.py
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # type: ignore
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class OneLegRMAPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env       = 24
    max_iterations          = 3000
    save_interval           = 200
    experiment_name         = "one_leg_rma"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],   # same as Baseline
        critic_hidden_dims=[256, 128, 64],  # critic input is 22-D (handled internally)
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

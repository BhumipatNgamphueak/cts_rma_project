# tasks/cts/agents/rsl_rl_ppo_cfg.py
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # type: ignore
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CTSPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for CTS GO2 locomotion (concurrent teacher-student).

    SIM2SIM FIX (#4): tighten L_rec to bring the student encoder closer to the
    teacher encoder. With the previous λ_rec=1.0 / warmup=500 / lr=3e-4 final
    L_rec=0.21 — student z didn't fully match teacher z. At MuJoCo deployment
    this gap manifests as actor confusion. Stronger reconstruction loss makes
    student z reliably interchangeable with teacher z. Affects CTS only —
    Baseline / RMA training is unchanged.
    """
    num_steps_per_env       = 24
    max_iterations          = 5000
    save_interval           = 200
    experiment_name         = "cts_go2"
    empirical_normalization = False

    # Stronger L_rec parameters (read by CTSRunner.__init__ via train_cfg.pop).
    cts_lambda_rec = 5.0     # was 1.0  — 5× stronger reconstruction loss
    cts_rec_warmup = 1000    # was 500  — slower ramp gives PPO time to settle
    cts_rec_lr     = 5e-4    # was 3e-4 — slightly higher LR for E^s

    policy = RslRlPpoActorCriticCfg(
        class_name="CTSActorCritic",   # injected into runner namespace by train.py
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # latent_dim and history_len are injected by train.py via train_dict
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

"""
CTS Phase 2 — Train the student policy via teacher distillation.

Student sees 14-D obs (deployable on real hardware).  A frozen teacher
is loaded from a Phase-1 checkpoint. The student receives an imitation
reward that decays from alpha=1.0 → 0.1 over the first 1000 iterations,
so it transitions from pure behavioral cloning to RL-guided locomotion.

Usage:
    python scripts/one_leg/cts/train_student.py \\
        --teacher_checkpoint logs/one_leg/cts/teacher/<run>/model_final.pt \\
        --num_envs 1024 --headless
"""
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--teacher_checkpoint", type=str, required=True,
                    help="Path to the Phase-1 teacher model_final.pt")
parser.add_argument("--num_envs",       type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--experiment",     type=str, default="one_leg_cts_student")
parser.add_argument("--seed",           type=int, default=42)
parser.add_argument("--checkpoint",     type=str, default=None,
                    help="Resume student from a previous checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, torch, gymnasium as gym
from datetime import datetime
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper           # type: ignore
from rsl_rl.runners     import OnPolicyRunner                # type: ignore

import cts_rma_project.tasks  # noqa
from cts_rma_project.tasks.one_leg.cts.one_leg_cts_env       import OneLegCTSStudentEnv
from cts_rma_project.tasks.one_leg.cts.one_leg_cts_env_cfg   import OneLegCTSStudentEnvCfg
from cts_rma_project.tasks.one_leg.cts.agents.rsl_rl_ppo_cfg import OneLegCTSStudentPPOCfg


def main():
    device = args_cli.device or "cuda"

    # ── Student environment ───────────────────────────────────────────────
    env_cfg = OneLegCTSStudentEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device

    env     = gym.make("OneLeg-CTS-Student-v0", cfg=env_cfg)
    raw_env: OneLegCTSStudentEnv = env.unwrapped
    env     = RslRlVecEnvWrapper(env)

    # ── Load frozen teacher ───────────────────────────────────────────────
    print(f"[INFO] Loading teacher from {args_cli.teacher_checkpoint}")
    teacher_ckpt = torch.load(args_cli.teacher_checkpoint, map_location=device)

    # Build a minimal teacher network matching the Phase-1 architecture
    from rsl_rl.modules import ActorCritic  # type: ignore
    teacher_net = ActorCritic(
        num_actor_obs=22,   # teacher saw 22-D
        num_critic_obs=22,
        num_actions=3,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    ).to(device)
    teacher_net.load_state_dict(teacher_ckpt["model_state_dict"])
    raw_env.load_teacher(teacher_net.actor)

    # ── PPO runner ────────────────────────────────────────────────────────
    runner_cfg = OneLegCTSStudentPPOCfg()
    runner_cfg.max_iterations  = args_cli.max_iterations
    runner_cfg.experiment_name = args_cli.experiment
    runner_cfg.seed            = args_cli.seed

    log_dir = os.path.join("logs", "one_leg", "cts", "student",
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=device)
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)

    # Decay alpha after each iteration
    original_learn = runner.learn

    def learn_with_decay(**kwargs):
        for i in range(args_cli.max_iterations):
            original_learn(num_learning_iterations=1,
                           init_at_random_ep_len=(i == 0))
            raw_env.step_iteration()

    learn_with_decay()

    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"[INFO] Student saved to {final_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

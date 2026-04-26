"""
CTS Phase 1 — Train the teacher policy.

Teacher sees full 22-D obs (14 prop + 8 privileged).  Trained with
standard PPO. The saved checkpoint is used in Phase 2 to guide the student.

Usage:
    python scripts/one_leg/cts/train_teacher.py --num_envs 1024 --headless
"""
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",       type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--experiment",     type=str, default="one_leg_cts_teacher")
parser.add_argument("--seed",           type=int, default=42)
parser.add_argument("--checkpoint",     type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, gymnasium as gym
from datetime import datetime
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper           # type: ignore
from rsl_rl.runners     import OnPolicyRunner                # type: ignore

import cts_rma_project.tasks  # noqa
from cts_rma_project.tasks.one_leg.cts.one_leg_cts_env_cfg    import OneLegCTSTeacherEnvCfg
from cts_rma_project.tasks.one_leg.cts.agents.rsl_rl_ppo_cfg  import OneLegCTSTeacherPPOCfg


def main():
    device = args_cli.device or "cuda"

    env_cfg = OneLegCTSTeacherEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device

    env = gym.make("OneLeg-CTS-Teacher-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = OneLegCTSTeacherPPOCfg()
    runner_cfg.max_iterations  = args_cli.max_iterations
    runner_cfg.experiment_name = args_cli.experiment
    runner_cfg.seed            = args_cli.seed

    log_dir = os.path.join("logs", "one_leg", "cts", "teacher",
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=device)
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)

    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)

    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"[INFO] Teacher saved to {final_path}")
    print(f"[INFO] Run Phase 2 with: --teacher_checkpoint {final_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# tasks/cts/cts_runner.py
"""
CTS Curriculum Runner.

Wraps RSL-RL's OnPolicyRunner and periodically advances (or decays) a
scalar curriculum_level ∈ [0, 1] based on the agent's command-tracking
performance.  The level controls the maximum velocity command range that
the environment will sample.

Curriculum schedule
-------------------
  vel_max = lerp(CMD_VEL_START, CMD_VEL_FINAL, curriculum_level)

  Every CURRICULUM_INTERVAL training iterations:
    - Read the mean episode reward from the runner's internal buffer.
    - If mean_reward > ADVANCE_THRESHOLD  → level += ADVANCE_STEP
    - If mean_reward < DECAY_THRESHOLD    → level -= DECAY_STEP
    - Clamp to [0, 1].
  Then update the command-manager term's range attributes in the live env.
"""
from __future__ import annotations

import os
import torch
from collections import deque

from rsl_rl.runners import OnPolicyRunner  # type: ignore

# ── Curriculum hyper-parameters ──────────────────────────────────────────────
CMD_VEL_START      = 0.5    # m/s  at curriculum level 0
CMD_VEL_FINAL      = 1.5    # m/s  at curriculum level 1
CMD_VEL_LAT_RATIO  = 0.5    # lateral range = ratio × forward range

CURRICULUM_INTERVAL = 50    # check / update every N iterations
ADVANCE_THRESHOLD   = 0.75  # mean episode reward above this → advance
DECAY_THRESHOLD     = 0.35  # mean episode reward below this → decay
ADVANCE_STEP        = 0.02
DECAY_STEP          = 0.01
# ─────────────────────────────────────────────────────────────────────────────


class CTSRunner:
    """
    Curriculum Training System runner.

    Usage::

        runner = CTSRunner(env, ppo_cfg_dict, log_dir="logs/cts", device="cuda")
        runner.learn(num_learning_iterations=3000)
        runner.save("cts_final.pt")
    """

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cuda",
    ):
        self._inner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
        self.env     = env
        self.device  = device
        self.log_dir = log_dir or "logs/cts"

        # Curriculum state
        self.curriculum_level: float = 0.0

        os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = True):
        """Run PPO training with periodic curriculum updates."""
        iterations_done = 0
        first_block = True

        while iterations_done < num_learning_iterations:
            block = min(CURRICULUM_INTERVAL, num_learning_iterations - iterations_done)
            self._inner.learn(block, init_at_random_ep_len=first_block)
            first_block = False
            iterations_done += block

            self._update_curriculum()
            print(
                f"[CTS] iter {iterations_done:5d}/{num_learning_iterations} "
                f"| curriculum_level={self.curriculum_level:.3f} "
                f"| vel_max={self._current_vel_max():.2f} m/s"
            )

    def save(self, path: str):
        self._inner.save(path)

    def load(self, path: str):
        self._inner.load(path)

    def get_inference_policy(self, device=None):
        return self._inner.get_inference_policy(device=device or self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _current_vel_max(self) -> float:
        return CMD_VEL_START + self.curriculum_level * (CMD_VEL_FINAL - CMD_VEL_START)

    def _mean_episode_reward(self) -> float | None:
        """Read the mean episode reward from the runner's reward buffer."""
        buf = getattr(self._inner, "rewbuffer", None)
        if buf and len(buf) > 0:
            return sum(buf) / len(buf)
        # Fallback: read from the storage if available
        storage = getattr(getattr(self._inner, "alg", None), "storage", None)
        if storage is not None and hasattr(storage, "rewards"):
            return float(storage.rewards.mean())
        return None

    def _update_curriculum(self):
        mean_rew = self._mean_episode_reward()
        if mean_rew is None:
            return

        if mean_rew > ADVANCE_THRESHOLD:
            self.curriculum_level = min(1.0, self.curriculum_level + ADVANCE_STEP)
        elif mean_rew < DECAY_THRESHOLD:
            self.curriculum_level = max(0.0, self.curriculum_level - DECAY_STEP)

        self._apply_curriculum_to_env()

    def _apply_curriculum_to_env(self):
        """Update the live command-manager ranges to match the current level."""
        vel_max = self._current_vel_max()
        lat_max = vel_max * CMD_VEL_LAT_RATIO
        try:
            # gymnasium Env.unwrapped already traverses the full wrapper chain
            raw_env = self.env.unwrapped
            cmd_term = raw_env.command_manager._terms["base_velocity"]
            cmd_term.cfg.ranges.lin_vel_x = (-vel_max, vel_max)
            cmd_term.cfg.ranges.lin_vel_y = (-lat_max, lat_max)
        except (AttributeError, KeyError):
            pass  # env may not expose command manager in all wrappers

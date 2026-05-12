# tasks/cts/cts_runner.py
"""
CTSRunner — extends OnPolicyRunner with L_rec distillation pass for GO2.

Training loop (each iteration):
  1. Standard rollout collection (PPO.act calls CTSActorCritic.act(obs_1851D)).
  2. CTSActorCritic routes teacher/student via obs[:, -1] flag internally.
  3. compute_returns with critic_obs (63D [ot,xt], privileged).
  4. Pre-collect L_rec data (history, xt) from storage before PPO clears it.
  5. Standard PPO update (clears storage) — updates E^t via alg.optimizer.
  6. L_rec pass AFTER PPO — updates E^s only (E^t detached):
       zs = E^s(history),  zt = detach(E^t(xt))
       L_rec = MSE(zs, zt) * lambda_rec
       backprop through E^s only.

Optimizer ownership:
  alg.optimizer  → teacher_encoder + actor + critic  (PPO owns E^t)
  _rec_optimizer → student_conv + student_fc          (L_rec owns E^s)
  Sharing E^t across two optimizers causes conflicting momentum states →
  use detach(E^t(xt)) as the L_rec target so E^t has a single owner.
"""
from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

from rsl_rl.runners import OnPolicyRunner


class CTSRunner(OnPolicyRunner):
    """OnPolicyRunner + L_rec distillation loss for concurrent CTS on GO2."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self._lambda_rec = float(train_cfg.pop("cts_lambda_rec", 1.0))
        self._rec_warmup = int(train_cfg.pop("cts_rec_warmup", 500))
        self._rec_lr     = float(train_cfg.pop("cts_rec_lr", 3e-4))
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        if not hasattr(self, "logger_type"):
            self.logger_type = "tensorboard"

        # Student encoder only — E^t stays with alg.optimizer (PPO)
        self._rec_optimizer = torch.optim.Adam(
            list(self.policy.student_conv.parameters())
            + list(self.policy.student_fc.parameters()),
            lr=self._rec_lr,
        )

    @property
    def policy(self):
        return self.alg.policy

    # ── L_rec: collect data from storage before PPO clears it ────────────
    def _collect_rec_data(self) -> list:
        pairs = []
        # Use the policy's actual priv_dim so INT(16)/EXT(10) ablations slice
        # the correct columns (FULL=26 -> 37:63 as before).
        priv_dim = getattr(self.policy, "_priv_dim", 26)
        crit_xt_end = 37 + priv_dim
        generator = self.alg.storage.mini_batch_generator(self.alg.num_mini_batches, 1)
        for batch in generator:
            obs_b, crit_b = batch[0], batch[1]
            student_mask  = obs_b[:, -1] <= 0.5
            if not student_mask.any():
                continue
            h_dim = self.policy._history_len * 37   # H × _OBS_DIM (37 for GO2)
            pairs.append((
                obs_b[student_mask, :h_dim].clone(),                    # history (Bs, H*37)
                crit_b[student_mask, 37:crit_xt_end].clone(),            # xt only (Bs, priv_dim)
            ))
        return pairs

    # ── L_rec: update E^s after PPO ──────────────────────────────────────
    def _update_rec_loss(self, rec_data: list, current_iter: int) -> float:
        ramp    = min(1.0, current_iter / max(1, self._rec_warmup))
        eff_lam = self._lambda_rec * ramp

        # Cap rec LR at initial value — PPO's adaptive schedule can spike upward
        ppo_lr = self.alg.optimizer.param_groups[0]["lr"]
        rec_lr = min(ppo_lr, self._rec_lr)
        for pg in self._rec_optimizer.param_groups:
            pg["lr"] = rec_lr

        total_loss, count = 0.0, 0
        for hist_b, xt_b in rec_data:
            zs = self.policy.encode_student(hist_b)
            with torch.no_grad():
                zt = self.policy.encode_teacher(xt_b)   # detached target

            rec_loss = torch.nn.functional.mse_loss(zs, zt) * eff_lam
            self._rec_optimizer.zero_grad()
            rec_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.student_conv.parameters())
                + list(self.policy.student_fc.parameters()),
                self.alg.max_grad_norm,
            )
            self._rec_optimizer.step()
            total_loss += rec_loss.item()
            count      += 1

        return total_loss / count if count > 0 else 0.0

    # ── Override learn() to inject L_rec ─────────────────────────────────
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            from torch.utils.tensorboard import SummaryWriter
            self.logger_type = "tensorboard"
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get("critic", obs).to(self.device)
        obs = obs.to(self.device)
        self.train_mode()

        rewbuffer         = deque(maxlen=100)
        rewbuffer_teacher = deque(maxlen=100)
        rewbuffer_student = deque(maxlen=100)
        lenbuffer         = deque(maxlen=100)
        ep_infos          = []
        cur_reward_sum     = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)

        is_teacher = self.env.unwrapped.is_teacher   # (N,) bool

        start_iter = self.current_learning_iteration
        tot_iter   = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # ── Rollout ───────────────────────────────────────────────────
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs)
                    obs_new, rewards, dones, infos = self.env.step(
                        actions.to(self.env.device))
                    obs_new        = obs_new.to(self.device)
                    rewards        = rewards.to(self.device)
                    dones          = dones.to(self.device)
                    obs_new        = self.obs_normalizer(obs_new)
                    privileged_obs = self.privileged_obs_normalizer(
                        infos["observations"].get("critic", obs_new).to(self.device))

                    self.alg.process_env_step(rewards, dones, infos)

                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])

                    cur_reward_sum     += rewards
                    cur_episode_length += 1
                    new_ids   = (dones > 0).nonzero(as_tuple=False)
                    done_envs = new_ids[:, 0]
                    rewbuffer.extend(cur_reward_sum[done_envs].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[done_envs].cpu().numpy().tolist())
                    t_mask = is_teacher[done_envs]
                    rewbuffer_teacher.extend(
                        cur_reward_sum[done_envs[t_mask]].cpu().numpy().tolist())
                    rewbuffer_student.extend(
                        cur_reward_sum[done_envs[~t_mask]].cpu().numpy().tolist())
                    cur_reward_sum[done_envs]     = 0
                    cur_episode_length[done_envs] = 0
                    obs = obs_new

                collection_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(privileged_obs)

            # ── L_rec data collection before PPO clears storage ───────────
            rec_data = self._collect_rec_data()

            # ── PPO update ────────────────────────────────────────────────
            loss_dict = self.alg.update()

            # ── L_rec pass AFTER PPO ─────────────────────────────────────
            mean_rec_loss = self._update_rec_loss(rec_data, it - start_iter)
            loss_dict["L_rec"] = mean_rec_loss

            learn_time = time.time() - start
            self.current_learning_iteration = it

            # ── Logging ───────────────────────────────────────────────────
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if self.writer is not None:
                    if len(rewbuffer_teacher) > 0:
                        self.writer.add_scalar(
                            "Train/mean_reward_teacher",
                            statistics.mean(rewbuffer_teacher), it)
                    if len(rewbuffer_student) > 0:
                        self.writer.add_scalar(
                            "Train/mean_reward_student",
                            statistics.mean(rewbuffer_student), it)
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        # Final save
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, "model_final.pt"))

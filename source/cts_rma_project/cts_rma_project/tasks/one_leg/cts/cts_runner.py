# tasks/one_leg/cts/cts_runner.py
"""
CTSRunner — extends OnPolicyRunner with an L_rec distillation pass.

Training loop (each iteration):
  1. Standard rollout collection (PPO.act calls CTSActorCritic.act(obs_76D)).
  2. CTSActorCritic routes teacher/student envs internally via obs[:, -1] flag.
  3. compute_returns with critic_obs (39D [ot,xt], privileged).
  4. Pre-collect L_rec data (hist, xt) from storage before PPO clears it.
  5. Standard PPO update (clears storage) — updates Et via alg.optimizer.
  6. L_rec pass AFTER PPO — updates Es only (Et is detached):
       zs = Es(hist),  zt = detach(Et(xt))
       L_rec = MSE(zs, zt) * lambda_rec
       backprop through Es only.

Optimizer ownership (critical for stability):
  alg.optimizer   → owns Et (teacher_encoder) + actor + critic
  _rec_optimizer  → owns Es (student_conv + student_fc)
  Sharing Et across two Adam optimizers causes conflicting momentum/variance
  states → Et oscillates between what PPO wants and what L_rec wants → reward drops.
  Detaching zt gives Et a single owner (PPO) and Es a single owner (L_rec).
"""
from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

from rsl_rl.runners import OnPolicyRunner


class CTSRunner(OnPolicyRunner):
    """OnPolicyRunner + L_rec distillation loss for concurrent CTS."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self._lambda_rec  = float(train_cfg.pop("cts_lambda_rec", 1.0))
        self._rec_warmup  = int(train_cfg.pop("cts_rec_warmup", 500))   # iters before full L^rec
        # Separate, lower LR for student encoder — never synced above this value.
        # PPO's adaptive LR can spike upward; if _rec_optimizer follows it up,
        # Es makes huge jumps → actor for student envs destabilises → reward collapses.
        self._rec_lr      = float(train_cfg.pop("cts_rec_lr", 3e-4))
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        if not hasattr(self, "logger_type"):
            self.logger_type = "tensorboard"

        # Only student encoder is updated by _rec_optimizer.
        # teacher_encoder is owned exclusively by alg.optimizer (PPO).
        self._rec_optimizer = torch.optim.Adam(
            list(self.policy.student_conv.parameters())
            + list(self.policy.student_fc.parameters()),
            lr=self._rec_lr,
        )

    @property
    def policy(self):
        return self.alg.policy

    # ══════════════════════════════════════════════════════════════════════
    # L_rec pass: iterate storage, compute and backprop reconstruction loss
    # ══════════════════════════════════════════════════════════════════════
    def _collect_rec_data(self) -> list:
        """Pre-collect (hist, xt) pairs from storage before PPO clears it."""
        pairs = []
        generator = self.alg.storage.mini_batch_generator(
            self.alg.num_mini_batches, 1)
        for batch in generator:
            obs_b, crit_b = batch[0], batch[1]
            student_mask = obs_b[:, -1] <= 0.5
            if not student_mask.any():
                continue
            pairs.append((
                obs_b[student_mask, :75].clone(),       # history (Bs, 75)
                crit_b[student_mask, 15:48].clone(),    # xt only (Bs, 33)
            ))
        return pairs

    def _update_rec_loss(self, rec_data: list, current_iter: int) -> float:
        """Run L_rec on pre-collected data AFTER PPO update.

        Called after self.alg.update() so PPO uses clean rollout (no stale
        log-prob ratio from L_rec changing Et before PPO runs).
        Lambda is linearly ramped up over the first _rec_warmup iterations so
        random student gradients don't corrupt Et early in training.
        """
        # Linear warmup: lambda ramps 0→_lambda_rec over _rec_warmup iters
        ramp    = min(1.0, current_iter / max(1, self._rec_warmup))
        eff_lam = self._lambda_rec * ramp

        # Follow PPO LR decay but never exceed the initial rec_lr.
        # PPO's adaptive schedule can increase LR when KL is low — if
        # _rec_optimizer followed it up, Es would make huge jumps and
        # destabilise the actor for student envs → reward collapses.
        ppo_lr = self.alg.optimizer.param_groups[0]["lr"]
        rec_lr = min(ppo_lr, self._rec_lr)
        for pg in self._rec_optimizer.param_groups:
            pg["lr"] = rec_lr

        total_loss = 0.0
        count      = 0

        for hist_b, xt_b in rec_data:
            zs = self.policy.encode_student(hist_b)          # (Bs, Z)
            with torch.no_grad():
                zt = self.policy.encode_teacher(xt_b)        # (Bs, Z) — detached target

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

    # ══════════════════════════════════════════════════════════════════════
    # Override learn() to inject L_rec between compute_returns and update
    # ══════════════════════════════════════════════════════════════════════
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # Writer setup mirrors OnPolicyRunner (base may already have done this)
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
        lenbuffer          = deque(maxlen=100)
        ep_infos           = []
        cur_reward_sum     = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)

        # Teacher/student mask from the underlying env
        is_teacher = self.env.unwrapped.is_teacher  # (N,) bool

        start_iter = self.current_learning_iteration
        tot_iter   = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # ── Rollout collection ────────────────────────────────────────
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs)
                    obs_new, rewards, dones, infos = self.env.step(actions.to(self.env.device))
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
                    new_ids      = (dones > 0).nonzero(as_tuple=False)
                    done_envs    = new_ids[:, 0]              # flat env indices
                    done_rewards = cur_reward_sum[done_envs].cpu().numpy().tolist()
                    rewbuffer.extend(done_rewards)
                    lenbuffer.extend(cur_episode_length[done_envs].cpu().numpy().tolist())

                    # Split by teacher / student
                    t_mask = is_teacher[done_envs]
                    rewbuffer_teacher.extend(cur_reward_sum[done_envs[t_mask]].cpu().numpy().tolist())
                    rewbuffer_student.extend(cur_reward_sum[done_envs[~t_mask]].cpu().numpy().tolist())

                    cur_reward_sum[done_envs]     = 0
                    cur_episode_length[done_envs] = 0

                    obs = obs_new

                collection_time = time.time() - start
                start = time.time()

                # Compute returns
                self.alg.compute_returns(privileged_obs)

            # ── Pre-collect L_rec data before PPO clears storage ─────────
            rec_data = self._collect_rec_data()

            # ── Standard PPO update (clean rollout, no L_rec interference) ─
            loss_dict = self.alg.update()

            # ── L_rec pass AFTER PPO — Et shaped without polluting PPO ────
            mean_rec_loss = self._update_rec_loss(rec_data, it - start_iter)
            loss_dict["L_rec"] = mean_rec_loss

            learn_time = time.time() - start
            self.current_learning_iteration = it

            # ── Logging via base class (gives rich table + TensorBoard) ───
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

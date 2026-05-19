"""
OOD evaluation for GO2 locomotion policies (Baseline / RMA / CTS) at DR×2.0.

What each method actually is
-----------------------------
  baseline : Standard PPO.  Actor sees o_t (37-D).  Critic sees o_t (37-D).
  rma      : Asymmetric actor-critic.  Actor sees o_t (37-D).  Critic sees
             [o_t, x_t] (61-D) during training, giving better value estimates.
             At deployment the actor is IDENTICAL to Baseline (same 37-D input).
  cts      : Concurrent teacher-student.  Student encoder maps a 50-step
             proprioceptive history (H×37-D) to latent z (8-D).  Actor sees
             [o_t, z] (45-D) — more information than Baseline/RMA at test time.

Fairness note
-------------
  Baseline vs RMA  : identical actor capacity and deployment input (37-D).
                     Only training signal differs (privileged critic for RMA).
                     ✓ Fair — isolates effect of privileged critic.
  Baseline/RMA vs CTS : CTS actor has extra latent z from history.
                        CTS has additional CNN encoder (more computation).
                        ✓ Fair as deployment comparison — no privileged sim
                        info used at test time for any method.

Usage:
    /home/drl-68/IsaacLab/isaaclab.sh -p scripts/eval_ood_go2.py \\
        --method baseline \\
        --checkpoint logs/baseline/<run>/model_final.pt \\
        --num_episodes 100 --num_envs 64 --headless

    /home/drl-68/IsaacLab/isaaclab.sh -p scripts/eval_ood_go2.py \\
        --method rma \\
        --checkpoint logs/rma/<run>/model_final.pt \\
        --num_episodes 100 --num_envs 64 --headless

    /home/drl-68/IsaacLab/isaaclab.sh -p scripts/eval_ood_go2.py \\
        --method cts \\
        --checkpoint logs/cts/<run>/model_final.pt \\
        --latent_dim 8 --history_len 50 \\
        --num_episodes 100 --num_envs 64 --headless
"""
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GO2 OOD evaluation")
parser.add_argument("--method",       type=str, required=True,
                    choices=["baseline", "rma", "rma_teacher", "cts"])
parser.add_argument("--checkpoint",   type=str, required=True)
parser.add_argument("--dr_scale",     type=float, default=2.0,
                    help="DR half-width multiplier (1.0=training range, 2.0=OOD×2)")
parser.add_argument("--num_episodes", type=int, default=100,
                    help="N = 100 matches the spec sheet (Sim2Sim Result Metrics).")
parser.add_argument("--num_envs",     type=int, default=64)
parser.add_argument("--episode_length_s", type=float, default=10.0,
                    help="Episode length T in seconds (spec sheet: T=10s). "
                         "Overrides cfg.episode_length_s for eval; training is unaffected.")
parser.add_argument("--vel_rmse_threshold", type=float, default=0.3,
                    help="Velocity-tracking RMSE threshold (m/s) for the Success/Partial split. "
                         "Spec sheet: 0.3 m/s.")
parser.add_argument("--latent_dim",   type=int, default=8,
                    help="Encoder bottleneck z dimension (8/16/32/64/128 for ablation)")
parser.add_argument("--priv_mode",    type=str, default="FULL",
                    choices=["FULL", "INT", "EXT"],
                    help="Privileged subset the RMA/CTS checkpoint was trained with "
                         "(FULL=26/INT=16/EXT=10). Ignored for baseline.")
parser.add_argument("--history_len",  type=int, default=50)
parser.add_argument("--adapt_module", type=str, default=None,
                    help="(RMA only) path to Phase 2 adapt_module_final.pt; omit for z=0")
parser.add_argument("--results_file", type=str, default=None,
                    help="CSV path to append the aggregated row")
parser.add_argument("--save_raw_dir", type=str, default=None,
                    help="If set, also write per-episode JSON + per-step NPZ for this "
                         "(method,priv,latent,dr) under this directory. Enables "
                         "post-hoc analysis / bootstrapping / time-series plots.")
parser.add_argument("--no_terrain",   action="store_true",
                    help="Replace generated terrain with a flat ground plane "
                         "and disable the height scanner.")
parser.add_argument("--no_dist",      action="store_true",
                    help="Disable push_robot and impulse disturbance events.")
parser.add_argument("--no_impulse",   action="store_true",
                    help="Disable ONLY impulse_reset/impulse_interval, keep "
                         "push_robot. Matches v2 training (push_robot only; "
                         "impulses were added after v2 was trained).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math, os, sys, csv, statistics
import torch
import numpy as np
import gymnasium as gym
from datetime import datetime
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore

import cts_rma_project.tasks  # noqa

# Shared gait-metric library (single source of truth for the 8 behaviour metrics).
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from gait_metrics import (
    GAIT_METRIC_NAMES,
    compute_episode_metrics as _compute_episode_gait_metrics,
    mean_std_across_episodes as _mean_std_gait,
)


# ── Tracking metrics (mirror unitree_rl_lab/scripts/rsl_rl/eval_ood.py) ──────
# Use the same per-step formulas the OpenTopic reference uses, so Isaac OOD and
# MuJoCo sim2sim report apples-to-apples behaviour metrics.
_TRACK_STD = math.sqrt(0.25)   # std**2 = 0.25 — matches SharedRewardsCfg track_*_exp

def _tracking_exp(actual: torch.Tensor, cmd: torch.Tensor, std: float = _TRACK_STD) -> torch.Tensor:
    return torch.exp(-torch.sum((actual - cmd) ** 2, dim=-1) / (std ** 2))


# ── DR scaling ────────────────────────────────────────────────────────────────
# Authoritative training ranges — these MUST mirror tasks/shared/shared_env_cfg.py
# ::SharedEventCfg. (The previous version of this table was copied from the
# single-leg project and referenced event terms that no longer exist; it has been
# rewritten to match the current Go2 SharedEventCfg.)
_TRAIN_RANGES = {
    "friction":      (0.30, 1.70),   # static & dynamic friction (randomize_material)
    "restitution":   (0.00, 0.15),   # restitution             (randomize_material)
    "mass_scale":    (0.85, 1.15),   # base mass scale          (randomize_payload)
    "inertia_scale": (0.70, 1.30),   # base inertia scale       (randomize_base_inertia)
    "kp_scale":      (0.70, 1.30),   # actuator Kp scale        (randomize_gains)
    "kd_scale":      (0.65, 1.35),   # actuator Kd scale        (randomize_gains)
    "com_halfwidth": 0.08,           # base COM offset ± m      (randomize_com)
    "delay_ms":      (0.0, 30.0),    # action delay [ms]        (randomize_action_delay)
}
# Disturbances (push_robot, impulse_reset, impulse_interval) and reset-state
# randomisation (reset_base, reset_joints) are NOT scaled in OOD evaluation.


def _scale(lo: float, hi: float, s: float) -> tuple:
    """Widen [lo, hi] about its midpoint by factor s (s=1.0 -> training range)."""
    c = (lo + hi) / 2.0
    h = (hi - lo) / 2.0
    return (c - s * h, c + s * h)


def _apply_dr_scale(cfg, scale: float):
    """Scale the Go2 SharedEventCfg DR ranges by `scale` in place. Returns cfg."""
    ev = cfg.events

    # friction (static & dynamic share the same range)
    f_lo, f_hi = _scale(*_TRAIN_RANGES["friction"], scale)
    f_lo = max(f_lo, 0.01)
    ev.randomize_material.params["static_friction_range"]  = (f_lo, f_hi)
    ev.randomize_material.params["dynamic_friction_range"] = (f_lo, f_hi)

    # restitution
    r_lo, r_hi = _scale(*_TRAIN_RANGES["restitution"], scale)
    r_lo = max(r_lo, 0.0); r_hi = min(r_hi, 1.0)
    ev.randomize_material.params["restitution_range"] = (r_lo, r_hi)
    # (track_material reads back from PhysX — no range params to set)

    # base mass scale
    m_lo, m_hi = _scale(*_TRAIN_RANGES["mass_scale"], scale)
    ev.randomize_payload.params["mass_scale_range"] = (max(m_lo, 0.1), m_hi)

    # base inertia scale
    i_lo, i_hi = _scale(*_TRAIN_RANGES["inertia_scale"], scale)
    ev.randomize_base_inertia.params["scale_range"] = (max(i_lo, 0.1), i_hi)

    # actuator gains (single event with separate Kp/Kd ranges)
    kp_lo, kp_hi = _scale(*_TRAIN_RANGES["kp_scale"], scale)
    kd_lo, kd_hi = _scale(*_TRAIN_RANGES["kd_scale"], scale)
    ev.randomize_gains.params["kp_scale_range"] = (max(kp_lo, 0.1), kp_hi)
    ev.randomize_gains.params["kd_scale_range"] = (max(kd_lo, 0.1), kd_hi)

    # base COM offset (symmetric about 0 -> scale the half-width)
    com_hw = _TRAIN_RANGES["com_halfwidth"] * scale
    ev.randomize_com.params["com_range"] = (-com_hw, com_hw)

    # action delay
    d_lo, d_hi = _scale(*_TRAIN_RANGES["delay_ms"], scale)
    ev.randomize_action_delay.params["delay_range_ms"] = (max(d_lo, 0.0), d_hi)

    return cfg


# ── Policy loaders ────────────────────────────────────────────────────────────
def _load_baseline(ckpt: dict, device: str):
    """Load standard rsl_rl ActorCritic for Baseline."""
    from rsl_rl.modules import ActorCritic
    sd          = ckpt.get("model_state_dict", ckpt.get("actor_critic", ckpt))
    actor_keys  = sorted(k for k in sd if k.startswith("actor.") and "weight" in k)
    hidden_dims = [sd[k].shape[0] for k in actor_keys[:-1]]
    num_actions = sd[actor_keys[-1]].shape[0]
    obs_dim     = sd[actor_keys[0]].shape[1]
    policy = ActorCritic(
        num_actor_obs=obs_dim, num_critic_obs=obs_dim, num_actions=num_actions,
        actor_hidden_dims=hidden_dims, critic_hidden_dims=hidden_dims,
        activation="elu", init_noise_std=1.0,
    ).to(device)
    policy.load_state_dict(sd, strict=False)
    return policy


def _load_rma(ckpt: dict, device: str, latent_dim: int = 8,
              adapt_module_path: str = None, priv_dim: int = 26):
    """Load RMAActorCritic (Phase 1 checkpoint).

    At inference the actor uses z=0 unless adapt_module_path is provided,
    in which case the AdaptationModule provides z_hat from obs/action history.
    History management is handled externally in the eval loop when adapt_module is used.
    """
    from cts_rma_project.tasks.rma.rma_network import RMAActorCritic, AdaptationModule
    sd = ckpt.get("model_state_dict", ckpt.get("actor_critic", ckpt))
    policy = RMAActorCritic(
        num_actor_obs=37, num_critic_obs=37 + priv_dim, num_actions=12,
        env_factor_dim=priv_dim, latent_dim=latent_dim,
    ).to(device)
    policy.load_state_dict(sd, strict=False)

    adapt_mod = None
    if adapt_module_path:
        adapt_mod = AdaptationModule(
            state_dim=37, action_dim=12, embed_dim=32,
            latent_dim=latent_dim, history_len=50,
        ).to(device)
        adapt_mod.load_state_dict(torch.load(adapt_module_path, map_location=device))
        adapt_mod.eval()
        print(f"[eval] RMA adaptation module loaded from {adapt_module_path}")

    return policy, adapt_mod


def _load_rma_teacher(ckpt: dict, device: str, latent_dim: int = 8, priv_dim: int = 26):
    """Load RMAActorCritic in teacher mode (num_actor_obs=37+priv_dim → _teacher_mode=True).

    At inference: z = env_encoder(obs[:,37:37+priv_dim]) — privileged info used directly.
    This gives the Phase-1 upper-bound performance (oracle access to x_t).
    """
    from cts_rma_project.tasks.rma.rma_network import RMAActorCritic
    sd = ckpt.get("model_state_dict", ckpt.get("actor_critic", ckpt))
    policy = RMAActorCritic(
        num_actor_obs=37 + priv_dim, num_critic_obs=37 + priv_dim, num_actions=12,
        env_factor_dim=priv_dim, latent_dim=latent_dim,
    ).to(device)
    policy.load_state_dict(sd, strict=False)
    return policy


def _load_cts(ckpt: dict, device: str, latent_dim: int = 8, history_len: int = 50,
              priv_dim: int = 26):
    from cts_rma_project.tasks.cts.cts_network import CTSActorCritic
    sd = ckpt.get("model_state_dict", ckpt.get("actor_critic", ckpt))
    policy = CTSActorCritic(
        num_actor_obs=history_len * 37 + 1, num_critic_obs=37 + priv_dim,
        num_actions=12, latent_dim=latent_dim, history_len=history_len, priv_dim=priv_dim,
    ).to(device)
    policy.load_state_dict(sd, strict=False)
    return policy


# ── Env factory ───────────────────────────────────────────────────────────────
def _make_env(method: str, num_envs: int, dr_scale: float, device: str,
              history_len: int = 50, priv_mode: str = "FULL",
              episode_length_s: float = 10.0,
              no_terrain: bool = False,
              no_dist: bool = False,
              no_impulse: bool = False) -> RslRlVecEnvWrapper:
    if method == "baseline":
        from cts_rma_project.tasks.baseline.baseline_env_cfg import BaselineEnvCfg
        cfg    = BaselineEnvCfg()
        gym_id = "Template-Baseline-GO2-v0"
    elif method == "rma":
        from cts_rma_project.tasks.rma.rma_env_cfg import RMAEnvCfg
        cfg           = RMAEnvCfg()
        cfg.priv_mode = priv_mode
        gym_id        = "Template-RMA-GO2-v0"
    elif method == "rma_teacher":
        from cts_rma_project.tasks.rma.rma_env_cfg import RMATeacherEnvCfg
        from cts_rma_project.tasks.shared.mdp import PRIV_DIMS
        cfg           = RMATeacherEnvCfg()
        cfg.priv_mode = priv_mode
        priv_dim_t    = PRIV_DIMS[priv_mode]
        cfg.observation_space = 37 + priv_dim_t
        cfg.state_space       = 37 + priv_dim_t
        gym_id        = "Template-RMA-Teacher-GO2-v0"
    else:  # cts
        from cts_rma_project.tasks.cts.cts_env_cfg import CTSEnvCfg
        cfg               = CTSEnvCfg()
        cfg.teacher_ratio = 0.0
        cfg.history_len   = history_len
        cfg.priv_mode     = priv_mode
        cfg.observation_space = history_len * 37 + 1
        gym_id            = "Template-CTS-GO2-v0"

    cfg.scene.num_envs    = num_envs
    cfg.sim.device        = device
    cfg.episode_length_s  = episode_length_s
    cfg.observations.policy.enable_corruption = False
    _apply_dr_scale(cfg, dr_scale)

    if no_terrain:
        from isaaclab.terrains import TerrainImporterCfg
        cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
        )
        cfg.scene.height_scanner = None
        print("[eval] Terrain: FLAT ground plane (no generated terrain)")

    if no_dist:
        cfg.events.push_robot       = None
        cfg.events.impulse_interval = None
        cfg.events.impulse_reset    = None
        print("[eval] Disturbances: DISABLED (push_robot, impulse_*)")

    if no_impulse and not no_dist:
        cfg.events.impulse_interval = None
        cfg.events.impulse_reset    = None
        print("[eval] Impulses DISABLED, push_robot KEPT (v2-training-faithful)")

    # RMA Phase 2: disable height scanner for non-terrain priv modes
    if method == "rma" and priv_mode not in ("TERR", "FULL_T"):
        cfg.scene.height_scanner = None

    return RslRlVecEnvWrapper(gym.make(gym_id, cfg=cfg))


# ── Main ──────────────────────────────────────────────────────────────────────
_PRIV_DIMS = {"FULL": 26, "INT": 16, "EXT": 10}


def main():
    device      = args_cli.device or "cuda"
    method      = args_cli.method
    dr_scale    = args_cli.dr_scale
    num_eps     = args_cli.num_episodes
    num_envs    = args_cli.num_envs
    latent_dim  = args_cli.latent_dim
    history_len = args_cli.history_len
    priv_mode   = args_cli.priv_mode.upper()
    priv_dim    = _PRIV_DIMS[priv_mode]

    ckpt      = torch.load(args_cli.checkpoint, map_location=device)
    ckpt_iter = ckpt.get("iter", "?")
    terrain_tag = "flat"    if args_cli.no_terrain else "terrain"
    dist_tag    = "no_dist" if args_cli.no_dist    else "dist"
    print(f"[eval] method={method}  ckpt_iter={ckpt_iter}  dr={dr_scale:.1f}x  "
          f"priv={priv_mode}  latent={latent_dim}  {terrain_tag}  {dist_tag}")

    adapt_mod = None
    if method == "baseline":
        policy = _load_baseline(ckpt, device)
    elif method == "rma":
        policy, adapt_mod = _load_rma(ckpt, device, latent_dim, args_cli.adapt_module, priv_dim)
    elif method == "rma_teacher":
        policy = _load_rma_teacher(ckpt, device, latent_dim, priv_dim)
    else:
        policy = _load_cts(ckpt, device, latent_dim, history_len, priv_dim)
    policy.eval()

    # RMA adaptation module history buffers
    rma_state_hist  = None
    rma_action_hist = None
    if adapt_mod is not None:
        rma_state_hist  = torch.zeros(num_envs, 50, 37, device=device)
        rma_action_hist = torch.zeros(num_envs, 50, 12, device=device)

    env       = _make_env(method, num_envs, dr_scale, device, history_len, priv_mode,
                          episode_length_s=args_cli.episode_length_s,
                          no_terrain=args_cli.no_terrain,
                          no_dist=args_cli.no_dist,
                          no_impulse=args_cli.no_impulse)
    max_steps = int(env.unwrapped.max_episode_length)
    raw_env   = env.unwrapped
    robot     = raw_env.scene["robot"]
    contact_sensor = raw_env.scene.sensors["contact_forces"]
    step_dt   = float(raw_env.step_dt)
    print(f"[eval] max_steps={max_steps}  step_dt={step_dt:.4f}s")

    # Resolve foot body indices once (canonical order [FL, FR, RL, RR] to match MuJoCo).
    _FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_body_ids = [robot.body_names.index(n) for n in _FOOT_NAMES]
    foot_body_ids_t = torch.tensor(foot_body_ids, device=device, dtype=torch.long)
    # Foot collision radius — same physical model (Unitree Go2). MuJoCo geom size = 0.022 m.
    foot_radius   = 0.022

    obs, _     = env.get_observations()
    obs        = obs.to(device)

    # Per-episode buffers
    ep_rewards:   list[float] = []
    ep_lengths:   list[float] = []
    ep_lin_track: list[float] = []
    ep_ang_track: list[float] = []
    ep_track_err: list[float] = []   # = vel_rmse per spec sheet
    ep_fwd_disp:  list[float] = []
    ep_gait:      list[dict]  = []
    ep_outcomes:  list[str]   = []   # "success" / "partial" / "fail"
    per_ep_raw:   list[dict]  = []   # populated only if --save_raw_dir is set
    success_count = 0
    partial_count = 0
    fall_count    = 0

    # Per-env (in-flight) accumulators
    cur_reward    = torch.zeros(num_envs, device=device)
    cur_length    = torch.zeros(num_envs, device=device)
    cur_lin_track = torch.zeros(num_envs, device=device)
    cur_ang_track = torch.zeros(num_envs, device=device)
    cur_track_err = torch.zeros(num_envs, device=device)
    cur_fwd_disp  = torch.zeros(num_envs, device=device)

    # Gait-metric per-env, per-step GPU buffers (sliced + sent to CPU on done).
    T = max_steps
    g_contacts   = torch.zeros(T, num_envs, 4,  device=device)
    g_foot_z     = torch.zeros(T, num_envs, 4,  device=device)
    g_foot_xy_s  = torch.zeros(T, num_envs, 4,  device=device)
    g_foot_speed = torch.zeros(T, num_envs, 4,  device=device)
    g_actions    = torch.zeros(T, num_envs, 12, device=device)
    g_tau        = torch.zeros(T, num_envs, 12, device=device)
    g_base_z     = torch.zeros(T, num_envs,     device=device)
    g_base_xy    = torch.zeros(T, num_envs, 2,  device=device)
    g_cmd_xy     = torch.zeros(T, num_envs, 2,  device=device)
    env_idx_all  = torch.arange(num_envs, device=device)

    print(f"[eval] collecting {num_eps} episodes ...")
    with torch.inference_mode():
        while len(ep_rewards) < num_eps and simulation_app.is_running():
            if method == "rma_teacher":
                xt      = obs[:, 37:37 + priv_dim]
                z_true  = policy.env_encoder(xt)
                actions = policy.act_inference(obs, z_override=z_true)
            elif adapt_mod is not None:
                z_hat   = adapt_mod(rma_state_hist, rma_action_hist)
                actions = policy.act_inference(obs, z_override=z_hat)
            else:
                actions = policy.act_inference(obs)
            obs, rewards, dones, infos = env.step(actions)
            obs     = obs.to(device)
            rewards = rewards.to(device)
            dones   = dones.to(device)

            # ── Per-step tracking metrics (matches OpenTopic eval_ood.py) ────
            cmd         = raw_env.command_manager.get_command("base_velocity")
            v_xy_actual = robot.data.root_lin_vel_b[:, :2]
            wz_actual   = robot.data.root_ang_vel_b[:, 2:3]
            v_xy_cmd    = cmd[:, :2]
            wz_cmd      = cmd[:, 2:3]
            lin_track   = _tracking_exp(v_xy_actual, v_xy_cmd)
            ang_track   = _tracking_exp(wz_actual,   wz_cmd)
            track_err   = torch.norm(v_xy_actual - v_xy_cmd, dim=-1)

            if adapt_mod is not None:
                ot = obs[:, :37]
                rma_state_hist  = torch.roll(rma_state_hist,  -1, dims=1)
                rma_action_hist = torch.roll(rma_action_hist, -1, dims=1)
                rma_state_hist[:, -1, :]  = ot
                rma_action_hist[:, -1, :] = actions
                done_ids_reset = (dones > 0).nonzero(as_tuple=False)[:, 0]
                if done_ids_reset.numel():
                    rma_state_hist[done_ids_reset]  = 0.0
                    rma_action_hist[done_ids_reset] = 0.0

            cur_reward    += rewards
            cur_length    += 1
            cur_lin_track += lin_track
            cur_ang_track += ang_track
            cur_track_err += track_err
            # Forward displacement: integrate per-step projection of body-frame
            # linear velocity onto commanded direction (m). For episodes whose cmd
            # changes mid-episode this still accumulates displacement toward the
            # *currently* commanded direction.
            cmd_speed = torch.linalg.norm(v_xy_cmd, dim=-1).clamp(min=1e-6)
            fwd_proj  = (v_xy_actual * v_xy_cmd).sum(dim=-1) / cmd_speed
            cur_fwd_disp += fwd_proj * step_dt

            # ── Per-step gait buffers (see scripts/gait_metrics.py) ──────────
            # contacts: 0/1 per foot from contact sensor (>1 N on z component)
            contacts_t  = (contact_sensor.data.net_forces_w_history[:, 0, foot_body_ids_t, 2] > 1.0).float()
            foot_pos_w  = robot.data.body_pos_w[:, foot_body_ids_t]              # (N, 4, 3)
            foot_vel_w  = robot.data.body_lin_vel_w[:, foot_body_ids_t]          # (N, 4, 3)
            foot_xy_sp  = torch.linalg.norm(foot_vel_w[:, :, :2], dim=-1)         # (N, 4)
            foot_sp     = torch.linalg.norm(foot_vel_w,            dim=-1)        # (N, 4)
            base_pos_w  = robot.data.root_pos_w                                    # (N, 3)
            tau_t       = robot.data.applied_torque                                # (N, 12)
            # Write into (T, N, …) buffers at per-env slot t_idx = cur_length - 1
            t_idx = (cur_length.long() - 1).clamp_(min=0, max=T - 1)
            g_contacts  [t_idx, env_idx_all]      = contacts_t
            g_foot_z    [t_idx, env_idx_all]      = foot_pos_w[:, :, 2]
            g_foot_xy_s [t_idx, env_idx_all]      = foot_xy_sp
            g_foot_speed[t_idx, env_idx_all]      = foot_sp
            g_actions   [t_idx, env_idx_all]      = actions.to(device).float()
            g_tau       [t_idx, env_idx_all]      = tau_t
            g_base_z    [t_idx, env_idx_all]      = base_pos_w[:, 2]
            g_base_xy   [t_idx, env_idx_all]      = base_pos_w[:, :2]
            g_cmd_xy    [t_idx, env_idx_all]      = v_xy_cmd

            # success vs fall: time_outs[i] is True if the episode ended by hitting
            # the time limit (= surviving until episode_length_s) — exactly matches
            # OpenTopic's eval_ood.py. Anything else (base contact, bad orientation)
            # is a fall.
            time_outs = infos.get(
                "time_outs",
                torch.zeros(num_envs, dtype=torch.bool, device=device),
            )

            done_ids = (dones > 0).nonzero(as_tuple=False)[:, 0]
            for idx in done_ids.tolist():
                if len(ep_rewards) >= num_eps:
                    break
                steps = max(int(cur_length[idx].item()), 1)
                vel_rmse_ep = (cur_track_err[idx] / steps).item()   # episode-mean RMSE
                ep_rewards.append(cur_reward[idx].item())
                ep_lengths.append(float(steps))
                ep_lin_track.append((cur_lin_track[idx] / steps).item())
                ep_ang_track.append((cur_ang_track[idx] / steps).item())
                ep_track_err.append(vel_rmse_ep)
                ep_fwd_disp.append(cur_fwd_disp[idx].item())
                # ── Three-class outcome per the spec sheet ─────────────────────
                # success  = survived to T AND vel_rmse < threshold (0.3 m/s)
                # partial  = survived to T but vel_rmse ≥ threshold
                # fail     = fell (any non-time-out termination)
                if not bool(time_outs[idx]):
                    outcome = "fail"; fall_count += 1
                elif vel_rmse_ep < args_cli.vel_rmse_threshold:
                    outcome = "success"; success_count += 1
                else:
                    outcome = "partial"; partial_count += 1
                ep_outcomes.append(outcome)
                # ── Compute gait metrics for this episode (shared formula) ───
                T_i      = min(steps, T)
                cmd_mean = g_cmd_xy[:T_i, idx].mean(dim=0).cpu().numpy()
                _ep_contacts      = g_contacts  [:T_i, idx].cpu().numpy()
                _ep_foot_z        = g_foot_z    [:T_i, idx].cpu().numpy()
                _ep_foot_xy_speed = g_foot_xy_s [:T_i, idx].cpu().numpy()
                _ep_foot_speed    = g_foot_speed[:T_i, idx].cpu().numpy()
                _ep_actions       = g_actions   [:T_i, idx].cpu().numpy()
                _ep_tau           = g_tau       [:T_i, idx].cpu().numpy()
                _ep_base_z        = g_base_z    [:T_i, idx].cpu().numpy()
                _ep_base_xy       = g_base_xy   [:T_i, idx].cpu().numpy()
                _ep_cmd_xy        = g_cmd_xy    [:T_i, idx].cpu().numpy()
                ep_gait.append(_compute_episode_gait_metrics(
                    contacts      = _ep_contacts,
                    foot_z        = _ep_foot_z,
                    foot_xy_speed = _ep_foot_xy_speed,
                    foot_speed    = _ep_foot_speed,
                    actions       = _ep_actions,
                    tau           = _ep_tau,
                    base_z        = _ep_base_z,
                    base_xy       = _ep_base_xy,
                    cmd_xy        = cmd_mean,
                    foot_radius   = foot_radius,
                ))
                if args_cli.save_raw_dir:
                    per_ep_raw.append({
                        "reward":         cur_reward[idx].item(),
                        "length":         steps,
                        "outcome":        outcome,
                        "vel_rmse":       vel_rmse_ep,
                        "fwd_disp":       cur_fwd_disp[idx].item(),
                        "mean_lin_track": (cur_lin_track[idx] / steps).item(),
                        "mean_ang_track": (cur_ang_track[idx] / steps).item(),
                        "gait":           dict(ep_gait[-1]),
                        "contacts":       _ep_contacts,
                        "foot_z":         _ep_foot_z,
                        "foot_xy_speed":  _ep_foot_xy_speed,
                        "foot_speed":     _ep_foot_speed,
                        "actions":        _ep_actions,
                        "tau":            _ep_tau,
                        "base_z":         _ep_base_z,
                        "base_xy":        _ep_base_xy,
                        "cmd_xy":         _ep_cmd_xy,
                    })
                cur_reward[idx]    = 0.0
                cur_length[idx]    = 0.0
                cur_lin_track[idx] = 0.0
                cur_ang_track[idx] = 0.0
                cur_track_err[idx] = 0.0
                cur_fwd_disp[idx]  = 0.0
                # Per-env gait buffer slots will be overwritten on the next steps
                # of the new episode; no explicit zeroing needed.
                if len(ep_rewards) % 10 == 0:
                    print(f"  episodes: {len(ep_rewards):3d}/{num_eps}  "
                          f"success: {success_count}/{len(ep_rewards)}  "
                          f"({100*success_count/max(len(ep_rewards),1):.1f}%)", end="\r")

    env.close()
    print()

    n          = len(ep_rewards)
    mean_rew, std_rew = statistics.mean(ep_rewards), (statistics.stdev(ep_rewards) if n > 1 else 0.0)
    mean_len, std_len = statistics.mean(ep_lengths), (statistics.stdev(ep_lengths) if n > 1 else 0.0)
    mean_lt,  std_lt  = statistics.mean(ep_lin_track), (statistics.stdev(ep_lin_track) if n > 1 else 0.0)
    mean_at,  std_at  = statistics.mean(ep_ang_track), (statistics.stdev(ep_ang_track) if n > 1 else 0.0)
    mean_te,  std_te  = statistics.mean(ep_track_err), (statistics.stdev(ep_track_err) if n > 1 else 0.0)
    mean_fd,  std_fd  = statistics.mean(ep_fwd_disp), (statistics.stdev(ep_fwd_disp) if n > 1 else 0.0)
    success      = 100.0 * success_count / n
    partial_rate = 100.0 * partial_count / n
    fall_rate    = 100.0 * fall_count    / n
    # Survival = success + partial (the robot didn't fall — strict spec-sheet def).
    survival_rate = 100.0 * (success_count + partial_count) / n
    mean_ep_s = mean_len * step_dt
    std_ep_s  = std_len  * step_dt
    gait_agg  = _mean_std_gait(ep_gait)
    # Threshold checks from the spec sheet.
    surv_pass = "PASS" if survival_rate >= 80.0 else "FAIL"
    rmse_pass = "PASS" if mean_te <= args_cli.vel_rmse_threshold else "FAIL"

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  GO2 OOD Evaluation — DR×{dr_scale:.1f}    "
          f"(T={args_cli.episode_length_s:.0f}s, spec-sheet metrics)")
    print(f"{sep}")
    print(f"  Method        : {method.upper()}  priv={priv_mode}  latent={latent_dim}")
    print(f"  Checkpoint    : {os.path.basename(args_cli.checkpoint)}")
    print(f"  Total episodes: {n}    (spec sheet N=100)")
    print(f"{sep}")
    print(f"  Survival rate : {survival_rate:6.1f} %     [{surv_pass} ≥ 80%]   "
          f"({success_count + partial_count}/{n} didn't fall)")
    print(f"   ├ Success   : {success:6.1f} %     ({success_count}/{n})    "
          f"(time-out AND vel_rmse < {args_cli.vel_rmse_threshold} m/s)")
    print(f"   ├ Partial   : {partial_rate:6.1f} %     ({partial_count}/{n})    "
          f"(time-out AND vel_rmse ≥ {args_cli.vel_rmse_threshold} m/s)")
    print(f"   └ Fail      : {fall_rate:6.1f} %     ({fall_count}/{n} fell)")
    print(f"  Episode len   : {mean_ep_s:6.2f} ± {std_ep_s:.2f} s   ({mean_len:.0f} ± {std_len:.0f} steps)")
    print(f"  Cum. reward   : {mean_rew:+8.2f} ± {std_rew:.2f}")
    print(f"  Vel-track RMSE: {mean_te:.4f} ± {std_te:.4f} m/s    [{rmse_pass} < {args_cli.vel_rmse_threshold} m/s]")
    print(f"  Fwd. displ.   : {mean_fd:+.3f} ± {std_fd:.3f} m       (monotone with reward)")
    print(f"  Lin vel track : {mean_lt:.4f} ± {std_lt:.4f}   (exp(-err²/0.25), 1.0 = perfect)")
    print(f"  Ang vel track : {mean_at:.4f} ± {std_at:.4f}   (exp(-err²/0.25), 1.0 = perfect)")
    print(f"{sep}")
    print(f"  Gait quality  :  (shared with scripts/sim2sim/sim2sim_go2.py — see gait_metrics.py)")
    print(f"   gait_adh     : {gait_agg['gait_adh']:.4f} ± {gait_agg['gait_adh_std']:.4f}     (1.0 = perfect trot)")
    print(f"   clear_err    : {gait_agg['clear_err']:.4f} ± {gait_agg['clear_err_std']:.4f}     (0.0 = perfect swing-foot clearance)")
    print(f"   slip_rate    : {gait_agg['slip_rate']:.4f} ± {gait_agg['slip_rate_std']:.4f}     (0.0 = no foot slip)")
    print(f"   smoothness   : {gait_agg['smoothness']:.4f} ± {gait_agg['smoothness_std']:.4f}     (0.0 = no action jerk)")
    print(f"   base_z_var   : {gait_agg['base_z_var']:.6f} ± {gait_agg['base_z_var_std']:.6f}     (0.0 = perfectly stable trunk)")
    print(f"   contact_sym  : {gait_agg['contact_sym']:.4f} ± {gait_agg['contact_sym_std']:.4f}     (1.0 = perfect trot alternation)")
    print(f"   stride_var   : {gait_agg['stride_var']:.4f} ± {gait_agg['stride_var_std']:.4f}     (0.0 = perfectly consistent stride)")
    print(f"   jtorque_var  : {gait_agg['jtorque_var']:.4f} ± {gait_agg['jtorque_var_std']:.4f}     (0.0 = perfectly steady torques)")
    print(f"{sep}\n")

    if args_cli.results_file:
        rpath    = args_cli.results_file
        os.makedirs(os.path.dirname(os.path.abspath(rpath)), exist_ok=True)
        new_file = not os.path.exists(rpath)
        gait_cols   = [f"{c}{suf}" for c in GAIT_METRIC_NAMES for suf in ("", "_std")]
        gait_values = [f"{gait_agg[c]:.6f}" for c in gait_cols]
        with open(rpath, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["sim", "method", "priv_mode", "latent_dim", "dr_scale",
                            "terrain", "disturbance", "episode_length_s",
                            "mean_reward", "std_reward",
                            "mean_length", "std_length",
                            "success_rate", "partial_rate", "fall_rate", "survival_rate",
                            "mean_lin_track", "std_lin_track",
                            "mean_ang_track", "std_ang_track",
                            "mean_track_err", "std_track_err",
                            "mean_fwd_disp", "std_fwd_disp",
                            *gait_cols,
                            "episodes", "checkpoint", "timestamp"])
            lat_col = "N/A" if method == "baseline" else str(latent_dim)
            prv_col = "BASE" if method == "baseline" else priv_mode
            w.writerow(["isaac", method.upper(), prv_col, lat_col, f"{dr_scale:.1f}",
                        terrain_tag, dist_tag, f"{args_cli.episode_length_s:.1f}",
                        f"{mean_rew:.4f}", f"{std_rew:.4f}",
                        f"{mean_len:.1f}", f"{std_len:.1f}",
                        f"{success:.1f}", f"{partial_rate:.1f}",
                        f"{fall_rate:.1f}", f"{survival_rate:.1f}",
                        f"{mean_lt:.4f}", f"{std_lt:.4f}",
                        f"{mean_at:.4f}", f"{std_at:.4f}",
                        f"{mean_te:.4f}", f"{std_te:.4f}",
                        f"{mean_fd:.4f}", f"{std_fd:.4f}",
                        *gait_values,
                        n,
                        os.path.basename(args_cli.checkpoint),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        print(f"[eval] results saved → {rpath}")

    # ── Raw per-episode + per-step dump (for future analysis) ───────────────
    if args_cli.save_raw_dir and per_ep_raw:
        import json
        os.makedirs(args_cli.save_raw_dir, exist_ok=True)
        lat_tag = "NA" if method == "baseline" else str(latent_dim)
        prv_tag = "BASE" if method == "baseline" else priv_mode
        tag = f"isaac_{method}_{prv_tag}_l{lat_tag}_dr{dr_scale:.1f}".replace("/", "_")
        # JSON: per-episode scalars (small, lossless, human-readable)
        json_path = os.path.join(args_cli.save_raw_dir, f"{tag}.json")
        with open(json_path, "w") as f:
            json.dump({
                "schema_version": 1,
                "sim": "isaac", "method": method.upper(), "priv_mode": prv_tag,
                "latent_dim": lat_tag, "dr_scale": dr_scale,
                "episode_length_s": args_cli.episode_length_s,
                "vel_rmse_threshold": args_cli.vel_rmse_threshold,
                "checkpoint": os.path.basename(args_cli.checkpoint),
                "episodes": [{
                    "reward":         e["reward"],
                    "length":         e["length"],
                    "outcome":        e["outcome"],
                    "vel_rmse":       e["vel_rmse"],
                    "fwd_disp":       e["fwd_disp"],
                    "mean_lin_track": e["mean_lin_track"],
                    "mean_ang_track": e["mean_ang_track"],
                    **e["gait"],
                } for e in per_ep_raw],
            }, f, indent=2)
        # NPZ: per-step time series, episodes concatenated along time + ep_lengths
        npz_path = os.path.join(args_cli.save_raw_dir, f"{tag}.npz")
        import numpy as _np
        np.savez_compressed(npz_path,
            ep_lengths   = _np.array([e["length"] for e in per_ep_raw], dtype=_np.int32),
            contacts     = _np.concatenate([e["contacts"]      for e in per_ep_raw], 0).astype(_np.float32),
            foot_z       = _np.concatenate([e["foot_z"]        for e in per_ep_raw], 0).astype(_np.float32),
            foot_xy_speed= _np.concatenate([e["foot_xy_speed"] for e in per_ep_raw], 0).astype(_np.float32),
            foot_speed   = _np.concatenate([e["foot_speed"]    for e in per_ep_raw], 0).astype(_np.float32),
            actions      = _np.concatenate([e["actions"]       for e in per_ep_raw], 0).astype(_np.float32),
            tau          = _np.concatenate([e["tau"]           for e in per_ep_raw], 0).astype(_np.float32),
            base_z       = _np.concatenate([e["base_z"]        for e in per_ep_raw], 0).astype(_np.float32),
            base_xy      = _np.concatenate([e["base_xy"]       for e in per_ep_raw], 0).astype(_np.float32),
            cmd_xy       = _np.concatenate([e["cmd_xy"]        for e in per_ep_raw], 0).astype(_np.float32),
        )
        print(f"[eval] raw data → {json_path}")
        print(f"[eval]            {npz_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

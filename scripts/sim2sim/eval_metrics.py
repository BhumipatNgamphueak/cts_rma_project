"""eval_metrics.py — Multi-condition Sim2Sim experiment for cts_rma_project.

Adapted from OpenTopic's `scripts/sim2sim/eval_metrics.py` (4-condition design)
to evaluate Baseline / RMA / CTS policies trained in Isaac Lab and deployed in
MuJoCo via the existing `sim2sim_go2.py` primitives.

Differences from OpenTopic's version:
  * 37-dim proprioceptive obs (not 48-dim) — uses sim2sim_go2.get_obs()
  * Decimation=2 / 100 Hz policy (not decimation=4 / 50 Hz)
  * Flat ±23.5 N·m torque clip (not Go2HV velocity-dependent curve)
  * Methods: baseline, rma, cts (no TXL)
  * No ext_force / push DR (sim2sim_go2.EpisodeDR doesn't model these)

What is preserved from OpenTopic:
  * Per-step rich behaviour metrics
      vel_rmse, fwd_disp, gait_adh, clear_err, slip, smoothness,
      base_z_var, contact_sym, stride_var, jtorque_var
  * 4 conditions: Cond1=Isaac1× (ref), Cond2=Isaac2× (ref), Cond3=MuJoCo1×, Cond4=MuJoCo2×
  * JSON + TXT report writers

Usage:
    python scripts/sim2sim/eval_metrics.py \\
        --baseline_ckpt logs/baseline/<run>/model_final.pt \\
        --rma_ckpt      logs/rma/<run>/model_final.pt \\
        --cts_ckpt      logs/cts/<run>/model_final.pt \\
        --num_episodes 30 \\
        --isaac_baseline_1x 1900 --isaac_baseline_2x 1500 \\
        --isaac_rma_1x      2100 --isaac_rma_2x      1700 \\
        --isaac_cts_1x      2200 --isaac_cts_2x      1800 \\
        --output sim2sim_report.txt --seed 42
"""

import argparse
import collections
import datetime
import json
import math
import os
import sys

import numpy as np
import torch
import mujoco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sim2sim_go2 as s2s

# ── Evaluation constants ──────────────────────────────────────────────────────
T_SUCCESS_S           = 10.0   # survival window for "fully survived" classification
FOOT_CLEARANCE_TARGET = 0.05   # m above ground (target swing foot height)
TROT_A = np.array([1., 0., 0., 1.])   # ideal trot phase A: FL + RR
TROT_B = np.array([0., 1., 1., 0.])   # ideal trot phase B: FR + RL


# ── Per-step metric helpers (adapted from OpenTopic eval_metrics.py) ─────────

def _foot_lin_vel(d: mujoco.MjData) -> np.ndarray:
    """World-frame linear velocity of each foot (4×3). Foot geom is on calf body
    in the menagerie model so we read d.cvel of the calf and rotate to world."""
    vels = np.zeros((4, 3))
    for fi in range(4):
        bid    = s2s.LEG_BODY_IDS[fi][2]              # calf body
        R_calf = d.xmat[bid].reshape(3, 3)            # local → world
        vels[fi] = R_calf @ d.cvel[bid, 3:6]          # cvel[3:6] = linear vel (local)
    return vels


def _gait_adherence(contacts: np.ndarray) -> float:
    """1.0 = perfect trot (FL+RR or FR+RL synchronised), 0 = anti-trot."""
    c = np.array(contacts, dtype=np.float64)
    err = min(np.sum((TROT_A - c) ** 2), np.sum((TROT_B - c) ** 2))
    return float(np.exp(-err))


def _clearance_error(d, contacts, foot_radius) -> float:
    """During swing (contact=0), penalise (z - target_z)² × |xy_speed|.
    Encourages legs that lift TO the target height while moving forward."""
    target_z = foot_radius + FOOT_CLEARANCE_TARGET
    fv = _foot_lin_vel(d)
    total = 0.0
    for fi in range(4):
        if contacts[fi] < 0.5:
            fz  = float(d.geom_xpos[s2s.FOOT_GEOM_IDS[fi], 2])
            vxy = float(np.linalg.norm(fv[fi, :2]))
            total += (target_z - fz) ** 2 * vxy
    return total


def _slip(contacts, prev_contacts, d) -> float:
    """At each touchdown (contact transitions 0→1), foot speed should be near 0.
    Returns sum of slip speeds at touchdown events this step."""
    fv = _foot_lin_vel(d)
    total = 0.0
    for fi in range(4):
        if contacts[fi] > 0.5 and prev_contacts[fi] < 0.5:
            total += float(np.linalg.norm(fv[fi]))
    return total


def _smoothness(action_dq) -> float:
    """Discrete second-derivative norm: ||a[t-2] - 2 a[t-1] + a[t]||²."""
    if len(action_dq) < 3:
        return 0.0
    a, b, c = list(action_dq)[0], list(action_dq)[1], list(action_dq)[2]
    return float(np.sum((a - 2.0 * b + c) ** 2))


def _contact_sym(contact_hist) -> float:
    """Fraction of steps where (FL+RR) and (FR+RL) groups are in opposite states."""
    if len(contact_hist) < 4:
        return 0.0
    c  = np.array(contact_hist)
    g1 = (c[:, 0] + c[:, 3]) > 0.5
    g2 = (c[:, 1] + c[:, 2]) > 0.5
    return float((g1 != g2).mean())


def _stride_var(events, cmd_dir) -> float:
    """Variance of forward stride lengths across all feet."""
    per_foot  = [[] for _ in range(4)]
    last_xy   = [None] * 4
    for fi, bxy in events:
        if last_xy[fi] is not None:
            per_foot[fi].append(abs(float(np.dot(bxy - last_xy[fi], cmd_dir))))
        last_xy[fi] = bxy
    all_s = [s for foot in per_foot for s in foot]
    return float(np.var(all_s)) if len(all_s) >= 2 else 0.0


# ── Inference helper (handles all three methods) ─────────────────────────────

def _policy_step(method, policy, obs_np, obs_hist, history_len, device):
    """Run one inference step. Returns (action_np, obs_hist).

    obs_hist is updated in place for CTS; passed through unchanged otherwise.
    """
    if method == "cts":
        obs_hist = torch.roll(obs_hist, -1, dims=1)
        obs_hist[0, -1, :] = torch.from_numpy(obs_np).to(device)
        flag      = torch.zeros(1, 1, device=device)
        policy_in = torch.cat([obs_hist.reshape(1, -1), flag], dim=1)
        action_t  = policy.act_inference(policy_in)
    elif method == "rma":
        obs_t    = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        action_t = policy.act_inference(obs_t)
    else:  # baseline
        obs_t    = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        action_t = policy(obs_t)
    return action_t.squeeze(0).cpu().numpy(), obs_hist


# ── Single-episode runner with metric collection ─────────────────────────────

def run_episode(m, d, policy, method, device,
                joint_perm, inv_joint_perm,
                vel_cmd, max_steps,
                spawn_h, dr, foot_radius,
                history_len, no_dr=False, random_init=False):
    """Run one episode with full metric collection.

    Returns dict with: n_steps, survived, outcome, reward,
    vel_rmse, fwd_disp, gait_adh, clear_err, slip_rate,
    smoothness, base_z_var, contact_sym, stride_var, jtorque_var.
    """
    # DR + reset
    if no_dr:
        dr.reset_to_nominal()
        delay_steps = 1
    else:
        params = dr.sample()
        dr.apply(params)
        delay_steps = max(1, round(params["delay_ms"] / (s2s.POLICY_DT * 1000)))
    mujoco.mj_forward(m, d)

    s2s._reset_pose(m, d, spawn_h, randomize=random_init, vel_cmd=vel_cmd)

    # Buffers
    null_action = np.zeros(12, dtype=np.float32)
    action_buf  = collections.deque([null_action.copy()] * (delay_steps + 1),
                                     maxlen=delay_steps + 1)
    obs_hist = None
    if method == "cts" and history_len is not None:
        # Keep history at zeros — matches training cts_env._reset_idx.
        obs_hist = torch.zeros(1, history_len, 37, device=device)

    rs = s2s.RewardState()

    # Per-step metric accumulators
    action_dq      = collections.deque(maxlen=3)
    contacts_hist  = []
    vel_rmse_list  = []
    base_z_list    = []
    gait_adh_list  = []
    clear_err_list = []
    smooth_list    = []
    torques_all    = []
    contact_events = []
    prev_contacts  = np.zeros(4)
    slip_total, slip_events = 0.0, 0
    cum_reward     = 0.0
    fwd_integrated = 0.0

    done_reason = "timeout"
    step        = 0

    with torch.no_grad():
        while True:
            obs_np = s2s.get_obs(m, d, vel_cmd, joint_perm)
            action_np, obs_hist = _policy_step(
                method, policy, obs_np, obs_hist, history_len, device,
            )
            action_buf.appendleft(action_np.copy())
            delayed = action_buf[-1]

            d.ctrl[:] = s2s.compute_target_q(delayed, inv_joint_perm)
            for _ in range(s2s.DECIMATION):
                mujoco.mj_step(m, d)
            tau = d.actuator_force.copy()

            cum_reward += s2s.compute_step_reward(m, d, vel_cmd, action_np, tau, rs)

            # ── metrics ─────────────────────────────────────────────────────
            contacts = s2s.get_foot_contacts(m, d)
            R        = d.xmat[s2s.BASE_BODY_ID].reshape(3, 3)
            lv_b     = R.T @ d.qvel[0:3]
            base_xy  = np.array([d.qpos[0], d.qpos[1]])
            fwd_integrated += lv_b[0] * s2s.POLICY_DT

            rmse = math.sqrt((vel_cmd[0] - lv_b[0])**2 + (vel_cmd[1] - lv_b[1])**2)
            vel_rmse_list.append(rmse)
            base_z_list.append(float(d.qpos[2]))
            gait_adh_list.append(_gait_adherence(contacts))
            clear_err_list.append(_clearance_error(d, contacts, foot_radius))
            contacts_hist.append(contacts.copy())
            torques_all.append(tau.copy())

            action_dq.appendleft(action_np.copy())
            if step >= 2:
                smooth_list.append(_smoothness(action_dq))

            sl = _slip(contacts, prev_contacts, d)
            if sl > 0:
                slip_total  += sl
                slip_events += 1
            for fi in range(4):
                if contacts[fi] > 0.5 and prev_contacts[fi] < 0.5:
                    contact_events.append((fi, base_xy.copy()))
            prev_contacts = contacts.copy()

            step += 1
            done, done_reason = s2s.is_done(m, d, step, max_steps)
            if done:
                break

    tau_arr = np.array(torques_all)
    cmd_dir = np.array([vel_cmd[0], vel_cmd[1]])
    spd = float(np.linalg.norm(cmd_dir))
    cmd_dir = cmd_dir / spd if spd > 1e-6 else np.array([1.0, 0.0])

    survived = (done_reason == "timeout")

    return {
        "n_steps":     step,
        "survived":    survived,
        "outcome":     "success" if survived else "fail",
        "reason":      done_reason,
        "reward":      cum_reward,
        "vel_rmse":    float(np.mean(vel_rmse_list))   if vel_rmse_list   else 9.99,
        "fwd_disp":    fwd_integrated,
        "gait_adh":    float(np.mean(gait_adh_list))   if gait_adh_list   else 0.0,
        "clear_err":   float(np.mean(clear_err_list))  if clear_err_list  else 0.0,
        "slip_rate":   slip_total / slip_events         if slip_events > 0 else 0.0,
        "smoothness":  float(np.mean(smooth_list))      if smooth_list     else 0.0,
        "base_z_var":  float(np.var(base_z_list))       if len(base_z_list) > 1 else 0.0,
        "contact_sym": _contact_sym(contacts_hist),
        "stride_var":  _stride_var(contact_events, cmd_dir),
        "jtorque_var": float(np.mean(np.var(tau_arr, axis=0))) if len(tau_arr) > 1 else 0.0,
    }


# ── Condition runner ──────────────────────────────────────────────────────────

def run_condition(m, d, policy, method, device,
                  joint_perm, inv_joint_perm, vel_cmd,
                  max_steps, spawn_h, dr, foot_radius,
                  history_len, n_ep, label, seed):
    """Run n_ep episodes for one (method, dr_scale) condition."""
    np.random.seed(seed)
    print(f"\n  [{label}]  {n_ep} episodes × {max_steps} steps …", flush=True)
    results = []
    for ep in range(n_ep):
        r = run_episode(m, d, policy, method, device,
                        joint_perm, inv_joint_perm, vel_cmd,
                        max_steps, spawn_h, dr, foot_radius,
                        history_len)
        results.append(r)
        sr  = sum(x["survived"] for x in results) / len(results) * 100
        rmu = float(np.mean([x["reward"] for x in results]))
        print(f"    ep {ep+1:3d}/{n_ep}  survival={sr:.0f}%  "
              f"reward={rmu:+.1f}  rmse={results[-1]['vel_rmse']:.3f}", flush=True)
    return results


def _agg(results, key):
    vals = [r[key] for r in results]
    return float(np.mean(vals)), float(np.std(vals))


# ── JSON writer ───────────────────────────────────────────────────────────────

def write_json(cond_results: dict, args, episode_length_s, out_path):
    def _cond_summary(results):
        metrics = {}
        for key in ["reward", "vel_rmse", "fwd_disp", "gait_adh", "clear_err",
                    "slip_rate", "smoothness", "base_z_var", "contact_sym",
                    "stride_var", "jtorque_var"]:
            mu, sd = _agg(results, key)
            metrics[key] = {"mean": mu, "std": sd}
        return {
            "n_episodes":    len(results),
            "survival_rate": sum(r["survived"] for r in results) / len(results),
            **metrics,
        }

    data = {
        "generated":        datetime.datetime.now().isoformat(),
        "vel_cmd":          [args.vel_x, args.vel_y, args.vel_yaw],
        "num_episodes":     args.num_episodes,
        "episode_length_s": episode_length_s,
        "seed":             args.seed,
        "conditions":       {k: _cond_summary(v) for k, v in cond_results.items()},
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[json]   → {os.path.abspath(out_path)}")


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(cond_results: dict, args, episode_length_s, out_path):
    W = 78
    lines = []

    def hdr(t=""):
        lines.append("=" * W)
        if t:
            lines.append(f"  {t}")
            lines.append("=" * W)

    def sec(t):
        lines.append("")
        lines.append("-" * W)
        lines.append(f"  {t}")
        lines.append("-" * W)

    methods_present = []
    for m_ in ["baseline", "rma", "cts"]:
        if f"{m_}_1x" in cond_results:
            methods_present.append(m_)

    LABELS = {
        "baseline_1x": "Baseline 1×DR", "baseline_2x": "Baseline 2×DR",
        "rma_1x":      "RMA      1×DR", "rma_2x":      "RMA      2×DR",
        "cts_1x":      "CTS      1×DR", "cts_2x":      "CTS      2×DR",
    }

    # Header
    hdr("Sim2Sim Evaluation Report — GO2 Locomotion (cts_rma_project)")
    lines.append(f"  Generated      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Episodes/cond  : {args.num_episodes}")
    lines.append(f"  Episode length : {episode_length_s:.1f} s")
    lines.append(f"  T_success      : {T_SUCCESS_S:.0f} s")
    lines.append(f"  Vel command    : vx={args.vel_x:+.2f} m/s  vy={args.vel_y:+.2f}  wz={args.vel_yaw:+.2f}")
    lines.append(f"  Methods        : {', '.join(m_.upper() for m_ in methods_present)}")
    lines.append(f"  Seed           : {args.seed}")
    lines.append("")
    lines.append("  4 TEST CONDITIONS:")
    lines.append("    Cond 1 — 1×DR in Isaac Lab   (reference, user-provided)")
    lines.append("    Cond 2 — 2×DR in Isaac Lab   (reference, user-provided)")
    lines.append("    Cond 3 — 1×DR in MuJoCo      (this script)")
    lines.append("    Cond 4 — 2×DR in MuJoCo      (this script)")
    lines.append("")
    lines.append("  Isaac Lab reference rewards (Conds 1 & 2):")
    for method in methods_present:
        r1 = getattr(args, f"isaac_{method}_1x", None)
        r2 = getattr(args, f"isaac_{method}_2x", None)
        s1 = f"{r1:+.1f}" if r1 is not None else "—"
        s2 = f"{r2:+.1f}" if r2 is not None else "—"
        lines.append(f"    {method.upper():<10}  1×DR={s1:>9}    2×DR={s2:>9}")

    # Per-condition table
    sec("Per-condition summary (Cond 3 & 4 — MuJoCo episodes)")
    lines.append(f"  {'Condition':<16} {'Survival':>9} {'Reward':>10} {'StdR':>7}  "
                 f"{'vRMSE':>7} {'GaitAdh':>8} {'ClearErr':>9} {'Slip':>7} {'Smooth':>8}")
    lines.append("  " + "-" * (W - 2))
    for method in methods_present:
        for dr_key in ("1x", "2x"):
            cond = f"{method}_{dr_key}"
            res  = cond_results[cond]
            sr   = sum(r["survived"] for r in res) / len(res) * 100
            mr, sr_  = _agg(res, "reward")
            v, _v    = _agg(res, "vel_rmse")
            g, _g    = _agg(res, "gait_adh")
            c, _c    = _agg(res, "clear_err")
            s, _s    = _agg(res, "slip_rate")
            sm, _sm  = _agg(res, "smoothness")
            lines.append(f"  {LABELS[cond]:<16} {sr:>8.0f}% {mr:>+10.1f} {sr_:>7.1f}  "
                         f"{v:>7.3f} {g:>8.3f} {c:>9.4f} {s:>7.3f} {sm:>8.3f}")

    # Sim2Sim transfer gap (MuJoCo - Isaac, per method per scale)
    sec("Sim2Sim transfer gap  (MuJoCo reward − Isaac reward)")
    lines.append(f"  {'Method':<10} {'1×DR Δ':>14} {'2×DR Δ':>14}     "
                 f"(negative = MuJoCo worse than Isaac)")
    lines.append("  " + "-" * (W - 2))
    for method in methods_present:
        r1_isaac = getattr(args, f"isaac_{method}_1x", None)
        r2_isaac = getattr(args, f"isaac_{method}_2x", None)
        r1_mu  = _agg(cond_results[f"{method}_1x"], "reward")[0]
        r2_mu  = _agg(cond_results[f"{method}_2x"], "reward")[0]
        d1 = f"{r1_mu - r1_isaac:+.1f}" if r1_isaac is not None else "—"
        d2 = f"{r2_mu - r2_isaac:+.1f}" if r2_isaac is not None else "—"
        lines.append(f"  {method.upper():<10} {d1:>14} {d2:>14}")

    # OOD degradation (1× → 2× drop, in MuJoCo)
    sec("OOD degradation in MuJoCo  (2×DR vs 1×DR, lower drop = more robust)")
    lines.append(f"  {'Method':<10} {'ΔReward':>10} {'ΔSurvival':>12} {'ΔvRMSE':>10}")
    lines.append("  " + "-" * (W - 2))
    for method in methods_present:
        r1 = cond_results[f"{method}_1x"]
        r2 = cond_results[f"{method}_2x"]
        dr_   = _agg(r2, "reward")[0]   - _agg(r1, "reward")[0]
        dsr   = (sum(x["survived"] for x in r2) - sum(x["survived"] for x in r1)) \
                / len(r1) * 100
        drmse = _agg(r2, "vel_rmse")[0] - _agg(r1, "vel_rmse")[0]
        lines.append(f"  {method.upper():<10} {dr_:>+10.1f} {dsr:>+11.0f}% {drmse:>+10.3f}")

    lines.append("")
    lines.append("=" * W)

    output = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(output)
    print(f"\n[report] → {os.path.abspath(out_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-condition sim2sim experiment for cts_rma_project")

    # Checkpoints (at least one required)
    parser.add_argument("--baseline_ckpt", default=None)
    parser.add_argument("--rma_ckpt",      default=None)
    parser.add_argument("--cts_ckpt",      default=None)
    parser.add_argument("--scene_xml",     default=None)
    parser.add_argument("--latent_dim",    type=int, default=8)
    parser.add_argument("--history_len",   type=int, default=50)

    # Experiment design
    parser.add_argument("--num_episodes",   type=int,   default=30)
    parser.add_argument("--dr_scale_train", type=float, default=1.0,
                        help="Cond 3 DR scale (training range)")
    parser.add_argument("--dr_scale_ood",   type=float, default=2.0,
                        help="Cond 4 DR scale (OOD)")
    parser.add_argument("--vel_x",          type=float, default=0.5)
    parser.add_argument("--vel_y",          type=float, default=0.0)
    parser.add_argument("--vel_yaw",        type=float, default=0.0)

    # Isaac Lab reference rewards (Conds 1 & 2)
    parser.add_argument("--isaac_baseline_1x", type=float, default=None)
    parser.add_argument("--isaac_baseline_2x", type=float, default=None)
    parser.add_argument("--isaac_rma_1x",      type=float, default=None)
    parser.add_argument("--isaac_rma_2x",      type=float, default=None)
    parser.add_argument("--isaac_cts_1x",      type=float, default=None)
    parser.add_argument("--isaac_cts_2x",      type=float, default=None)

    parser.add_argument("--output", default="sim2sim_report.txt")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    methods = {}
    if args.baseline_ckpt: methods["baseline"] = args.baseline_ckpt
    if args.rma_ckpt:      methods["rma"]      = args.rma_ckpt
    if args.cts_ckpt:      methods["cts"]      = args.cts_ckpt
    if not methods:
        parser.error("at least one of --baseline_ckpt / --rma_ckpt / --cts_ckpt required")

    device = args.device if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scene_xml = args.scene_xml or s2s.SCENE_XML
    if not os.path.exists(scene_xml):
        raise FileNotFoundError(f"Scene XML not found: {scene_xml}")

    print(f"\n{'='*60}")
    print(f"  Sim2Sim Multi-Condition Experiment — cts_rma_project")
    print(f"  episodes={args.num_episodes}  methods={list(methods)}")
    print(f"  Cond3 = MuJoCo dr_scale={args.dr_scale_train:.1f}×")
    print(f"  Cond4 = MuJoCo dr_scale={args.dr_scale_ood:.1f}×")
    print(f"{'='*60}\n")

    # ── MuJoCo setup (shared) ─────────────────────────────────────────────────
    m = mujoco.MjModel.from_xml_path(scene_xml)
    d = mujoco.MjData(m)
    s2s.init_model_ids(m)
    s2s.fix_model_physics(m)
    spawn_h        = s2s.compute_spawn_height(m, d)
    joint_perm     = s2s.verify_joint_order(m)
    inv_joint_perm = np.argsort(joint_perm)
    foot_radius    = float(m.geom_size[s2s.FOOT_GEOM_IDS[0], 0])
    nominal_masses = m.body_mass.copy()
    vel_cmd        = np.array([args.vel_x, args.vel_y, args.vel_yaw], dtype=np.float32)
    max_steps      = int(s2s.MAX_EPISODE_S / s2s.POLICY_DT)
    episode_length_s = max_steps * s2s.POLICY_DT

    print(f"Episode: {max_steps} steps × {s2s.POLICY_DT*1000:.0f} ms = {episode_length_s:.1f} s\n")

    # ── Run all conditions ────────────────────────────────────────────────────
    cond_results = {}
    for method, ckpt in methods.items():
        print(f"\n{'─'*60}  {method.upper()}")
        policy, hist_len = s2s.load_policy(
            method, ckpt, device,
            latent_dim=args.latent_dim, history_len=args.history_len,
        )

        for dr_scale, dr_key, cond_n in [
            (args.dr_scale_train, "1x", 3),
            (args.dr_scale_ood,   "2x", 4),
        ]:
            dr = s2s.EpisodeDR(m, nominal_masses, dr_scale)
            label = f"Cond {cond_n} — {method.upper()} MuJoCo {dr_key.replace('x','×')}DR"
            cond_results[f"{method}_{dr_key}"] = run_condition(
                m, d, policy, method, device,
                joint_perm, inv_joint_perm, vel_cmd,
                max_steps, spawn_h, dr, foot_radius,
                hist_len,
                n_ep=args.num_episodes,
                label=label,
                seed=args.seed,
            )
            res = cond_results[f"{method}_{dr_key}"]
            sr  = sum(r["survived"] for r in res) / len(res) * 100
            mu_r, sd_r = _agg(res, "reward")
            print(f"  Result: reward={mu_r:+.1f}±{sd_r:.1f}  "
                  f"survival={sr:.0f}%  "
                  f"vel_rmse={_agg(res,'vel_rmse')[0]:.3f} m/s")

    # ── Write report ──────────────────────────────────────────────────────────
    write_report(cond_results, args, episode_length_s, args.output)
    json_path = os.path.splitext(args.output)[0] + ".json"
    write_json(cond_results, args, episode_length_s, json_path)


if __name__ == "__main__":
    main()

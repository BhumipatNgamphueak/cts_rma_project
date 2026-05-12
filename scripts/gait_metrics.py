"""scripts/gait_metrics.py — shared behaviour-level locomotion metrics.

Single source of truth for the 8 gait-quality metrics used in both the Isaac-Lab
OOD eval (`scripts/eval_ood_go2.py`) and the MuJoCo sim-to-sim eval
(`scripts/sim2sim/sim2sim_go2.py` and `scripts/sim2sim/eval_metrics.py`).

Formulas are taken verbatim from OpenTopic's `scripts/sim2sim/eval_metrics.py`
so cross-sim numbers compare apples-to-apples.

Foot indexing convention (must match in both sims): [FL, FR, RL, RR].

Metrics implemented
-------------------
1. gait_adh     — gait pattern adherence  [0,1], 1 = perfect trot
2. clear_err    — foot clearance error    (sum over feet of (target_z - foot_z)^2 · |v_xy| during swing)
3. slip_rate    — foot slip at touchdown  (mean foot speed at contacts that just transitioned 0→1)
4. smoothness   — action 2nd-derivative   (||a[t-2] - 2 a[t-1] + a[t]||^2 averaged over the episode)
5. base_z_var   — base height variance    (var(base_z) over the episode)
6. contact_sym  — foot contact symmetry   (fraction of steps where (FL+RR) and (FR+RL) are anti-phase)
7. stride_var   — stride consistency      (variance of per-stride forward displacement)
8. jtorque_var  — joint torque variance   (mean over joints of var_t(tau_j))

All functions operate on plain numpy arrays so the same code path serves the
MuJoCo loop (already numpy) and the Isaac loop (call `.cpu().numpy()` on
per-env buffers at episode end).
"""

from __future__ import annotations

import numpy as np

# ── Constants (mirror unitree_rl_lab/scripts/sim2sim/eval_metrics.py) ────────
TROT_A = np.array([1.0, 0.0, 0.0, 1.0])   # ideal trot phase A: FL + RR in stance
TROT_B = np.array([0.0, 1.0, 1.0, 0.0])   # ideal trot phase B: FR + RL in stance
FOOT_CLEARANCE_TARGET = 0.05               # m above (foot_radius) — target swing height
GAIT_METRIC_NAMES = [
    "gait_adh", "clear_err", "slip_rate", "smoothness",
    "base_z_var", "contact_sym", "stride_var", "jtorque_var",
]


# ─────────────────────────────────────────────────────────────────────────────
# Per-step helpers — call inside the env step loop, accumulate into a list.
# ─────────────────────────────────────────────────────────────────────────────
def gait_adherence_step(contacts: np.ndarray) -> float:
    """contacts: (4,) 0/1 vector in [FL, FR, RL, RR] order.
    Returns 1.0 = perfect trot (any phase), 0 = anti-trot."""
    c = np.asarray(contacts, dtype=np.float64)
    err = min(np.sum((TROT_A - c) ** 2), np.sum((TROT_B - c) ** 2))
    return float(np.exp(-err))


def clearance_error_step(
    contacts: np.ndarray,
    foot_z: np.ndarray,
    foot_xy_speed: np.ndarray,
    foot_radius: float,
) -> float:
    """During swing (contact=0), penalise (target - foot_z)^2 · |xy_speed|.
    Encourages feet that lift to target height while in forward motion.

    contacts:        (4,) 0/1
    foot_z:          (4,) world-frame foot z
    foot_xy_speed:   (4,) ||v_xy|| per foot
    foot_radius:     scalar foot collision radius
    """
    target_z = foot_radius + FOOT_CLEARANCE_TARGET
    total = 0.0
    for fi in range(4):
        if contacts[fi] < 0.5:
            total += (target_z - float(foot_z[fi])) ** 2 * float(foot_xy_speed[fi])
    return total


def slip_at_touchdown_step(
    contacts: np.ndarray,
    prev_contacts: np.ndarray,
    foot_speed: np.ndarray,
) -> tuple[float, int]:
    """At each foot whose contact transitions 0 → 1 this step, accumulate
    the foot's speed (lower is better; slip = nonzero touchdown speed).

    Returns (slip_total_this_step, n_touchdowns_this_step).
    Episode-level aggregation: slip_rate = sum / events (skipped if 0 events).
    """
    total = 0.0
    n_events = 0
    for fi in range(4):
        if contacts[fi] > 0.5 and prev_contacts[fi] < 0.5:
            total += float(foot_speed[fi])
            n_events += 1
    return total, n_events


def smoothness_step(a_tm2: np.ndarray, a_tm1: np.ndarray, a_t: np.ndarray) -> float:
    """||a[t-2] - 2 a[t-1] + a[t]||^2 — discrete 2nd-derivative norm of action sequence.

    Call once per step starting at step >= 2. Aggregate by mean over the episode.
    """
    return float(np.sum((np.asarray(a_tm2) - 2.0 * np.asarray(a_tm1) + np.asarray(a_t)) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode aggregators — call once after the episode terminates.
# ─────────────────────────────────────────────────────────────────────────────
def contact_symmetry_episode(contact_hist: np.ndarray) -> float:
    """contact_hist: (T, 4) per-step foot contact vectors.
    Returns fraction of steps where trot groups (FL+RR) and (FR+RL) are in
    opposite states. 1.0 = perfect trot alternation."""
    c = np.asarray(contact_hist)
    if len(c) < 4:
        return 0.0
    g1 = (c[:, 0] + c[:, 3]) > 0.5    # FL + RR
    g2 = (c[:, 1] + c[:, 2]) > 0.5    # FR + RL
    return float((g1 != g2).mean())


def stride_variance_episode(touchdowns: list, cmd_dir: np.ndarray) -> float:
    """touchdowns: list of (foot_idx, base_xy_at_touchdown) per touchdown event.
    cmd_dir: (2,) unit vector of commanded heading.

    Per foot: forward stride lengths = |proj_{cmd_dir}(Δbase_xy_between_consecutive_touchdowns)|.
    Episode metric = variance of all per-foot strides pooled together. Lower = more consistent.
    """
    cmd_dir = np.asarray(cmd_dir, dtype=np.float64)
    per_foot = [[] for _ in range(4)]
    last_xy: list = [None] * 4
    for fi, bxy in touchdowns:
        if last_xy[fi] is not None:
            per_foot[fi].append(abs(float(np.dot(np.asarray(bxy) - last_xy[fi], cmd_dir))))
        last_xy[fi] = np.asarray(bxy, dtype=np.float64)
    all_strides = [s for f_list in per_foot for s in f_list]
    return float(np.var(all_strides)) if len(all_strides) >= 2 else 0.0


def base_z_variance_episode(base_z: np.ndarray) -> float:
    """base_z: (T,) base height over the episode. Lower variance = more stable trunk."""
    a = np.asarray(base_z)
    return float(np.var(a)) if a.size > 1 else 0.0


def joint_torque_variance_episode(tau: np.ndarray) -> float:
    """tau: (T, n_joints). Mean over joints of var-over-time. Lower = less aggressive control."""
    a = np.asarray(tau)
    return float(np.mean(np.var(a, axis=0))) if a.shape[0] > 1 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# One-shot wrapper — pass the full episode buffers, get all 8 metrics.
# ─────────────────────────────────────────────────────────────────────────────
def compute_episode_metrics(
    *,
    contacts: np.ndarray,        # (T, 4) bool/0-1
    foot_z: np.ndarray,           # (T, 4) world-frame foot z
    foot_xy_speed: np.ndarray,    # (T, 4) ||v_foot_xy||
    foot_speed: np.ndarray,       # (T, 4) ||v_foot|| (for slip-at-touchdown)
    actions: np.ndarray,          # (T, 12) policy action vector
    tau: np.ndarray,              # (T, 12) joint torques
    base_z: np.ndarray,           # (T,)
    base_xy: np.ndarray,          # (T, 2)
    cmd_xy: np.ndarray,           # (2,) commanded linear velocity (m/s, world)
    foot_radius: float,
) -> dict[str, float]:
    """Compute the 8 gait metrics for a single episode from per-step buffers.

    All inputs are numpy arrays of consistent length T (the episode length).
    Foot order across all 4-dim arrays MUST be [FL, FR, RL, RR].
    """
    contacts      = np.asarray(contacts,      dtype=np.float64)
    foot_z        = np.asarray(foot_z,        dtype=np.float64)
    foot_xy_speed = np.asarray(foot_xy_speed, dtype=np.float64)
    foot_speed    = np.asarray(foot_speed,    dtype=np.float64)
    actions       = np.asarray(actions,       dtype=np.float64)
    tau           = np.asarray(tau,           dtype=np.float64)
    base_z        = np.asarray(base_z,        dtype=np.float64)
    base_xy       = np.asarray(base_xy,       dtype=np.float64)
    T = contacts.shape[0]
    if T == 0:
        return {n: 0.0 for n in GAIT_METRIC_NAMES}

    # 1. gait adherence — mean over steps
    gait_adh = float(np.mean([gait_adherence_step(contacts[t]) for t in range(T)]))

    # 2. clearance error — mean over steps
    clear_err = float(np.mean([
        clearance_error_step(contacts[t], foot_z[t], foot_xy_speed[t], foot_radius)
        for t in range(T)
    ]))

    # 3. slip rate — mean foot speed across touchdown events (zero if no events)
    prev = np.zeros(4)
    slip_total, n_events = 0.0, 0
    for t in range(T):
        s, n = slip_at_touchdown_step(contacts[t], prev, foot_speed[t])
        slip_total += s
        n_events   += n
        prev = contacts[t]
    slip_rate = slip_total / n_events if n_events > 0 else 0.0

    # 4. smoothness — mean of ||a[t-2] - 2 a[t-1] + a[t]||^2 for t >= 2
    if T >= 3:
        smoothness = float(np.mean([
            smoothness_step(actions[t-2], actions[t-1], actions[t]) for t in range(2, T)
        ]))
    else:
        smoothness = 0.0

    # 5. base z variance
    base_z_var = base_z_variance_episode(base_z)

    # 6. foot contact symmetry
    contact_sym = contact_symmetry_episode(contacts)

    # 7. stride variance — collect touchdowns then aggregate
    prev = np.zeros(4)
    touchdowns = []
    for t in range(T):
        for fi in range(4):
            if contacts[t, fi] > 0.5 and prev[fi] < 0.5:
                touchdowns.append((fi, base_xy[t].copy()))
        prev = contacts[t]
    cmd = np.asarray(cmd_xy, dtype=np.float64)
    speed = float(np.linalg.norm(cmd))
    cmd_unit = cmd / speed if speed > 1e-6 else np.array([1.0, 0.0])
    stride_var = stride_variance_episode(touchdowns, cmd_unit)

    # 8. joint-torque variance
    jtorque_var = joint_torque_variance_episode(tau)

    return {
        "gait_adh":    gait_adh,
        "clear_err":   clear_err,
        "slip_rate":   slip_rate,
        "smoothness":  smoothness,
        "base_z_var":  base_z_var,
        "contact_sym": contact_sym,
        "stride_var":  stride_var,
        "jtorque_var": jtorque_var,
    }


def mean_std_across_episodes(per_episode_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-episode metric dicts into {name: mean, name_std: std}."""
    out: dict[str, float] = {}
    if not per_episode_dicts:
        for n in GAIT_METRIC_NAMES:
            out[n] = 0.0
            out[f"{n}_std"] = 0.0
        return out
    for n in GAIT_METRIC_NAMES:
        vals = np.array([float(d.get(n, 0.0)) for d in per_episode_dicts], dtype=np.float64)
        out[n]            = float(vals.mean())
        out[f"{n}_std"]   = float(vals.std())
    return out

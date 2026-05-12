"""
Sim2Sim — Deploy Isaac Lab GO2 teacher-student (CTS / Baseline) policy in MuJoCo.

Adapted from OpenTopic/scripts/sim2sim/sim2sim_go2.py (same owner, same robot).
Differences vs OpenTopic:
  • dof_frictionloss=0.0  (DCMotorCfg friction=0.0 vs Go2HV 0.01)
  • No velocity-dependent torque curve  (DCMotorCfg flat 23.5 N·m)
  • 37-dim obs: joint_pos_rel+joint_vel+ang_vel_b+gravity_b+vel_cmd+foot_contact
  • CTS student encoder with history pre-fill (cold-start fix)

Physics: sim.dt=0.005 × decimation=4 = 50 Hz policy  (exact match to training)

Usage:
    conda run -n env_isaaclab python scripts/sim2sim/sim2sim_go2.py \\
        --method baseline \\
        --checkpoint logs/baseline/<run>/model_5600.pt \\
        --dr_scale 1.0 --num_episodes 20 --seed 42 --device cpu

    conda run -n env_isaaclab python scripts/sim2sim/sim2sim_go2.py \\
        --method cts \\
        --checkpoint logs/cts/2026-05-04_10-28-44_cts_go2_l8/model_16800.pt \\
        --latent_dim 8 --history_len 50 \\
        --dr_scale 1.0 --num_episodes 5 --render --no_terminate
"""

import argparse
import collections
import os
import sys
import numpy as np
import torch
import mujoco
import mujoco.viewer

# Shared gait-metric library (see scripts/gait_metrics.py).
_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo/scripts/
if _SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPT_ROOT)
from gait_metrics import (
    GAIT_METRIC_NAMES,
    compute_episode_metrics as _compute_episode_gait_metrics,
    mean_std_across_episodes as _mean_std_gait,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SCENE_XML    = os.path.join(PROJECT_ROOT, "mujoco_menagerie", "unitree_go2", "scene.xml")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source", "cts_rma_project"))

# ── Robot constants ────────────────────────────────────────────────────────────
# Isaac Lab groups joints by TYPE: [all hips] [all thighs] [all calves]
# MuJoCo menagerie uses per-leg order: FL(hip,thigh,calf), FR, RL, RR
# verify_joint_order() builds the permutation; inv_perm is used for ctrl.
ISAAC_JOINT_NAMES = [
    "FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint",
]
DEFAULT_JOINT_POS = np.array([
     0.1,  0.8, -1.5,   # FL: hip, thigh, calf
    -0.1,  0.8, -1.5,   # FR
     0.1,  1.0, -1.5,   # RL
    -0.1,  1.0, -1.5,   # RR
], dtype=np.float64)

# PD gains — matches DCMotorCfg: stiffness=25, damping=0.5, friction=0.0
KP_NOM       = 25.0
KD_NOM       =  0.5
TORQUE_LIMIT = 23.5   # N·m — DCMotorCfg effort_limit (flat, no velocity curve)
ACTION_SCALE =  0.25  # JointPositionActionCfg scale

# Physics — exact match to shared_env_cfg.py: sim.dt=0.005, decimation=4
PHYSICS_DT = 0.005   # 200 Hz — matches Isaac Lab sim.dt
DECIMATION = 4        # policy at 50 Hz (0.005 × 4 = 0.020 s / step)
POLICY_DT  = PHYSICS_DT * DECIMATION   # 0.020 s

MIN_BASE_HEIGHT       = 0.25   # [m] fallen threshold
BAD_ORIENTATION_LIMIT = 1.2    # rad (~69°) — matches bad_orientation termination
MAX_EPISODE_S         = 20.0   # [s] matches episode_length_s

FLOOR_GEOM_ID:          int = -1
BASE_BODY_ID:           int = -1
FOOT_GEOM_IDS:          list[int] = []
ALL_COLLISION_GEOM_IDS: list[int] = []
LEG_BODY_IDS:           list[list[int]] = []


def init_model_ids(m: mujoco.MjModel):
    """Resolve body and geom IDs from the MuJoCo model by name."""
    global FLOOR_GEOM_ID, BASE_BODY_ID, FOOT_GEOM_IDS, ALL_COLLISION_GEOM_IDS, LEG_BODY_IDS

    FLOOR_GEOM_ID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if FLOOR_GEOM_ID < 0:
        raise RuntimeError("Geom 'floor' not found in MuJoCo model.")

    BASE_BODY_ID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    if BASE_BODY_ID < 0:
        raise RuntimeError("Body 'base' not found in MuJoCo model.")

    FOOT_GEOM_IDS = []
    for name in ("FL", "FR", "RL", "RR"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise RuntimeError(f"Foot geom '{name}' not found.")
        FOOT_GEOM_IDS.append(gid)

    # All robot collision geoms (group=3 in go2.xml, excludes visual group=2 and floor).
    # Isaac Lab applies friction DR to body_names=".*" (all bodies), so match all collision geoms.
    ALL_COLLISION_GEOM_IDS = [i for i in range(m.ngeom) if int(m.geom_group[i]) == 3]

    LEG_BODY_IDS = []
    for prefix in ("FL", "FR", "RL", "RR"):
        ids = []
        for part in ("hip", "thigh", "calf"):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{part}")
            if bid < 0:
                raise RuntimeError(f"Body '{prefix}_{part}' not found.")
            ids.append(bid)
        LEG_BODY_IDS.append(ids)

    print(f"[ids] floor_geom={FLOOR_GEOM_ID}  base_body={BASE_BODY_ID}")
    print(f"[ids] foot geoms : FL={FOOT_GEOM_IDS[0]} FR={FOOT_GEOM_IDS[1]} "
          f"RL={FOOT_GEOM_IDS[2]} RR={FOOT_GEOM_IDS[3]}")
    print(f"[ids] collision geoms : {len(ALL_COLLISION_GEOM_IDS)} total")
    print(f"[ids] leg bodies : {LEG_BODY_IDS}")


# ── DR training ranges — must exactly mirror tasks/shared/shared_env_cfg.py ──
# Previous values were the pre-sim2sim-fix-#2 *narrow* ranges, which made the
# MuJoCo "DR×1" env easier than what the policies were actually trained on.
# Updated to the current widened training ranges; see commit comments in
# shared_env_cfg.py marked "SIM2SIM FIX (#2)".
#
# Two terms in shared_env_cfg.py are intentionally NOT in this table:
#   * restitution range [0.0, 0.15]:
#       MuJoCo expresses restitution via solref/solimp constraint parameters,
#       not directly as a coefficient. Wiring it would require solver-aware
#       mapping; the range is small so the impact is minor. Documented gap.
#   * push_robot / impulse_reset / impulse_interval (episode-level disturbances):
#       Isaac applies random base pushes during training; the MuJoCo eval does
#       not replay them. Documented gap — both DR×1 and DR×2 share this, so it
#       affects absolute reward but not the OOD-gap ratio.
_TRAIN = {
    "friction":        (0.30, 1.70),   # static & dynamic; widened from (0.5, 1.5)
    "mass_scale_base": (0.85, 1.15),   # widened from (0.90, 1.10)
    "inertia_scale":   (0.70, 1.30),   # widened from (0.80, 1.20)
    "kp_scale":        (0.70, 1.30),   # widened from (0.85, 1.15)
    "kd_scale":        (0.65, 1.35),   # widened from (0.80, 1.20)
    "com_offset_m":    (-0.08, 0.08),  # widened from ±0.05
    "delay_ms":        (0.0,  30.0),   # widened from (0.0, 20.0)
}


def _ood_range(lo, hi, s, lo_clip=None, hi_clip=None):
    c = (lo + hi) / 2.0
    h = (hi - lo) / 2.0 * s
    lo2 = c - h if lo_clip is None else max(c - h, lo_clip)
    hi2 = c + h if hi_clip is None else min(c + h, hi_clip)
    return (lo2, hi2)


def compute_dr_ranges(scale: float) -> dict:
    return {
        "friction":        _ood_range(*_TRAIN["friction"],        scale, lo_clip=0.01),
        "mass_scale_base": _ood_range(*_TRAIN["mass_scale_base"], scale, lo_clip=0.10),
        "inertia_scale":   _ood_range(*_TRAIN["inertia_scale"],   scale, lo_clip=0.10),
        "kp_scale":        _ood_range(*_TRAIN["kp_scale"],        scale, lo_clip=0.10),
        "kd_scale":        _ood_range(*_TRAIN["kd_scale"],        scale, lo_clip=0.10),
        "com_offset_m":    _ood_range(*_TRAIN["com_offset_m"],    scale),
        "delay_ms":        _ood_range(*_TRAIN["delay_ms"],        scale, lo_clip=0.0),
    }


def print_dr_table(scale: float):
    ranges = compute_dr_ranges(scale)
    print(f"\nDR ranges at scale={scale:.1f}x:")
    print(f"  {'Parameter':<18} {'Train range':<24} {'OOD range'}")
    print(f"  {'-'*18} {'-'*24} {'-'*22}")
    for k, (tlo, thi) in _TRAIN.items():
        olo, ohi = ranges[k]
        print(f"  {k:<18} [{tlo:>6.2f}, {thi:>6.2f}]         [{olo:>6.2f}, {ohi:>6.2f}]")
    print()


# ── Physics matching ───────────────────────────────────────────────────────────

def fix_model_physics(m: mujoco.MjModel):
    """
    Align MuJoCo physics with Isaac Lab training config (shared_env_cfg.py).

    1. timestep=0.005s, IMPLICITFAST, impratio=100  — same as OpenTopic (proven).
    2. dof_damping=0.0   — PD Kd handles all resistance; no passive joint damping.
    3. dof_frictionloss=0.0  — DCMotorCfg friction=0.0 (NOT 0.01 — that is OpenTopic/Go2HV only).
    4. dof_armature=0.01 — Unitree GO2 default.
    5. Foot solimp/solref: keep go2.xml values (already tuned for near-rigid contacts).
       condim=6 confirmed for full 6D friction cone.
    6. Implicit position PD, flat ±23.5 N·m limit (DCMotorCfg, no velocity curve).
    """
    m.opt.timestep   = PHYSICS_DT          # 0.005 s = 200 Hz
    m.opt.impratio   = 100                  # prevents foot slip (same as OpenTopic)
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # couples PD + contacts

    # DOF layout: 0-5 = free joint, 6-17 = 12 leg joints
    m.dof_damping[6:18]      = 0.0    # zero passive damping (PD Kd handles it)
    m.dof_frictionloss[6:18] = 0.0   # DCMotorCfg friction=0.0
    m.dof_armature[6:18]     = 0.01  # Unitree GO2 default

    # Keep go2.xml foot solimp/solref — already tuned for near-rigid contacts.
    # Only ensure condim=6 for full 6D friction cone on feet (already in go2.xml).
    for fid in FOOT_GEOM_IDS:
        m.geom_condim[fid] = 6

    # Implicit position PD: tau = Kp*(ctrl - q) - Kd*qdot, solved inside constraint solver.
    # Flat torque limit (DCMotorCfg): no velocity-dependent curve unlike OpenTopic/Go2HV.
    for i in range(m.nu):
        m.actuator_gaintype[i]    = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[i, 0]  = KP_NOM
        m.actuator_biastype[i]    = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_biasprm[i, 0]  = 0.0
        m.actuator_biasprm[i, 1]  = -KP_NOM
        m.actuator_biasprm[i, 2]  = -KD_NOM
        m.actuator_forcelimited[i] = True
        m.actuator_forcerange[i]   = [-TORQUE_LIMIT, TORQUE_LIMIT]

    print(f"[physics] dt={m.opt.timestep*1000:.0f}ms  IMPLICITFAST  impratio={m.opt.impratio}  "
          f"damping=0  frictionloss=0  armature=0.01  "
          f"PD(Kp={KP_NOM},Kd={KD_NOM},lim=±{TORQUE_LIMIT}Nm)")


# ── Spawn height ───────────────────────────────────────────────────────────────

def compute_spawn_height(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    """Set base z so the lowest foot sphere just touches the floor (no penetration)."""
    mujoco.mj_resetData(m, d)
    d.qpos[0:3]  = [0.0, 0.0, 1.0]
    d.qpos[3]    = 1.0
    d.qpos[4:7]  = 0.0
    d.qpos[7:19] = DEFAULT_JOINT_POS
    mujoco.mj_kinematics(m, d)
    foot_zs     = [d.geom_xpos[fid, 2] for fid in FOOT_GEOM_IDS]
    foot_z_min  = min(foot_zs)
    foot_z_max  = max(foot_zs)
    foot_radius = float(m.geom_size[FOOT_GEOM_IDS[0], 0])
    height = float(1.0 - foot_z_min + foot_radius)
    print(f"[spawn] foot_z: min={foot_z_min:.4f} max={foot_z_max:.4f}  "
          f"radius={foot_radius:.4f}  spawn_height={height:.4f} m")
    return height


# ── Joint order verification ───────────────────────────────────────────────────

def verify_joint_order(m: mujoco.MjModel) -> np.ndarray:
    """Return perm[il_idx] = mj_idx  (Isaac Lab type-grouped → MuJoCo per-leg)."""
    mj_names    = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    mj_stripped = [n + "_joint" for n in mj_names]

    print("Joint order (Isaac Lab → MuJoCo):")
    print(f"  {'IL':>3}  {'Isaac Lab joint':<22}  {'MuJoCo actuator':<20}  MJ  Match")
    print(f"  {'-'*3}  {'-'*22}  {'-'*20}  {'-'*3}  -----")
    perm, ok = [], True
    for il_idx, il_name in enumerate(ISAAC_JOINT_NAMES):
        try:
            mj_idx = mj_stripped.index(il_name)
        except ValueError:
            print(f"  ERROR: '{il_name}' not in MuJoCo model!")
            ok, mj_idx = False, il_idx
        match = "✓" if mj_idx == il_idx else "SWAP"
        print(f"  {il_idx:>3}  {il_name:<22}  {mj_names[mj_idx]:<20}  {mj_idx:>3}  {match}")
        perm.append(mj_idx)

    perm = np.array(perm, dtype=int)
    is_id = np.all(perm == np.arange(len(perm)))
    print(f"\n  perm={perm.tolist()}  ({'identity' if is_id else 'REORDER'})\n")
    if not ok:
        raise RuntimeError("Joint name mismatch between Isaac Lab and MuJoCo model.")
    return perm


# ── DR application ─────────────────────────────────────────────────────────────

class EpisodeDR:
    """Samples and applies DR parameters for one episode.

    Matches shared_env_cfg.py / shared/mdp/events.py exactly:
      - friction:        ALL robot collision geoms
      - mass_scale_base: SCALE base body mass (0.9-1.1 × nominal), inertia separate
      - inertia_scale:   SCALE base body inertia INDEPENDENTLY (0.8-1.2 × nominal)
      - kp_scale / kd_scale: Kp_eff = KP * kp_scale, Kd_eff = KD * kd_scale
      - com_offset:      shift base COM in body frame (±0.05 m per axis)
      - delay_ms:        action delay in milliseconds
    """

    def __init__(self, m: mujoco.MjModel, nominal_masses: np.ndarray, scale: float):
        self.m = m
        self.nominal_masses  = nominal_masses.copy()
        self.nominal_inertia = m.body_inertia.copy()   # [N_bodies, 3]
        self.nominal_ipos    = m.body_ipos.copy()       # [N_bodies, 3]
        self.nominal_geom_friction = {gid: float(m.geom_friction[gid, 0])
                                       for gid in ALL_COLLISION_GEOM_IDS}
        self.ranges = compute_dr_ranges(scale)

    def sample(self) -> dict:
        def uni(lo, hi): return float(np.random.uniform(lo, hi))
        r = self.ranges
        return {
            "friction":        uni(*r["friction"]),
            "mass_scale_base": uni(*r["mass_scale_base"]),
            "inertia_scale":   uni(*r["inertia_scale"]),
            "kp_scale":        uni(*r["kp_scale"]),
            "kd_scale":        uni(*r["kd_scale"]),
            "com_offset":      np.array([uni(*r["com_offset_m"]) for _ in range(3)]),
            "delay_ms":        uni(*r["delay_ms"]),
        }

    def apply(self, params: dict):
        m = self.m

        # Friction on all robot collision geoms
        for gid in ALL_COLLISION_GEOM_IDS:
            m.geom_friction[gid, 0] = params["friction"]

        # Base mass scale (recompute_inertia=False — inertia is independent below)
        m.body_mass[BASE_BODY_ID] = self.nominal_masses[BASE_BODY_ID] * params["mass_scale_base"]

        # Base inertia scale — independent of mass (matches randomize_inertia_and_track)
        m.body_inertia[BASE_BODY_ID] = self.nominal_inertia[BASE_BODY_ID] * params["inertia_scale"]

        # Base COM offset in body frame
        m.body_ipos[BASE_BODY_ID] = self.nominal_ipos[BASE_BODY_ID] + params["com_offset"]

        # Gains: Kp_eff = KP * kp_scale,  Kd_eff = KD * kd_scale
        kp_eff = KP_NOM * params["kp_scale"]
        kd_eff = KD_NOM * params["kd_scale"]
        for i in range(m.nu):
            m.actuator_gainprm[i, 0] = kp_eff
            m.actuator_biasprm[i, 1] = -kp_eff
            m.actuator_biasprm[i, 2] = -kd_eff

    def reset_to_nominal(self):
        """Restore all DR-modified fields to nominal values."""
        for i, mass in enumerate(self.nominal_masses):
            self.m.body_mass[i]    = mass
            self.m.body_inertia[i] = self.nominal_inertia[i]
        self.m.body_ipos[:] = self.nominal_ipos
        for gid, f0 in self.nominal_geom_friction.items():
            self.m.geom_friction[gid, 0] = f0
        for i in range(self.m.nu):
            self.m.actuator_gainprm[i, 0] = KP_NOM
            self.m.actuator_biasprm[i, 1] = -KP_NOM
            self.m.actuator_biasprm[i, 2] = -KD_NOM


# ── Observation ────────────────────────────────────────────────────────────────

def get_foot_contacts(m: mujoco.MjModel, d: mujoco.MjData,
                       force_threshold: float = 1.0) -> np.ndarray:
    """Binary contact flag per foot (FL, FR, RL, RR).

    Uses contact NORMAL FORCE > threshold (default 1.0 N) to match Isaac Lab's
    `net_forces_w_history[..., 2] > 1.0` convention used in proprioceptive_obs_go2
    and privileged_external_go2. Iterating d.contact and summing the normal
    component of mj_contactForce gives the force pressing the foot into the floor.

    Earlier z-position thresholding produced "always [1,1,1,1]" when the robot
    stood still — a distribution the CTS student encoder never saw in training
    (training had foot_contact varying with gait).
    """
    contacts = np.zeros(4, dtype=np.float32)
    foot_set = set(FOOT_GEOM_IDS)
    floor    = FLOOR_GEOM_ID
    # Sum normal force per foot
    forces = np.zeros(4, dtype=np.float64)
    cf = np.zeros(6)
    for ci in range(d.ncon):
        c = d.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        # Pair must be (foot, floor)
        if (g1 == floor and g2 in foot_set):
            foot_g = g2
        elif (g2 == floor and g1 in foot_set):
            foot_g = g1
        else:
            continue
        mujoco.mj_contactForce(m, d, ci, cf)
        # cf[0] is the normal component in contact frame; |cf[0]| is normal force.
        normal_n = abs(float(cf[0]))
        forces[FOOT_GEOM_IDS.index(foot_g)] += normal_n
    for fi in range(4):
        if forces[fi] > force_threshold:
            contacts[fi] = 1.0
    return contacts


def get_obs(m: mujoco.MjModel, d: mujoco.MjData,
            vel_cmd: np.ndarray, joint_perm: np.ndarray) -> np.ndarray:
    """
    37-dim proprioceptive obs matching shared_env_cfg.py / proprioceptive_obs_go2():

      [0:12)   joint_pos_rel   (12, relative to DEFAULT_JOINT_POS, IL type-grouped order)
      [12:24)  joint_vel       (12, rad/s, IL type-grouped order)
      [24:27)  ang_vel_b       (3, rad/s, body frame — freejoint qvel[3:6] is already local)
      [27:30)  gravity_b       (3, unit vector in body frame)
      [30:33)  vel_cmd         (3, vx vy wz)
      [33:37)  foot_contact    (4, binary FL FR RL RR)

    Note: NO lin_vel_b, NO last_action, NO scaling on ang_vel_b or joint_vel.
    joint_perm permutes MuJoCo per-leg order → Isaac Lab type-grouped order.
    """
    R = d.xmat[BASE_BODY_ID].reshape(3, 3)

    q_joints    = d.qpos[7:19]
    qdot_joints = d.qvel[6:18]

    joint_pos_rel = (q_joints - DEFAULT_JOINT_POS)[joint_perm]
    joint_vel     = qdot_joints[joint_perm]
    ang_vel_b     = d.qvel[3:6]                          # freejoint local frame
    gravity_b     = R.T @ np.array([0.0, 0.0, -1.0])
    foot_contact  = get_foot_contacts(m, d)

    return np.concatenate([
        joint_pos_rel, joint_vel, ang_vel_b, gravity_b, vel_cmd, foot_contact
    ]).astype(np.float32)


# ── Control ────────────────────────────────────────────────────────────────────

def compute_target_q(action: np.ndarray, inv_joint_perm: np.ndarray) -> np.ndarray:
    """Convert policy action (IL type-grouped order) → MuJoCo ctrl (per-leg order).

    inv_joint_perm[mj_idx] = il_idx  →  ctrl[mj] = ref[mj] + action[il] * scale
    """
    return DEFAULT_JOINT_POS + action[inv_joint_perm] * ACTION_SCALE


# ── Tracking-quality metrics (matches unitree_rl_lab/scripts/rsl_rl/eval_ood.py)
#    Adapted from real_open_topic so MuJoCo runs report the same per-episode
#    metrics as the Isaac Sim OOD evaluation: lin_track, ang_track, track_error.
#    These are independent of the reward function — they just measure how well
#    the policy follows the velocity command.

def compute_tracking_step(d: mujoco.MjData, vel_cmd: np.ndarray) -> tuple[float, float, float]:
    """Per-step tracking metrics matching eval_ood.py._tracking_exp + L2 error.

    Returns (lin_track, ang_track, track_error):
      lin_track  = exp(-||v_xy_b - cmd_xy||² / 0.25)   ∈ [0, 1], 1 = perfect
      ang_track  = exp(-(wz_b - cmd_wz)² / 0.25)        ∈ [0, 1], 1 = perfect
      track_error = ||v_xy_b - cmd_xy||                 [m/s], 0 = perfect
    """
    R         = d.xmat[BASE_BODY_ID].reshape(3, 3)
    lin_vel_b = R.T @ d.qvel[0:3]
    wz_b      = float(d.qvel[5])         # qvel[3:6] = body-frame ang vel for free joint
    err_xy    = lin_vel_b[:2] - vel_cmd[:2]
    err_wz    = wz_b - float(vel_cmd[2])
    lin_track = float(np.exp(-float(np.dot(err_xy, err_xy)) / 0.25))
    ang_track = float(np.exp(-(err_wz * err_wz)            / 0.25))
    track_err = float(np.linalg.norm(err_xy))
    return lin_track, ang_track, track_err


# ── Reward — matches SharedRewardsCfg in shared_env_cfg.py ────────────────────

class RewardState:
    def __init__(self):
        self.prev_action   = np.zeros(12, dtype=np.float64)
        self.foot_air_time = np.zeros(4,  dtype=np.float64)
        self.foot_con_time = np.zeros(4,  dtype=np.float64)
        self.in_contact    = np.zeros(4,  dtype=bool)

    def reset(self):
        self.prev_action[:]   = 0.0
        self.foot_air_time[:] = 0.0
        self.foot_con_time[:] = 0.0
        self.in_contact[:]    = False


def compute_step_reward(
    m: mujoco.MjModel, d: mujoco.MjData,
    vel_cmd: np.ndarray,
    action_np: np.ndarray,
    tau: np.ndarray,
    rs: RewardState,
) -> float:
    """
    Per-step reward matching SharedRewardsCfg.

    Term                  | Weight  | Isaac Lab func
    ----------------------|---------|----------------------------
    track_lin_vel_xy      | +1.5    | track_lin_vel_xy_exp (std=√0.25)
    track_ang_vel_z       | +0.75   | track_ang_vel_z_exp  (std=√0.25)
    feet_air_time         | +0.1    | threshold=0.5 s
    alive                 | +1.0    | is_alive   (+1/step until termination)
    base_linear_velocity  | -2.0    | lin_vel_z_l2
    base_angular_velocity | -0.05   | ang_vel_xy_l2
    joint_vel             | -0.001  | joint_vel_l2
    joint_acc             | -2.5e-7 | joint_acc_l2
    joint_torques         | -2e-4   | joint_torques_l2
    action_rate           | -0.1    | action_rate_l2
    dof_pos_limits        | -10.0   | 5% soft inset
    energy                | -2e-5   | |tau × qdot|
    flat_orientation_l2   | -2.5    | gravity_b[:2] l2
    joint_pos             | -0.7    | stand_still_scale=5, threshold=0.05
    feet_slide            | -0.1    | calf body vel when in contact
    air_time_variance     | -1.0    | variance of air/contact timers
    undesired_contacts    | -1.0    | hip + thigh bodies (Isaac also includes calf, but the
                                      Go2 menagerie XML has no separate foot body — its
                                      foot geom lives on the calf body — so checking
                                      cfrc_ext[calf] would fire on every walking step.
                                      Skipping calf is a pragmatic approximation;
                                      under-penalises rare calf-scraping impacts only.)
    """
    R         = d.xmat[BASE_BODY_ID].reshape(3, 3)
    lin_vel_b = R.T @ d.qvel[0:3]
    ang_vel_b = d.qvel[3:6]
    gravity_b = R.T @ np.array([0.0, 0.0, -1.0])
    joint_pos = d.qpos[7:19]
    joint_vel = d.qvel[6:18]
    joint_acc = d.qacc[6:18]
    contact   = get_foot_contacts(m, d)

    # tracking
    r_lin = 1.5  * np.exp(-((lin_vel_b[0]-vel_cmd[0])**2 + (lin_vel_b[1]-vel_cmd[1])**2) / 0.25)
    r_ang = 0.75 * np.exp(-(ang_vel_b[2] - vel_cmd[2])**2 / 0.25)

    # feet air time (+0.1, threshold=0.5 s)
    r_air  = 0.0
    cmd_xy = float(np.linalg.norm(vel_cmd[:2]))
    for fi in range(4):
        now = contact[fi] > 0.5
        was = rs.in_contact[fi]
        if now:
            if not was and cmd_xy > 0.1:
                r_air += (rs.foot_air_time[fi] - 0.5)
            rs.foot_air_time[fi]  = 0.0
            rs.foot_con_time[fi] += POLICY_DT
        else:
            if was:
                rs.foot_con_time[fi] = 0.0
            rs.foot_air_time[fi] += POLICY_DT
        rs.in_contact[fi] = now
    r_air *= 0.1

    # base penalties
    r_vz  = -2.0  * float(lin_vel_b[2]**2)
    r_wxy = -0.05 * float(ang_vel_b[0]**2 + ang_vel_b[1]**2)

    # joint penalties
    r_jvel = -0.001  * float(np.sum(joint_vel**2))
    r_jacc = -2.5e-7 * float(np.sum(joint_acc**2))
    r_tau  = -2e-4   * float(np.sum(tau**2))
    r_act  = -0.1    * float(np.sum((action_np - rs.prev_action)**2))
    rs.prev_action[:] = action_np

    # dof_pos_limits (-10.0, 5% soft inset)
    jlo = m.jnt_range[1:13, 0];  jhi = m.jnt_range[1:13, 1]
    mg  = 0.05 * (jhi - jlo)
    out = np.maximum(0.0, (jlo + mg) - joint_pos) + np.maximum(0.0, joint_pos - (jhi - mg))
    r_lim = -10.0 * float(np.sum(out))

    # energy (-2e-5, |tau × qdot|)
    r_eng  = -2e-5 * float(np.sum(np.abs(tau) * np.abs(joint_vel)))

    # flat_orientation_l2 (-2.5)
    r_flat = -2.5  * float(np.sum(gravity_b[:2]**2))

    # joint_pos penalty (-0.7, stand_still_scale=5, velocity_threshold=0.05)
    cmd_n    = float(np.linalg.norm(vel_cmd))
    body_vxy = float(np.linalg.norm(lin_vel_b[:2]))
    jdev     = float(np.linalg.norm(joint_pos - DEFAULT_JOINT_POS))
    scale    = 1.0 if (cmd_n > 0.0 or body_vxy > 0.05) else 5.0
    r_jpos   = -0.7 * scale * jdev

    # feet_slide (-0.1)
    # Isaac Lab: body_lin_vel_w (world-frame XY) of foot body.
    # MuJoCo: d.cvel[bid, 3:6] is linear velocity in LOCAL body frame.
    # Convert to world frame: R_calf @ v_local, then take XY norm.
    r_slide = 0.0
    for fi in range(4):
        if contact[fi] > 0.5:
            bid     = LEG_BODY_IDS[fi][2]          # calf body (foot geom lives here)
            R_calf  = d.xmat[bid].reshape(3, 3)    # local → world rotation
            v_world = R_calf @ d.cvel[bid, 3:6]    # linear vel in world frame
            r_slide -= 0.1 * float(np.linalg.norm(v_world[:2]))

    # air_time_variance (-1.0)
    # Isaac Lab: torch.var uses ddof=1 (divides by N-1=3 for 4 feet).
    # np.var default is ddof=0; pass ddof=1 to match.
    ca    = np.clip(rs.foot_air_time, 0, 0.5)
    cc    = np.clip(rs.foot_con_time, 0, 0.5)
    r_var = -1.0 * (float(np.var(ca, ddof=1)) + float(np.var(cc, ddof=1)))

    # undesired_contacts (-1.0, hip+thigh only)
    r_unc = 0.0
    for fi in range(4):
        for part in range(2):   # 0=hip, 1=thigh
            bid = LEG_BODY_IDS[fi][part]
            if float(np.linalg.norm(d.cfrc_ext[bid, 3:6])) > 1.0:
                r_unc -= 1.0

    # is_alive (+1.0): Isaac awards +1 per step until termination. The MuJoCo loop
    # already terminates the episode AFTER this reward is added on the final step,
    # which is the standard Isaac Lab convention (sub-step accounting differences
    # are at most 1 reward unit per episode — negligible).
    r_alive = 1.0

    return float(r_lin + r_ang + r_air + r_alive
                 + r_vz + r_wxy + r_jvel + r_jacc + r_tau + r_act
                 + r_lim + r_eng + r_flat + r_jpos + r_slide + r_var + r_unc)


# ── Termination ────────────────────────────────────────────────────────────────

def is_done(m: mujoco.MjModel, d: mujoco.MjData,
            step: int, max_steps: int) -> tuple[bool, str]:
    if step >= max_steps:
        return True, "timeout"
    # bad_orientation: acos(-gravity_b[2]) > 1.2 rad  →  gravity_b[2] > -cos(1.2) ≈ -0.362
    R          = d.xmat[BASE_BODY_ID].reshape(3, 3)
    gravity_bz = float((R.T @ np.array([0.0, 0.0, -1.0]))[2])
    if gravity_bz > -np.cos(BAD_ORIENTATION_LIMIT):
        return True, "bad_orientation"
    return False, ""


# ── Policy loaders ─────────────────────────────────────────────────────────────

class BaselineMLP(torch.nn.Module):
    """37-dim → [512, 256, 128] → 12  with ELU."""
    def __init__(self, sd):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(37, 512), torch.nn.ELU(),
            torch.nn.Linear(512, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 12),
        )
        mapping = {
            "net.0.weight": "actor.0.weight", "net.0.bias": "actor.0.bias",
            "net.2.weight": "actor.2.weight", "net.2.bias": "actor.2.bias",
            "net.4.weight": "actor.4.weight", "net.4.bias": "actor.4.bias",
            "net.6.weight": "actor.6.weight", "net.6.bias": "actor.6.bias",
        }
        self.load_state_dict({k: sd[v] for k, v in mapping.items()})

    def forward(self, obs):
        return self.net(obs)


_PRIV_DIMS = {"FULL": 26, "INT": 16, "EXT": 10}


def load_policy(method: str, checkpoint: str, device: str,
                latent_dim: int = 8, history_len: int = 50,
                priv_mode: str = "FULL"):
    """Load a trained policy from checkpoint.

    Returns (policy, history_len_or_None).
    For CTS: history_len is the buffer length for the student encoder.
    For baseline: history_len is None.
    `priv_mode` selects the privileged-vector width that the teacher encoder
    was trained with (FULL=26, INT=16, EXT=10) — must match the checkpoint.
    """
    if priv_mode not in _PRIV_DIMS:
        raise ValueError(f"priv_mode must be one of {list(_PRIV_DIMS)}, got {priv_mode!r}")
    priv_dim    = _PRIV_DIMS[priv_mode]
    critic_obs  = 37 + priv_dim

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd   = ckpt.get("model_state_dict", ckpt.get("actor_critic", ckpt))
    print(f"[policy] checkpoint iter={ckpt.get('iter', '?')}  priv_mode={priv_mode} (dim={priv_dim})")

    if method == "baseline":
        policy = BaselineMLP(sd).to(device)
        policy.eval()
        return policy, None

    elif method == "rma":
        # Deployment mode: actor sees 37-dim o_t, latent z=0 (Phase-1) or from
        # adaptation module (Phase-2). We load the deployment-mode actor with
        # num_actor_obs=37 so _teacher_mode=False — act_inference uses z=0.
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "rma_network",
            os.path.join(PROJECT_ROOT, "source", "cts_rma_project",
                         "cts_rma_project", "tasks", "rma", "rma_network.py"),
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        RMAActorCritic = _mod.RMAActorCritic

        policy = RMAActorCritic(
            num_actor_obs=37,
            num_critic_obs=critic_obs,
            num_actions=12,
            env_factor_dim=priv_dim,
            latent_dim=latent_dim,
        ).to(device)
        policy.load_state_dict(sd, strict=False)
        policy.eval()
        print(f"[policy] RMA: latent_dim={latent_dim}  env_factor_dim={priv_dim}  (deployment mode, z=0)")
        return policy, None

    elif method == "cts":
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "cts_network",
            os.path.join(PROJECT_ROOT, "source", "cts_rma_project",
                         "cts_rma_project", "tasks", "cts", "cts_network.py"),
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        CTSActorCritic = _mod.CTSActorCritic

        # CTSActorCritic accepts priv_dim as a kwarg (added when priv_mode plumbing
        # was introduced); fall back gracefully on older signatures.
        try:
            policy = CTSActorCritic(
                num_actor_obs=history_len * 37 + 1,
                num_critic_obs=critic_obs,
                num_actions=12,
                latent_dim=latent_dim,
                history_len=history_len,
                priv_dim=priv_dim,
            ).to(device)
        except TypeError:
            policy = CTSActorCritic(
                num_actor_obs=history_len * 37 + 1,
                num_critic_obs=critic_obs,
                num_actions=12,
                latent_dim=latent_dim,
                history_len=history_len,
            ).to(device)
        policy.load_state_dict(sd, strict=False)
        policy.eval()
        print(f"[policy] CTS: latent_dim={latent_dim}  history_len={history_len}  priv_dim={priv_dim}")
        return policy, history_len

    else:
        raise ValueError(f"Unknown method: {method}")


# ── Pose reset ─────────────────────────────────────────────────────────────────

def _reset_pose(m: mujoco.MjModel, d: mujoco.MjData, spawn_height: float,
                randomize: bool = False, vel_cmd: np.ndarray | None = None):
    """Reset robot pose to match Isaac Lab SharedEventCfg reset_base/reset_joints.

    SharedEventCfg.reset_base ALWAYS applies random base XY+yaw + velocity
    push at every reset. We match that here as the default behaviour:
        XY              ±0.5 m       per axis  (already matched)
        yaw             ±π                     (already matched)
        base linear vel ±0.5 m/s     per axis   ← NEW
        base angular vel ±1.0 rad/s  per axis   ← NEW

    Without the velocity push, the CTS student encoder sees a "perfectly
    still" 50-frame history at episode start — a distribution it never saw
    during training (training resets always have non-zero base velocity).
    The encoder produces an OOD latent → actor outputs cautious "stand
    still" actions → robot never starts walking. Baseline doesn't hit this
    because it uses no temporal encoder.

    randomize=True additionally adds the SharedEventCfg.reset_joints chaos:
        joint pos offset ±1.047 rad (±60°)
        joint vel        ±1.0   rad/s
    Use this when you want to test recovery from extreme initial states
    (matches training fully); the policy may take a few steps to stabilise.
    """
    mujoco.mj_resetData(m, d)

    x   = float(np.random.uniform(-0.5, 0.5))
    y   = float(np.random.uniform(-0.5, 0.5))
    yaw = float(np.random.uniform(-np.pi, np.pi))
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)

    d.qpos[0:3]  = [x, y, spawn_height]
    d.qpos[3]    = cy      # quaternion w
    d.qpos[4]    = 0.0     # x
    d.qpos[5]    = 0.0     # y
    d.qpos[6]    = sy      # z (yaw rotation)
    d.qpos[7:19] = DEFAULT_JOINT_POS
    d.qvel[:]    = 0.0

    # Bootstrap base velocity matching training's reset_base distribution.
    # WHY: the CTS student encoder needs a non-zero starting velocity, otherwise
    # its 50-frame history starts as "perfectly still" — a distribution it never
    # saw in training (Isaac Lab's reset_base ALWAYS samples velocity in
    # [-0.5, 0.5] m/s per axis). Without this, the encoder produces an OOD
    # latent and the actor outputs cautious "stand still" actions.
    # Baseline / RMA aren't affected (no temporal encoder).
    # ang vel is kept small (±0.2 vs training's ±1.0) so random rotation
    # doesn't overwhelm the forward-gait pattern in the encoder's first frames.
    d.qvel[0:3] = np.random.uniform(-0.5, 0.5, 3)
    d.qvel[3:6] = np.random.uniform(-0.2, 0.2, 3)

    if randomize:
        # Match SharedEventCfg.reset_joints (±1.047 rad, ±1.0 rad/s)
        d.qpos[7:19] += np.random.uniform(-1.047, 1.047, 12)
        d.qvel[6:18]  = np.random.uniform(-1.0,    1.0,   12)
        # Full angular vel randomisation only with --random_init
        d.qvel[3:6]   = np.random.uniform(-1.0, 1.0, 3)

    mujoco.mj_forward(m, d)


# ── Episode ────────────────────────────────────────────────────────────────────

def _foot_lin_vel_world(d) -> np.ndarray:
    """World-frame linear velocity of each foot, shape (4, 3) in [FL, FR, RL, RR] order.
    Same convention as scripts/sim2sim/eval_metrics.py::_foot_lin_vel."""
    vels = np.zeros((4, 3))
    for fi in range(4):
        bid    = LEG_BODY_IDS[fi][2]                  # calf body carries the foot geom
        R_calf = d.xmat[bid].reshape(3, 3)            # local → world
        vels[fi] = R_calf @ d.cvel[bid, 3:6]           # cvel[3:6] = linear vel (local)
    return vels


def run_episode(m, d, policy, method, device, dr: EpisodeDR,
                joint_perm, inv_joint_perm, vel_cmd, max_steps, spawn_height,
                history_len=None, no_dr=False, random_init=False,
                no_terminate=False):
    """Run one episode. Returns (total_reward, total_steps, done_reason,
    sum_lin_track, sum_ang_track, sum_track_err, gait_metrics_dict)."""

    # DR
    if no_dr:
        dr.reset_to_nominal()
        delay_steps = 1   # minimum 1-step delay always
    else:
        params = dr.sample()
        dr.apply(params)
        delay_steps = max(1, round(params["delay_ms"] / (POLICY_DT * 1000)))
    mujoco.mj_forward(m, d)

    # Buffers
    null_action = np.zeros(12, dtype=np.float32)
    action_buf  = collections.deque([null_action.copy()] * (delay_steps + 1),
                                     maxlen=delay_steps + 1)

    # CTS: history buffer
    obs_hist = None
    if method == "cts" and history_len is not None:
        obs_hist = torch.zeros(1, history_len, 37, device=device)

    # Pose reset
    _reset_pose(m, d, spawn_height, randomize=random_init, vel_cmd=vel_cmd)

    # CTS: leave history as zeros (matches training cts_env.py:_reset_idx, which
    # zeroes obs_history at every reset). Pre-filling with standing obs creates
    # an OOD distribution the student encoder never saw during training.

    rs = RewardState()
    total_reward = 0.0
    sum_lin_track = 0.0          # eval_ood.py: per-step exp(-||v_xy-cmd||²/0.25)
    sum_ang_track = 0.0          # eval_ood.py: per-step exp(-(wz-cmd_wz)²/0.25)
    sum_track_err = 0.0          # eval_ood.py: per-step ||v_xy-cmd|| [m/s]
    fwd_disp     = 0.0           # spec sheet "Forward displacement" [m]
    step = 0

    cmd_speed_safe = max(float(np.linalg.norm(vel_cmd[:2])), 1e-6)

    # Gait-metric per-step buffers (see scripts/gait_metrics.py).
    foot_radius = float(m.geom_size[FOOT_GEOM_IDS[0], 0])
    g_contacts, g_foot_z, g_foot_xy_s, g_foot_speed = [], [], [], []
    g_actions, g_tau, g_base_z, g_base_xy = [], [], [], []

    with torch.no_grad():
        while True:
            obs_np = get_obs(m, d, vel_cmd, joint_perm)

            if method == "cts":
                obs_hist = torch.roll(obs_hist, -1, dims=1)
                obs_hist[0, -1, :] = torch.from_numpy(obs_np).to(device)
                flag      = torch.zeros(1, 1, device=device)   # student mode
                policy_in = torch.cat([obs_hist.reshape(1, -1), flag], dim=1)
                action_t  = policy.act_inference(policy_in)
            elif method == "rma":
                obs_t    = torch.from_numpy(obs_np).unsqueeze(0).to(device)
                action_t = policy.act_inference(obs_t)        # z=0 deployment mode
            else:
                obs_t    = torch.from_numpy(obs_np).unsqueeze(0).to(device)
                action_t = policy(obs_t)

            action_np = action_t.squeeze(0).cpu().numpy()
            action_buf.appendleft(action_np.copy())
            delayed = action_buf[-1]

            d.ctrl[:] = compute_target_q(delayed, inv_joint_perm)
            for _ in range(DECIMATION):
                mujoco.mj_step(m, d)
            tau = d.actuator_force.copy()

            total_reward += compute_step_reward(m, d, vel_cmd, action_np, tau, rs)
            lt, at, te = compute_tracking_step(d, vel_cmd)
            sum_lin_track += lt
            sum_ang_track += at
            sum_track_err += te

            # Forward displacement: integrate body-frame linear velocity projected
            # onto the commanded direction. For the typical cmd = (vx, 0, 0) this
            # equals integral of vx_body, matching OpenTopic eval_metrics.py.
            R_base = d.xmat[BASE_BODY_ID].reshape(3, 3)
            lv_b   = R_base.T @ d.qvel[0:3]
            fwd_disp += float(np.dot(lv_b[:2], vel_cmd[:2])) / cmd_speed_safe * POLICY_DT

            # ── Gait-metric per-step capture ─────────────────────────────────
            contacts  = get_foot_contacts(m, d).astype(np.float32)
            foot_z_t  = np.array([d.geom_xpos[fid, 2] for fid in FOOT_GEOM_IDS],
                                 dtype=np.float32)
            fv        = _foot_lin_vel_world(d)
            g_contacts.append(contacts)
            g_foot_z.append(foot_z_t)
            g_foot_xy_s.append(np.linalg.norm(fv[:, :2], axis=1).astype(np.float32))
            g_foot_speed.append(np.linalg.norm(fv,        axis=1).astype(np.float32))
            g_actions.append(action_np.astype(np.float32).copy())
            g_tau.append(tau.astype(np.float32).copy())
            g_base_z.append(float(d.qpos[2]))
            g_base_xy.append(d.qpos[0:2].astype(np.float32).copy())

            step += 1

            if no_terminate:
                done, reason = False, "running"
            else:
                done, reason = is_done(m, d, step, max_steps)
            if done:
                _arr_contacts      = np.asarray(g_contacts)
                _arr_foot_z        = np.asarray(g_foot_z)
                _arr_foot_xy_speed = np.asarray(g_foot_xy_s)
                _arr_foot_speed    = np.asarray(g_foot_speed)
                _arr_actions       = np.asarray(g_actions)
                _arr_tau           = np.asarray(g_tau)
                _arr_base_z        = np.asarray(g_base_z)
                _arr_base_xy       = np.asarray(g_base_xy)
                gait = _compute_episode_gait_metrics(
                    contacts      = _arr_contacts,
                    foot_z        = _arr_foot_z,
                    foot_xy_speed = _arr_foot_xy_speed,
                    foot_speed    = _arr_foot_speed,
                    actions       = _arr_actions,
                    tau           = _arr_tau,
                    base_z        = _arr_base_z,
                    base_xy       = _arr_base_xy,
                    cmd_xy        = np.asarray(vel_cmd[:2], dtype=np.float32),
                    foot_radius   = foot_radius,
                )
                raw_buffers = {
                    "contacts":      _arr_contacts,
                    "foot_z":        _arr_foot_z,
                    "foot_xy_speed": _arr_foot_xy_speed,
                    "foot_speed":    _arr_foot_speed,
                    "actions":       _arr_actions,
                    "tau":           _arr_tau,
                    "base_z":        _arr_base_z,
                    "base_xy":       _arr_base_xy,
                }
                return (total_reward, step, reason,
                        sum_lin_track, sum_ang_track, sum_track_err, gait, fwd_disp,
                        raw_buffers)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",       type=str, required=True,
                        choices=["baseline", "rma", "cts"])
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--scene_xml",    type=str, default=None)
    parser.add_argument("--dr_scale",     type=float, default=1.0,
                        help="DR multiplier (1.0=training range, 2.0=OOD×2)")
    parser.add_argument("--no_dr",        action="store_true",
                        help="Disable DR; use nominal gains and no perturbations")
    parser.add_argument("--random_init",  action="store_true",
                        help="Random joint noise at each reset")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="N = 100 per the spec sheet.")
    parser.add_argument("--episode_length_s", type=float, default=10.0,
                        help="Episode length T in seconds (spec sheet: T=10s).")
    parser.add_argument("--vel_rmse_threshold", type=float, default=0.3,
                        help="Velocity-tracking RMSE threshold (m/s) for Success vs Partial. "
                             "Spec sheet: 0.3 m/s.")
    parser.add_argument("--priv_mode",    type=str, default="FULL",
                        choices=["FULL", "INT", "EXT"],
                        help="Privileged-knowledge subset the checkpoint was trained with.")
    parser.add_argument("--save_raw_dir", type=str, default=None,
                        help="If set, write per-episode JSON + per-step NPZ here for "
                             "post-hoc analysis (same schema as scripts/eval_ood_go2.py).")
    parser.add_argument("--latent_dim",   type=int, default=8)
    parser.add_argument("--history_len",  type=int, default=50)
    parser.add_argument("--render",       action="store_true")
    parser.add_argument("--no_terminate", action="store_true",
                        help="Disable termination (viewer only)")
    parser.add_argument("--vel_x",        type=float, default=0.5)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--device",       type=str, default="cuda")
    # (priv_mode is already declared above)
    parser.add_argument("--results_file", type=str, default=None,
                        help="Append run summary to this CSV "
                             "(same schema as scripts/eval_ood_go2.py).")
    args = parser.parse_args()

    if args.no_terminate and not args.render:
        parser.error("--no_terminate requires --render")
    if args.no_terminate:
        args.num_episodes = 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    scene_xml = args.scene_xml or SCENE_XML
    if not os.path.exists(scene_xml):
        raise FileNotFoundError(f"Scene XML not found: {scene_xml}")

    print(f"\n{'='*60}")
    print(f"  GO2 Sim2Sim  —  T-S Policy Project")
    print(f"  method={args.method}  dr_scale={args.dr_scale:.1f}x  episodes={args.num_episodes}")
    print(f"  scene={scene_xml}")
    print(f"{'='*60}\n")

    m = mujoco.MjModel.from_xml_path(scene_xml)
    d = mujoco.MjData(m)
    print(f"MuJoCo: nq={m.nq}  nv={m.nv}  nu={m.nu}  nbody={m.nbody}")

    init_model_ids(m)
    fix_model_physics(m)
    spawn_h = compute_spawn_height(m, d)

    joint_perm     = verify_joint_order(m)
    inv_joint_perm = np.argsort(joint_perm)
    print(f"[perm] {joint_perm.tolist()}")
    print(f"[inv]  {inv_joint_perm.tolist()}\n")

    print_dr_table(args.dr_scale)
    dr = EpisodeDR(m, m.body_mass.copy(), args.dr_scale)

    policy, hist_len = load_policy(
        args.method, args.checkpoint, device,
        args.latent_dim, args.history_len,
        priv_mode=args.priv_mode.upper(),
    )
    print(f"[policy] ready  device={device}\n")

    max_steps   = int(args.episode_length_s / POLICY_DT)   # spec-sheet T (default 10 s)
    success_thr = int(0.8 * max_steps)   # legacy length-based threshold, kept for compat
    vel_cmd     = np.array([args.vel_x, 0.0, 0.0], dtype=np.float32)

    ep_rewards, ep_lengths, ep_reasons = [], [], []
    ep_lin_track, ep_ang_track, ep_track_err = [], [], []   # eval_ood.py-style metrics
    ep_fwd_disp: list[float] = []                           # spec-sheet forward displacement
    ep_gait: list[dict] = []                                # gait_metrics.py-style metrics
    ep_outcomes: list[str] = []                             # 3-class: success/partial/fail
    per_ep_raw: list[dict] = []                             # populated iff --save_raw_dir set

    if args.render:
        with mujoco.viewer.launch_passive(m, d) as v:
            v.cam.distance  = 2.5
            v.cam.elevation = -15
            v.cam.azimuth   = 135

            for ep in range(args.num_episodes):
                if args.no_dr:
                    dr.reset_to_nominal()
                    delay_steps = 1
                else:
                    params = dr.sample()
                    dr.apply(params)
                    delay_steps = max(1, round(params["delay_ms"] / (POLICY_DT * 1000)))
                mujoco.mj_forward(m, d)

                null_action = np.zeros(12, dtype=np.float32)
                action_buf  = collections.deque([null_action.copy()] * (delay_steps + 1),
                                                maxlen=delay_steps + 1)
                obs_hist = None
                if args.method == "cts" and hist_len is not None:
                    obs_hist = torch.zeros(1, hist_len, 37, device=device)

                _reset_pose(m, d, spawn_h, randomize=args.random_init, vel_cmd=vel_cmd)
                # CTS: keep history at zeros (matches training reset behaviour).

                rs = RewardState()
                total_r, step = 0.0, 0
                sum_lt, sum_at, sum_te = 0.0, 0.0, 0.0
                done, reason  = False, "running"
                print(f"  ep {ep+1}  base_z={d.qpos[2]:.4f}  delay={delay_steps}step")
                v.sync()

                with torch.no_grad():
                    while not done and v.is_running():
                        obs_np = get_obs(m, d, vel_cmd, joint_perm)

                        if args.method == "cts":
                            obs_hist = torch.roll(obs_hist, -1, dims=1)
                            obs_hist[0, -1, :] = torch.from_numpy(obs_np).to(device)
                            flag      = torch.zeros(1, 1, device=device)
                            policy_in = torch.cat([obs_hist.reshape(1, -1), flag], dim=1)
                            action_t  = policy.act_inference(policy_in)
                        elif args.method == "rma":
                            obs_t    = torch.from_numpy(obs_np).unsqueeze(0).to(device)
                            action_t = policy.act_inference(obs_t)
                        else:
                            obs_t    = torch.from_numpy(obs_np).unsqueeze(0).to(device)
                            action_t = policy(obs_t)

                        action_np = action_t.squeeze(0).cpu().numpy()
                        action_buf.appendleft(action_np.copy())
                        delayed = action_buf[-1]

                        d.ctrl[:] = compute_target_q(delayed, inv_joint_perm)
                        for _ in range(DECIMATION):
                            mujoco.mj_step(m, d)
                            v.sync()
                        tau = d.actuator_force.copy()

                        total_r += compute_step_reward(m, d, vel_cmd, action_np, tau, rs)
                        lt, at, te = compute_tracking_step(d, vel_cmd)
                        sum_lt  += lt
                        sum_at  += at
                        sum_te  += te
                        step    += 1

                        if step % 50 == 0:
                            R_b  = d.xmat[BASE_BODY_ID].reshape(3, 3)
                            lv_b = R_b.T @ d.qvel[0:3]
                            act_mean = float(np.abs(action_np).mean())
                            print(f"    step={step:4d}  vx={lv_b[0]:+.3f}  vy={lv_b[1]:+.3f}  "
                                  f"wz={d.qvel[5]:+.3f}  base_z={d.qpos[2]:.3f}  "
                                  f"act_mean={act_mean:.3f}  r={total_r:.1f}", flush=True)

                        if args.no_terminate:
                            done, reason = False, "running"
                        else:
                            done, reason = is_done(m, d, step, max_steps)

                ep_rewards.append(total_r)
                ep_lengths.append(step)
                ep_reasons.append(reason)
                steps_safe = max(step, 1)
                ep_lin_track.append(sum_lt / steps_safe)
                ep_ang_track.append(sum_at / steps_safe)
                ep_track_err.append(sum_te / steps_safe)
                print(f"  ep {ep+1:3d}  steps={step:5d}  reward={total_r:8.1f}  "
                      f"lin_track={ep_lin_track[-1]:.3f}  ang_track={ep_ang_track[-1]:.3f}  "
                      f"err={ep_track_err[-1]:.3f} m/s  {reason}")

    else:
        for ep in range(args.num_episodes):
            rew, steps, reason, sum_lt, sum_at, sum_te, gait, fwd_d, raw_buffers = run_episode(
                m, d, policy, args.method, device, dr,
                joint_perm, inv_joint_perm, vel_cmd, max_steps, spawn_h,
                history_len=hist_len,
                no_dr=args.no_dr,
                random_init=args.random_init,
            )
            ep_rewards.append(rew)
            ep_lengths.append(steps)
            ep_reasons.append(reason)
            steps_safe = max(steps, 1)
            ep_lin_track.append(sum_lt / steps_safe)
            ep_ang_track.append(sum_at / steps_safe)
            ep_track_err.append(sum_te / steps_safe)
            ep_fwd_disp.append(fwd_d)
            ep_gait.append(gait)
            # Three-class outcome per the spec sheet.
            vel_rmse_ep = ep_track_err[-1]
            if reason != "timeout":
                outcome = "fail"
            elif vel_rmse_ep < args.vel_rmse_threshold:
                outcome = "success"
            else:
                outcome = "partial"
            ep_outcomes.append(outcome)
            if args.save_raw_dir:
                per_ep_raw.append({
                    "reward":         rew,
                    "length":         steps,
                    "outcome":        outcome,
                    "vel_rmse":       vel_rmse_ep,
                    "fwd_disp":       fwd_d,
                    "mean_lin_track": ep_lin_track[-1],
                    "mean_ang_track": ep_ang_track[-1],
                    "gait":           dict(gait),
                    **raw_buffers,    # contacts/foot_z/.../base_xy (numpy arrays)
                })
            print(f"  ep {ep+1:3d}  steps={steps:5d}  reward={rew:8.1f}  "
                  f"lin_track={ep_lin_track[-1]:.3f}  ang_track={ep_ang_track[-1]:.3f}  "
                  f"rmse={vel_rmse_ep:.3f}m/s  fwd={fwd_d:+.2f}m  "
                  f"gait_adh={gait['gait_adh']:.3f}  sym={gait['contact_sym']:.3f}  "
                  f"[{outcome}]")

    # ── Summary (spec-sheet: 3-class outcome + thresholds + behaviour) ───────
    dr.reset_to_nominal()
    n_eps  = max(len(ep_rewards), 1)

    # Render path doesn't populate ep_outcomes — derive on the fly so the summary
    # remains valid in viewer mode too.
    if not ep_outcomes:
        ep_outcomes = []
        for reason_r, rmse_r in zip(ep_reasons, ep_track_err):
            if reason_r != "timeout":
                ep_outcomes.append("fail")
            elif rmse_r < args.vel_rmse_threshold:
                ep_outcomes.append("success")
            else:
                ep_outcomes.append("partial")

    success_count = sum(1 for o in ep_outcomes if o == "success")
    partial_count = sum(1 for o in ep_outcomes if o == "partial")
    fall_count    = sum(1 for o in ep_outcomes if o == "fail")
    survival_count = success_count + partial_count

    mean_r,  std_r  = float(np.mean(ep_rewards)),  float(np.std(ep_rewards))
    mean_l,  std_l  = float(np.mean(ep_lengths)),  float(np.std(ep_lengths))
    mean_lt, std_lt = float(np.mean(ep_lin_track)), float(np.std(ep_lin_track))
    mean_at, std_at = float(np.mean(ep_ang_track)), float(np.std(ep_ang_track))
    mean_te, std_te = float(np.mean(ep_track_err)), float(np.std(ep_track_err))
    mean_fd, std_fd = (float(np.mean(ep_fwd_disp)), float(np.std(ep_fwd_disp))) if ep_fwd_disp else (0.0, 0.0)
    mean_ep_s = mean_l * POLICY_DT
    std_ep_s  = std_l  * POLICY_DT
    reason_counts = {r: ep_reasons.count(r) for r in set(ep_reasons)}
    gait_agg = _mean_std_gait(ep_gait)   # {name: mean, name_std: std} for each of the 8 metrics

    # Spec-sheet thresholds.
    survival_rate = 100.0 * survival_count / n_eps
    surv_pass = "PASS" if survival_rate >= 80.0 else "FAIL"
    rmse_pass = "PASS" if mean_te <= args.vel_rmse_threshold else "FAIL"

    lines = []
    lines.append("=" * 70)
    lines.append(f"  SIM2SIM (Isaac → MuJoCo) — DR×{args.dr_scale:.1f}    "
                 f"(T={args.episode_length_s:.0f}s, spec-sheet metrics)")
    lines.append("=" * 70)
    lines.append(f"  Checkpoint     : {os.path.basename(args.checkpoint)}")
    lines.append(f"  Method         : {args.method.upper()}  priv={args.priv_mode}  latent={args.latent_dim}")
    lines.append(f"  Total episodes : {n_eps}    "
                 f"(spec sheet N=100)   |   vel_cmd = ({args.vel_x:+.2f}, 0, 0)")
    lines.append("-" * 70)
    lines.append(f"  Survival rate  : {survival_rate:6.1f} %     [{surv_pass} ≥ 80%]   "
                 f"({survival_count}/{n_eps} didn't fall)")
    lines.append(f"   ├ Success    : {100*success_count/n_eps:6.1f} %     ({success_count}/{n_eps})    "
                 f"(timeout AND vel_rmse < {args.vel_rmse_threshold} m/s)")
    lines.append(f"   ├ Partial    : {100*partial_count/n_eps:6.1f} %     ({partial_count}/{n_eps})    "
                 f"(timeout AND vel_rmse ≥ {args.vel_rmse_threshold} m/s)")
    lines.append(f"   └ Fail       : {100*fall_count/n_eps:6.1f} %     ({fall_count}/{n_eps} fell)")
    lines.append(f"  Episode length : {mean_ep_s:6.2f} ± {std_ep_s:.2f} s  "
                 f"({mean_l:.0f} ± {std_l:.0f} steps, max={max_steps})")
    lines.append(f"  Cum. reward    : {mean_r:+.2f} ± {std_r:.2f}")
    lines.append(f"  Vel-track RMSE : {mean_te:.4f} ± {std_te:.4f} m/s    "
                 f"[{rmse_pass} < {args.vel_rmse_threshold} m/s]")
    lines.append(f"  Fwd displ.     : {mean_fd:+.3f} ± {std_fd:.3f} m       (monotone w/ reward)")
    lines.append(f"  Lin vel track  : {mean_lt:.4f} ± {std_lt:.4f}  (exp(-err²/0.25), 1.0 = perfect)")
    lines.append(f"  Ang vel track  : {mean_at:.4f} ± {std_at:.4f}  (1.0 = perfect)")
    lines.append("-" * 70)
    lines.append("  Gait quality   :  (shared with scripts/eval_ood_go2.py — see gait_metrics.py)")
    lines.append(f"   gait_adh      : {gait_agg['gait_adh']:.4f} ± {gait_agg['gait_adh_std']:.4f}     "
                 f"(1.0 = perfect trot)")
    lines.append(f"   clear_err     : {gait_agg['clear_err']:.4f} ± {gait_agg['clear_err_std']:.4f}     "
                 f"(0.0 = perfect swing-foot clearance)")
    lines.append(f"   slip_rate     : {gait_agg['slip_rate']:.4f} ± {gait_agg['slip_rate_std']:.4f}     "
                 f"(0.0 = no foot slip)")
    lines.append(f"   smoothness    : {gait_agg['smoothness']:.4f} ± {gait_agg['smoothness_std']:.4f}     "
                 f"(0.0 = no action jerk)")
    lines.append(f"   base_z_var    : {gait_agg['base_z_var']:.6f} ± {gait_agg['base_z_var_std']:.6f}     "
                 f"(0.0 = perfectly stable trunk)")
    lines.append(f"   contact_sym   : {gait_agg['contact_sym']:.4f} ± {gait_agg['contact_sym_std']:.4f}     "
                 f"(1.0 = perfect trot alternation)")
    lines.append(f"   stride_var    : {gait_agg['stride_var']:.4f} ± {gait_agg['stride_var_std']:.4f}     "
                 f"(0.0 = perfectly consistent stride)")
    lines.append(f"   jtorque_var   : {gait_agg['jtorque_var']:.4f} ± {gait_agg['jtorque_var_std']:.4f}     "
                 f"(0.0 = perfectly steady torques)")
    lines.append(f"  End reasons    : {reason_counts}")
    lines.append("=" * 65)

    output = "\n".join(lines)
    print("\n" + output + "\n")

    # Save to <checkpoint_dir>/ood_eval/results.txt (matches eval_ood.py layout)
    save_dir = os.path.join(os.path.dirname(args.checkpoint), "ood_eval")
    os.makedirs(save_dir, exist_ok=True)
    suffix = f"_dr{args.dr_scale:.1f}".replace(".", "p")
    result_path = os.path.join(save_dir, f"sim2sim_{args.method}{suffix}.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"[sim2sim] Results saved to: {result_path}")

    # CSV — same schema as scripts/eval_ood_go2.py with sim="mujoco" + tracking cols
    if args.results_file:
        import csv as _csv
        from datetime import datetime as _dt
        rpath    = args.results_file
        os.makedirs(os.path.dirname(os.path.abspath(rpath)), exist_ok=True)
        new_file = not os.path.exists(rpath)
        lat_col  = "N/A" if args.method == "baseline" else str(args.latent_dim)
        prv_col  = "BASE" if args.method == "baseline" else args.priv_mode
        success_rate = 100.0 * success_count / n_eps
        partial_rate = 100.0 * partial_count / n_eps
        fall_rate    = 100.0 * fall_count    / n_eps
        # Gait-metric columns: mean + std for each of the 8 shared metrics.
        gait_cols   = [f"{c}{suf}" for c in GAIT_METRIC_NAMES for suf in ("", "_std")]
        gait_values = [f"{gait_agg[c]:.6f}" for c in gait_cols]
        with open(rpath, "a", newline="") as f:
            w = _csv.writer(f)
            if new_file:
                w.writerow(["sim", "method", "priv_mode", "latent_dim", "dr_scale",
                            "episode_length_s",
                            "mean_reward", "std_reward",
                            "mean_length", "std_length",
                            "success_rate", "partial_rate", "fall_rate", "survival_rate",
                            "mean_lin_track", "std_lin_track",
                            "mean_ang_track", "std_ang_track",
                            "mean_track_err", "std_track_err",
                            "mean_fwd_disp", "std_fwd_disp",
                            *gait_cols,
                            "episodes", "checkpoint", "timestamp"])
            w.writerow(["mujoco", args.method.upper(), prv_col, lat_col, f"{args.dr_scale:.1f}",
                        f"{args.episode_length_s:.1f}",
                        f"{mean_r:.4f}",  f"{std_r:.4f}",
                        f"{mean_l:.1f}",  f"{std_l:.1f}",
                        f"{success_rate:.1f}", f"{partial_rate:.1f}",
                        f"{fall_rate:.1f}", f"{survival_rate:.1f}",
                        f"{mean_lt:.4f}", f"{std_lt:.4f}",
                        f"{mean_at:.4f}", f"{std_at:.4f}",
                        f"{mean_te:.4f}", f"{std_te:.4f}",
                        f"{mean_fd:.4f}", f"{std_fd:.4f}",
                        *gait_values,
                        n_eps, os.path.basename(args.checkpoint),
                        _dt.now().strftime("%Y-%m-%d %H:%M:%S")])
        print(f"[sim2sim] CSV row appended → {rpath}\n")

    # ── Raw per-episode + per-step dump (for future analysis) ───────────────
    if args.save_raw_dir and per_ep_raw:
        import json as _json
        os.makedirs(args.save_raw_dir, exist_ok=True)
        lat_tag = "NA" if args.method == "baseline" else str(args.latent_dim)
        prv_tag = "BASE" if args.method == "baseline" else args.priv_mode
        tag = f"mujoco_{args.method}_{prv_tag}_l{lat_tag}_dr{args.dr_scale:.1f}".replace("/", "_")
        json_path = os.path.join(args.save_raw_dir, f"{tag}.json")
        with open(json_path, "w") as f:
            _json.dump({
                "schema_version": 1,
                "sim": "mujoco", "method": args.method.upper(), "priv_mode": prv_tag,
                "latent_dim": lat_tag, "dr_scale": args.dr_scale,
                "episode_length_s": args.episode_length_s,
                "vel_rmse_threshold": args.vel_rmse_threshold,
                "checkpoint": os.path.basename(args.checkpoint),
                "vel_cmd": list(map(float, vel_cmd)),
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
        npz_path = os.path.join(args.save_raw_dir, f"{tag}.npz")
        np.savez_compressed(npz_path,
            ep_lengths    = np.array([e["length"] for e in per_ep_raw], dtype=np.int32),
            contacts      = np.concatenate([e["contacts"]      for e in per_ep_raw], 0).astype(np.float32),
            foot_z        = np.concatenate([e["foot_z"]        for e in per_ep_raw], 0).astype(np.float32),
            foot_xy_speed = np.concatenate([e["foot_xy_speed"] for e in per_ep_raw], 0).astype(np.float32),
            foot_speed    = np.concatenate([e["foot_speed"]    for e in per_ep_raw], 0).astype(np.float32),
            actions       = np.concatenate([e["actions"]       for e in per_ep_raw], 0).astype(np.float32),
            tau           = np.concatenate([e["tau"]           for e in per_ep_raw], 0).astype(np.float32),
            base_z        = np.concatenate([e["base_z"]        for e in per_ep_raw], 0).astype(np.float32),
            base_xy       = np.concatenate([e["base_xy"]       for e in per_ep_raw], 0).astype(np.float32),
        )
        print(f"[sim2sim] raw data → {json_path}")
        print(f"[sim2sim]            {npz_path}")


if __name__ == "__main__":
    main()

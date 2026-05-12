"""
Sim2Sim Step 1 — Spawn GO2 in MuJoCo.

Loads the Unitree GO2 from mujoco_menagerie, initialises it at the Isaac Lab
default joint pose, and opens the interactive viewer.

Usage (from project root, with env_isaaclab active):
    conda run -n env_isaaclab python scripts/sim2sim/spawn_go2.py
"""

import os
import numpy as np
import mujoco
import mujoco.viewer

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SCENE_XML    = os.path.join(PROJECT_ROOT, "mujoco_menagerie", "unitree_go2", "scene.xml")

# ── Isaac Lab default joint positions (matches training) ──────────────────────
# Order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#        RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
DEFAULT_JOINT_POS = np.array([
     0.1,  0.8, -1.5,   # FL
    -0.1,  0.8, -1.5,   # FR
     0.1,  1.0, -1.5,   # RL
    -0.1,  1.0, -1.5,   # RR
], dtype=np.float64)

# PD gains matching Isaac Lab DCMotorCfg
KP = 25.0   # position gain  [N·m / rad]
KD =  0.5   # damping gain   [N·m·s / rad]
TORQUE_LIMIT = 23.5  # [N·m]

def build_model():
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    return m, d


def compute_spawn_height(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    """Compute base z so the lowest foot sphere just touches the floor."""
    foot_geom_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                     for n in ("FL", "FR", "RL", "RR")]
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0.0, 0.0, 1.0]
    d.qpos[3] = 1.0
    d.qpos[4:7] = 0.0
    d.qpos[7:19] = DEFAULT_JOINT_POS
    mujoco.mj_kinematics(m, d)
    foot_z_min  = min(d.geom_xpos[fid, 2] for fid in foot_geom_ids)
    foot_radius = float(m.geom_size[foot_geom_ids[0], 0])
    return float(1.0 - foot_z_min + foot_radius)


def reset(m: mujoco.MjModel, d: mujoco.MjData, spawn_height: float):
    mujoco.mj_resetData(m, d)
    d.qpos[0] = 0.0
    d.qpos[1] = 0.0
    d.qpos[2] = spawn_height
    d.qpos[3] = 1.0
    d.qpos[4:7] = 0.0
    d.qpos[7:19] = DEFAULT_JOINT_POS
    mujoco.mj_forward(m, d)


def pd_control(m: mujoco.MjModel, d: mujoco.MjData,
               q_target: np.ndarray) -> np.ndarray:
    """PD torque control: τ = Kp*(q_t - q) - Kd*qdot."""
    q    = d.qpos[7:19]
    qdot = d.qvel[6:18]
    tau  = KP * (q_target - q) - KD * qdot
    return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)


def get_joint_names(m: mujoco.MjModel):
    return [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(m.nu)
    ]


def main():
    m, d = build_model()
    spawn_height = compute_spawn_height(m, d)
    reset(m, d, spawn_height)

    print("[GO2 Spawn] Model loaded successfully.")
    print(f"  nq={m.nq}  nv={m.nv}  nu={m.nu}")
    print(f"  Actuators: {get_joint_names(m)}")
    print(f"  Default joint pos: {DEFAULT_JOINT_POS.round(3)}")
    print(f"  Spawn height: {spawn_height:.4f} m")
    print("[GO2 Spawn] Opening MuJoCo viewer — close window to exit.\n")

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.distance  = 2.5
        v.cam.elevation = -15
        v.cam.azimuth   = 135
        while v.is_running():
            d.ctrl[:] = pd_control(m, d, DEFAULT_JOINT_POS)
            mujoco.mj_step(m, d)
            v.sync()


if __name__ == "__main__":
    main()

"""
Hang GO2 100 m in the sky — no ground contact — and verify actuator forces.

Expected result:
  ncon = 0  (no contacts)
  d.actuator_force[i] ≈ Kp*(ctrl[i] - q[i]) - Kd*qdot[i]   (non-zero if policy output is non-zero)

If forces are near-zero → actuator model broken (wrong gaintype/biastype).
If forces are non-zero → contact WAS canceling torques on the ground.

Usage:
  conda run -n env_isaaclab python scripts/sim2sim/test_sky.py <checkpoint>
  conda run -n env_isaaclab python scripts/sim2sim/test_sky.py <checkpoint> --render
"""
import argparse
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer
import sys, os

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source", "cts_rma_project"))
sys.path.insert(0, SCRIPT_DIR)

from sim2sim_go2 import (
    SCENE_XML, DEFAULT_JOINT_POS,
    init_model_ids, fix_model_physics,
    compute_target_q, verify_joint_order, get_obs, load_policy,
    KP_NOM, KD_NOM, DECIMATION, PHYSICS_DT,
)

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--render",    action="store_true")
parser.add_argument("--real_time", action="store_true",
                    help="Slow simulation to real-time speed in render mode")
parser.add_argument("--vel_x", type=float, default=1.0,
                    help="Commanded forward velocity [m/s] (default 1.0)")
parser.add_argument("--vel_y", type=float, default=0.0)
parser.add_argument("--ang_z", type=float, default=0.0)
args = parser.parse_args()

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)
init_model_ids(m)
fix_model_physics(m)

# ── Zero gravity so the robot stays pinned in the sky ────────────────────────
m.opt.gravity[:] = 0.0   # disable gravity — body stays put, only legs move

SKY_Z = 3.0
mujoco.mj_resetData(m, d)
d.qpos[0:3]  = [0.0, 0.0, SKY_Z]
d.qpos[3]    = 1.0
d.qpos[4:7]  = 0.0
d.qpos[7:19] = DEFAULT_JOINT_POS
d.qvel[:]    = 0.0
mujoco.mj_forward(m, d)

print(f"\n[sky test] Robot z = {d.qpos[2]:.1f} m  ncon = {d.ncon}")
assert d.ncon == 0, "Unexpected contact — robot should be floating in sky"

joint_perm = verify_joint_order(m)
policy, _  = load_policy("baseline", args.checkpoint, "cpu")
vel_cmd    = np.array([args.vel_x, args.vel_y, args.ang_z], dtype=np.float32)
print(f"[sky test] vel_cmd = vx={args.vel_x}  vy={args.vel_y}  wz={args.ang_z}")

print(f"\nActuator model: Kp={KP_NOM}  Kd={KD_NOM}  DECIMATION={DECIMATION}")

POLICY_DT = PHYSICS_DT * DECIMATION

def run_step(step):
    obs     = get_obs(m, d, vel_cmd, joint_perm, DEFAULT_JOINT_POS)
    obs_t   = torch.from_numpy(obs).unsqueeze(0)
    action  = policy(obs_t).squeeze(0).numpy()

    target_q  = compute_target_q(action, joint_perm)
    d.ctrl[:] = target_q
    for _ in range(DECIMATION):
        mujoco.mj_step(m, d)
    # Pin base: zero out freejoint velocity & reset position so the body
    # stays exactly where it spawned — reaction forces from leg motion won't drift it
    d.qpos[0:3] = [0.0, 0.0, SKY_Z]
    d.qpos[3]   = 1.0
    d.qpos[4:7] = 0.0
    d.qvel[0:6] = 0.0

    q    = d.qpos[7:19]
    qdot = d.qvel[6:18]
    expected = np.clip(KP_NOM * (target_q - q) - KD_NOM * qdot, -23.5, 23.5)

    force_mean    = float(np.mean(np.abs(d.actuator_force)))
    force_max     = float(np.max(np.abs(d.actuator_force)))
    expected_mean = float(np.mean(np.abs(expected)))
    ctrl_dev      = float(np.mean(np.abs(target_q - q)))

    if step % 20 == 0:
        print(f"step={step:4d}  ncon={d.ncon}  z={d.qpos[2]:.2f}m"
              f"  force_mean={force_mean:.3f}  expected_mean={expected_mean:.3f}"
              f"  force_max={force_max:.2f}  ctrl_dev={ctrl_dev:.4f}")
        if step == 0:
            leg = ["FL","FR","RL","RR"]
            for label, arr in [("action  ", action), ("act_force", d.actuator_force), ("expected ", expected)]:
                print(f"  {label}: ", end="")
                for i, l in enumerate(leg):
                    v = arr[i*3:(i+1)*3]
                    print(f"{l}=[{v[0]:+.1f},{v[1]:+.1f},{v[2]:+.1f}]", end=" ")
                print()

    return action

if args.render:
    with mujoco.viewer.launch_passive(m, d) as v:
        # Camera: zoom out and look UP slightly so robot is visible at 5 m
        v.cam.distance  = 4.0
        v.cam.elevation = -5
        v.cam.azimuth   = 135
        v.cam.lookat[:] = [0.0, 0.0, SKY_Z]   # point camera at sky spawn height

        step = 0
        print("\n[render] viewer open — robot hanging at 5 m, policy running, no ground contact\n")
        with torch.no_grad():
            while v.is_running():
                t0 = time.time()
                run_step(step)
                v.sync()
                step += 1
                if args.real_time:
                    elapsed = time.time() - t0
                    if elapsed < POLICY_DT:
                        time.sleep(POLICY_DT - elapsed)
else:
    print(f"\n{'step':>4}  {'ncon':>4}  {'ctrl_dev_mean':>14}  "
          f"{'act_force_mean':>15}  {'act_force_max':>14}  {'expected_mean':>14}")
    print("-" * 80)
    with torch.no_grad():
        for step in range(25):
            obs     = get_obs(m, d, vel_cmd, joint_perm, DEFAULT_JOINT_POS)
            obs_t   = torch.from_numpy(obs).unsqueeze(0)
            action  = policy(obs_t).squeeze(0).numpy()

            target_q = compute_target_q(action, joint_perm)
            d.ctrl[:] = target_q
            for _ in range(DECIMATION):
                mujoco.mj_step(m, d)

            q    = d.qpos[7:19]
            qdot = d.qvel[6:18]
            expected = np.clip(KP_NOM * (target_q - q) - KD_NOM * qdot, -23.5, 23.5)

            ctrl_dev      = float(np.mean(np.abs(target_q - q)))
            force_mean    = float(np.mean(np.abs(d.actuator_force)))
            force_max     = float(np.max(np.abs(d.actuator_force)))
            expected_mean = float(np.mean(np.abs(expected)))

            print(f"{step:4d}  {d.ncon:4d}  {ctrl_dev:14.4f}  "
                  f"{force_mean:15.4f}  {force_max:14.4f}  {expected_mean:14.4f}")

            if step == 0:
                leg = ["FL", "FR", "RL", "RR"]
                for label, arr in [("action  ", action),
                                   ("act_force", d.actuator_force),
                                   ("expected ", expected)]:
                    print(f"  {label}: ", end="")
                    for i, l in enumerate(leg):
                        v = arr[i*3:(i+1)*3]
                        print(f"{l}=[{v[0]:+.1f},{v[1]:+.1f},{v[2]:+.1f}]", end=" ")
                    print()

    print("\n[Interpretation]")
    print("  act_force_mean ≈ expected_mean  → actuator model correct, contact was the issue")
    print("  act_force_mean ≈ 0              → actuator gaintype/biastype not set correctly")

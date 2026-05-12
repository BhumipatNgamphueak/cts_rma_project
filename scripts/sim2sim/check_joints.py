"""Check whether joint angles show a gait pattern or symmetric standing."""
import numpy as np
import torch
import mujoco
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'cts_rma_project'))
sys.path.insert(0, os.path.dirname(__file__))
from sim2sim_go2 import (
    SCENE_XML, DEFAULT_JOINT_POS, init_model_ids, fix_model_physics,
    compute_spawn_height, verify_joint_order, get_obs, _reset_pose,
    load_policy,
)

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)
init_model_ids(m)
fix_model_physics(m)
spawn_h = compute_spawn_height(m, d)
joint_perm = verify_joint_order(m)

policy, _ = load_policy('baseline', 'logs/baseline/2026-05-03_20-05-35_baseline_go2/model_18600.pt', 'cpu')

vel_cmd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_reset_pose(m, d, spawn_h)

# Apply Isaac Lab-style reset: random joint offsets ±1.047 rad + initial base velocity
rng = np.random.default_rng(42)
d.qpos[7:19] += rng.uniform(-1.047, 1.047, 12)   # joint offset
d.qvel[0:3]   = rng.uniform(-0.5, 0.5, 3)         # base linear vel
d.qvel[3:6]   = rng.uniform(-1.0, 1.0, 3)         # base angular vel
mujoco.mj_forward(m, d)
print("=== With Isaac Lab random initial conditions ===")

print(f"{'step':>4}  {'FL_hip':>7} {'FL_thigh':>8} {'FL_calf':>7} | {'FR_hip':>7} {'FR_thigh':>8} {'FR_calf':>7} | vx")
with torch.no_grad():
    for step in range(300):
        obs = get_obs(m, d, vel_cmd, joint_perm, DEFAULT_JOINT_POS)
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        action = policy(obs_t).squeeze(0).cpu().numpy()
        target_q = DEFAULT_JOINT_POS + action[joint_perm] * 0.25
        tau = 25.0 * (target_q - d.qpos[7:19]) - 0.5 * d.qvel[6:18]
        d.ctrl[:] = np.clip(tau, -23.5, 23.5)
        for _ in range(5):
            mujoco.mj_step(m, d)
        if step % 10 == 0:
            q = d.qpos[7:13]
            vx = d.qvel[0]
            print(f"{step:4d}  {q[0]:+.3f}   {q[1]:+.3f}    {q[2]:+.3f} | {q[3]:+.3f}   {q[4]:+.3f}    {q[5]:+.3f} | {vx:+.4f}")

# v3 — Terrain + Terrain-Height Privileged Observation

**Apply after v2 training completes.** This patch adds:
1. A generated cobblestone terrain (replaces flat ground)
2. A 77-dim height-scan privileged observation (size 1.0 × 0.6 m grid at 0.1 m resolution)
3. A new `priv_mode = "FULL_T"` that includes terrain heights → privileged dim 103

Baseline observation (37-dim proprio) is unchanged. Only RMA/CTS teacher/critic see terrain.

**The change is applied in `shared_env_cfg.py` (affects all three methods identically), so the comparison stays fair.**

---

## Files to edit

| # | File | What |
|---|---|---|
| 1 | `tasks/shared/shared_env_cfg.py` | Terrain generator + scene + height_scanner sensor |
| 2 | `tasks/shared/mdp/observations.py` | New `privileged_terrain_go2` term + `FULL_T` mode |
| 3 | `tasks/shared/mdp/__init__.py` | Export the new term |
| 4 | `tasks/cts/cts_env_cfg.py` | Accept `priv_mode="FULL_T"` (already works via PRIV_DIMS) |
| 5 | `tasks/rma/rma_env_cfg.py` | Same |
| 6 | `tasks/baseline/baseline_env_cfg.py` | **NO CHANGE** — baseline doesn't see terrain |
| 7 | `scripts/cts/train.py` | already supports `--priv_mode FULL_T` via PRIV_DIMS |
| 8 | `scripts/rma/train_phase1.py` | same |

---

## 1) `tasks/shared/shared_env_cfg.py`

### 1a. Add terrain generator (top of file, near other imports)

```python
import isaaclab.terrains as terrain_gen
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# Cobblestone road — moderate roughness (matches OpenTopic velocity_env_cfg.py)
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        # To add rougher tiles later:
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.3, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),
    },
)
```

### 1b. Replace `SharedSceneCfg.terrain` and add `height_scanner`

```python
@configclass
class SharedSceneCfg(InteractiveSceneCfg):
    # ── REPLACED: flat plane → generated terrain ────────────────────────────
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    # ── NEW: height scanner — 11×7 grid (1.0m × 0.6m at 0.1m res) ──────────
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
```

### 1c. Add `update_period` for the height scanner in `__post_init__`

In `SharedEnvCfg.__post_init__` (search for `self.sim.dt`), add:

```python
def __post_init__(self):
    super().__post_init__()
    self.sim.dt      = 0.005
    self.decimation  = 2
    self.episode_length_s = 20.0
    self.action_space = 12
    # NEW: tick height scanner once per policy step (every decimation×dt = 0.01 s)
    self.scene.height_scanner.update_period = self.decimation * self.sim.dt
```

---

## 2) `tasks/shared/mdp/observations.py`

### 2a. Add the terrain-height term (after `privileged_external_go2`)

```python
def privileged_terrain_go2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg:  SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Terrain height map relative to robot base z, in body-yaw frame.

    Layout: 11 × 7 = 77 height samples in a 1.0 m × 0.6 m grid centered under
    the base, with the +x cells in the robot's forward direction (yaw-aligned
    by RayCasterCfg(attach_yaw_only=True)). Each value is

        h_rel = (base_z - 0.5) - ray_hit_z

    so values around 0 = ground at default standing height, positive = bump,
    negative = hole. The −0.5 offset matches Isaac Lab's stock `mdp.height_scan`
    convention (used in unitree_rl_lab); keeping the convention lets us reuse
    the same numerical scale the literature reports.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    asset  = env.scene[asset_cfg.name]
    # Sensor outputs ray hit positions (N, R, 3) in world frame; we need scalar
    # height clearance per ray. Isaac Lab provides .data.pos_w (the sensor
    # origin) and .data.ray_hits_w (the hits).
    base_z   = asset.data.root_pos_w[:, 2:3]                  # (N, 1)
    hit_z    = sensor.data.ray_hits_w[..., 2]                  # (N, R)
    return (base_z - 0.5 - hit_z).clamp(-1.0, 1.0)            # (N, R)


def privileged_full_terrain_go2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    height_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """x^int(16) ⊕ x^ext(10) ⊕ x^terr(77) = 103."""
    return torch.cat([
        privileged_internal_go2(env),
        privileged_external_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg),
        privileged_terrain_go2(env, sensor_cfg=height_cfg, asset_cfg=asset_cfg),
    ], dim=-1)
```

### 2b. Extend `PRIV_DIMS` and `privileged_subset_go2`

```python
# Replace:
PRIV_DIMS = {"FULL": 26, "INT": 16, "EXT": 10}
# With:
PRIV_DIMS = {
    "FULL":   26,
    "INT":    16,
    "EXT":    10,
    "TERR":   77,    # terrain heights only
    "FULL_T": 103,   # FULL + terrain
}
```

```python
def privileged_subset_go2(
    env, mode="FULL",
    asset_cfg=SceneEntityCfg("robot"),
    sensor_cfg=SceneEntityCfg("contact_forces"),
):
    m = (mode or "FULL").upper()
    if m == "INT":    return privileged_internal_go2(env)
    if m == "EXT":    return privileged_external_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
    if m == "TERR":   return privileged_terrain_go2(env)
    if m == "FULL_T": return privileged_full_terrain_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
    return privileged_full_go2(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
```

---

## 3) `tasks/shared/mdp/__init__.py`

Export the new symbols:

```python
from .observations import (
    proprioceptive_obs_go2,
    privileged_internal_go2,
    privileged_external_go2,
    privileged_terrain_go2,        # NEW
    privileged_full_go2,
    privileged_full_terrain_go2,   # NEW
    privileged_subset_go2,
    combined_obs_subset,
    PRIV_DIMS,
    # ... keep existing exports
)
```

---

## 4) `tasks/cts/cts_env_cfg.py` — already works

The CTSEnvCfg reads `PRIV_DIMS[priv_mode]` so setting `priv_mode="FULL_T"` automatically picks up 103. No code changes needed — just pass `--priv_mode FULL_T` at train time.

Optional: bump the default if you want terrain by default:
```python
priv_mode: str = "FULL_T"  # was "FULL"
```

## 5) `tasks/rma/rma_env_cfg.py` — same as CTS

Reads `PRIV_DIMS[priv_mode]` automatically. Optional default change.

---

## 6) `scripts/cts/train.py` — already works

Your edited `train.py` reads `PRIV_DIMS[priv_mode]`, so:

```bash
# v3 training command — terrain + FULL_T privileged
/home/drl-68/IsaacLab/isaaclab.sh -p scripts/cts/train.py \
    --num_envs 4096 --max_iterations 25000 --seed 42 \
    --latent_dim 8 --history_len 50 --priv_mode FULL_T \
    --lambda_rec 5.0 \
    --experiment cts_go2_v3_terrain_l8 --device cuda:0 --headless
```

Same pattern for `train_phase1.py` (RMA) and `train.py` (Baseline — but baseline gets the new terrain in physics, NOT in obs).

---

## 7) Sanity-check before launching v3 training

After applying patch sections 1–3, run:

```bash
cd /home/drl-68/t_s_policy/cts_rma_project
conda run -n env_isaaclab python -c "
from cts_rma_project.tasks.shared.mdp import PRIV_DIMS, privileged_full_terrain_go2
print('PRIV_DIMS:', PRIV_DIMS)
assert PRIV_DIMS['FULL_T'] == 103, 'FULL_T should be 103'
print('OK — privileged shape: 16+10+77 = 103')
"
```

If that prints `OK`, the patch is wired correctly and v3 training can start.

---

## 8) v3 training (after patch is applied)

Three terminals, mirror v2 commands but with `--priv_mode FULL_T` and `_v3_terrain` experiment names:

```bash
# Terminal 1 — Baseline (terrain in physics, no terrain in obs)
ITERS=25000; SEED=42
/home/drl-68/IsaacLab/isaaclab.sh -p scripts/baseline/train.py \
    --num_envs 4096 --max_iterations $ITERS --seed $SEED \
    --experiment baseline_go2_v3_terrain --device cuda:0 --headless

# Terminal 2 — RMA Phase 1 (terrain heights in critic+teacher)
/home/drl-68/IsaacLab/isaaclab.sh -p scripts/rma/train_phase1.py \
    --num_envs 4096 --max_iterations $ITERS --seed $SEED \
    --latent_dim 8 --priv_mode FULL_T \
    --experiment rma_go2_v3_terrain_l8 --device cuda:0 --headless

# After RMA Phase 1 → Phase 2 (1000 iters)
PHASE1=$(ls -t logs/rma/*rma_go2_v3_terrain_l8/model_final.pt | head -1)
/home/drl-68/IsaacLab/isaaclab.sh -p scripts/rma/train_phase2.py \
    --checkpoint "$PHASE1" --num_envs 4096 --num_iterations 1000 \
    --device cuda:0 --headless

# Terminal 3 — CTS
/home/drl-68/IsaacLab/isaaclab.sh -p scripts/cts/train.py \
    --num_envs 4096 --max_iterations $ITERS --seed $SEED \
    --latent_dim 8 --history_len 50 --priv_mode FULL_T \
    --lambda_rec 5.0 \
    --experiment cts_go2_v3_terrain_l8 --device cuda:0 --headless
```

---

## 9) Sim2sim deployment with terrain — NOT in this patch

Running v3 policies in MuJoCo will need:
1. A MuJoCo terrain mesh that matches the Isaac Lab terrain (or close enough).
2. A `get_height_scan_mujoco()` function to fake/sample the 77-dim height obs at deployment.

This is a separate effort. For Baseline / CTS-student / RMA-Phase-2 the proprioceptive obs is unchanged, so they CAN deploy on flat MuJoCo (just with possibly degraded behaviour because they trained expecting rough terrain). For testing teacher/critic in MuJoCo you'd need the height-scan path.

---

## 10) Comparison fairness

All three v3 runs use the **same `SharedSceneCfg`** (terrain + height_scanner attached to every env), so they train on identical worlds.

| Method | What it sees |
|---|---|
| Baseline | 37-dim proprio only. Terrain affects physics but not obs. |
| RMA Phase 1 | 37-dim proprio + 103-dim privileged (incl. terrain) → encoder → z |
| RMA Phase 2 | 37-dim proprio + history → adaptation module → ẑ |
| CTS teacher | 37-dim proprio + 103-dim privileged → encoder → z |
| CTS student | 37-dim proprio history → CNN encoder → z (must infer terrain from history) |

This is the standard sim-to-real / privileged-distillation setup. The comparison stays fair: same env, same training, same evaluation.

Expected finding: **terrain raises the bar for sim2sim transfer** (more dynamics, irregular contacts). RMA/CTS *should* benefit more than Baseline from the privileged terrain access, IF the adaptation/student paths can recover the heights from history.

---

## Checklist (apply in order after v2 finishes)

- [ ] Confirm v2 runs all completed (`logs/*/2026-05-10_17-07-*/model_final.pt` exist for all 3)
- [ ] Run v2 `eval_metrics.py` and save the report
- [ ] Apply patch sections 1, 2, 3 above
- [ ] Run sanity check (section 7)
- [ ] Launch v3 trainings (section 8, three terminals)
- [ ] After all v3 finishes → run `eval_metrics.py` again for v3 (separate output dir)
- [ ] Compare v2-vs-v3 in the paper / report

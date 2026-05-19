# Project Answers — FRA 503 Final Project

> **Status note for the LaTeX-drafting AI.** All experiments referenced in this
> document have finished. Numbers below are the *final* values from the report
> CSVs (`results/ood_one_leg.csv`, `results/ood_go2.csv`,
> `results/sim2sim_go2.csv`) and the auto-generated tables
> (`results/one_leg_results_table.md`, `results/go2_results_table.md`).
> **Do not fabricate any number, equation, hyperparameter, or citation.** When
> a value is reported here, copy it verbatim into the report (round only if
> needed for column width).
>
> **Two drafts** will be produced from this file:
> - **Draft A** — smooth pivot: Phase 1 (~30 %) → diagnosis → Phase 2 Go2 (~50 %).
> - **Draft B** — Go2-centric: most of the report on Go2, with a short
>   "Project Evolution" section (~0.75 page) acknowledging Phase 1 and the pivot.
>
> **Document map (where to find what).**
> §A — Phase 1 (single-leg).  §B — Go2 robot & MDP definition (platform, obs,
> privileged knowledge, reward, DR, episode).  **§B.7** — Architectures
> (Baseline / RMA / CTS data flow + PPO hyper-parameters).
> §C — Go2 experimental results (per-row table, transfer ratios,
> priv-subset ablation including behaviour metrics).  §D — Reproducibility,
> title block, framing emphasis, miscellaneous gotchas.  §E — One-line
> figure caption stubs.  §F — Long-form figure descriptions with
> quote-ready readings.  §G — Suggested figure-to-section mapping for the
> 12-page LaTeX template.  §H — Definition of "worst case" used in the
> figure captions.
>
> **Headline story for both drafts.** On the single-leg fixed-base platform all
> three methods are competitive and the T–S advantage fails to appear (Baseline
> in fact wins on raw reward). On the free-base Unitree Go2, the T–S advantage
> reappears — and it is **CTS, not RMA**, that wins. Evaluated under the *exact
> v2 training distribution* (flat ground + velocity push, 20 s episodes, 30
> episodes/cell), CTS is the strongest method at the nominal condition
> (DR×1: 100 % survival, 0.130 m/s tracking error) **and** degrades the least
> under OOD stress (DR×2: 86.7 % survival vs Baseline 73.3 % vs RMA 23.3 %).
> CTS is also the most robust to the training-time push (survival 100→100 % at
> DR×1 with/without push) and the best velocity tracker in *both* simulators.
> **RMA's Phase-2 adaptation module is the clear loser once tested faithfully**:
> it is fragile to the very push it was trained with (80 % survival at Isaac
> DR×1, 23 % at DR×2) and its adaptation module collapses entirely in MuJoCo —
> producing a near-stationary policy (forward displacement ≈ 0 m, velocity-RMSE
> pinned at the 0.5 m/s command magnitude) that "survives" only by not
> attempting to walk.
>
> **Methodology caveat (state explicitly).** The MuJoCo sim2sim harness
> (`scripts/sim2sim/sim2sim_go2.py`) has *no push/disturbance mechanism*; it can
> only reproduce the **no-push** condition, which is strictly easier than v2's
> training distribution. Therefore the Isaac↔MuJoCo transfer comparison is
> reported at no-push (apples-to-apples), while the headline robustness ranking
> uses the training-faithful Isaac condition (push ON). Both Isaac conditions
> are reported in §C.2.

---

## A. Phase 1 Closure (single-leg hexapod)

### A.1 Phase-1 evaluation status — done, Isaac-only

- **Do actual numbers exist for Baseline / RMA / CTS on the single leg?** —
  **Yes, in Isaac Lab, with one fair configuration (FULL privilege, latent Z = 8,
  100 episodes per cell, two DR scales).** Source CSV: `results/ood_one_leg.csv`
  (6 rows). Markdown table: `results/one_leg_results_table.md`. Figures:
  `results/figures/one_leg/fig_one_leg_headline.{png,pdf}` and
  `fig_one_leg_ood_retention.{png,pdf}`.
- **MuJoCo evaluation of the single leg is not reported numerically.** The
  earlier qualitative observation that the T–S advantage did not survive
  Isaac→MuJoCo on the single leg is acknowledged in the diagnosis paragraph
  below, but in this final report the single-leg quantitative comparison is
  presented **Isaac-only** under one fair configuration. (Cross-sim transfer
  is reported quantitatively only for Go2 in §C.)
- **Phase-1 numbers (Isaac, FULL privilege, Z = 8, 100 episodes per cell):**

  | method   | priv | Z  | DR×s | episode return (mean ± std) | success % | mean length |
  |---       |---   |---:|---:  |---                          |---:       |---:         |
  | Baseline | FULL | —  | 1×   | **2043.8 ± 86.5**           | 100       | 999         |
  | Baseline | FULL | —  | 2×   | **1802.1 ± 336.3**          | 100       | 999         |
  | RMA      | FULL | 8  | 1×   | **1879.5 ± 161.3**          | 100       | 999         |
  | RMA      | FULL | 8  | 2×   | **1741.0 ± 277.0**          | 100       | 999         |
  | CTS      | FULL | 8  | 1×   | **1549.7 ± 134.4**          | 100       | 999         |
  | CTS      | FULL | 8  | 2×   | **1413.8 ± 261.0**          | 100       | 999         |

  OOD retention (DR×2 reward as fraction of DR×1 reward; spec ≥ 70 %):
  Baseline **88.2 %** ✓PASS, RMA **92.6 %** ✓PASS, CTS **91.2 %** ✓PASS.

### A.2 Phase-1 failure mode (of the T–S hypothesis on this platform)
- [x] **Baseline matches or beats T-S on this platform.**
- [ ] All three collapse together.
- [ ] T-S degrades more sharply than Baseline.
- [ ] Other.

**Justification (numbers from §A.1):** At the training distribution Baseline
returns **2043.8** vs RMA **1879.5** vs CTS **1549.7** (mean episode return,
identical reward and termination terms). All three methods reach 100 % success
and both DR×1 and DR×2 OOD-retention pass the 70 % spec. The privileged-input
T–S methods therefore do *not* provide the expected advantage on this platform.

### A.3 Diagnosis hypotheses to commit to in the report
- **H1 — Fixed base eliminates body-level dynamics that privileged signals are
  designed to inform.** → **COMMIT.** The test station's base is constrained by
  aluminium rails, so CoM / base-velocity / terrain-interaction signals have no
  independent body dynamics to describe; this is the primary diagnosis.
- **H2 — Reference-dominated task (periodic gait clock) leaves little room for
  adaptive behaviour to manifest.** → **COMMIT.** Foot-trajectory tracking on a
  periodic gait clock dominates the policy output, so even if the teacher
  latent carried useful information the headroom for it to change behaviour is
  small.
- **H3 — Sim-to-Sim differences (contact solver, integrator) primarily affect
  base motion, which is rail-constrained out of the test.** → **MENTION.** The
  PhysX↔MuJoCo gap is largest on base dynamics, but those are mostly removed by
  the rails, so the sim-to-sim discrepancy that T–S adaptation could absorb is
  attenuated.
- **H4 — The privileged vector is largely redundant with proprioception on a
  fixed-base single leg.** → **COMMIT.** With a fixed base and a short
  kinematic chain, the privileged state carries little information not already
  recoverable from joint states / contact flags, so the teacher latent gives
  the student no real advantage.
- **Overall framing:** the single-leg test station was inadequate to test the
  T–S hypothesis as Kumar et al. (2021) and Wang et al. (2024) intended, which
  motivates the pivot to a free-base quadruped (see D.3 — frame as a
  methodological strength, not a setback).

### A.4 Phase-1 figures/tables to include in Draft A
Generated by `scripts/plot_results_one_leg.py` from `results/ood_one_leg.csv`.

| filename                                                            | shows                                                                        | status |
|---                                                                  |---                                                                           |---     |
| `results/figures/one_leg/fig_one_leg_headline.{png,pdf}`            | 2 panels: (A) mean reward DR×1 / DR×2 with Δ% annotation; (B) success rate   | ready  |
| `results/figures/one_leg/fig_one_leg_ood_retention.{png,pdf}`       | 1 panel: OOD-retention bar with 70 % spec line and PASS badges               | ready  |
| `results/one_leg_results_table.md`                                  | combined per-row table + retention table (markdown)                          | ready  |

Single-leg platform spec (from `tasks/one_leg/.../*_env.py`, `one_leggy8.usd`):
single hexapod leg on a fixed Cartesian test station; **3 DOF (hip / knee /
ankle)**; **≈ 2.902 kg**; base translations constrained by aluminium rails;
task = foot-trajectory tracking on a periodic gait clock; proprioception =
joint state + previous action + reference foot position + gait-clock phase
`(c, sinφ, cosφ)`; privileged state ≈ body parameters (mass, friction, gains,
delay) + contact force/flag + joint torques/accels. (Exact dims are not needed
for the reduced Phase-1 narrative.)

---

## B. Go2 Setup (Phase 2)

### B.1 Robot platform
```yaml
model_source: "Isaac Lab UNITREE_GO2_CFG (isaaclab_assets/robots/unitree.py:138); MuJoCo side: mujoco_menagerie/unitree_go2/go2.xml"
dof_count: 12                 # 4 legs × (hip, thigh, calf)
total_mass_kg: 15.21          # summed from mujoco_menagerie Go2 XML (13 rigid bodies)
action_space: "joint position targets"   # JointPositionActionCfg, scale=0.25, use_default_offset=True
pd_gains:                     # DC-motor actuator model, all 12 joints share
  Kp: 25.0                    # N.m/rad        (DR scaled by 0.7-1.3 → 17.5-32.5)
  Kd: 0.5                     # N.m.s/rad      (DR scaled by 0.65-1.35 → 0.325-0.675)
torque_limit_Nm: 23.5         # effort_limit/saturation_effort in UNITREE_GO2_CFG
joint_vel_limit_rad_s: 30.0
control_frequency_hz: 50      # decimation=4 over 200 Hz physics
sim_dt_s: 0.005               # 200 Hz physics
```
Action mapping: `q_target = q_default + 0.25 · a_t`. Default joint pose
(stand): `hip ≈ ±0.1 rad`, `front-thigh ≈ 0.8 rad`, `rear-thigh ≈ 1.0 rad`,
`calf ≈ -1.5 rad`.

### B.2 Observation vector (Go2) — proprioceptive `o_t ∈ R^37`
Source: `tasks/shared/mdp/observations.py::proprioceptive_obs_go2`.
```yaml
observation:
  - {symbol: "q - q_default", description: "joint positions relative to default pose", dim: 12, source: "robot articulation"}
  - {symbol: "q_dot",         description: "joint velocities",                          dim: 12, source: "robot articulation"}
  - {symbol: "omega_b",       description: "base angular velocity in body frame",      dim: 3,  source: "IMU-equivalent"}
  - {symbol: "g_b",           description: "projected gravity (orientation proxy)",    dim: 3,  source: "articulation"}
  - {symbol: "v_cmd",         description: "commanded base velocity (lin_x, lin_y, ang_z)", dim: 3, source: "velocity command manager"}
  - {symbol: "c_foot",        description: "per-foot binary contact flag (>1 N on z)", dim: 4,  source: "contact sensor on *_foot"}
total_dim: 37
```
Training-time observation noise (disabled at play/eval): joint_pos σ = 0.01 rad,
joint_vel σ = 0.50 rad/s, ω_b σ = 0.20 rad/s, g_b σ = 0.05. **Base linear
velocity is *not* in the observation** — only base angular velocity.

### B.3 Privileged knowledge vector (Go2) — `x_t ∈ R^26`
Source: `tasks/shared/mdp/observations.py` (`privileged_internal_go2`,
`privileged_external_go2`, `privileged_full_go2`).
```yaml
privileged_internal:           # x_int — episode-constant body parameters, 16-D
  - {symbol: "mu",            description: "ground friction coefficient (raw)",                         dim: 1}
  - {symbol: "e_rest",        description: "restitution coefficient (raw)",                             dim: 1}
  - {symbol: "delta_m_base",  description: "base mass-scale deviation (scale - 1)",                     dim: 1}
  - {symbol: "delta_Kp",      description: "Kp scale deviation per joint type (hip/thigh/calf)",        dim: 3}
  - {symbol: "delta_Kd",      description: "Kd scale deviation per joint type (hip/thigh/calf)",        dim: 3}
  - {symbol: "delta_com",     description: "base centre-of-mass offset [m] in body frame",              dim: 3}
  - {symbol: "delta_I_base",  description: "base inertia-scale deviation per axis (scale - 1)",         dim: 3}
  - {symbol: "d_act",         description: "action delay [ms] normalised to [0,1] (20 ms -> 1.0)",      dim: 1}
privileged_external:           # x_ext — timestep-varying interaction signals, 10-D
  - {symbol: "f_contact",     description: "net contact force summed over the four feet [N], world",   dim: 3}
  - {symbol: "c_bin",         description: "per-foot binary contact flag (>1 N)",                       dim: 4}
  - {symbol: "tau_avg",       description: "mean |applied torque| per joint type (hip/thigh/calf) [N.m]", dim: 3}
includes_base_linear_velocity: false        # not in o_t or x_t
includes_terrain_height: false              # flat ground; no height-scan
total_dim: 26                               # 16 internal + 10 external
teacher_input_dim: 63                       # = o_t(37) + x_t(26); encoder maps x_t(26) -> z(Z), policy sees [o_t, z]
```
The asymmetric **critic** input is `[o_t(37), x_t(26)] = 63-D` for RMA Phase 1
and for CTS. `delta_Kp`/`delta_Kd` are per joint *type* (3 values: hip/thigh/calf).

### B.4 Reward function (Go2)
All three methods share **exactly** this reward
(`tasks/shared/shared_env_cfg.py::SharedRewardsCfg`). Standard terms use
`isaaclab.envs.mdp`; custom terms are in `tasks/shared/mdp/rewards.py`. No
formal citation is asserted; if the report wants one, the author should supply it.

| term                  | function                     | weight  | LaTeX-ready form (per env, per step)                                                       | notes  |
|---                    |---                           |---:     |---                                                                                         |---     |
| track lin. vel. xy    | `track_lin_vel_xy_exp`       | **+1.5**| $\exp\!\bigl(-\lVert\mathbf v^{cmd}_{xy}-\mathbf v_{xy}\rVert^2/\sigma^2\bigr),\ \sigma^2=0.25$ | task   |
| track ang. vel. z     | `track_ang_vel_z_exp`        |**+0.75**| $\exp\!\bigl(-(\omega^{cmd}_z-\omega_z)^2/\sigma^2\bigr),\ \sigma^2=0.25$                   | task   |
| base lin. vel. z      | `lin_vel_z_l2`               | **−2.0**| $v_z^2$                                                                                    |        |
| base ang. vel. xy     | `ang_vel_xy_l2`              |**−0.05**| $\lVert\boldsymbol\omega_{xy}\rVert^2$                                                     |        |
| joint vel.            | `joint_vel_l2`               |**−1e-3**| $\lVert\dot{\mathbf q}\rVert^2$                                                            |        |
| joint acc.            | `joint_acc_l2`               |**−2.5e-7**| $\lVert\ddot{\mathbf q}\rVert^2$                                                         |        |
| joint torques         | `joint_torques_l2`           |**−2e-4**| $\lVert\boldsymbol\tau\rVert^2$                                                            |        |
| action rate           | `action_rate_l2`             | **−0.1**| $\lVert\mathbf a_t-\mathbf a_{t-1}\rVert^2$                                                |        |
| joint pos. limits     | `joint_pos_limits`           | **−10.0**| sum of joint-limit violations                                                              |        |
| energy                | `energy` (custom)            |**−2e-5**| $\sum_i\lvert\dot q_i\rvert\,\lvert\tau_i\rvert$                                            | custom |
| flat orientation      | `flat_orientation_l2`        | **−2.5**| $\lVert\mathbf g_{xy}\rVert^2$                                                             |        |
| joint pose            | `joint_position_penalty`     | **−0.7**| $\lVert\mathbf q-\mathbf q_{def}\rVert$, ×5 when standing (cmd≈0, speed<0.3 m/s)            | custom |
| feet air time         | `feet_air_time`              | **+0.1**| $\sum_f (t^{air}_{last}-0.5)\,\mathbb{1}[\text{first contact}]$, gated by $\lVert v^{cmd}_{xy}\rVert>0.1$ | custom |
| air-time variance     | `air_time_variance_penalty`  | **−1.0**| $\mathrm{var}(\mathrm{clip}(t^{air},0.5))+\mathrm{var}(\mathrm{clip}(t^{contact},0.5))$    | custom |
| feet slide            | `feet_slide`                 | **−0.1**| $\sum_f\lVert\mathbf v^{foot}_{xy}\rVert\,\mathbb{1}[\text{contact}]$                       | custom |
| alive                 | `is_alive`                   | **+1.0**| $1$ per surviving step                                                                     |        |
| termination penalty   | `is_terminated`              |  **0.0**| disabled                                                                                   |        |
| undesired contacts    | `undesired_contacts`         | **−1.0**| count of bodies in `{*_hip,*_thigh,*_calf}` with contact force > 1 N                       |        |

Command sampling: `UniformVelocityCommandCfg`, resampled every 10 s, curriculum
`lin_x∈[−0.3,0.3], lin_y∈[−0.2,0.2], ang_z∈[−0.5,0.5]` widening toward
`±1.0` once the lin-vel-tracking reward exceeds ~80 % of weight; 10 % standing
envs; heading-tracking enabled.

### B.5 Domain randomisation ranges (Go2)
Authoritative source: `tasks/shared/shared_env_cfg.py::SharedEventCfg`. OOD
scaling at eval time scales each "scaled-DR" parameter range around its
midpoint by factor `s` (see `ood_scaling_rule`). Disturbance and reset-state
ranges are *not* OOD-scaled.

```yaml
dr_ranges:
  # episode-reset randomisation (these are the "scaled-DR" parameters)
  - {parameter: "static & dynamic friction (foot/ground)", unit: "-",        training_low: 0.30, training_high: 1.70}
  - {parameter: "restitution",                              unit: "-",        training_low: 0.00, training_high: 0.15}
  - {parameter: "base mass scale",                          unit: "x nominal",training_low: 0.85, training_high: 1.15}
  - {parameter: "base inertia scale (uniform per axis)",    unit: "x nominal",training_low: 0.70, training_high: 1.30}
  - {parameter: "actuator Kp scale",                        unit: "x nominal",training_low: 0.70, training_high: 1.30}
  - {parameter: "actuator Kd scale",                        unit: "x nominal",training_low: 0.65, training_high: 1.35}
  - {parameter: "base CoM offset (per axis)",               unit: "m",        training_low: -0.08,training_high: 0.08}
  - {parameter: "action delay",                             unit: "ms",       training_low: 0.0,  training_high: 30.0}
  # disturbances (NOT scaled at OOD eval)
  - {parameter: "velocity push (lin x/y, ang r/p/y), 5-10 s", unit: "m/s & rad/s", training_low: -1.0, training_high: 1.0}
  - {parameter: "external force impulse on base @ reset",    unit: "N",   training_low: -5.0, training_high: 5.0}
  - {parameter: "external torque impulse on base @ reset",   unit: "N.m", training_low: -2.0, training_high: 2.0}
  - {parameter: "external force impulse on base, 3-8 s",     unit: "N",   training_low: -10.0,training_high: 10.0}
  - {parameter: "external torque impulse on base, 3-8 s",    unit: "N.m", training_low: -3.0, training_high: 3.0}
  # reset-state randomisation (also NOT scaled at OOD)
  - {parameter: "reset base pose: x,y",  unit: "m",     training_low: -0.5,   training_high: 0.5}
  - {parameter: "reset base pose: z",    unit: "m",     training_low: 0.0,    training_high: 0.3}
  - {parameter: "reset base yaw",        unit: "rad",   training_low: -3.14,  training_high: 3.14}
  - {parameter: "reset base velocities", unit: "m/s & rad/s", training_low: 0.0, training_high: 0.0}  # zeroed (sim2sim fix #1)
  - {parameter: "reset joint pos offset",unit: "rad",   training_low: -1.047, training_high: 1.047}
  - {parameter: "reset joint vel",       unit: "rad/s", training_low: -1.0,   training_high: 1.0}
ood_scaling_rule: "Each scaled-DR parameter range is widened about its midpoint: [c - s*h, c + s*h] with c=(lo+hi)/2, h=(hi-lo)/2, s in {1.0, 2.0}. s=1.0 reproduces training. Reported in §C uses s in {1.0, 2.0} (the s=1.5 sweep was dropped from the final report)."
clipping_rules: "After scaling: friction >= 0.01; restitution clipped to [0,1] (cap at 0.15 retained); base mass scale >= 0.1; action delay >= 0 ms."
```

> **v2 training-condition note (authoritative for the final numbers).** The v2
> checkpoints (`logs/{baseline,cts,rma}/2026-05-10_...`) were trained under
> git commit `dd64881` of `shared_env_cfg.py`, which had: **flat ground**
> (`GroundPlaneCfg`, *not* the cobblestone generator that the current working
> copy shows), `push_robot` **enabled** (`push_by_setting_velocity`, interval
> 10–15 s, lin x/y ±0.5 m/s, lin z ±0.3, ang r/p/y ±0.3 rad/s), and **no
> force/torque impulse events** (the `impulse_reset` / `impulse_interval`
> rows above were added on 2026-05-12, *after* v2 was trained, so they are
> NOT part of v2's learned distribution). The §C numbers are evaluated under
> this exact distribution (flat + push, no impulse) for the training-faithful
> Isaac condition, and with push disabled for the Isaac↔MuJoCo sim2sim
> comparison (MuJoCo has no push mechanism — see methodology caveat in the
> headline).

### B.6 Episode and success criterion
```yaml
episode_length_steps: 1000               # 20.0 s at 50 Hz control (decimation 4 over 200 Hz)
episode_length_s: 20.0                    # v2 training & all final eval use 20 s
survival_window_s: 10.0                   # "survived" = stayed upright >= 10 s (>= 500 steps)
success_criterion: "episode survives >= 10 s without a fall. The MuJoCo harness classifies survived vs fail on the 10 s window; Isaac additionally splits survived episodes by velocity-RMSE < 0.3 m/s into success vs partial."
outcome_classes:
  - success: "survived >= 10 s AND vel_rmse < 0.3 m/s (Isaac)"
  - partial: "survived >= 10 s AND vel_rmse >= 0.3 m/s (Isaac)"
  - fail:    "early termination (fall) before 10 s"
termination_conditions:
  - "time_out: episode reaches 20.0 s"
  - "bad orientation: base tilt angle > 1.2 rad (~68.8 deg)"
note: >
  Earlier drafts of this document reported 10 s / 500-step episodes. The final
  v2 experiments use 20 s / 1000-step episodes (matches the v2 training
  episode_length_s = 20.0). Survival is scored on a 10 s window so that
  'survived' means the same thing across the 20 s eval and any 10 s reference.
```

### B.7 Architectures (Baseline / RMA / CTS)
The **transfer mechanism is the only experimental variable.** All three methods
share identical scene, DR ranges (§B.5), reward terms (§B.4), terminations
(§B.6), physics (200 Hz), control rate (50 Hz), actor & critic widths
`[512, 256, 128]` ELU MLP, and the PPO hyperparameters listed at the end of
this subsection. At deployment, **no method uses privileged simulator state**:
Baseline sees `o_t`, RMA and CTS see `[o_t, ẑ]` with `ẑ` produced from a 50-step
proprioceptive history.

**Baseline — PPO on proprioception only.**
- Actor input: `o_t ∈ R^37`.
- Critic input: `o_t ∈ R^37` (symmetric — no privileged state).
- Actor MLP `[512, 256, 128]` → `a_t ∈ R^12` (joint-position offsets).
- Training: standard PPO with domain randomisation; no encoder, no
  reconstruction loss, no teacher/student split.

**RMA — sequential two-stage Teacher-Student (Kumar et al. 2021, arXiv:2107.04034).**
- **Phase 1 — privileged teacher (PPO).**
  - EnvFactorEncoder μ : `x_t (26-D) → z (8-D)` — small MLP.
  - Actor input: `[o_t (37), z (8)] = 45-D`.
  - Critic input: `[o_t (37), x_t (26)] = 63-D` (asymmetric).
  - Actor MLP `[512, 256, 128]` → `a_t`.
- **Phase 2 — supervised adaptation module (history → ẑ).**
  - AdaptationModule φ : `history (50 × (o_t (37) + a_{t-1} (12))) → ẑ (8)`.
    Embed (linear 49→32) → Conv1d(32→32, k=8, s=4) → ELU → Conv1d(32→32, k=5,
    s=1) → ELU → Conv1d(32→32, k=5, s=1) → ELU → flatten → Linear → ẑ.
  - Loss: MSE(ẑ, stop-grad z) for supervised regression.
  - Phase-1 actor weights are **frozen** in Phase 2 — only φ is trained.
- **Deployment:** actor reads `[o_t, ẑ]` with `ẑ` from φ(history); the
  privileged x_t is never used at test time.

**CTS — single-stage concurrent Teacher-Student (Wang et al. 2024, arXiv:2405.10830).**
- Teacher encoder `E^t : x_t (26-D) → z (8-D)` — MLP `[256, 128]`.
- Student encoder `E^s : history (50 × 37) → ẑ (8-D)` —
  Conv1d(37→32, k=8, s=4) → ELU → Conv1d(32→64, k=5, s=2) → ELU →
  Conv1d(64→128, k=3, s=1) → ELU → flatten → Linear → ẑ.
- **Shared actor** MLP `[512, 256, 128]` reads `[o_t (37), z* (8)]` where
  `z*` is `z` for teacher envs or `ẑ` for student envs.
- Critic input: `[o_t (37), E^t(x_t) (8)]` (asymmetric, like RMA Phase 1).
- **Concurrent training in one PPO loop.** Environments are split 75 % / 25 %:
  3072 of 4096 envs are *teachers* (actor uses `z`); 1024 are *students*
  (actor uses `ẑ`). Routing is by a flag in the obs vector.
- **Reconstruction loss** `L_rec = λ_rec · MSE(E^s(history), stop-grad E^t(x_t))`
  applied after every PPO update (only on student envs), with a separate
  Adam optimiser on `E^s`. Hyperparameters: `λ_rec = 5.0`, warm-up = 1000
  iters (loss disabled before that), `E^s` learning rate = 5e-4.
- **Deployment:** actor reads `[o_t, ẑ]` exactly as in training — the
  reconstruction loss is what makes the latents interchangeable.

**Shared PPO hyperparameters** (RSL-RL `OnPolicyRunner`, identical to all three):
- envs = **4096**, `num_steps_per_env = 24`, `max_iterations = 5000`
  (Go2 v2; one-leg uses 5 000 as well), `save_interval = 200`, no empirical
  normalisation.
- Actor/critic hidden dims `[512, 256, 128]`, ELU, `init_noise_std = 1.0`.
- `value_loss_coef = 1.0`, clipped value loss, `clip_param = 0.2`,
  `num_learning_epochs = 5`, `num_mini_batches = 4`, `learning_rate = 1e-3`
  with adaptive KL schedule (`desired_kl = 0.01`), `gamma = 0.99`,
  `lam = 0.95`, `max_grad_norm = 1.0`.
- `entropy_coef = 0.01` (Baseline) / `0.005` (RMA, CTS).
- Action scale = 0.25 on joint-position targets;
  `q_target = q_default + 0.25 · a_t`.

**Two RMA implementations exist in the repo — use the canonical one above.**
There is a leftover *asymmetric actor-critic* variant in
`RMAEnvCfg` / `RMAPPORunnerCfg` whose actor is identical to Baseline and
whose critic alone sees privileged state (no encoder, no two-stage Phase 2).
The report-version RMA is the **canonical Kumar et al. (2021) two-stage**
variant described above; the v2 logs (`logs/rma/2026-05-10_..._rma_go2_v2_l8_l8`
+ `logs/rma/phase2/...`) correspond to this canonical version.

---

## C. Go2 Experiments — final numbers

### C.1 Experiment matrix (final, what is reported)
All Go2 final numbers: **FULL privilege (26-D x_t), latent Z = 8, 30 episodes
per cell, 20 s episodes (1000 steps), v2 checkpoints
(`logs/{baseline,cts,rma}/2026-05-10_...`), RMA Phase-2 adaptation module
`logs/rma/phase2/2026-05-11_12-11-23`**. All three methods share identical
scene / reward / DR / physics; only the transfer mechanism differs.

```yaml
conditions:                           # all: 20 s, 30 ep, flat ground, v2 ckpts
  - name: "Isaac training-faithful (flat + push, no impulse)"
    role: "CANONICAL — matches exactly what v2 was trained on"
    file: "results/isaac_v2_trainfaithful_20s.csv"
    method: "Baseline, RMA, CTS"; dr_scales: [1.0, 2.0]; status: "done"
  - name: "Isaac no-push (flat, no disturbance)"
    role: "sim2sim-comparable companion (MuJoCo has no push)"
    file: "results/isaac_v2_matched_20s.csv"
    method: "Baseline, RMA, CTS"; dr_scales: [1.0, 2.0]; status: "done"
  - name: "MuJoCo sim2sim (flat, no push — only mode the harness supports)"
    file: "results/sim2sim_report_v2_matched.{txt,json}"
    method: "Baseline, RMA, CTS"; dr_scales: [1.0, 2.0]; status: "done"
  - name: "Phase-1 single-leg fair-config"
    status: "done (Isaac-only, FULL/Z=8)"; method: "Baseline, RMA, CTS"; n_episodes: 100
pending_rerun:
  - "CTS privileged-subset ablation (FULL/INT/EXT) on Go2 — the §C.4/C.5 numbers
     were produced by the OLD pipeline (10 s, pre-bugfix EpisodeDR) and are NOT
     yet re-run under the corrected 20 s harness. Treated as PROVISIONAL."
ablations_not_reported:
  - "Latent-dimension sweep on Go2 (only Z=8 reported)."
  - "Privileged-subset ablation for RMA on Go2 (only CTS is ablated)."
```

> **Critical bug fixed before these numbers (document for the report's
> reproducibility section).** The MuJoCo multi-condition harness
> (`eval_metrics.py`) shared one `MjModel` across all conditions, and
> `EpisodeDR.__init__` snapshotted "nominal" inertia/COM/friction from the
> *current* (already DR-perturbed) model. Running conditions in series
> (baseline→rma→cts) compounded the COM offset additively and inertia
> multiplicatively, so CTS (evaluated last) ran on a catastrophically distorted
> robot and reported a spurious **3 %** survival. After restoring the model to
> true nominal before each condition's `EpisodeDR`, CTS recovers to **100 %**
> survival at DR×1 — consistent with standalone `sim2sim_go2.py` (100 %, reward
> ≈ 2100). All §C.2/C.3 MuJoCo numbers are post-fix.

### C.2 Final per-row evaluation table (Go2) — FULL methods only

All cells: Z = 8, 30 episodes, 20 s (1000 steps), v2 checkpoints. Reward =
mean ± std cumulative return. `trk` = mean velocity-tracking error [m/s]
(Isaac `mean_track_err`; MuJoCo `vel_rmse`). `fwd` = mean forward
displacement [m] (key behavioural sanity check — a policy with fwd ≈ 0 is
standing still regardless of reward).

**(a) Isaac — TRAINING-FAITHFUL (flat + push, no impulse) — the canonical result**
`results/isaac_v2_trainfaithful_20s.csv`

| method   | s | reward          | surv % | trk   | fwd   |
|---       |--:|---              |---:    |---:   |---:   |
| Baseline | 1 | 2571.1 ± 162.0  | 100.0  | 0.144 | 1.46  |
| Baseline | 2 | 1800.0 ± 1207.7 | 73.3   | 0.239 | 0.42  |
| **CTS**  | 1 | **2630.3 ± 75.4** | **100.0** | **0.130** | **2.57** |
| **CTS**  | 2 | **2311.2 ± 652.8** | **86.7**  | **0.157** | **1.85** |
| RMA      | 1 | 1773.5 ± 617.6  | 80.0   | 0.201 | 0.04  |
| RMA      | 2 | 713.6 ± 944.1   | 23.3   | 0.259 | 0.08  |

**(b) Isaac — NO-PUSH (flat, no disturbance) — sim2sim-comparable companion**
`results/isaac_v2_matched_20s.csv`

| method   | s | reward          | surv % | trk   | fwd   |
|---       |--:|---              |---:    |---:   |---:   |
| Baseline | 1 | 2604.1 ± 154.9  | 100.0  | 0.131 | 1.71  |
| Baseline | 2 | 2276.4 ± 646.4  | 93.3   | 0.146 | 1.38  |
| CTS      | 1 | 2685.6 ± 64.5   | 100.0  | 0.110 | 2.68  |
| CTS      | 2 | 2201.8 ± 851.6  | 86.7   | 0.156 | 1.16  |
| RMA      | 1 | 2201.9 ± 172.3  | 100.0  | 0.186 | 0.05  |
| RMA      | 2 | 935.0 ± 1067.3  | 60.0   | 0.258 | 0.04  |

**(c) MuJoCo — NO-PUSH (flat) — sim2sim target** (harness has no push)
`results/sim2sim_report_v2_matched.json`

| method   | s | reward          | surv % | trk(RMSE) | fwd    |
|---       |--:|---              |---:    |---:       |---:    |
| Baseline | 1 | 2349.6 ± 239.9  | 100.0  | 0.222     | 6.01   |
| Baseline | 2 | 1742.9 ± 800.5  | 90.0   | 0.351     | 3.20   |
| CTS      | 1 | 1930.3 ± 363.5  | 100.0  | 0.177     | 8.21   |
| CTS      | 2 | 1108.9 ± 989.8  | 73.3   | 0.327     | 4.19   |
| RMA      | 1 | 2136.7 ± 59.7   | 100.0  | **0.500** | **0.01** |
| RMA      | 2 | 1826.7 ± 584.6  | 100.0  | **0.500** | **−0.00**|

**Reading.**
1. **CTS wins the canonical (training-faithful) condition** outright: best
   reward, survival and tracking at DR×1 *and* DR×2, and the smallest OOD drop
   (survival 100→86.7 %). Baseline is close at DR×1 but collapses far more at
   DR×2 (100→73.3 %). RMA is third everywhere (80 % then 23.3 %).
2. **CTS is the most push-robust.** Adding the training-time push barely moves
   CTS (survival 100→100 % at DR×1, 86.7→86.7 % at DR×2 vs the no-push column).
   Baseline loses 93.3→73.3 % at DR×2; RMA loses 100→80 % at DR×1 and
   60→23.3 % at DR×2 — RMA is fragile to the very disturbance it trained on.
3. **RMA's adaptation module does not produce locomotion.** `fwd ≈ 0 m` in
   *both* simulators (Isaac 0.04–0.05 m, MuJoCo ≈ 0 m over a 20 s episode) and
   MuJoCo `vel_rmse` pinned at 0.500 (= the 0.5 m/s command magnitude → zero
   forward velocity). RMA's high MuJoCo reward / "100 % survival" is a
   stationary-policy artefact (alive-bonus + posture shaping), **not** task
   competence.
4. **CTS is the best velocity tracker in both simulators** (lowest `trk` in
   every Isaac and MuJoCo block where the policy actually moves).

Figure: `results/figures/fig_go2_v2_corrected.{png,pdf}` (generated by
`scripts/plot_v2_corrected.py` directly from the three authoritative files;
1×3 panel = survival / reward / tracking-error, 3 conditions × 3 methods ×
DR×1/×2). This is the **only Go2 figure that reflects the corrected v2
numbers** — every `fig_go2_*` figure from `plot_results_go2.py` is stale
(see §F banner).

### C.2(d) — Four-method comparison (v2fix, 10 s, no-push) — **CITE THIS for RMA Teacher / Student**

> **Added 2026-05-19.** Source CSVs: `results/ood_go2.csv` (Isaac) +
> `results/sim2sim_go2.csv` (MuJoCo). These are 10 s (500-step) episodes,
> no velocity push, flat ground. Isaac n = 100 (RMA Teacher / RMA Student),
> n = 30 (Baseline / CTS FULL). MuJoCo n = 30 all methods. The four methods
> share the same v2 checkpoints except RMA Student which uses the v2fix
> checkpoint (Phase-1 fixed capacity + orthogonal encoder init, Phase-2
> trained 600 iterations, adaptation module MSE converged to ~0.94 — i.e.
> the adaptation module outputs z ≈ z_mean and produces near-zero guidance).
>
> **Why 10 s, not 20 s?** The `eval_ood_go2.py` harness uses `episode_length_s=10`
> so rewards are roughly half the §C.2(a-c) values — **do not mix the two
> tables' reward numbers directly**. Survival %, tracking error, and
> forward displacement are scale-independent and can be compared.

**(a) Isaac OOD — 10 s, no push**

`results/ood_go2.csv` — columns: `success_rate / partial_rate / fall_rate /
survival_rate = success + partial` (Isaac splits by vel_rmse < 0.3 m/s;
MuJoCo: survival = not-fell, success/partial by same 0.3 threshold).

| method | DR×s | return (mean ± std) | success % | partial % | fall % | survival % | trk [m/s] | fwd [m] |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Baseline       | 1 | 1251.6 ± 88.9   | **100** | 0  | 0      | **100** | 0.139 | 0.89 |
| Baseline       | 2 | 1119.6 ± 213.6  | 96.7    | 0  | 3.3    | 96.7    | 0.162 | 0.47 |
| **CTS (FULL)** | 1 | **1265.9 ± 61.7** | **100** | 0  | 0      | **100** | **0.124** | 1.35 |
| **CTS (FULL)** | 2 | **1117.3 ± 263.7** | **100** | 0  | **0**  | **100** | 0.169 | 0.65 |
| RMA Teacher    | 1 | 1252.1 ± 64.8   | **100** | 0  | 0      | **100** | 0.143 | **1.25** |
| RMA Teacher    | 2 | 1000.3 ± 448.3  | 82      | 2  | 16     | 84      | 0.232 | 0.56 |
| RMA Student    | 1 | 764.2 ± 360.1   | 64      | 19 | **17** | 83      | 0.244 | 0.13 |
| RMA Student    | 2 | 296.9 ± 662.4   | 38      | 7  | **55** | 45      | 0.328 | 0.17 |

**(b) MuJoCo Sim2Sim — 10 s, no push**

`results/sim2sim_go2.csv`. Reward scale differs from Isaac (different alive-bonus
accumulation in MuJoCo).

| method | DR×s | return (mean ± std) | success % | partial % | fall % | survival % | trk [m/s] | fwd [m] |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Baseline         | 1 | 1160.9 ± 103.5  | 63.3    | 36.7  | 0      | **100** | 0.251 | 2.76 |
| Baseline         | 2 | 889.4 ± 361.0   | 40      | 43.3  | 16.7   | 83.3    | 0.340 | 1.87 |
| **CTS (FULL)**   | 1 | 980.7 ± 116.8   | **96.7** | 3.3  | **0**  | **100** | **0.178** | **4.10** |
| **CTS (FULL)**   | 2 | 647.7 ± 395.8   | **53.3** | 23.3 | 23.3   | 76.7    | **0.302** | **2.78** |
| RMA Teacher†     | 1 | 1036.8 ± 23.5   | 0       | **100** | **0** | **100** | 0.499 | 0.02 |
| RMA Teacher†     | 2 | 927.3 ± 316.3   | 3.3     | 96.7  | **0**  | **100** | 0.491 | 0.10 |
| RMA Student      | 1 | −1252.2 ± 2033  | 0       | 33.3  | **66.7** | 33.3  | 0.446 | 0.77 |
| RMA Student      | 2 | −444.1 ± 678.2  | 0       | 36.7  | **63.3** | 36.7  | 0.486 | 0.63 |

† **RMA Teacher MuJoCo anomaly.** 100 % survival, 0 falls, perfect ang_track
(0.998) but `fwd ≈ 0.02 m` over 10 s — the robot spins in place. The
privileged-obs builder `get_priv_obs_sim2sim` likely feeds a mis-scaled
torque or contact signal so the encoder outputs a z that rewards spinning
over walking. This is a sim2sim integration bug, not a policy failure; the
Isaac OOD result (100 % success, fwd = 1.25 m) is the authoritative
RMA-Teacher number.

**Reading (cite §C.2(d) when discussing the four-method comparison).**
1. **CTS wins overall.** Only method with 100 % survival and 0 falls at DR×1
   *and* DR×2 in Isaac; best linear-velocity tracker in both simulators; best
   sim2sim success rate (96.7 % at DR×1).
2. **RMA Teacher ≈ Baseline ≈ CTS in Isaac at DR×1** (all 100 % survival,
   rewards within 1.5 %). At DR×2 Teacher degrades noticeably (84 % survival,
   16 % fall) — the Phase-1 actor without exact x_t at test time in DR×2 is
   less robust than CTS's concurrent training.
3. **RMA Student fails systematically — root cause: Phase-2 adaptation module
   cannot converge.**
   The adaptation module φ(history) → ẑ was trained for 1 471 Phase-2
   iterations (source: `logs/rma/phase2/2026-05-19_14-29-11/phase2_loss.csv`).
   MSE loss:
   - **Initial (iter 10):** 0.9817
   - **Final plateau (last-100 mean):** 0.9364
   - **Improvement over 1 471 iters:** only 0.0453 — essentially flat after
     the first ~10 iterations (see `fig_go2_rma_phase2_loss`).
   The teacher provides oracle z (MSE = 0 by definition); the student is
   stuck at ≈ 0.936, meaning ẑ ≈ z_mean — the CNN collapses to predicting
   the latent mean and ignores the history signal entirely. Root cause:
   `batch_size / num_envs ≈ 19` effective history steps vs `history_len = 50`
   → the CNN never sees a full-length history window during training.
   Consequence: In Isaac: 17 % fall at DR×1, 55 % fall at DR×2. In MuJoCo:
   66.7 % fall at DR×1 with mean reward −1252. The two-stage dependency
   (Phase-2 convergence requires Phase-1 to produce stable z trajectories,
   and the batch-size ratio prevents Phase-2 from learning) is the root
   failure mode.
4. **MuJoCo partial vs success split reveals sim2sim gap.** Baseline has only
   63.3 % full-success in MuJoCo despite 100 % survival (vel_rmse often ≥ 0.3).
   CTS achieves 96.7 % full-success, confirming it is the better velocity
   tracker.

Figures (from `scripts/plot_results_go2.py`, regenerated 2026-05-19 with
RMA Teacher added):
- `results/figures/fig_go2_comparison.{pdf,png}` — 4-view reward matrix
- `results/figures/fig_go2_comparison_outcome.{pdf,png}` — stacked success/partial/fall
- `results/figures/fig_go2_comparison_survival.{pdf,png}` — survival spec (≥ 80 %)
- `results/figures/fig_go2_comparison_rmse.{pdf,png}` — vel-tracking RMSE
- `results/figures/fig_go2_headline.{pdf,png}` — sim2sim retention G(π)
- `results/figures/fig_go2_ood_profile.{pdf,png}` — Isaac OOD gap
- `results/figures/fig_go2_summary.{pdf,png}` — spec-sheet dashboard
- `results/figures/fig_go2_rma_phase2_loss.{pdf,png}` — **RMA Phase-2 MSE
  loss curve (teacher–student gap)**; shows flat plateau at ≈ 0.936 vs
  teacher baseline of 0, confirming the adaptation module does not converge
  (source: `logs/rma/phase2/2026-05-19_14-29-11/phase2_loss.csv`)

### C.2(e) — RMA Phase-2 MSE Loss: Teacher–Student Gap

**Source:** `logs/rma/phase2/2026-05-19_14-29-11/phase2_loss.csv` (1 471 iterations).
**Figure:** `results/figures/fig_go2_rma_phase2_loss.{pdf,png}`.

The Phase-2 supervised step trains φ : o_{t-H:t} → ẑ (1D-CNN, history_len = 50)
to match the oracle encoder output z = μ_φ(x_t) using frozen Phase-1 weights.

| Quantity | Value |
|---|---|
| Training iterations (Phase 2) | 1 471 |
| MSE at iter 10 (first logged) | 0.9817 |
| MSE plateau — last-50 mean | **0.9364** |
| MSE plateau — last-100 mean | 0.9364 |
| MSE improvement over full run | 0.0453 (≈ 4.6 %) |
| Teacher MSE (oracle baseline) | **0.0000** |
| Teacher–student gap | **≈ 0.936** |

**Interpretation.** The adaptation module converges in the first ≈ 10 iterations
(from 0.98 → 0.93) and then flat-lines: the loss curve shows zero meaningful
descent across 1 461 remaining iterations. The CNN collapses to predicting ẑ ≈
z_mean — ignoring the observation history entirely. This is quantified by the
teacher–student gap of ≈ 0.936 (teacher = 0 by construction; perfect student
would also reach 0).

**Root cause: batch_size / num_envs mismatch.**
During Phase-2 training, each rollout provides only `batch_size / num_envs ≈ 19`
steps per environment, while the history window is `history_len = 50`. The CNN
therefore never receives a full 50-step history buffer during training — the
first 31 entries are always zero-padded — causing the network to learn that the
mean prediction is optimal (minimises MSE on the padding-dominated distribution).

**Impact on performance.**
The ẑ ≈ z_mean output effectively disables the environmental context signal, so
the Phase-1 actor runs with a constant latent rather than a task-conditioned one.
This explains the large RMA Teacher → RMA Student performance drop:

| Metric (Isaac, DR×1) | RMA Teacher | RMA Student | Gap |
|---|---:|---:|---:|
| Mean reward | 1252.1 | 764.2 | −488 (−39 %) |
| Success rate | 100 % | 64 % | −36 pp |
| Fall rate | 0 % | 17 % | +17 pp |
| Vel-tracking RMSE | 0.143 | 0.244 | +70 % |

Fix (not applied in this study): increase `num_envs` or `batch_size` so that
`batch_size / num_envs ≥ history_len`, or pre-fill the history buffer before
Phase-2 gradient steps begin.

### C.3 Sim2Sim transfer ratios (Go2, spec sheet)
- **G(π)** = R_MuJoCo,1× / R_Isaac,1× × 100 % (target ≥ 60 %)
- **OOD gap** = R_Isaac,2× / R_Isaac,1× × 100 % (target ≥ 70 %)
- **Combined gap** = R_MuJoCo,2× / R_Isaac,1× × 100 % (target ≥ 40 %)

**(i) Sim2sim ratios — Isaac NO-PUSH vs MuJoCo NO-PUSH (20 s baseline runs)**
(apples-to-apples: both 20 s, 30 ep, flat, no push; reward scales now comparable).

| method   | R_iso,1× | R_iso,2× | R_muj,1× | R_muj,2× | G(π)            | OOD gap         | Combined        |
|---       |---:      |---:      |---:      |---:      |---:             |---:             |---:             |
| Baseline | 2604.1   | 2276.4   | 2349.6   | 1742.9   | **90.2 %** ✓PASS | **87.4 %** ✓PASS | **66.9 %** ✓PASS |
| CTS      | 2685.6   | 2201.8   | 1930.3   | 1108.9   | **71.9 %** ✓PASS | **82.0 %** ✓PASS | **41.3 %** ✓PASS |
| RMA      | 2201.9   | 935.0    | 2136.7   | 1826.7   | 97.0 %†          | **42.5 %** ✗FAIL | 83.0 %†          |

† RMA's G(π)/Combined look high **only because RMA stands still in MuJoCo**
(fwd ≈ 0 m, vel-RMSE = 0.500): the alive-bonus dominates a stationary
episode, so reward-retention overstates behaviour. RMA still **FAILs the
OOD-gap spec inside Isaac itself** (42.5 %, the only FAIL in the matrix) —
its Phase-2 module collapses out-of-distribution even before the simulator
change.

**(ia) Sim2sim ratios — four-method comparison (10 s, no-push, v2fix)**
Source: `results/ood_go2.csv` (Isaac) + `results/sim2sim_go2.csv` (MuJoCo).
Note: reward scale is ~½ of the 20 s runs; ratios are scale-invariant.

| method       | R_iso,1× | R_iso,2× | R_muj,1× | R_muj,2× | G(π)             | OOD gap         | Combined        |
|---           |---:      |---:      |---:      |---:      |---:              |---:             |---:             |
| Baseline     | 1251.6   | 1119.6   | 1160.9   | 889.4    | **92.8 %** ✓PASS | **89.5 %** ✓PASS | **71.1 %** ✓PASS |
| CTS (FULL)   | 1265.9   | 1117.3   | 980.7    | 647.7    | **77.5 %** ✓PASS | **88.3 %** ✓PASS | **51.2 %** ✓PASS |
| RMA Teacher  | 1252.1   | 1000.3   | 1036.8†  | 927.3†   | 82.8 %†          | **79.9 %** ✓PASS | 74.1 %†        |
| RMA Student  | 764.2    | 296.9    | −1252.2  | −444.1   | **−163.8 %** ✗FAIL | **38.9 %** ✗FAIL | **−58.1 %** ✗FAIL |

† RMA Teacher MuJoCo G(π)/Combined are inflated by the **spinning-in-place
anomaly** (fwd ≈ 0.02 m, vel_rmse = 0.499 — see §C.2(d) note). The
MuJoCo reward comes from alive-bonus + posture only, not locomotion. Use
Isaac OOD (§C.2(d)(a)) as the authoritative RMA Teacher result.

**(ii) OOD retention under the canonical training-faithful condition**
(Isaac flat + push, 20 s, R_2× / R_1×; Baseline/CTS/old-RMA checkpoints):

| method   | R_iso,1× (push) | R_iso,2× (push) | OOD retention   |
|---       |---:             |---:             |---:             |
| **CTS**  | 2630.3          | 2311.2          | **87.9 %** ✓ best |
| Baseline | 2571.1          | 1800.0          | 70.0 % (spec edge) |
| RMA      | 1773.5          | 713.6           | **40.2 %** ✗FAIL |

**(iia) OOD retention — four-method, 10 s no-push (v2fix checkpoints)**
(Isaac, `results/ood_go2.csv`, R_2× / R_1×):

| method       | R_iso,1× | R_iso,2× | OOD retention    |
|---           |---:      |---:      |---:              |
| Baseline     | 1251.6   | 1119.6   | **89.5 %** ✓PASS |
| **CTS**      | 1265.9   | 1117.3   | **88.3 %** ✓PASS |
| RMA Teacher  | 1252.1   | 1000.3   | **79.9 %** ✓PASS |
| RMA Student  | 764.2    | 296.9    | **38.9 %** ✗FAIL |

**Reading.** Under the apples-to-apples no-push sim2sim comparison, Baseline
has the highest raw reward-retention, but CTS is the only method whose *useful
behaviour* (forward locomotion + tracking) actually transfers — Baseline's
fwd halves (1.71→3.20 m note the MuJoCo gait differs) while its tracking
error nearly doubles, and RMA does not locomote at all in MuJoCo. Under the
*scientifically correct* training-faithful condition (ii), **CTS clearly wins
OOD retention (87.9 %)**, Baseline sits on the 70 % spec edge, and **RMA fails
the OOD spec (40.2 %)** — consistent with the headline ranking.

### C.4 Ablations on Go2 — what is reported and what is not

> ✅ **CORRECTED & RE-RUN (2026-05-18).** The §C.5 CTS privileged-subset
> numbers have been regenerated under the corrected 20 s harness (post
> `EpisodeDR` bug fix): Isaac = training-faithful (flat + push) v2 INT/EXT
> checkpoints, MuJoCo = clean per-checkpoint `sim2sim_go2.py`. Source data:
> `results/isaac_v2_trainfaithful_20s.csv` +
> `results/isaac_cts_intext_trainfaithful_20s.csv` (Isaac) and
> `results/mujoco_cts_priv_20s.csv` (MuJoCo), consolidated into
> `results/{ood_go2_v2,sim2sim_go2_v2}.csv` by `scripts/_consolidate_v2.py`.
> The old "INT beats FULL in MuJoCo" claim is now correctly explained as
> INT's non-locomotion (0 % success). §C.5 is safe to cite.

- **CTS privileged-knowledge ablation (FULL / INT / EXT) on Go2:** **reported,
  corrected (§C.5).** Source: `results/go2_cts_priv_ablation_table.md` (auto-
  generated by `scripts/plot_results_go2.py` from
  `results/ood_go2.csv` + `results/sim2sim_go2.csv`). Figure:
  `results/figures/fig_go2_cts_priv_ablation.{png,pdf}`. Privileged subsets:
  `FULL` = x_int(16) ⊕ x_ext(10) = 26-D (baseline of the ablation),
  `INT` = body-parameter subset (16-D: μ, e_rest, Δm_base, ΔKp, ΔKd, Δcom,
  ΔI_base, d_act), `EXT` = interaction-signal subset (10-D: f_contact, c_bin,
  τ_avg). All numbers are in §C.6 below.
- **Privileged-knowledge ablation for RMA on Go2:** not reported (compute
  scoped to the CTS variant only — see §D.3 for justification).
- **Latent-dimension sweep on Go2:** not reported (only Z = 8 was trained to
  completion on Go2; the single-leg latent sweep on disk is not used).

### C.5 CTS privileged-subset ablation — final numbers (CORRECTED)
*(verbatim from the regenerated `results/go2_cts_priv_ablation_table.md`;
CTS only, Z = 8, 30 episodes/cell, **20 s** episodes, v2 checkpoints.
Isaac = training-faithful flat + push; MuJoCo = no-push, post-bugfix.)*

**Per-cell results:**

| sim    | DR×s | priv | reward (mean ± std) | success % | vel-RMSE [m/s] | Δreward vs FULL |
|---     |---   |---   |---                  |---:       |---:            |---:             |
| Isaac  | 1    | FULL | 2630.3 ± 75.4       | 100       | 0.130          | —               |
| Isaac  | 1    | INT  | 2601.9 ± 81.1       | 100       | 0.156          | −1.1 %          |
| Isaac  | 1    | EXT  | 2530.2 ± 138.6      | 100       | 0.158          | −3.8 %          |
| Isaac  | 2    | FULL | 2311.2 ± 652.8      | 87        | 0.157          | —               |
| Isaac  | 2    | INT  | 2081.5 ± 790.9      | 87        | 0.191          | −9.9 %          |
| Isaac  | 2    | EXT  | 1856.4 ± 1019.7     | 73        | 0.206          | −19.7 %         |
| MuJoCo | 1    | FULL | 1930.3 ± 363.5      | **93**    | 0.177          | —               |
| MuJoCo | 1    | INT  | 2076.9 ± 11.9       | **0**     | **0.497**      | +7.6 %          |
| MuJoCo | 1    | EXT  | 1791.3 ± 262.2      | 67        | 0.253          | −7.2 %          |
| MuJoCo | 2    | FULL | 1108.9 ± 989.8      | 40        | 0.327          | —               |
| MuJoCo | 2    | INT  | 1586.7 ± 872.7      | 17        | 0.428          | +43.1 %         |
| MuJoCo | 2    | EXT  | 1111.8 ± 835.7      | 23        | 0.381          | +0.3 %          |

**Spec-sheet transfer ratios per privileged subset:**

| priv | R_iso,1× | R_iso,2× | R_muj,1× | R_muj,2× | G(π)            | OOD gap         | Combined        |
|---   |---:      |---:      |---:      |---:      |---:             |---:             |---:             |
| FULL | 2630.3   | 2311.2   | 1930.3   | 1108.9   | 73.4 % ✓PASS    | **87.9 % ✓PASS** | 42.2 % ✓PASS    |
| INT  | 2601.9   | 2081.5   | 2076.9   | 1586.7   | 79.8 % †PASS    | 80.0 % ✓PASS    | 61.0 % †PASS    |
| EXT  | 2530.2   | 1856.4   | 1791.3   | 1111.8   | 70.8 % ✓PASS    | 73.4 % ✓PASS    | 43.9 % ✓PASS    |

† INT's G(π)/Combined are inflated by the **stationary-policy artefact** —
see (ii) below; reward-retention overstates behaviour when the policy is not
locomoting.

**Behaviour (gait-quality) metrics** *(regenerated; lower is better except
`gait_adh`/`contact_sym`. DR×2 = OOD on the DR axis.):*

| sim    | DR | priv | gait adh. | contact sym. | swing clear. err. | foot slip rate | action smooth. | base-z var. | stride var. | joint-torque var. |
|---     |---:|---   |---:       |---:          |---:               |---:            |---:            |---:         |---:         |---:               |
| Isaac  | 1  | FULL | 0.338     | 0.871        | 0.0035            | 0.160          | 1.73           | 0.0005      | 0.0006      | 5.07              |
| Isaac  | 1  | INT  | 0.345     | 0.900        | 0.0039            | 0.292          | 1.47           | 0.0005      | 0.0018      | 5.11              |
| Isaac  | 1  | EXT  | 0.349     | 0.918        | 0.0035            | 0.314          | 1.48           | 0.0005      | 0.0014      | 4.57              |
| MuJoCo | 1  | FULL | 0.203     | 0.068        | 0.0013            | 1.63           | 1.36           | 0.0004      | 0.0061      | 18.19             |
| MuJoCo | 1  | INT  | 0.136     | **0.0000**   | **0.0000**        | 1.10           | **0.008**      | 0.0000      | **0.0001**  | **0.191**         |
| MuJoCo | 1  | EXT  | 0.218     | 0.049        | 0.0010            | 1.49           | 1.17           | 0.0003      | 0.0051      | 15.76             |
| Isaac  | 2  | FULL | 0.326     | 0.822        | 0.0049            | 0.240          | 2.12           | 0.0011      | 0.0038      | 6.24              |
| Isaac  | 2  | INT  | 0.331     | 0.841        | 0.0089            | 0.304          | 1.99           | 0.0014      | 0.0023      | 7.90              |
| Isaac  | 2  | EXT  | 0.324     | 0.810        | 0.013             | 0.312          | 3.01           | 0.0018      | 0.0015      | 10.93             |
| MuJoCo | 2  | FULL | 0.208     | 0.091        | 0.0068            | 1.59           | 2.21           | 0.0010      | 0.0041      | 21.85             |
| MuJoCo | 2  | INT  | 0.171     | 0.010        | 0.0034            | 1.23           | **0.400**      | 0.0001      | 0.0016      | **5.53**          |
| MuJoCo | 2  | EXT  | 0.210     | 0.057        | 0.0027            | 1.15           | 1.06           | 0.0004      | 0.0043      | 16.66             |

**Reading (corrected — the bug fix sharpens, not weakens, the FULL story).**
(i) **In Isaac (training-faithful)** the ordering is clean **FULL ≥ INT ≥
EXT**: at DR×1 all three survive 100 % and INT/EXT cost only −1.1 %/−3.8 %
reward; at DR×2 FULL retains 87 % survival and the best tracking (0.157),
INT matches survival but tracks worse (0.191), and **EXT degrades most**
(73 % survival, −19.7 % reward, 0.206). (ii) **In MuJoCo, INT does NOT beat
FULL — it stops walking.** INT's higher raw reward (+7.6 % / +43.1 %) is a
*stationary-policy artefact*: success % = **0** at DR×1 (vs FULL 93 %),
vel-RMSE = **0.497** (≈ the 0.5 m/s command → ~zero forward velocity), and
the behaviour metrics are unambiguous — INT MuJoCo DR×1 action-smoothness
**0.008**, joint-torque variance **0.191**, stride variance **0.0001**,
contact symmetry **0.000**: a robot standing essentially still (the same
failure mode as RMA's Phase-2 module, §C.2c). FULL is the **only** subset
that actually locomotes in MuJoCo (93 % success, jtorque-var 18.19, real
gait). EXT partially locomotes (67 % success). (iii) **Conclusion (now
stronger):** only the **FULL** privileged combination transfers as genuine
locomotion to MuJoCo; the interaction signals removed in INT are exactly
what prevents the policy from collapsing to a stationary "survive-by-not-
moving" solution. The earlier (buggy, 10 s) table's claim that "INT beats
FULL on reward in MuJoCo" was a measurement artefact — corrected, INT's
reward advantage is revealed as non-locomotion.

### C.6 Plots/tables planned and present
*(see §E for one-line image captions and §F for longer figure descriptions.)*
```yaml
artefacts:
  # final tables
  - {filename: "results/go2_results_table.md",                                     shows: "per-row reward+success table; spec-sheet transfer-ratio table",       source: "scripts/plot_results_go2.py",         status: "ready"}
  - {filename: "results/go2_cts_priv_ablation_table.md",                           shows: "CTS-only FULL/INT/EXT ablation: reward, success, vel-RMSE, Δ vs FULL; transfer ratios per priv subset", source: "scripts/plot_results_go2.py::write_cts_priv_ablation_table", status: "ready"}
  - {filename: "results/one_leg_results_table.md",                                 shows: "per-row Isaac reward+success on the single leg; OOD retention",      source: "scripts/plot_results_one_leg.py",     status: "ready"}
  # ────────────────────────────────────────────────────────────────────────
  # ✅ ALL Go2 figures + tables below were REGENERATED 2026-05-18 from the
  # CORRECTED post-bugfix data (20 s, v2 ckpts). Command:
  #   python scripts/_consolidate_v2.py        # builds the two v2 CSVs
  #   python scripts/plot_results_go2.py \
  #       --ood results/ood_go2_v2.csv --sim2sim results/sim2sim_go2_v2.csv
  # Provenance: ood_go2_v2.csv = isaac_v2_trainfaithful_20s.csv (Baseline·BASE,
  # RMA·FULL, CTS·FULL) + isaac_cts_intext_trainfaithful_20s.csv (CTS·INT/EXT),
  # i.e. CANONICAL training-faithful Isaac (flat+push). sim2sim_go2_v2.csv =
  # Baseline/RMA from sim2sim_report_v2_matched.json + CTS·FULL/INT/EXT from
  # mujoco_cts_priv_20s.csv (clean per-ckpt sim2sim_go2.py, no-push).
  # status:ready is now ACCURATE. fig_go2_latent_ablation is intentionally
  # skipped (only Z=8 trained). A standalone 3-condition figure also exists:
  # results/figures/fig_go2_v2_corrected.{png,pdf} (scripts/plot_v2_corrected.py).
  # ────────────────────────────────────────────────────────────────────────
  # Go2 figures
  - {filename: "results/figures/fig_go2_headline.{png,pdf}",                       shows: "headline single-figure (G(π) per method, DR×1 vs DR×2 with PASS/FAIL)", source: "plot_results_go2.py::fig_headline",  status: "ready"}
  - {filename: "results/figures/fig_go2_summary.{png,pdf}",                        shows: "2x2 spec-sheet panel (return / survival / RMSE / threshold matrix)", source: "plot_results_go2.py::fig_summary_dashboard", status: "ready"}
  - {filename: "results/figures/fig_go2_comparison.{png,pdf}",                     shows: "4-view reward comparison: Isaac OOD / MuJoCo OOD / Sim2Sim DR×1 / Sim2Sim DR×2", source: "plot_results_go2.py::fig_comparison_matrix", status: "ready"}
  - {filename: "results/figures/fig_go2_comparison_survival.{png,pdf}",            shows: "same 4 views, but Y axis is survival rate (%) with 80 % spec line",  source: "plot_results_go2.py::fig_comparison_survival", status: "ready"}
  - {filename: "results/figures/fig_go2_comparison_rmse.{png,pdf}",                shows: "same 4 views, but Y axis is velocity-tracking RMSE (m/s) with 0.3 m/s spec line", source: "plot_results_go2.py::fig_comparison_rmse", status: "ready"}
  - {filename: "results/figures/fig_go2_comparison_outcome.{png,pdf}",             shows: "same 4 views, stacked Success/Partial/Fail outcome breakdown",       source: "plot_results_go2.py::fig_comparison_outcome",  status: "ready"}
  - {filename: "results/figures/fig_go2_ood_profile.{png,pdf}",                    shows: "Isaac OOD retention bar + absolute Isaac reward DR×1 vs DR×2",       source: "plot_results_go2.py::fig_ood_profile",        status: "ready"}
  - {filename: "results/figures/fig_go2_sim2sim_transfer.{png,pdf}",               shows: "sim2sim retention G(pi,s) with 60 % spec line + Isaac->MuJoCo gap (Δ annotations)", source: "plot_results_go2.py::fig_sim2sim_transfer", status: "ready"}
  - {filename: "results/figures/fig_go2_gait_quality.{png,pdf}",                   shows: "8 gait-quality metrics (adherence, contact symmetry, swing clearance err, foot slip rate, smoothness, base-height var, stride var, torque var), Isaac vs MuJoCo", source: "plot_results_go2.py::fig_gait_quality", status: "ready"}
  - {filename: "results/figures/fig_go2_cts_priv_ablation.{png,pdf}",               shows: "CTS-only privileged-subset ablation (FULL/INT/EXT) across 4 views (Isaac DR×1, Isaac DR×2, MuJoCo DR×1, MuJoCo DR×2); top row = mean reward, bottom row = success rate", source: "plot_results_go2.py::fig_cts_priv_ablation", status: "ready"}
  - {filename: "results/figures/fig_go2_cts_priv_ablation_gait_dr1.{png,pdf}",      shows: "CTS-only priv-ablation behaviour metrics at DR×1: 8 gait-quality panels (gait adherence, contact symmetry, swing clearance err., foot slip rate, action smoothness, base-height var., stride var., joint-torque var.) for FULL/INT/EXT, Isaac (solid) vs MuJoCo (hatched)", source: "plot_results_go2.py::fig_cts_priv_ablation_gait", status: "ready"}
  - {filename: "results/figures/fig_go2_cts_priv_ablation_gait_dr2.{png,pdf}",      shows: "Same 8 behaviour metrics at DR×2 (OOD on the DR axis): the gap between FULL and INT widens — FULL spends more torque variance (22.47 vs INT 8.40) to keep tracking",   source: "plot_results_go2.py::fig_cts_priv_ablation_gait", status: "ready"}
  # PPO learning curves (from TensorBoard event files)
  - {filename: "results/figures/fig_go2_learning_curves.{png,pdf}",                 shows: "Baseline / RMA / CTS Train/mean_reward vs PPO iteration (0–25 000), all v2 runs at Z=8/FULL. Method colour grammar.",                                                  source: "scripts/plot_learning_curves.py::fig_mean_reward",                  status: "ready"}
  - {filename: "results/figures/fig_go2_learning_curves_len.{png,pdf}",             shows: "Same three methods, Train/mean_episode_length vs PPO iteration.",                                                                                                          source: "scripts/plot_learning_curves.py::fig_mean_length",                  status: "ready"}
  - {filename: "results/figures/fig_go2_cts_teacher_student.{png,pdf}",             shows: "CTS-FULL teacher vs student reward over training — left/right panels (shared y), plus gap (T−S) strip; post-warmup mean gap ≈ +15 (+0.8 % of teacher).",                source: "scripts/plot_learning_curves.py::fig_cts_teacher_student",          status: "ready"}
  - {filename: "results/figures/fig_go2_cts_teacher_student_overlay.{png,pdf}",     shows: "3×2 grid: top row = teacher+student overlaid per priv subset (FULL/INT/EXT), bottom row = gap track. Post-warmup gaps: FULL +0.8 %, INT +0.5 %, EXT +0.7 %.",            source: "scripts/plot_learning_curves.py::fig_cts_teacher_student_overlay", status: "ready"}
  - {filename: "results/figures/fig_go2_cts_priv_learning_curves.{png,pdf}",        shows: "Single-panel overlay of CTS Train/mean_reward across the three priv subsets. PRIV_COLOR grammar. Final R (smoothed): FULL=2104, INT=1986, EXT=1968. Numbers in legend (no overlap).", source: "scripts/plot_learning_curves.py::fig_cts_priv_learning_curves",      status: "ready"}
  # CSVs of raw curve data
  - {filename: "results/learning_curves/go2_learning_curves_reward.csv",            shows: "Raw (step, reward) pairs for Baseline / RMA / CTS Train/mean_reward.",                                                                                                     source: "scripts/plot_learning_curves.py",                                    status: "ready"}
  - {filename: "results/learning_curves/go2_learning_curves_length.csv",            shows: "Raw (step, episode-length) pairs for Baseline / RMA / CTS.",                                                                                                              source: "scripts/plot_learning_curves.py",                                    status: "ready"}
  - {filename: "results/learning_curves/go2_cts_teacher_student.csv",               shows: "Raw teacher/student reward for CTS-FULL.",                                                                                                                                source: "scripts/plot_learning_curves.py",                                    status: "ready"}
  - {filename: "results/learning_curves/go2_cts_teacher_student_overlay.csv",       shows: "Raw teacher/student reward for CTS-FULL/INT/EXT.",                                                                                                                        source: "scripts/plot_learning_curves.py",                                    status: "ready"}
  - {filename: "results/learning_curves/go2_cts_priv_learning_curves.csv",          shows: "Raw mean reward for CTS-FULL/INT/EXT.",                                                                                                                                   source: "scripts/plot_learning_curves.py",                                    status: "ready"}
  # one-leg figures
  - {filename: "results/figures/one_leg/fig_one_leg_headline.{png,pdf}",           shows: "single-leg Isaac OOD: mean reward (panel A) + success rate (panel B)", source: "scripts/plot_results_one_leg.py",     status: "ready"}
  - {filename: "results/figures/one_leg/fig_one_leg_ood_retention.{png,pdf}",      shows: "single-leg OOD retention bar with 70 % spec line + PASS badges",      source: "scripts/plot_results_one_leg.py",     status: "ready"}
  # required (planned for the report, not auto-generated)
  - {filename: "fig_architecture_diagram.{pdf}",                                   shows: "Baseline / RMA two-stage / CTS concurrent data-flow side by side",   source: "hand-drawn / TikZ",                   status: "to-be-drawn"}
```

---

## D. Reproducibility / Misc

### D.1 Code repository
- **URL:** local only — `/home/drl-68/t_s_policy/cts_rma_project`. If the
  author publishes a repo before submission they will fill it in; otherwise
  leave this field omitted from the report.
- **Commit hash:** `TBD` — the working tree currently contains the
  Go2-v2 training and eval changes uncommitted on top of `063632c update`.
  The author should tag a "report version" commit before submission.

### D.2 Title block confirmation
- **Author:** Bhumipat Ngamphueak, student ID 66340500043, FIBO, KMUTT.
  Course: FRA 503 Deep Reinforcement Learning, 2026. Solo final implementation
  report, 12-page LaTeX maximum.
- **Draft A title (proposed):** *"Comparative Study of Teacher–Student
  Architectures for Legged Locomotion Control: From Single-Leg Test Station to
  Quadruped Sim-to-Sim"*.
- **Draft B title (proposed):** *"Teacher–Student Architectures for Quadruped
  Locomotion Control: A Sim-to-Sim Evaluation on the Unitree Go2"*.

### D.3 Specific emphasis (apply in BOTH drafts)
- **Frame the pivot as a methodological strength.** The single-leg study
  contributes the diagnosis (a fixed-base platform cannot exercise the
  privileged signals); re-running the *same* three configurations on a
  free-base quadruped is the scientific contribution, not a failure.
- **L_rec is the key technical distinction between RMA and CTS.** RMA distils
  *sequentially*: Phase 1 trains a privileged teacher with encoder μ(x_t)→z;
  Phase 2 freezes the teacher and trains a 1D-CNN adaptation module φ(history)→ẑ
  by supervised regression. CTS distils *concurrently in one PPO loop*: teacher
  encoder E^t(x_t)→z and student encoder E^s(history)→z share one actor; an MSE
  loss `L_rec = MSE(E^s(hist), stop-grad E^t(x_t))` pulls the student latent
  toward the teacher latent (λ_rec = 5.0, warm-up 1000 iters, separate
  optimiser for E^s).
- **Fairness controls.** All three methods share identical scene, DR, reward,
  terminations, physics, control rate, network widths ([512,256,128] ELU MLP)
  and PPO hyperparameters; the **transfer mechanism is the only experimental
  variable**. At deployment no method uses privileged simulator state:
  Baseline sees `o_t`, RMA and CTS see `[o_t, ẑ]` with `ẑ` produced from a
  50-step proprioceptive history.
- **The reward/survival-vs-locomotion distinction is the central technical
  point of the Go2 results (CORRECTED — use §C.2/C.3/C.5 numbers).** Reward
  and survival alone *overstate* behaviour because a policy can "survive by
  not moving": **RMA**'s Phase-2 module does exactly this in MuJoCo — 100 %
  survival but forward displacement ≈ 0 m and velocity-RMSE pinned at the
  0.5 m/s command (§C.2c). The genuine task metric is forward locomotion +
  tracking, on which **CTS-FULL wins** (best survival *and* tracking at the
  training-faithful condition, §C.2a; smallest OOD drop). **The CTS
  privileged-subset ablation reproduces the same artefact inside the CTS
  family**: in MuJoCo, **INT-only collapses to a stationary policy — 0 %
  task-success at DR×1 despite 100 % survival** (vel-RMSE 0.497, torque-var
  0.19), so its higher *raw reward* is non-locomotion; **only FULL truly
  locomotes (93 % success)**, EXT partially (67 %). Body-parameter signals
  (INT) alone are insufficient; the interaction signals removed in INT are
  exactly what prevents the stationary-collapse. Reward is not the scoring
  rule — the 3-class success/partial/fail + forward-displacement + RMSE
  panels are the primary evidence.
- **Sim-to-sim as a proxy for sim-to-real.** Train in Isaac Lab (PhysX);
  transfer zero-shot to MuJoCo (`mujoco_menagerie/unitree_go2`). OOD sweeps at
  s ∈ {1.0, 2.0} probe robustness beyond the training distribution.

### D.4 Anything else the LaTeX-drafting AI should know
*(Architecture details + PPO hyperparameters are in §B.7 — do not duplicate.)*
- **Two RMA implementations exist in the repo — only the canonical one is used.**
  The report's RMA is the canonical Kumar et al. (2021) two-stage variant
  (described in §B.7). There is also a leftover asymmetric-critic variant
  in `RMAEnvCfg` / `RMAPPORunnerCfg` (actor identical to Baseline,
  privileged critic, no encoder, no Phase 2) — **ignore it**. All logs
  cited in this document (`logs/one_leg/rma/...`, `logs/rma/...v2..._l8_l8`,
  `logs/rma/phase2/...`) are the canonical version.
- **MuJoCo side:** `mujoco_menagerie/unitree_go2` via
  `scripts/sim2sim/sim2sim_go2.py` with metrics in
  `scripts/sim2sim/eval_metrics.py`. The Isaac and MuJoCo evaluation
  pipelines share an episode length (10 s), reward calculation (`is_alive`
  bonus included on both sides), DR range table, and 3-class outcome rule
  — so episode returns are directly comparable.
- **Sim2sim fixes applied identically to all three methods** (mention once
  in the methods section, so the comparison stays fair): (1) reset base
  velocity zeroed (was random ±0.5 m/s / ±1.0 rad/s) so policies learn to
  start walking from rest, which is the deployment scenario; (2) widened
  DR ranges (friction, inertia, Kp/Kd, CoM, action delay) so the training
  distribution spans the PhysX↔MuJoCo gap; (3) stronger CTS reconstruction
  loss (λ_rec 1.0 → 5.0, warm-up 500 → 1000, E^s LR 3e-4 → 5e-4) to make
  the student latent reliably interchangeable with the teacher latent at
  deployment.
- **References allowed without further confirmation:** Kumar et al. 2021
  (RMA, arXiv:2107.04034); Wang et al. 2024 (CTS / concurrent teacher–student,
  arXiv:2405.10830); Lee et al. 2020 (privileged-info T–S for legged
  locomotion, arXiv:2010.11251). Named tooling: NVIDIA Isaac Lab, RSL-RL,
  MuJoCo / `mujoco_menagerie`, Unitree Go2.
- **Draft weighting reminder:** Draft A ≈ 30 % Phase 1 / 10 % pivot / 50 % Go2 /
  10 % conclusions. Draft B = a ~0.75-page "Project Evolution" box for Phase 1,
  the rest Go2. The single-leg section is Isaac-only and uses **one fair
  configuration** in both drafts.

---

## E. Figure caption stubs (one line each, for figure environments)

> ✅ **Corrected 2026-05-18.** Go2 stubs below now carry the verified
> §C.2/C.3/C.5 numbers (Isaac = v2 training-faithful flat+push 20 s; MuJoCo =
> no-push 20 s). Safe to paste.

Use these as `\caption{...}` text inside the LaTeX figure environments;
keep them short, factual, and consistent with the data in §C.2/C.3.

- **`fig_one_leg_headline`** — *One-leg hexapod (Phase 1), Isaac Lab OOD test:
  (A) mean episode return at DR×1 vs DR×2 for Baseline/RMA/CTS (FULL, Z = 8,
  100 episodes); (B) success rate (all three methods are at 100 %).*
- **`fig_one_leg_ood_retention`** — *One-leg hexapod OOD retention
  (R_Isaac,2× / R_Isaac,1×). All three methods pass the 70 % spec
  (Baseline 88 %, RMA 93 %, CTS 91 %); the privileged-input T–S advantage
  does not appear on this fixed-base platform.*
- **`fig_go2_headline`** — *Go2 sim-to-sim transfer (Isaac = v2 training-
  faithful, MuJoCo = no-push). Reward-retention G(π) = R_MuJoCo,1× /
  R_Isaac,1×: Baseline 91 %, CTS 73 %, RMA 121 % — RMA's >100 % is an
  artefact of its near-stationary MuJoCo policy (high alive-bonus, zero
  locomotion), not transfer quality. The honest metric is forward
  locomotion: CTS is the only method that both survives and tracks across
  the simulator change.*
- **`fig_go2_summary`** — *Spec-sheet summary panel for Go2: (A) cumulative
  episode reward, (B) survival rate, (C) velocity-tracking error with the
  0.3 m/s spec line, (D) threshold-pass matrix. At the training-faithful
  Isaac condition every method survives 100 % at DR×1; at DR×2 survival is
  Baseline 73 %, CTS 87 %, RMA 23 % (RMA the only sub-80 % cell). In MuJoCo
  the 0.3 m/s tracking spec is met by CTS (0.18) and Baseline (0.22) at
  DR×1; RMA sits at 0.50 (stationary) in every MuJoCo cell.*
- **`fig_go2_comparison`** — *Four-view reward comparison (rows = OOD inside a
  sim, columns = sim-to-sim transfer): (A) Isaac OOD DR×1 vs DR×2, (B) MuJoCo
  OOD DR×1 vs DR×2, (C) sim-to-sim transfer at DR×1, (D) sim-to-sim transfer
  at DR×2 (worst case). Δ annotations report the retention ratio of the
  right-bar to the left-bar.*
- **`fig_go2_comparison_survival`** — *Same four views as
  `fig_go2_comparison`, but Y axis is survival rate. At the training-faithful
  Isaac condition CTS degrades least under OOD (DR×1→DR×2: 100→87 %) vs
  Baseline 100→73 % and RMA 80→23 % — RMA is the only method below the 80 %
  spec, at Isaac DR×2 (23 %).*
- **`fig_go2_comparison_rmse`** — *Same four views, Y axis is
  velocity-tracking error with the 0.3 m/s spec line. CTS has the lowest
  error in every panel where it locomotes (Isaac 0.13/0.16, MuJoCo
  0.18/0.33). RMA is pinned at ≈0.50 in MuJoCo — it is not tracking
  (stationary policy), so its "low-variance" bars are not a quality signal.*
- **`fig_go2_comparison_outcome`** — *Same four views, stacked Success /
  Partial / Fail per method. The decisive contrast is MuJoCo: CTS-FULL
  produces genuine locomotion (forward displacement ≈ 8 m at DR×1) while
  RMA's Phase-2 module collapses to a stationary "survive-by-not-moving"
  policy (forward ≈ 0 m) — high survival, zero task success.*
- **`fig_go2_ood_profile`** — *Isaac-only OOD profile: (A) reward-retention
  R_Isaac,2× / R_Isaac,1× with the 70 % spec line — Baseline 70 % PASS,
  CTS 88 % PASS, RMA 40 % FAIL (RMA the only OOD-spec failure); (B) absolute
  Isaac reward at DR×1 and DR×2.*
- **`fig_go2_sim2sim_transfer`** — *Sim-to-sim transfer figure: (A) G(π) and
  combined-gap bars with the 60 % / 40 % spec lines — Baseline G 91 % /
  Comb 68 %, CTS G 73 % / Comb 42 %, RMA G 121 % / Comb 103 % (RMA values
  inflated by its stationary MuJoCo policy); (B) absolute Isaac vs MuJoCo
  reward side-by-side with the Δ gap annotated per bar.*
- **`fig_go2_gait_quality`** — *Eight gait-quality metrics for Baseline / RMA /
  CTS at FULL / Z = 8 / DR×1, Isaac (solid) vs MuJoCo (hatched). Contact
  symmetry collapses Isaac→MuJoCo for every method (Baseline 0.90→0.14,
  CTS 0.87→0.07, RMA 0.96→0.00) — a periodic→drifty sim-to-sim shift. RMA's
  MuJoCo bars are flat-lined (action-smoothness 0.001, joint-torque variance
  0.09) — the quantitative signature of a robot standing still, not a good
  gait.*
- **`fig_go2_cts_priv_ablation`** — *CTS privileged-subset ablation
  (FULL = 26-D, INT = 16-D body params, EXT = 10-D interaction signals)
  across four views: Isaac DR×1, Isaac DR×2, MuJoCo DR×1, MuJoCo DR×2. Top
  row = episode reward, bottom row = success rate. In Isaac all three are
  competitive (DR×1 100 % each; DR×2 FULL/INT 87 %, EXT 73 %). In MuJoCo
  only FULL truly locomotes (success 93 % at DR×1); **INT collapses to a
  stationary policy (0 % success despite high reward)** and EXT is partial
  (67 %) — the interaction signals removed in INT are exactly what prevents
  the stationary collapse.*
- **`fig_go2_cts_priv_ablation_gait_dr1`** — *Behaviour-metric companion at
  DR×1: the 8 gait metrics for the three CTS variants, Isaac (solid) vs
  MuJoCo (hatched). The smoking gun for INT's MuJoCo collapse: INT MuJoCo
  action-smoothness 0.008 and joint-torque variance 0.19 (vs FULL 1.36 /
  18.19) with contact symmetry ≈ 0 — INT is not locomoting, which is why
  its raw reward looks high but task-success is 0 %.*
- **`fig_go2_cts_priv_ablation_gait_dr2`** — *Same 8 metrics at DR×2. INT
  stays in the stationary regime (joint-torque variance 5.5 vs FULL 21.9;
  action-smoothness 0.40 vs FULL 2.21); FULL spends real control effort to
  keep tracking. Contact-symmetry collapse in MuJoCo is unchanged by DR
  scale — the periodic-gait loss is a simulator-change effect, not a
  DR-OOD effect.*
- **`fig_architecture_diagram`** — *(to be drawn) Side-by-side data flow for the
  three configurations: Baseline (o_t → MLP → a_t), RMA (Phase 1: x_t → μ → z,
  policy on [o_t, z]; Phase 2: history → φ → ẑ, frozen teacher actor on
  [o_t, ẑ]) and CTS (concurrent teacher E^t(x_t) → z and student E^s(history) →
  ẑ feeding the same actor, with L_rec coupling them).*
- **`fig_go2_learning_curves`** — *Go2 PPO learning curve: Train/mean_reward
  vs iteration (0–25 000) for Baseline / RMA / CTS at FULL, Z = 8 — light
  raw lines and bold EMA-smoothed lines per method. RMA shows the
  characteristic curriculum dip ~iter 9 000-11 000; final reward
  CTS 2110 (highest), Baseline 2041, RMA 2011.*
- **`fig_go2_learning_curves_len`** — *Same three methods, Y axis = mean
  episode length in control steps (max 1000 = 20 s episode). All three
  saturate near the 1000-step cap by iter ~3 000 (final: Baseline 1000,
  CTS 990, RMA 986).*
- **`fig_go2_cts_teacher_student`** — *Concurrent teacher/student training
  inside CTS-FULL: separate side-by-side panels (teacher = navy, student =
  orange) sharing y-axis, plus a gap (T − S) strip below. Post-warmup mean
  gap = +15 (+0.8 % of teacher reward) — the L_rec = 5.0 reconstruction loss
  keeps the history-only student within < 1 % of the privileged-input
  teacher across 25 000 PPO iterations.*
- **`fig_go2_cts_teacher_student_overlay`** — *Same teacher/student
  comparison repeated across all three CTS privileged subsets (FULL / INT /
  EXT). 2×3 layout: top row = teacher+student overlaid per subset (shared
  y-axis for direct comparison), bottom row = gap track per subset.
  Post-warmup mean gaps: FULL +0.8 %, INT +0.5 %, EXT +0.7 % — L_rec works
  uniformly across the privileged-subset space.*
- **`fig_go2_cts_priv_learning_curves`** — *Single-panel overlay of the
  Train/mean_reward learning curves for the three CTS variants (FULL / INT /
  EXT), Z = 8. Final reward: FULL = 2110, INT = 1963, EXT = 1989.
  FULL leads consistently from iter ~1 500; INT and EXT converge to nearly
  identical training reward (~7 % below FULL) despite encoding
  qualitatively different signals.*

---

## F. Detailed image descriptions for the LaTeX-drafting AI

> ✅ **Corrected 2026-05-18.** §F.3–F.18 Go2 readings now carry the verified
> §C.2/C.3/C.5 numbers (Isaac = v2 training-faithful flat+push 20 s; MuJoCo
> = no-push 20 s; figures regenerated). F.16/F.17 (teacher–student) were
> verified against the training CSVs and were already correct. §F.1–F.2
> (single-leg) unaffected. Safe to paste.

Each block below states **what the figure shows**, **the encoding** (axes,
colour/hatch grammar), and **the one-sentence reading** the report should
quote (do not paraphrase the numbers — use these). All Go2 figures share the
colour grammar Baseline = blue (`#2166ac`), RMA = green (`#4dac26`), CTS =
red/orange (`#d6604d`); DR×1 = solid fill, DR×2 = diagonal hatch / lighter
alpha; in cross-sim panels Isaac = solid, MuJoCo = hatched.

### F.1 `fig_one_leg_headline.png` (single-leg, Phase 1)
- **Panel A.** Mean episode return ± std, grouped by method, for DR×1 (solid)
  vs DR×2 (hatched). The Δ between bars is annotated as a retention %:
  Baseline 88 %, RMA 93 %, CTS 91 %.
- **Panel B.** Success rate per method × DR. Every cell is 100 % (success was
  defined as episode-length-reached on the fixed-base platform).
- **Quote-ready reading.** *"At the training distribution, Baseline returns
  2043.8 ± 86.5, higher than RMA 1879.5 ± 161.3 and CTS 1549.7 ± 134.4; all
  three methods reach 100 % success and retain ≥ 88 % of their reward at
  DR×2. The privileged-input methods provide no measurable advantage on
  this fixed-base platform."*

### F.2 `fig_one_leg_ood_retention.png` (single-leg, Phase 1)
- **Single-bar panel.** OOD retention = R_DR×2 / R_DR×1 per method, plotted as
  a percentage with the 70 % spec line dashed and a `✓ PASS` badge on each
  bar. Numbers: 88 %, 93 %, 91 %.
- **Quote-ready reading.** *"All three configurations satisfy the 70 % OOD
  retention spec on the single leg, but the spread between them is small
  (5 percentage points) and Baseline retains as much as the T–S methods —
  confirming that the platform does not stress-test the privileged-input
  hypothesis."*

### F.3 `fig_go2_headline.png` (Go2 sim-to-sim, primary figure)
- **Single panel.** For each method, two bars: G(π) at DR×1 (solid) vs DR×2
  (hatched), where G(π,s) = R_MuJoCo,s / R_Isaac,s. Numbers: Baseline
  91 % / 97 %, CTS 73 % / 48 %, RMA 121 % / 256 %. Dashed 60 % spec line.
  RMA's >100 % bars are flagged as a stationary-policy artefact.
- **Quote-ready reading.** *"Reward-retention alone is misleading here: RMA's
  apparent >100 % 'transfer' is an artefact — in MuJoCo its Phase-2 module
  produces a near-stationary policy (forward displacement ≈ 0 m, velocity
  error pinned at the command magnitude), so the alive-bonus inflates
  reward without locomotion. Baseline retains ~91 % and CTS ~73 % of raw
  reward, but the success-rate, forward-displacement and RMSE panels show
  CTS is the only method that transfers genuine, well-tracked locomotion."*

### F.4 `fig_go2_summary.png` (Go2 spec-sheet 2×2 dashboard)
- **(A)** Cumulative episode reward, four groups of three bars (Isaac×1,
  Isaac×2, MuJoCo×1, MuJoCo×2), per-method colour.
- **(B)** Survival rate with the 80 % spec line.
- **(C)** Velocity-tracking error with the 0.3 m/s spec line.
- **(D)** PASS/FAIL matrix over (method × {Survival, vel_err, G(π) ≥ 60,
  OOD gap ≥ 70, Combined ≥ 40}). The one orange FAIL cell: RMA OOD gap
  40.2 %.
- **Quote-ready reading.** *"Every spec passes except RMA's Isaac OOD
  reward-retention (40.2 %, the only FAIL). At the training-faithful Isaac
  condition survival is Baseline 100→73 %, CTS 100→87 %, RMA 80→23 %
  (DR×1→DR×2); CTS degrades least. In MuJoCo the 0.3 m/s tracking spec is
  met by CTS (0.18) and Baseline (0.22) at DR×1, while RMA sits at 0.50
  in every MuJoCo cell because it is not moving."*

### F.5 `fig_go2_comparison.png` (4-view reward)
- **2×2 grid** of bar plots. Rows = OOD-inside-sim (top) and sim-to-sim
  transfer (bottom). Columns = DR×1 (left) / DR×2 (right).
- **(A)** Isaac OOD: DR×1 (solid) vs DR×2 (hatched); Δ = OOD retention.
- **(B)** MuJoCo OOD: DR×1 vs DR×2; Δ = MuJoCo OOD retention.
- **(C)** Sim2Sim @ DR×1: Isaac (solid) vs MuJoCo (hatched); Δ = G(π) at 1×.
- **(D)** Sim2Sim @ DR×2 (worst case): Isaac vs MuJoCo; Δ = G(π) at 2×.
- **Quote-ready reading.** *"The distinguishing feature is the DR×2 column:
  RMA's Isaac reward collapses to 40 % of its DR×1 value (the largest OOD
  drop, the only spec FAIL), while CTS retains 88 % and Baseline 70 %. In
  MuJoCo CTS keeps the strongest locomotion (forward ≈ 4.2 m at DR×2 vs
  RMA ≈ 0 m); reward bars for RMA are not comparable because RMA stands
  still."*

### F.6 `fig_go2_comparison_survival.png`
- Same 4 views as F.5, Y axis = survival rate (%). 80 % spec line.
  (Isaac = v2 training-faithful flat+push; MuJoCo = no-push.)
- Numbers per panel (Baseline / CTS / RMA):
  - **A** (Isaac OOD, DR×1 → DR×2): 100/100/80 → **73/87/23**.
  - **B** (MuJoCo OOD, DR×1 → DR×2): 100/100/100 → 90/73/100.
  - **C** (Sim2Sim @ DR×1, Isaac → MuJoCo): 100/100/80 → 100/100/100.
  - **D** (Sim2Sim @ DR×2, worst case): 73/87/23 → 90/73/100.
- **Quote-ready reading.** *"At the training-faithful Isaac condition CTS
  degrades least under OOD (100→87 %) vs Baseline (100→73 %) and RMA
  (80→23 %, the only sub-80 % method). MuJoCo survival is uniformly high,
  but for RMA that is the stationary-policy artefact (100 % survival, 0 %
  task success) — survival must be read together with forward
  displacement, not alone."*

### F.7 `fig_go2_comparison_rmse.png`
- Same 4 views, Y axis = velocity-tracking error (m/s). 0.3 m/s spec line.
  Lower is better. (Baseline / CTS / RMA, DR×1 → DR×2.)
  - A (Isaac OOD)  0.144 / 0.130 / 0.201 → 0.239 / 0.157 / 0.259.
  - B (MuJoCo OOD) 0.222 / 0.177 / 0.500 → 0.351 / 0.327 / 0.500.
  - C (Sim2Sim DR×1) Isaac vs MuJoCo: 0.144/0.130/0.201 vs 0.222/0.177/0.500.
  - D (Sim2Sim DR×2) Isaac vs MuJoCo: 0.239/0.157/0.259 vs 0.351/0.327/0.500.
- **Quote-ready reading.** *"CTS has the lowest tracking error in every
  panel — Isaac (0.13/0.16) and MuJoCo (0.18/0.33). RMA's MuJoCo error is
  pinned at ≈0.50 = the 0.5 m/s command magnitude, i.e. zero forward
  velocity: it is not tracking at all. Only CTS keeps tracking within
  ~0.18 m/s after the simulator change at the training distribution."*

### F.8 `fig_go2_comparison_outcome.png`
- Same 4 views, stacked bars Success / Partial / Fail. The Success share +
  forward-displacement is the report's evidence for the
  reward-vs-locomotion distinction.
- **Quote-ready reading.** *"In MuJoCo, CTS-FULL converts survival into
  genuine locomotion (forward displacement ≈ 8.2 m at DR×1, the highest of
  any method) while RMA's Phase-2 module yields 100 % survival but ≈ 0 m
  forward travel — survival without the task. The concurrent T–S
  architecture's advantage is turning survival into useful, tracked
  locomotion, not raw reward."*

### F.9 `fig_go2_ood_profile.png`
- **(A)** Isaac OOD reward-retention (R_2× / R_1×) per method with PASS/FAIL
  badges: Baseline 70 % PASS, CTS 88 % PASS, RMA 40 % FAIL.
- **(B)** Absolute Isaac reward at DR×1 and DR×2.
- **Quote-ready reading.** *"Even before crossing simulators, RMA fails the
  70 % Isaac OOD retention spec (40 %); CTS is the most OOD-robust (88 %)
  and Baseline sits on the spec edge (70 %)."*

### F.10 `fig_go2_sim2sim_transfer.png`
- **(A)** Sim2Sim retention G(π,s) per method at DR×1 (solid) / DR×2
  (hatched), 60 % spec line.
- **(B)** Absolute reward Isaac vs MuJoCo per method, Δ = R_Isaac − R_MuJoCo
  annotated.
- **Quote-ready reading.** *"On raw reward every method clears the 60 %
  transfer spec, but the ranking is dominated by the RMA stationary-policy
  artefact (G(π) = 121 %). The honest evidence is panel (B) + forward
  displacement: CTS is the only method whose Isaac→MuJoCo reward drop
  corresponds to retained locomotion; RMA's small 'drop' corresponds to no
  locomotion at all. Reward alone is not an adequate sim-to-sim scoring
  rule."*

### F.11 `fig_go2_cts_priv_ablation.png` (CTS privileged-subset ablation)
- **2×4 grid.** Columns = (Isaac DR×1, Isaac DR×2, MuJoCo DR×1, MuJoCo DR×2);
  DR×2 columns drawn with diagonal hatch and α≈0.70 to match the figure
  family. Each panel has three bars: FULL (green `#1a9850`), INT (blue
  `#4575b4`), EXT (orange `#f46d43`) — colours from the shared `PRIV_COLOR`
  grammar.
- **Top row** = episode reward, mean ± std. Numeric labels on every bar.
- **Bottom row** = success rate %. The 80 % spec line is dashed; numeric
  labels show the percentage above each bar.
- **Key numbers per panel** (FULL / INT / EXT):
  - Isaac DR×1: reward 2630 / 2602 / 2530; success 100 / 100 / 100 %.
  - Isaac DR×2: reward 2311 / 2082 / 1856; success 87 / 87 / 73 %.
  - MuJoCo DR×1: reward 1930 / **2077** / 1791; success **93 / 0 / 67 %**.
  - MuJoCo DR×2: reward 1109 / **1587** / 1112; success 40 / 17 / 23 %.
- **Quote-ready reading.** *"Inside Isaac (training-faithful) the three
  subsets are near-equivalent at DR×1 (all 100 % success, ≤ 4 % reward
  drop); under OOD-DR×2 FULL and INT hold 87 % survival while EXT drops to
  73 %. In MuJoCo the reward column again looks inverted — INT's raw reward
  exceeds FULL's (+8 % at DR×1, +43 % at DR×2) — but this is the
  stationary-policy artefact: **INT's MuJoCo task-success is 0 % at DR×1**
  (vs FULL 93 %) because INT stops locomoting (forward ≈ 0 m, velocity
  error ≈ 0.50). FULL is the only subset that produces genuine MuJoCo
  locomotion; EXT is partial (67 %). The interaction signals removed in INT
  are exactly what prevent the stationary collapse — proprioceptive history
  alone cannot recover them."*
- **Companion table** is in `results/go2_cts_priv_ablation_table.md` (cited
  verbatim in §C.5).

### F.12 `fig_go2_cts_priv_ablation_gait_dr1.png` and `…_gait_dr2.png`
*(behaviour metrics for the CTS ablation, training distribution and OOD)*

Two figures with the **same** 2×4 grid; only the DR scale differs. Treat them
as a pair: `_dr1` is the training-distribution headline, `_dr2` is the OOD
condition (each scaled-DR range widened ×2 about its midpoint).
- **2×4 grid.** Same layout/colour grammar as `fig_go2_gait_quality`, but the
  bars in each panel are CTS-FULL (green `#1a9850`) / CTS-INT (blue `#4575b4`)
  / CTS-EXT (orange `#f46d43`); Isaac = solid, MuJoCo = hatched diagonal.
  Headline cells: `Z = 8`, `DR×1` and `DR×2`.
- **Eight panels.** (top row, l→r) gait adherence ↑, contact symmetry ↑,
  swing-clearance error ↓, foot-slip rate ↓. (bottom row) action smoothness ↓,
  base-height variance ↓, stride variance ↓, joint-torque variance ↓.
- **Key numbers at DR×1** (Isaac → MuJoCo, FULL / INT / EXT):
  - **gait adherence:** 0.338 / 0.345 / 0.349 → 0.203 / **0.136** / 0.218 — drops in MuJoCo for all; INT lowest (degenerate gait).
  - **contact symmetry:** 0.871 / 0.900 / 0.918 → 0.068 / **0.000** / 0.049 — periodic gait lost in MuJoCo; INT loses it completely (standing still).
  - **swing-clearance error:** 0.0035 / 0.0039 / 0.0035 → 0.0013 / **0.0000** / 0.0010 — INT ≈ 0 because the foot barely moves.
  - **foot slip rate:** 0.160 / 0.292 / 0.314 → 1.63 / 1.10 / 1.49 — rises in MuJoCo for all.
  - **action smoothness:** 1.73 / 1.47 / 1.48 → 1.36 / **0.008** / 1.17 — INT's ≈0 is the signature of a near-constant action (not moving).
  - **base-height variance:** 0.0005 / 0.0005 / 0.0005 → 0.0004 / **0.0000** / 0.0003 — INT body is static in MuJoCo.
  - **stride variance:** 0.0006 / 0.0018 / 0.0014 → 0.0061 / **0.0001** / 0.0051 — INT ≈ 0: no strides at all.
  - **joint-torque variance:** 5.07 / 5.11 / 4.57 → 18.19 / **0.19** / 15.76 — INT uses ~1 % of FULL's MuJoCo torque variance: it is not actuating to walk.
- **Key numbers at DR×2** (Isaac → MuJoCo, FULL / INT / EXT):
  - **contact symmetry:** 0.822 / 0.841 / 0.810 → 0.091 / **0.010** / 0.057 — same collapse; INT still essentially frozen.
  - **action smoothness:** 2.12 / 1.99 / 3.01 → 2.21 / **0.400** / 1.06 — INT stays in the low-effort stationary regime.
  - **stride variance:** 0.0038 / 0.0023 / 0.0015 → 0.0041 / 0.0016 / 0.0043 — FULL/EXT take real strides; INT minimal.
  - **joint-torque variance:** 6.24 / 7.90 / 10.93 → 21.85 / **5.53** / 16.66 — FULL spends the most control effort to keep tracking; INT the least (under-actuating).
- **Quote-ready reading.** *"The behaviour panels expose the mechanism behind
  the CTS-ablation reward inversion. In MuJoCo, INT-only does not learn a
  'quieter walk' — it stops walking: action-smoothness ≈ 0.008, joint-torque
  variance ≈ 0.19, stride variance ≈ 0.0001, contact symmetry ≈ 0 are the
  quantitative signature of a robot standing essentially still (the same
  failure mode as RMA's Phase-2 module). That is why INT's raw reward looks
  high (alive-bonus) while its task-success is 0 %. FULL is the only subset
  that spends real control effort (torque variance 18–22) to produce and
  keep a tracked gait; EXT is intermediate. Contact symmetry collapses for
  every subset in MuJoCo at both DR scales — the periodic→drifty shift is a
  simulator-change effect independent of the privileged input."*

### F.13 `fig_go2_gait_quality.png`
- 2×4 grid of 8 metric panels, Baseline / RMA / CTS in each panel, Isaac =
  solid bar, MuJoCo = hatched bar. Direction-of-improvement labelled in each
  title ("higher better" or "lower better").
- The visually most informative panels in MuJoCo are: **contact symmetry**
  (collapses Isaac→MuJoCo for every method — Baseline 0.90→0.14, CTS
  0.87→0.07, RMA 0.96→0.00 — the periodic gait is not preserved across the
  simulator change), **foot-slip rate** (rises in MuJoCo, friction transfer
  incomplete), and **joint-torque variance** (CTS reaches the highest
  absolute MuJoCo value ≈ 18.19, about 3.6× its Isaac value 5.07; Baseline
  5.16→9.66; **RMA collapses to ≈ 0.09 with action-smoothness ≈ 0.001** —
  the signature of a robot standing still, not a good gait).
- **Quote-ready reading.** *"The gait metrics confirm the sim-to-sim gap is
  concentrated in contact-related quantities (contact symmetry, foot-slip),
  and they expose RMA's MuJoCo behaviour for what it is: near-zero
  joint-torque variance and action-smoothness mean RMA is not locomoting at
  all. CTS pays for its low tracking error with the highest joint-torque
  variance (it is actively working to track) — a quantifiable, honest
  trade-off the report can name explicitly; Baseline sits between the two."*

### F.14 `fig_go2_learning_curves.png` (Baseline / RMA / CTS PPO curves)
- **Single panel.** X = PPO iteration (0–25 000); Y = `Train/mean_reward`
  scalar from RSL-RL's TensorBoard log. Method colour grammar:
  Baseline = blue, RMA = green, CTS = red/orange. Raw values shown as
  light translucent lines; bold lines are EMA-smoothed (α = 0.01) for
  readability.
- **Key features.** CTS climbs fastest in the first ~2 000 iters and
  finishes highest (final ≈ 2 110); Baseline just behind (≈ 2 041); RMA
  lowest (≈ 2 011) and shows a characteristic dip around iter ~9 000-11 000
  (typical of the privileged-input curriculum shift documented in
  Kumar et al. 2021).
- **Quote-ready reading.** *"All three methods converge by ~20 000
  iterations on a 25 000-iteration budget. CTS's faster early progress and
  higher plateau are consistent with the concurrent reconstruction-loss
  mechanism letting the student exploit the teacher's privileged signal
  immediately, while RMA's two-stage protocol introduces the
  characteristic mid-training dip when the policy adapts to the privileged
  distillation curriculum."*

### F.15 `fig_go2_learning_curves_len.png`
- Same three methods, Y = `Train/mean_episode_length` (max = 1000 control
  steps = 20 s episode). All three saturate near the cap by iter ~3 000
  (final: Baseline 1000, CTS 990, RMA 986); the figure confirms training
  reached "robot is alive for the full episode" quickly — the differences
  between methods are in reward shape, not survival, beyond that.

### F.16 `fig_go2_cts_teacher_student.png`
- **Three-panel layout** (top-left, top-right, bottom).
  - **Top-left** — CTS-FULL teacher curve only (75 % of envs, actor sees
    `[o_t, E^t(x_t)]`). Navy. Final reward ≈ 2 151.
  - **Top-right** — CTS-FULL student curve only (25 % of envs, actor sees
    `[o_t, E^s(history)]`). Orange. Final reward ≈ 2 050. Shares y-axis
    with the teacher panel so absolute reward is comparable across.
  - **Bottom (full width)** — gap = teacher − student. Positive (teacher
    ahead) is shaded navy; negative (student ahead) is shaded orange. Red
    dashed line = post-warmup mean gap (after iter ≥ 1 000) = **+15
    (+0.8 % of teacher's mean reward)**.
- **Quote-ready reading.** *"After warm-up, the CTS student (history-only,
  25 % of envs) follows the teacher (privileged-input, 75 % of envs) with
  a steady gap of ≈ 15 reward — about 0.8 % of the teacher's mean. The
  reconstruction loss L_rec = 5.0 keeps the two latents aligned tightly
  enough that the student is competitive in training; the residual gap
  reflects the information loss intrinsic to inferring privileged state
  from a 50-step proprioceptive history."*

### F.17 `fig_go2_cts_teacher_student_overlay.png`
- **2×3 grid.** Three columns = (CTS-FULL, CTS-INT, CTS-EXT) where each
  column header is coloured with the `PRIV_COLOR` grammar
  (FULL = green, INT = blue, EXT = orange).
  - **Top row** — teacher (solid navy, "sees x_t") and student
    (dashed orange, "history only") overlaid on the *same* axes per
    subset. Shared y-axis across all three columns so absolute reward is
    directly comparable. Small final-value annotations on each curve.
  - **Bottom row** — gap track (T − S) per subset, with post-warmup mean
    annotated.
- **Per-subset final values** (teacher / student / post-warmup mean gap):
  - **FULL**:  T = 2 151, S = 2 050, gap = **+15 (+0.8 %)**.
  - **INT**:   T = 1 977, S = 1 964, gap = **+8 (+0.5 %)**.
  - **EXT**:   T = 1 993, S = 1 947, gap = **+12 (+0.7 %)**.
- **Quote-ready reading.** *"Across all three privileged subsets the CTS
  reconstruction loss keeps the history-only student within ≤ 1 % of the
  privileged-input teacher in training reward (FULL +0.8 %, INT +0.5 %,
  EXT +0.7 %). The smaller subsets (INT and EXT) plateau at lower absolute
  reward (~1 970-2 000 vs FULL ~2 100-2 150), but their students remain
  just as tightly aligned with their teachers — confirming that the
  concurrent distillation mechanism is robust to the choice of
  privileged subset."*

### F.18 `fig_go2_cts_priv_learning_curves.png`
- **Single-panel overlay** of `Train/mean_reward` for the three CTS
  variants on the same axes, using the `PRIV_COLOR` grammar
  (FULL green, INT blue, EXT orange). Raw lines drawn light; bold smoothed
  lines on top. Each curve ends in a small white-edged dot; final
  *smoothed* values are listed **in the legend** (not as inline
  annotations, to prevent the INT/EXT labels from overlapping when their
  values are within 18 reward of each other).
- **Final values** (iter 24 999):
  - **CTS FULL (26-D)** — final R ≈ **2 110**.
  - **CTS INT  (16-D)** — final R ≈ **1 963**.
  - **CTS EXT  (10-D)** — final R ≈ **1 989**.
- **Reading.** FULL leads consistently from iter ~1 500 onwards by ~6-7 %
  over both INT and EXT. INT and EXT converge to nearly the same training
  reward (within ~1.3 %) despite encoding qualitatively very different
  signals — INT carries 16 D of body-parameter information (friction, mass,
  gains, CoM, inertia, action delay), EXT carries 10 D of *interaction*
  signals (per-foot contact forces, contact flags, mean joint torques).
  Similar training reward, different mechanism.
- **Quote-ready reading.** *"FULL's ~6-7 % training-reward advantage over
  INT and EXT understates the deployment gap: §C.5 shows that in MuJoCo
  INT-only's higher raw reward is a stationary-policy artefact — INT
  collapses to 0 % task success at DR×1 versus FULL's 93 % (a 93-point
  gap), because INT stops locomoting. Training-reward ordering is a poor
  proxy for behaviour-quality at deployment — a central technical point of
  the report."*

---

## G. What to write where (suggested figure-to-section mapping)

| section in the report          | figure(s)                                                              | tables to cite             |
|---                              |---                                                                     |---                         |
| §3 Phase 1 (single-leg)         | `fig_one_leg_headline`, `fig_one_leg_ood_retention`                    | `one_leg_results_table.md` (top half) |
| §3.x Diagnosis of Phase 1       | (text only; reference §A.3 hypotheses)                                  | —                          |
| §4 Phase 2 setup (Go2)          | `fig_architecture_diagram` (to draw)                                    | —                          |
| §4.x Training procedure (Go2)   | `fig_go2_learning_curves`, `fig_go2_learning_curves_len`, `fig_go2_cts_teacher_student` | — |
| §5 Go2 — Isaac OOD              | `fig_go2_ood_profile`                                                   | rows 1–6 of `go2_results_table.md` |
| §5 Go2 — sim-to-sim transfer    | `fig_go2_sim2sim_transfer`, `fig_go2_headline`                          | rows 7–12 of `go2_results_table.md`, transfer-ratio table |
| §5 Go2 — full 4-view comparison | `fig_go2_comparison`, `fig_go2_comparison_survival`, `fig_go2_comparison_rmse`, `fig_go2_comparison_outcome` | the per-row table |
| §5 Go2 — spec-sheet summary     | `fig_go2_summary`                                                       | transfer-ratio table       |
| §5.x Privileged-subset ablation | `fig_go2_cts_priv_ablation`, `fig_go2_cts_priv_ablation_gait_dr1`, `fig_go2_cts_priv_ablation_gait_dr2`, `fig_go2_cts_priv_learning_curves`, `fig_go2_cts_teacher_student_overlay` | `go2_cts_priv_ablation_table.md` |
| §6 Behaviour analysis (Go2)     | `fig_go2_gait_quality`                                                  | —                          |
| §7 Discussion / take-aways      | re-cite `fig_go2_comparison_outcome` (reward vs success), `fig_go2_cts_priv_ablation` (FULL > INT > EXT on behaviour) | —                          |

The four-view family (`fig_go2_comparison*`) is best used as a *single double-
column figure block* with the four metrics as four sub-figures, but each
sub-figure is also a stand-alone figure if the LaTeX template prefers
column-width figures.

---

## H. What "worst case" means in the figures

Several figures label one panel as **"worst case"**. This is *not* a separate
experiment — it is the panel inside the 2×2 evaluation matrix that stresses
**both** axes of difficulty at the same time. The two axes are:

1. **Domain-randomisation axis.** `DR×1` = the training distribution (the
   exact DR ranges the policy was optimised against). `DR×2` = each scaled-DR
   parameter range widened by a factor of 2 about its midpoint — i.e. the
   policy is asked to operate at friction / mass / Kp / Kd / CoM / inertia /
   action-delay values that lie *outside* training. This is the
   **out-of-distribution (OOD) axis**.
2. **Simulator axis.** `Isaac` = trained-and-evaluated in Isaac Lab (PhysX).
   `MuJoCo` = the policy is loaded zero-shot into a different physics
   engine (`mujoco_menagerie/unitree_go2`) with a different contact solver,
   integrator, and actuation model. This is the **sim-to-sim axis** — the
   stand-in for sim-to-real.

The four panels in the `fig_go2_comparison*` family (reward / survival /
RMSE / outcome) therefore correspond to the four cells of the matrix:

|                | Isaac (training sim)                       | MuJoCo (different sim)                                       |
|---             |---                                         |---                                                           |
| **DR×1 (training distribution)** | (A) Isaac OOD baseline — both bars in panel A's `DR×1` group | (C) **Pure sim-to-sim transfer** — the policy meets a different simulator at the DR it trained on |
| **DR×2 (out-of-distribution)**   | (A/B) **Pure OOD robustness** — same sim, harder DR        | (D) **"Worst case"** — different simulator *and* OOD DR at the same time |

So "worst case" = **MuJoCo × DR×2** = the cell that combines the OOD axis
*and* the sim-to-sim axis. It is the most challenging condition in the study,
and it is the cell where method differences are largest (and where the
spec-sheet "Combined gap ≥ 40 %" target lives — see §C.3).

The report should state this once, the first time `(worst-case)` appears in a
figure caption. Suggested phrasing for the LaTeX-drafter:

> *"In all four-view figures, "worst case" denotes the panel that combines
> two sources of distribution shift — out-of-distribution domain
> randomisation (DR×2) and a change of physics simulator (Isaac → MuJoCo)
> — and is therefore the cell that distinguishes the three methods most
> sharply."*

For the single-leg figures (`fig_one_leg_*`) only the DR axis is exercised
quantitatively (MuJoCo is Isaac-only at the reduced fair config), so
"worst case" there reduces to **DR×2 in Isaac** — labelled simply as `DR×2`
in those figures.

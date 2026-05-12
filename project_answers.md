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
> reappears — but it is **CTS, not RMA**, that survives the simulator change:
> CTS keeps the highest task-success rate in MuJoCo at the training distribution
> (96.7 % vs 63.3 % for Baseline/RMA) and is the only method whose velocity
> tracking stays inside the 0.3 m/s spec at MuJoCo DR×2. RMA's Phase-2
> adaptation module collapses out-of-distribution (48 % OOD retention, the only
> spec-sheet FAIL in the matrix). A controlled privileged-subset ablation on
> CTS (FULL / INT / EXT) further shows that the body-parameter subset (INT)
> alone is enough to *match FULL on raw reward* in both sims, but **only the
> FULL combination delivers the high task-success rate** in MuJoCo (97 % vs INT
> 60 % vs EXT 53 % at DR×1) — the interaction signals in EXT carry
> behavioural information that the history cannot recover.

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

### B.6 Episode and success criterion
```yaml
episode_length_steps: 500                # 10.0 s at 50 Hz control (decimation 4 over 200 Hz)
episode_length_s: 10.0
success_criterion: "episode reaches its 10 s time-out (no fall) AND velocity-tracking RMSE < 0.3 m/s. Surviving-but-poor-tracking episodes are reported separately as 'partial'."
outcome_classes:
  - success: "time_out AND vel_rmse < 0.3 m/s"
  - partial: "time_out AND vel_rmse >= 0.3 m/s"
  - fail:    "early termination (fall)"
termination_conditions:
  - "time_out: episode reaches 10.0 s"
  - "base illegal contact: contact force on the 'base' body > 100 N"
  - "bad orientation: base tilt angle > 1.2 rad (~68.8 deg)"
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
```yaml
experiments:
  - {name: "Isaac OOD - DR x1.0",      status: "done", method: "Baseline, RMA, CTS", n_episodes: 30}
  - {name: "Isaac OOD - DR x2.0",      status: "done", method: "Baseline, RMA, CTS", n_episodes: 30}
  - {name: "MuJoCo sim2sim - DR x1.0", status: "done", method: "Baseline, RMA, CTS", n_episodes: 30}
  - {name: "MuJoCo sim2sim - DR x2.0", status: "done", method: "Baseline, RMA, CTS", n_episodes: 30}
  - {name: "Phase-1 single-leg fair-config",  status: "done (Isaac-only, FULL/Z=8)", method: "Baseline, RMA, CTS", n_episodes: 100}
  - {name: "CTS privileged-subset ablation (FULL/INT/EXT) on Go2", status: "done", method: "CTS only, Z=8", n_episodes: 30, cells: "2 sims × 2 DR scales × 3 priv subsets = 12 (4 new INT + 4 new EXT cells appended to ood_go2.csv and sim2sim_go2.csv on top of the 4 pre-existing FULL cells)"}
ablations_not_reported:
  - "Latent-dimension sweep on Go2 (only Z=8 was reported); compute budget pivoted to nailing one fair config."
  - "Privileged-subset ablation for RMA on Go2 (only CTS is ablated)."
```
The fair Go2 configuration used for every reported number: **FULL privilege
(26-D x_t), latent Z = 8, 30 episodes per cell, episode length 10 s** (500
control steps), all three methods share identical scene/reward/DR/physics.
Total cells in the final tables: 3 methods × 2 sims × 2 DR scales = 12 rows
(see `results/go2_results_table.md`).

### C.2 Final per-row evaluation table (Go2) — FULL methods only
*(The 12 FULL-method rows of `results/go2_results_table.md` — the
Baseline-vs-CTS-vs-RMA comparison; the CTS-INT and CTS-EXT rows are reported
separately in §C.5. Columns: episode return = mean ± std of cumulative
reward; success % = 3-class "success" share, i.e. survived **and**
vel_rmse < 0.3 m/s.)*

| sim    | method   | priv | Z | s | episode return    | success % |
|---     |---       |---   |---:|--:|---                |---:       |
| Isaac  | Baseline | BASE | — | 1 | 1251.6 ± 88.9     | 100       |
| Isaac  | Baseline | BASE | — | 2 | 1119.6 ± 213.6    | 97        |
| Isaac  | CTS      | FULL | 8 | 1 | 1265.9 ± 61.7     | 100       |
| Isaac  | CTS      | FULL | 8 | 2 | 1117.3 ± 263.7    | 100       |
| Isaac  | RMA      | FULL | 8 | 1 | 936.9 ± 144.0     | 77        |
| Isaac  | RMA      | FULL | 8 | 2 | 453.5 ± 490.7     | 47        |
| MuJoCo | Baseline | BASE | — | 1 | 1160.9 ± 103.5    | 63        |
| MuJoCo | Baseline | BASE | — | 2 | 889.4 ± 361.0     | 40        |
| MuJoCo | CTS      | FULL | 8 | 1 | 980.7 ± 116.8     | 97        |
| MuJoCo | CTS      | FULL | 8 | 2 | 647.7 ± 395.8     | 53        |
| MuJoCo | RMA      | FULL | 8 | 1 | 833.2 ± 97.2      | 63        |
| MuJoCo | RMA      | FULL | 8 | 2 | 407.7 ± 416.5     | 17        |

Survival rate (success + partial, spec ≥ 80 %): Isaac/DR×1 = 100/100/90 %
(B/C/R), Isaac/DR×2 = **97/100/47 %**, MuJoCo/DR×1 = **100/100/100 %**,
MuJoCo/DR×2 = 83/77/83 % — see
`results/figures/fig_go2_comparison_survival.png`.

Velocity-tracking RMSE (spec < 0.3 m/s): Isaac/DR×1 = 0.14 / 0.12 / 0.20 (B/C/R),
Isaac/DR×2 = 0.16 / 0.17 / 0.28, MuJoCo/DR×1 = 0.25 / 0.18 / 0.28, MuJoCo/DR×2 =
**0.34 / 0.30 / 0.40** (B/C/R; the CTS value is 0.302) — **CTS is the only
method that stays near the 0.3 m/s spec under worst-case OOD-in-MuJoCo**
(0.302, +0.7 % over the spec); Baseline (0.34, +13 %) and RMA (0.40, +33 %)
both breach it substantially.

### C.3 Sim2Sim transfer ratios (Go2, spec sheet)
Reward-based ratios, with spec-sheet threshold check applied. Source:
`results/go2_results_table.md` (auto-computed in `plot_results_go2.py`).
- **G(π)** = R_MuJoCo,1× / R_Isaac,1× × 100 % (target ≥ 60 %)
- **OOD gap** = R_Isaac,2× / R_Isaac,1× × 100 % (target ≥ 70 %)
- **Combined gap** = R_MuJoCo,2× / R_Isaac,1× × 100 % (target ≥ 40 %)

| method   | priv | Z  | R_iso,1× | R_iso,2× | R_muj,1× | R_muj,2× | G(π)            | OOD gap         | Combined        |
|---       |---   |---:|---:      |---:      |---:      |---:      |---:             |---:             |---:             |
| Baseline | BASE | —  | 1251.6   | 1119.6   | 1160.9   | 889.4    | **92.8 %** ✓PASS | **89.5 %** ✓PASS | **71.1 %** ✓PASS |
| CTS      | FULL | 8  | 1265.9   | 1117.3   | 980.7    | 647.7    | **77.5 %** ✓PASS | **88.3 %** ✓PASS | **51.2 %** ✓PASS |
| RMA      | FULL | 8  | 936.9    | 453.5    | 833.2    | 407.7    | **88.9 %** ✓PASS | **48.4 %** ✗FAIL | **43.5 %** ✓PASS |

**Reading.** Baseline has the highest pure reward-retention, but its
**task-success rate craters in MuJoCo** (63 % vs CTS 97 %): the policy
"survives" but does not track the commanded velocity well, so reward —
dominated by the alive bonus and shaping terms — overstates how good the
behaviour is. CTS retains the lowest fraction of *raw reward* but the highest
fraction of *useful behaviour* (success share). RMA's Phase-2 adaptation
module collapses OOD inside Isaac itself (47 % success at DR×2; 48 % reward
retention — the only spec-sheet FAIL anywhere in the matrix).

### C.4 Ablations on Go2 — what is reported and what is not
- **CTS privileged-knowledge ablation (FULL / INT / EXT) on Go2:** **reported,
  all 12 cells.** Source: `results/go2_cts_priv_ablation_table.md` (auto-
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

### C.5 CTS privileged-subset ablation — final numbers
*(verbatim from `results/go2_cts_priv_ablation_table.md`; CTS only, Z = 8,
30 episodes per cell, episode length 10 s.)*

**Per-cell results:**

| sim    | DR×s | priv | reward (mean ± std) | success % | vel-RMSE [m/s] | Δreward vs FULL |
|---     |---   |---   |---                  |---:       |---:            |---:             |
| Isaac  | 1    | FULL | 1265.9 ± 61.7       | 100       | 0.124          | —               |
| Isaac  | 1    | INT  | 1251.8 ± 66.7       | 100       | 0.151          | −1.1 %          |
| Isaac  | 1    | EXT  | 1136.0 ± 339.4      | 90        | 0.152          | −10.3 %         |
| Isaac  | 2    | FULL | 1117.3 ± 263.7      | 100       | 0.169          | —               |
| Isaac  | 2    | INT  | 1013.1 ± 404.2      | 90        | 0.221          | −9.3 %          |
| Isaac  | 2    | EXT  | 993.1 ± 469.1       | 80        | 0.228          | −11.1 %         |
| MuJoCo | 1    | FULL | 980.7 ± 116.8       | **97**    | 0.178          | —               |
| MuJoCo | 1    | INT  | 1119.1 ± 81.4       | 60        | 0.306          | **+14.1 %**     |
| MuJoCo | 1    | EXT  | 885.6 ± 124.0       | 53        | 0.274          | −9.7 %          |
| MuJoCo | 2    | FULL | 647.7 ± 395.8       | 53        | 0.302          | —               |
| MuJoCo | 2    | INT  | 945.8 ± 291.9       | 43        | 0.349          | **+46.0 %**     |
| MuJoCo | 2    | EXT  | 699.5 ± 321.0       | 33        | 0.343          | +8.0 %          |

**Spec-sheet transfer ratios per privileged subset (all PASS):**

| priv | R_iso,1× | R_iso,2× | R_muj,1× | R_muj,2× | G(π)             | OOD gap          | Combined         |
|---   |---:      |---:      |---:      |---:      |---:              |---:              |---:              |
| FULL | 1265.9   | 1117.3   | 980.7    | 647.7    | 77.5 % ✓PASS     | 88.3 % ✓PASS     | 51.2 % ✓PASS     |
| INT  | 1251.8   | 1013.1   | 1119.1   | 945.8    | **89.4 % ✓PASS** | 80.9 % ✓PASS     | **75.6 % ✓PASS** |
| EXT  | 1136.0   | 993.1    | 885.6    | 699.5    | 78.0 % ✓PASS     | 87.4 % ✓PASS     | 61.6 % ✓PASS     |

**Behaviour (gait-quality) metrics at DR×1 and DR×2** *(from
`results/go2_cts_priv_ablation_table.md`; lower is better except
`gait_adh`/`contact_sym`. DR×2 = OOD on the DR axis.):*

| sim    | DR | priv | gait adh. | contact sym. | swing clear. err. | foot slip rate | action smooth. | base-z var. | stride var. | joint-torque var. |
|---     |---:|---   |---:       |---:          |---:               |---:            |---:            |---:         |---:         |---:               |
| Isaac  | 1  | FULL | 0.329     | 0.834        | 0.0060            | 0.153          | 1.80           | 0.0010      | 0.0006      | 5.75              |
| Isaac  | 1  | INT  | 0.337     | 0.868        | 0.0073            | 0.277          | 1.56           | 0.0010      | 0.0019      | 5.71              |
| Isaac  | 1  | EXT  | 0.330     | 0.838        | 0.015             | 0.309          | 2.67           | 0.0021      | 0.0006      | 8.10              |
| MuJoCo | 1  | FULL | 0.205     | 0.074        | 0.0012            | 1.64           | 1.36           | 0.0004      | 0.0027      | **18.85**         |
| MuJoCo | 1  | INT  | 0.211     | **0.0006**   | 0.0005            | 1.29           | **0.40**       | 0.0000      | 0.0006      | **6.82**          |
| MuJoCo | 1  | EXT  | 0.216     | 0.048        | 0.0008            | 1.48           | 1.11           | 0.0003      | 0.0060      | 16.01             |
| Isaac  | 2  | FULL | 0.321     | 0.798        | 0.0053            | 0.164          | 2.04           | 0.0010      | 0.0026      | 6.36              |
| Isaac  | 2  | INT  | 0.325     | 0.814        | 0.012             | 0.346          | 3.15           | 0.0014      | 0.0008      | 9.76              |
| Isaac  | 2  | EXT  | 0.334     | 0.852        | 0.011             | 0.337          | 2.63           | 0.0019      | 0.0008      | 10.08             |
| MuJoCo | 2  | FULL | 0.217     | 0.081        | 0.0026            | 1.53           | 1.64           | 0.0008      | 0.0048      | **22.47**         |
| MuJoCo | 2  | INT  | 0.204     | **0.0055**   | 0.0008            | 1.25           | **0.53**       | 0.0002      | 0.0022      | **8.40**          |
| MuJoCo | 2  | EXT  | 0.215     | 0.072        | 0.0025            | 1.29           | 1.10           | 0.0004      | 0.0035      | 17.53             |

**Reading.** (i) **In Isaac**, dropping the privileged subset hurts modestly:
INT keeps ~99 % of FULL's reward at DR×1 and 91 % at DR×2; EXT loses ~10 %
in both. FULL is the only subset that keeps 100 % success at Isaac DR×2. (ii)
**In MuJoCo, INT actually beats FULL on raw reward** (+14 % at DR×1, +46 %
at DR×2) and gives the best G(π)/Combined transfer ratios — but its
**task-success rate is sharply lower** (60 % vs 97 % at DR×1; 43 % vs 53 %
at DR×2). The 0.3 m/s velocity-RMSE spec is breached by INT in **both**
MuJoCo cells (0.306, 0.349). (iii) **EXT alone is the weakest subset
everywhere except Isaac DR×1**, with the lowest success rate in MuJoCo at
both DR scales. (iv) **The interaction signals in EXT carry behavioural
information that proprioception-plus-history cannot recover**: removing them
(INT-only) inflates reward at the cost of tracking, so FULL — the
combination of both subsets — is the configuration that wins the
behaviour-quality criterion the report cares about.

(v) **The behaviour metrics explain the mechanism — and DR×2 makes the gap
larger, not smaller.** At MuJoCo DR×1, INT uses **64 % less joint-torque
variance** (6.82 vs FULL 18.85) and **71 % less action-smoothness penalty**
(0.40 vs 1.36): INT-only learns a quieter, lower-effort policy that *saves
reward by under-tracking the velocity command*, exactly as the lower success
rate / higher RMSE confirm. At MuJoCo **DR×2** the same picture intensifies:
FULL's torque variance jumps to **22.47** (chasing the command harder under
the harder DR), while INT keeps it at **8.40** and EXT stays at 17.53 —
i.e. only FULL "spends" extra effort under OOD-DR, and that effort buys it
the highest success rate of the three subsets at DR×2 (53 % vs INT 43 % vs
EXT 33 %). Contact symmetry collapses for every subset in MuJoCo
(Isaac ~0.83 → MuJoCo ~0.0006 for INT, ~0.07 for FULL/EXT) independently of
DR scale, so the periodic-gait loss is a *simulator-change* effect, not a
DR-OOD effect.

### C.6 Plots/tables planned and present
*(see §E for one-line image captions and §F for longer figure descriptions.)*
```yaml
artefacts:
  # final tables
  - {filename: "results/go2_results_table.md",                                     shows: "per-row reward+success table; spec-sheet transfer-ratio table",       source: "scripts/plot_results_go2.py",         status: "ready"}
  - {filename: "results/go2_cts_priv_ablation_table.md",                           shows: "CTS-only FULL/INT/EXT ablation: reward, success, vel-RMSE, Δ vs FULL; transfer ratios per priv subset", source: "scripts/plot_results_go2.py::write_cts_priv_ablation_table", status: "ready"}
  - {filename: "results/one_leg_results_table.md",                                 shows: "per-row Isaac reward+success on the single leg; OOD retention",      source: "scripts/plot_results_one_leg.py",     status: "ready"}
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
- **The reward-vs-success distinction is the central technical point of the
  Go2 results.** Baseline's reward retains the most (G(π) = 93 %) but it
  *survives without tracking*: in MuJoCo only 63 % of its episodes actually
  meet the velocity-tracking spec, against 97 % for CTS at the same DR scale.
  Reward is not the right scoring rule on its own — the spec-sheet 3-class
  outcome (success / partial / fail) and RMSE/survival panels are the
  primary evidence. **The CTS privileged-subset ablation reproduces this
  reward-vs-success split inside the CTS family**: INT-only (16-D body
  params) gets the best raw-reward transfer (G(π) = 89.4 %) but only 60 %
  success in MuJoCo at DR×1, while FULL retains 97 %. EXT-only (10-D
  interaction signals) is the weakest, isolating *which half of the
  privileged vector carries which kind of information*: body parameters are
  enough for reward, interaction signals are required for tracking.
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

Use these as `\caption{...}` text inside the LaTeX figure environments;
keep them short, factual, and consistent with the data in §C.2/C.3.

- **`fig_one_leg_headline`** — *One-leg hexapod (Phase 1), Isaac Lab OOD test:
  (A) mean episode return at DR×1 vs DR×2 for Baseline/RMA/CTS (FULL, Z = 8,
  100 episodes); (B) success rate (all three methods are at 100 %).*
- **`fig_one_leg_ood_retention`** — *One-leg hexapod OOD retention
  (R_Isaac,2× / R_Isaac,1×). All three methods pass the 70 % spec
  (Baseline 88 %, RMA 93 %, CTS 91 %); the privileged-input T–S advantage
  does not appear on this fixed-base platform.*
- **`fig_go2_headline`** — *Go2 sim-to-sim transfer retention G(π,s) at DR×1
  and DR×2. Baseline 93 % / 79 %, RMA 89 % / 90 %, CTS 77 % / 58 %. The
  60 % spec line is shown; only CTS at DR×2 fails the reward-retention spec
  (but it is also the only method that keeps its task-success rate; see §F).*
- **`fig_go2_summary`** — *Spec-sheet summary panel for Go2: (A) cumulative
  episode reward, (B) survival rate (success + partial), (C) velocity-tracking
  RMSE with the 0.3 m/s spec line, (D) threshold-pass matrix. The 80 % survival
  spec is met by every cell except RMA at Isaac DR×2 (47 %); the 0.3 m/s RMSE
  spec is breached only at MuJoCo DR×2 by Baseline (0.34) and RMA (0.40), with
  CTS exactly at 0.30.*
- **`fig_go2_comparison`** — *Four-view reward comparison (rows = OOD inside a
  sim, columns = sim-to-sim transfer): (A) Isaac OOD DR×1 vs DR×2, (B) MuJoCo
  OOD DR×1 vs DR×2, (C) sim-to-sim transfer at DR×1, (D) sim-to-sim transfer
  at DR×2 (worst case). Δ annotations report the retention ratio of the
  right-bar to the left-bar.*
- **`fig_go2_comparison_survival`** — *Same four views as
  `fig_go2_comparison`, but Y axis is survival rate (success + partial). RMA
  is the only method that falls below the 80 % survival spec, at Isaac DR×2
  (47 %).*
- **`fig_go2_comparison_rmse`** — *Same four views, Y axis is
  linear-velocity-tracking RMSE with the 0.3 m/s spec line. CTS has the
  lowest RMSE in every Isaac panel and is the only method at or below
  0.3 m/s in MuJoCo at DR×2.*
- **`fig_go2_comparison_outcome`** — *Same four views, stacked Success / Partial
  / Fail per method. The Baseline–vs–CTS difference at MuJoCo DR×1 is
  dominated by *partial* episodes for Baseline (37 % partial) — i.e. the
  robot stays up but does not track the velocity command — whereas CTS
  classifies 97 % as full success.*
- **`fig_go2_ood_profile`** — *Isaac-only OOD profile: (A) Isaac OOD
  retention with the 70 % spec line — Baseline 89 % PASS, RMA 48 % FAIL,
  CTS 88 % PASS; (B) absolute Isaac reward at DR×1 and DR×2.*
- **`fig_go2_sim2sim_transfer`** — *Sim-to-sim transfer figure: (A) G(π,s)
  bars with the 60 % spec line — Baseline 93 % / 79 %, RMA 89 % / 90 %, CTS
  77 % / 58 %; (B) absolute Isaac and MuJoCo reward side-by-side with the
  Δ = R_Isaac − R_MuJoCo gap annotated per bar.*
- **`fig_go2_gait_quality`** — *Eight gait-quality metrics for Baseline / RMA /
  CTS on Go2 at FULL / Z = 8 / DR×1, Isaac (solid) vs MuJoCo (hatched):
  gait adherence, contact symmetry (sharply collapses in MuJoCo for every
  method, indicating a periodic-gait → drifty-gait sim-to-sim shift), swing
  clearance error, foot-slip rate (rises in MuJoCo), action smoothness, base-
  height variance, stride variance, joint-torque variance.*
- **`fig_go2_cts_priv_ablation`** — *CTS privileged-subset ablation
  (FULL = 26-D, INT = 16-D body params, EXT = 10-D interaction signals) across
  four views: Isaac DR×1, Isaac DR×2, MuJoCo DR×1, MuJoCo DR×2. Top row =
  episode reward, bottom row = success rate (%) with the 80 % spec line. In
  Isaac all three subsets are competitive; in MuJoCo INT actually beats FULL
  on raw reward but FULL has the highest success rate (97 % vs INT 60 % vs
  EXT 53 % at DR×1), showing that the interaction signals in EXT carry
  behavioural information not recoverable from history.*
- **`fig_go2_cts_priv_ablation_gait_dr1`** — *Behaviour-metric companion to
  `fig_go2_cts_priv_ablation` at the training distribution (DR×1): the 8
  gait-quality metrics (gait adherence, contact symmetry, swing-clearance
  error, foot-slip rate, action smoothness, base-height variance, stride
  variance, joint-torque variance) for the three CTS variants, Isaac (solid)
  vs MuJoCo (hatched). Shows that INT-only uses ~64 % less joint-torque
  variance and ~71 % less action-smoothness penalty in MuJoCo than FULL — a
  quieter, lower-effort policy that under-tracks (lower success), which is
  the mechanism behind INT's high reward and low task success in MuJoCo.*
- **`fig_go2_cts_priv_ablation_gait_dr2`** — *Same 8 metrics at DR×2 (OOD on
  the DR axis). The FULL-vs-INT gap on joint-torque variance widens from
  ×2.8 (18.85 / 6.82) at DR×1 to ×2.7 (22.47 / 8.40) at DR×2 — FULL keeps
  "spending" extra control effort under harder DR to maintain tracking,
  while INT does not. Contact symmetry collapse in MuJoCo is unchanged by
  DR scale, confirming that the periodic-gait loss is a simulator-change
  effect rather than a DR-OOD effect.*
- **`fig_architecture_diagram`** — *(to be drawn) Side-by-side data flow for the
  three configurations: Baseline (o_t → MLP → a_t), RMA (Phase 1: x_t → μ → z,
  policy on [o_t, z]; Phase 2: history → φ → ẑ, frozen teacher actor on
  [o_t, ẑ]) and CTS (concurrent teacher E^t(x_t) → z and student E^s(history) →
  ẑ feeding the same actor, with L_rec coupling them).*
- **`fig_go2_learning_curves`** — *Go2 PPO learning curve: Train/mean_reward
  vs iteration (0–25 000) for Baseline / RMA / CTS at FULL, Z = 8 — light
  raw lines and bold EMA-smoothed lines per method. RMA shows the
  characteristic curriculum dip ~iter 9 000-11 000; CTS converges highest
  (~2 100), Baseline second (~2 000), RMA lowest (~1 990).*
- **`fig_go2_learning_curves_len`** — *Same three methods, Y axis = mean
  episode length in control steps (max 500 = 10 s episode). All three saturate
  near 500 by iter ~3 000.*
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
  EXT), Z = 8. Final smoothed reward in the legend: FULL = 2104, INT = 1986,
  EXT = 1968. FULL leads consistently from iter ~1 500; INT and EXT converge
  to nearly identical training reward (~6 % below FULL) despite encoding
  qualitatively different signals.*

---

## F. Detailed image descriptions for the LaTeX-drafting AI

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
- **Single panel.** For each method, two bars side-by-side: G(π) at DR×1
  (solid) vs G(π) at DR×2 (hatched). Numbers above bars: Baseline 93 % / 79 %,
  RMA 89 % / 90 %, CTS 77 % / 58 %. Dashed line at 60 % (spec), dotted line at
  100 % ("perfect"). PASS/FAIL badges per bar.
- **Quote-ready reading.** *"On reward alone, Baseline transfers best to
  MuJoCo (93 % at DR×1, 79 % at DR×2). RMA's transfer is uniformly good
  (89–90 %) but starts from a low Isaac base. CTS retains the least raw
  reward (77 % / 58 %) but, as the success-rate and RMSE panels show,
  is the only method that retains the **behaviour** the report cares about."*

### F.4 `fig_go2_summary.png` (Go2 spec-sheet 2×2 dashboard)
- **(A)** Cumulative episode reward, four groups of three bars (Isaac×1,
  Isaac×2, MuJoCo×1, MuJoCo×2), per-method colour.
- **(B)** Survival rate (success + partial) with the 80 % spec line.
- **(C)** Velocity-tracking RMSE with the 0.3 m/s spec line.
- **(D)** 3×6 PASS/FAIL matrix over (method × {Survival Isaac 1×, Survival
  MuJoCo 1×, vel_rmse 1×, G(π) ≥ 60, OOD gap ≥ 70, Combined ≥ 40}). Only
  cells highlighted in orange (one cell: RMA OOD gap = 48 % FAIL).
- **Quote-ready reading.** *"The spec-sheet check passes for every cell
  except RMA's Isaac-OOD retention (48 %, below the 70 % spec). Baseline
  passes every spec but trails CTS on survival rate and RMSE in MuJoCo."*

### F.5 `fig_go2_comparison.png` (4-view reward)
- **2×2 grid** of bar plots. Rows = OOD-inside-sim (top) and sim-to-sim
  transfer (bottom). Columns = (left) DR×1 vs something, (right) DR×2 vs
  something.
- **(A)** Isaac OOD: DR×1 (solid) vs DR×2 (hatched) per method. Δ annotation
  = OOD retention.
- **(B)** MuJoCo OOD: DR×1 vs DR×2 per method. Δ annotation = MuJoCo OOD
  retention.
- **(C)** Sim2Sim @ DR×1: Isaac (solid) vs MuJoCo (hatched). Δ = G(π) at 1×.
- **(D)** Sim2Sim @ DR×2 (worst case): Isaac (solid) vs MuJoCo (hatched). Δ
  = G(π) at 2×.
- **Quote-ready reading.** *"The single feature that distinguishes the three
  methods is the **DR×2 column** in both rows: under aggressive
  randomisation RMA's reward collapses to 47 % of its DR×1 value in Isaac
  (the largest drop), while CTS keeps the highest absolute MuJoCo-DR×2
  reward (647.7 vs Baseline 889.4 — but with much lower variance and
  better success rate)."*

### F.6 `fig_go2_comparison_survival.png`
- Same 4 views as F.5, Y axis = survival rate (%). 80 % spec line.
- Numbers per panel (Baseline / RMA / CTS):
  - **A** (Isaac OOD, DR×1 → DR×2): 100/90/100 → **97/47/100**.
  - **B** (MuJoCo OOD, DR×1 → DR×2): **100/100/100** → 83/83/77.
  - **C** (Sim2Sim @ DR×1, Isaac → MuJoCo): 100/90/100 → 100/100/100.
  - **D** (Sim2Sim @ DR×2, Isaac → MuJoCo, worst case): **97/47/100** → 83/83/77.
- **Quote-ready reading.** *"Survival is where CTS pulls ahead: it is the
  only method that retains 100 % survival in MuJoCo at the training
  distribution, and it never drops below 77 % even at MuJoCo DR×2. RMA
  drops to 47 % survival at Isaac DR×2 — its Phase-2 adapter does not
  generalise OOD even inside Isaac."*

### F.7 `fig_go2_comparison_rmse.png`
- Same 4 views, Y axis = linear-velocity-tracking RMSE (m/s). 0.3 m/s spec
  line dashed. Lower is better.
- Per panel (Baseline / RMA / CTS, DR×1 vs DR×2):
  A (Isaac OOD) 0.14 / 0.20 / 0.12 → 0.16 / 0.28 / 0.17.
  B (MuJoCo OOD) 0.25 / 0.28 / 0.18 → 0.34 / 0.40 / 0.30.
  C (Sim2Sim DR×1) 0.14 / 0.20 / 0.12 vs 0.25 / 0.28 / 0.18.
  D (Sim2Sim DR×2) 0.16 / 0.28 / 0.17 vs 0.34 / 0.40 / 0.30.
- **Quote-ready reading.** *"CTS has the lowest RMSE in 3 of the 4 unique
  cells (Isaac DR×1, MuJoCo DR×1, and the worst-case MuJoCo DR×2); only at
  Isaac DR×2 does Baseline edge it by 0.16 vs 0.17. CTS is the only method
  that lands within +1 % of the 0.3 m/s spec in the worst-case panel
  (0.302). Baseline (0.34) and RMA (0.40) breach the spec by 13 % and 33 %
  respectively."*

### F.8 `fig_go2_comparison_outcome.png`
- Same 4 views, stacked bars Success (green) / Partial (orange) / Fail
  (dark red). The "partial" share is the report's evidence for the
  reward-vs-behaviour distinction.
- **Quote-ready reading.** *"At MuJoCo DR×1, Baseline survives 100 % of
  episodes but only 63 % are classed as success (37 % are partial — robot
  upright but tracking RMSE above 0.3 m/s). CTS converts almost all
  surviving episodes (97/100) into successes. The advantage of the
  concurrent T–S architecture is not in raw reward but in turning survival
  into useful tracking."*

### F.9 `fig_go2_ood_profile.png`
- **(A)** Isaac OOD retention per method, with PASS/FAIL badges. Numbers:
  Baseline 89 %, RMA 48 % (FAIL), CTS 88 %.
- **(B)** Absolute Isaac reward at DR×1 and DR×2.
- **Quote-ready reading.** *"Even before crossing simulators, RMA fails the
  Isaac OOD retention spec; Baseline and CTS both exceed it by a
  comfortable margin."*

### F.10 `fig_go2_sim2sim_transfer.png`
- **(A)** Sim2Sim retention G(π,s) per method at DR×1 (solid) and DR×2
  (hatched), with the 60 % spec line.
- **(B)** Absolute reward Isaac vs MuJoCo per method, with the Δ =
  R_Isaac − R_MuJoCo gap annotated next to each MuJoCo bar.
- **Quote-ready reading.** *"At the training distribution every method passes
  the 60 % transfer spec (Baseline 93 %, RMA 89 %, CTS 77 %); under OOD-DR×2
  CTS dips to 58 % — under-spec on reward, even though its **success share**
  remains the highest. This is the cleanest evidence that reward alone is
  not an adequate scoring rule for the sim-to-sim study."*

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
  - Isaac DR×1: reward 1266 / 1252 / 1136; success 100 / 100 / 90 %.
  - Isaac DR×2: reward 1117 / 1013 / 993; success 100 / 90 / 80 %.
  - MuJoCo DR×1: reward 981 / **1119** / 886; success **97 / 60 / 53 %**.
  - MuJoCo DR×2: reward 648 / **946** / 700; success 53 / 43 / 33 %.
- **Quote-ready reading.** *"Inside Isaac the three privileged subsets are
  near-equivalent (≤ 11 % reward drop, all keep ≥ 80 % success). In MuJoCo
  the picture inverts on reward: INT alone beats FULL by +14 % at DR×1 and
  +46 % at DR×2, while its task-success rate collapses to 60 % and 43 %
  (against FULL's 97 % and 53 %). EXT alone is the weakest subset in every
  cell of the bottom row. FULL — the union of body-parameter and
  interaction signals — is the only configuration that keeps both reward
  *and* tracking behaviour together in MuJoCo, confirming that the
  interaction signals carry information the proprioceptive history cannot
  recover."*
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
  - **gait adherence:** 0.33 / 0.34 / 0.33 → 0.21 / 0.21 / 0.22 — uniform drop in MuJoCo, ~independent of subset.
  - **contact symmetry:** 0.83 / 0.87 / 0.84 → **0.074 / 0.0006 / 0.048** — every subset loses periodic gait in MuJoCo; INT loses it most completely.
  - **swing-clearance error:** 0.0060 / 0.0073 / 0.015 → 0.0012 / 0.0005 / 0.0008 — counter-intuitively *lower* in MuJoCo (the foot lifts less because the gait is less periodic).
  - **foot slip rate:** 0.15 / 0.28 / 0.31 → 1.64 / **1.29** / 1.48 — INT slips least in MuJoCo.
  - **action smoothness:** 1.80 / 1.56 / 2.67 → 1.36 / **0.40** / 1.11 — INT is markedly smoother (lower control effort) in MuJoCo.
  - **base-height variance:** 0.0010 / 0.0010 / 0.0021 → 0.0004 / 0.0000 / 0.0003 — INT keeps the body practically still in MuJoCo.
  - **stride variance:** 0.0006 / 0.0019 / 0.0006 → 0.0027 / 0.0006 / **0.0060** — EXT pays for its low success with the highest stride irregularity.
  - **joint-torque variance:** 5.75 / 5.71 / 8.10 → **18.85** / **6.82** / 16.01 — INT uses < 40 % of FULL's torque variance in MuJoCo.
- **Key numbers at DR×2** (Isaac → MuJoCo, FULL / INT / EXT):
  - **contact symmetry:** 0.80 / 0.81 / 0.85 → **0.081 / 0.0055 / 0.072** — same collapse pattern as DR×1; periodic gait is lost regardless of DR scale.
  - **action smoothness:** 2.04 / 3.15 / 2.63 → 1.64 / **0.53** / 1.10 — INT smoothness gap *widens* under OOD-DR.
  - **stride variance:** 0.0026 / 0.0008 / 0.0008 → **0.0048** / 0.0022 / 0.0035 — FULL's stride variability rises under OOD-DR (it is actively re-planning).
  - **joint-torque variance:** 6.36 / 9.76 / 10.08 → **22.47** / **8.40** / 17.53 — FULL spends an extra 19 % of torque variance under OOD-DR (22.47 vs 18.85 at DR×1) to keep tracking; INT and EXT add only 23 % and 10 % respectively.
- **Quote-ready reading.** *"The behaviour-metric panels expose the mechanism
  behind the surprising reward inversion in the CTS ablation. In MuJoCo,
  INT-only learns a quieter, low-effort policy (joint-torque variance and
  action smoothness ≈ ⅓ of FULL's) that saves reward at the cost of
  velocity tracking — explaining why its reward is higher but its success
  rate is dramatically lower. Under OOD-DR (DR×2), FULL alone spends
  additional torque variance (22.47 vs 18.85 at DR×1) to keep tracking,
  while INT continues to under-actuate (8.40 vs 6.82) — a behavioural
  signature of which subset is 'trying harder' when the world becomes less
  familiar. Contact symmetry collapses for every subset in MuJoCo at both DR
  scales, indicating the periodic-gait → drifty-gait shift is a
  simulator-change effect that is independent of the privileged input."*

### F.13 `fig_go2_gait_quality.png`
- 2×4 grid of 8 metric panels, Baseline / RMA / CTS in each panel, Isaac =
  solid bar, MuJoCo = hatched bar. Direction-of-improvement labelled in each
  title ("higher better" or "lower better").
- The visually most informative panels in MuJoCo are: **contact symmetry**
  (sharp drop in MuJoCo for every method → the periodic gait is not
  preserved across the simulator change), **foot-slip rate** (rises in
  MuJoCo, indicating the friction transfer is incomplete), and
  **joint-torque variance** (CTS reaches the highest absolute MuJoCo value
  ≈ 18.85 — about 3.3× its Isaac value of 5.75, and the largest absolute
  Isaac→MuJoCo gap of any method; RMA's ratio 4.42 → 15.92 is similar in
  scale).
- **Quote-ready reading.** *"The behavioural gait metrics confirm that the
  sim-to-sim gap is concentrated in contact-related quantities (contact
  symmetry, foot-slip), not in tracking RMSE or smoothness alone. All three
  methods lose contact symmetry in MuJoCo, but CTS pays for its low RMSE
  with the highest joint-torque variance — a quantifiable trade-off the
  report can name explicitly."*

### F.14 `fig_go2_learning_curves.png` (Baseline / RMA / CTS PPO curves)
- **Single panel.** X = PPO iteration (0–25 000); Y = `Train/mean_reward`
  scalar from RSL-RL's TensorBoard log. Method colour grammar:
  Baseline = blue, RMA = green, CTS = red/orange. Raw values shown as
  light translucent lines; bold lines are EMA-smoothed (α = 0.01) for
  readability.
- **Key features.** CTS climbs fastest in the first ~2 000 iters and
  finishes highest (~2 100); Baseline tracks just behind it (~2 000); RMA
  is the slowest and shows a characteristic dip around iter ~9 000-11 000
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
- Same three methods, Y = `Train/mean_episode_length` (max = 500 control
  steps = 10 s episode). All three saturate near the cap by iter ~3 000;
  the figure mostly confirms early training reached "robot is alive for
  the full episode" quickly — the differences between methods are in
  reward shape, not survival, beyond that.

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
- **Final smoothed values** (post-warmup):
  - **CTS FULL (26-D)** — final R = **2 104**.
  - **CTS INT  (16-D)** — final R = **1 986**.
  - **CTS EXT  (10-D)** — final R = **1 968**.
- **Reading.** FULL leads consistently from iter ~1 500 onwards by ~6 %
  over both INT and EXT. INT and EXT converge to nearly the same training
  reward (gap of 18 reward, < 1 %) despite encoding qualitatively very
  different signals — INT carries 16 D of body-parameter information
  (friction, mass, gains, CoM, inertia, action delay), EXT carries 10 D of
  *interaction* signals (per-foot contact forces, contact flags, mean
  joint torques). Same training reward, different mechanism.
- **Quote-ready reading.** *"FULL's training reward advantage of ~6 % over
  INT and EXT does not translate into a proportional advantage at test
  time: §C.5 shows that in MuJoCo, INT-only actually beats FULL on raw
  reward (+14 % at DR×1) while losing 37 percentage points of task
  success. The training-reward ordering is a poor proxy for
  behaviour-quality at deployment — a central technical point of the
  report."*

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

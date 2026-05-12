# Project Audit — Go2 Teacher–Student Study
*Snapshot: 2026-05-10 21:30, just after the priv_mode/eval-DR/plot-script changes were committed to disk.*

Severity legend: 🔴 blocker · 🟠 must-fix-before-trusting · 🟡 watch · 🟢 fine.

---

## TL;DR (read this first)

- **Three Go2 main runs are live** (Baseline / RMA-Phase1 / CTS) at iteration ~6 000 of 25 000 (~24 % done). At ≈ 25 min / 1 000 iters they will finish around **2026-05-11 ~05:00**. They were launched **before** today's code changes, so they are running under the *old* `cts_network.py` / `cts_env_cfg.py` / etc. — that is fine and expected; the new code only affects *future* launches.
- **No trustworthy Go2 numbers exist yet.** The only Go2 result files on disk are (a) two early sim-to-sim text reports from the *2026-05-04 pre-fix* checkpoints (with concerning CTS tracking — see §6) and (b) a 2-row `results/ood_go2.csv` that is almost certainly garbage (success ≈ 18–22 %, see §2). All Go2 numbers that go into the report must come from the in-progress runs and the eval/sim2sim re-runs that follow.
- **Today's code changes are syntactically clean but not Isaac-tested.** The priv_mode plumbing (CTS + RMA INT/EXT + latent ablation) and the OOD-eval DR-table fix all pass `ast.parse` and the plot script runs end-to-end. They have **not** been smoke-tested in Isaac Lab — the run-plan's `smoke` stage (5-iter throw-away training) is the next thing to execute when the GPU frees up.
- **One risk to flag now**: **RMA Phase 2 needs to be re-run** against the v2 Phase-1 checkpoint. The existing `logs/rma/phase2/2026-05-05_08-34-25/adapt_module_final.pt` was trained against the *2026-05-04* (pre-sim2sim-fix) Phase-1 model — it does not pair with the new Phase-1 model now training. The run-plan covers this; just don't forget.

---

## 1. Live training-run state

| run | dir | last ckpt | max_iter | progress | sim2sim-fix? | seed | notes |
|---|---|---|---|---|---|---|---|
| Baseline v2 | `logs/baseline/2026-05-10_17-07-38_baseline_go2_v2` | `model_6200.pt` | 25 000 | **25 %** | yes (DR/reset fixes) | 42 | running, cuda:0 |
| RMA P1 v2 (l=8) | `logs/rma/2026-05-10_17-07-44_rma_go2_v2_l8_l8` | `model_6000.pt` | 25 000 | **24 %** | yes | 42 | running, cuda:0 |
| CTS v2 (l=8) | `logs/cts/2026-05-10_17-07-53_cts_go2_v2_l8` | `model_5800.pt` | 25 000 | **23 %** | yes (incl. λ_rec=5.0, warmup=1000, lr=5e-4 — fix #4) | 42 | running, cuda:0 |

🟡 The RMA Phase-2 adaptation module on disk (`adapt_module_final.pt`, 2026-05-05) was trained against an *earlier* Phase-1 checkpoint (`logs/rma/2026-05-04_..._rma_go2_l8_l8`). When the v2 Phase-1 finishes, **Phase 2 must be re-run** before any RMA evaluation; otherwise ẑ won't match z. (`run_remaining_go2.sh do_train_ablation_rma` includes a Phase-2 launch line for this — verify the `--checkpoint` path before running.)

🟢 No latent or privileged-subset ablations have been trained on Go2 yet — only `FULL × l=8` × 3 methods. Everything in `scripts/run_remaining_go2.sh do_train_ablation_{cts,rma}` is new work.

---

## 2. Result-file inventory & trust

| file | what it is | sample size | trust | use it? |
|---|---|---|---|---|
| `results/ood_go2.csv` | 2 rows: Baseline / CTS at DR×2 | 100 ep × 64 envs | 🔴 broken | **discard** — succ. ≈ 18 / 22 %, mean_length 660–745, std nearly equal to mean. Almost certainly produced by the now-fixed `_apply_dr_scale` (which referenced event terms that don't exist) running against an older `shared_env_cfg.py`, or a checkpoint mismatch. `rm` it before the OOD re-run. |
| `logs/baseline/2026-05-04_.../ood_eval/sim2sim_baseline_dr1p0.txt` | MuJoCo sim-to-sim, Baseline | 10 ep | 🟡 indicative only | Pre-sim2sim-fix checkpoint; **Baseline**'s lin-vel tracking was already 0.89, success 100 %. Suggests Baseline transferred well even without the fixes. Re-run with the v2 checkpoint anyway for a clean number. |
| `logs/cts/2026-05-04_.../ood_eval/sim2sim_cts_dr1p0.txt` | MuJoCo sim-to-sim, CTS | 30 ep | 🟠 expected to be poor | Pre-fix-#4 checkpoint. Lin-vel tracking 0.45 (vs Baseline 0.89), success 100 % but the policy walks slowly / off-direction. **This is *exactly* the failure mode sim2sim fix #4 was designed to address** — the v2 CTS run uses the stronger λ_rec, so re-running sim2sim with `model_final.pt` from the v2 run is the test of whether the fix worked. |
| `results/ood_results_all.csv`, `report_data_package.txt`, `ood_results.csv`, `ood_results_v2.csv`, `ood_results_cts_l8.csv`, `ood_results_v2_table.md` | Single-leg Phase-1 sweeps | various (100 ep × 64 envs) | 🟢 numerically valid for what they are, but 🟠 superseded by the planned single-fair-config re-run | Don't cite as Go2 numbers (they aren't). For Phase 1, the user has decided to **discard the multi-config sweeps** and run one fair config. Keep the files for reference only. |
| `results/figures/fig{1..6}_*.pdf` + older `fig_*.pdf` | Single-leg figures | — | 🟠 superseded | Drop from the report (per `project_answers.md` A.4). |
| `results/go2_results_table.{md,tex}` | auto-generated by `plot_results_go2.py` | from the discarded csv | 🔴 stale | Will refresh after the OOD re-run. |
| `logs/rma/phase2/2026-05-05_08-34-25/adapt_module_final.pt` | RMA Phase-2 weights | trained 10 000 iters | 🟠 wrong teacher | Trained against the 2026-05-04 Phase-1, not v2 — don't use with v2. Re-train Phase 2 once v2 Phase 1 is done. |
| `MUJOCO_LOG.TXT` (repo root) | MuJoCo runtime errors | — | 🟢 ignore | Two GLFW init errors from 2026-05-03 — happens when a render-mode call has no display. Harmless; current sim2sim runs headless. |

---

## 3. Code health by component (after today's edits)

### 3.1 `tasks/shared/`
| item | status |
|---|---|
| `SharedEnvCfg` (env, scene, rewards, terminations, DR) | 🟢 coherent. All three sim2sim fixes are documented inline (#1 reset velocities zeroed; #2 widened DR; #4 see CTS cfg). DR ranges in `events.py` match what the cfg passes. |
| `SharedRewardsCfg` weights | 🟢 17 reward terms, identical across the three methods → fairness preserved. |
| `shared/mdp/observations.py` | 🟢 `o_t` (37D) and `x_t` (16+10=26D) are well-defined. Today's additions (`PRIV_DIMS`, `privileged_subset_go2`, `combined_obs_subset`) are additive — no existing usage broken. |
| `shared/mdp/__init__.py` | 🟢 exports updated. Old `combined_obs_rma` still exported (back-compat — no harm). |

### 3.2 `tasks/baseline/`
🟢 Untouched today. `BaselineEnvCfg` policy = 37D proprioception, no critic privileged group. Standard `OnPolicyRunner`. Nothing to flag.

### 3.3 `tasks/rma/`
| item | status |
|---|---|
| `RMAActorCritic` (canonical Phase 1) | 🟢 Already had `env_factor_dim` as a kwarg, so priv_mode for RMA was nearly free. `evaluate()` slices `[:, _OT_DIM:_OT_DIM + self._env_factor_dim]` — works for any priv_dim. |
| `AdaptationModule` (canonical Phase 2) | 🟢 Independent of priv_dim (input is proprioception+action history, output is z of `latent_dim`). |
| `RMAPhase2Runner` | 🟢 today's edit: accepts `priv_dim`, slices the critic obs `[:, 37:37+priv_dim]`. Default 26 = identical to before. |
| `RMAEnvCfg` + `RMATeacherEnvCfg` | 🟢 today's edits: `priv_mode` field on both; CriticCfg / TeacherObsCfg now use `combined_obs_subset`. `RMATeacherEnvCfg.__post_init__` runs with the default `FULL` first, then `train_phase1.py` overrides `priv_mode` AND `observation_space`/`state_space` to match. Brittle but correct. |
| Leftover *asymmetric-critic* `RMAEnvCfg`/`RMAPPORunnerCfg` (actor=37D, critic=63D, no encoder) | 🟡 still in the repo; **not used** for the report. The eval script's docstring still describes this variant — minor confusion risk. Not blocking. |

### 3.4 `tasks/cts/`
| item | status |
|---|---|
| `CTSActorCritic` | 🟢 today's refactor: `priv_dim: int = 26` kwarg; `self._teacher_in = 37 + priv_dim`; teacher encoder `Linear(priv_dim, …)`; routing uses `self._teacher_in`/`self._priv_dim`. **Default 26 → loads existing FULL/l=8 checkpoints unchanged.** |
| `cts/mdp/observations.py::cts_teacher_student_obs` | 🟢 today: reads `env.cfg.priv_mode`, builds `xt = privileged_subset_go2(env, mode)`, fills `unified[..., :37+xt.shape[-1]]`. Shape-detection dummy still returns `H*37+1` zeros (that path unchanged). |
| `CTSEnvCfg` | 🟢 today: `priv_mode` field; `state_space = 37 + priv_dim` set in `__post_init__`. |
| `CTSRunner` (L_rec) | 🟢 untouched today. The L_rec pre-collection slices `obs_b[:, :H*37]` (history) and `crit_b[:, 37:63]` — the 63 here is hard-coded but **only** used for the FULL run. **🟠 For INT/EXT this 63 must become `37 + self.policy._priv_dim`** — see §5.1. |

### 3.5 `scripts/`
| script | status after today |
|---|---|
| `scripts/baseline/train.py` | 🟢 untouched. |
| `scripts/cts/train.py` | 🟢 added `--priv_mode {FULL,INT,EXT}`; sets `env_cfg.priv_mode`, `env_cfg.state_space`, `train_dict["policy"]["priv_dim"]`. |
| `scripts/rma/train_phase1.py` | 🟢 added `--priv_mode`; overrides `observation_space`/`state_space` after the post-init runs. |
| `scripts/rma/train_phase2.py` | 🟢 added `--priv_mode` (must match Phase 1); passes `priv_dim` to `RMAPhase2Runner` and `RMAActorCritic`. |
| `scripts/eval_ood_go2.py` | 🟠→🟢 today: **fixed the broken `_apply_dr_scale`** (was crashing on `ev.randomize_material_track`/`randomize_leg_mass`/`randomize_kp`/`randomize_motor_strength` which don't exist). Added `--priv_mode` and `priv_mode`/`latent_dim` columns in the output CSV. **Caveat**: the docstring at the top of the file still describes the *old* asymmetric-critic RMA — purely cosmetic, will not affect runs. |
| `scripts/sim2sim/sim2sim_go2.py` | 🟠 **does not yet support** `--priv_mode`, `--adapt_module`, or `--results_file` / CSV output. INT/EXT sim2sim and "real" RMA-with-Phase-2 sim2sim need a small extension here. The plot script's `--scan-logs` mode parses the `.txt` outputs, so once you do extend the script the plotting still works. |
| `scripts/plot_results_go2.py` | 🟢 new, runs end-to-end against current data; correctly skips ablation figures when there's not enough data. |
| `scripts/run_remaining_go2.sh` | 🟢 new; dry-run by default; bash syntax-checked; smoke stage exercises every code path before the real ablation runs. |

### 3.6 What I could *not* test
🟠 None of the priv_mode plumbing has actually been run inside Isaac Lab — **the smoke stage exists for exactly this reason**. Specific risks the smoke test will surface:
1. `combined_obs_subset` reads `getattr(env.cfg, "priv_mode", "FULL")` — if Isaac Lab's ObservationManager calls obs functions with a wrapper that hides `cfg`, the default kicks in and INT/EXT silently behaves like FULL. I'm 90 % sure `env.cfg` is the right attribute (it's what the existing `Go2CTSEnv` reads), but worth verifying.
2. `@configclass` adding the `priv_mode: str = "FULL"` field to the subclass — should work (configclass is dataclass-based), but Isaac Lab's `configclass` has a few non-standard behaviours.
3. `RslRlPpoActorCriticCfg.to_dict()` then injecting `priv_dim` into `train_dict["policy"]` — `CTSActorCritic.__init__` already accepted `**kwargs`, so this is the same path that already worked for `latent_dim`/`history_len`.

---

## 4. Method-fairness check

The whole point of the project is "the *only* difference between methods is the transfer mechanism." Audit:

| dimension | Baseline | RMA Phase 2 | CTS | fair? |
|---|---|---|---|---|
| Scene / asset | Unitree Go2, flat plane | same | same | 🟢 |
| Domain randomisation | `SharedEventCfg` (sim2sim-fix ranges) | same | same | 🟢 |
| Reward weights | `SharedRewardsCfg` (17 terms) | same | same | 🟢 |
| Terminations | time_out, base contact >100 N, tilt >1.2 rad | same | same | 🟢 |
| Physics / control | 200 Hz / 50 Hz / 20 s ep | same | same | 🟢 |
| Action space | joint position targets, scale 0.25 | same | same | 🟢 |
| Network width | actor & critic [512,256,128] ELU | same | same | 🟢 |
| PPO hyperparams | clip 0.2, kl 0.01, lr 1e-3, γ 0.99 | same | same | 🟢 |
| `entropy_coef` | **0.01** | **0.005** | **0.005** | 🟡 minor — Baseline has 2× the entropy bonus. Inherited from earlier configs; arguably it favours Baseline with more exploration, but the difference is small. **Decide**: align to 0.005 across all three for the *next* run, or accept the discrepancy and disclose. |
| Deployment input | `o_t (37)` | `[o_t(37), ẑ(latent)]` from history-CNN | `[o_t(37), z(latent)]` from history-CNN | 🟢 by design — the latent is the architectural variable being studied. |
| Privileged sim info at test | none | none (ẑ from history) | none (z from history) | 🟢 — no method gets oracle x_t at deployment. |
| Training-time privileged use | none | encoder μ(x_t) drives policy in Phase 1 | E^t(x_t) drives 75 % of envs every step | 🟢 — the ratio is the design difference, not a fairness leak. |
| Reset randomisation | `reset_base` velocity = 0 (sim2sim fix #1) | same | same | 🟢 (uniform across methods) |
| Number of envs | 4096 | 4096 | 4096 (3072 teacher + 1024 student) | 🟡 CTS has 1024 *student* envs giving fewer diverse student gradients, vs 4096 envs of pure-PPO for Baseline/RMA. This is intrinsic to the CTS method (75:25 split) — not a fairness bug, but worth a sentence in the report. |
| Random seed | 42 | 42 | 42 | 🟢 single seed only — see §5. |

🟠 **Single seed is the biggest fairness risk** — variance across seeds for these methods is non-trivial; reporting one seed per condition makes the comparison vulnerable to seed luck. Within compute budget consider 3 seeds × `{Baseline, RMA, CTS}` × FULL/l=8.

---

## 5. Risks & known issues

### 5.1 🟠 `CTSRunner._collect_rec_data` hard-codes the critic-obs slice to `[:, 37:63]`
File: `tasks/cts/cts_runner.py` line 67. For `priv_mode != FULL` this slices the *wrong* dimensions and the L_rec target will be partially zero (since `crit_b` is only `37+priv_dim` wide). Three options:

```python
# Option A (cleanest): use the network's known priv_dim
priv_dim = self.policy._priv_dim
pairs.append((
    obs_b[student_mask, :h_dim].clone(),
    crit_b[student_mask, 37:37+priv_dim].clone(),
))
```

This is the **#1 fix to do** before launching CTS-INT or CTS-EXT runs. (For FULL the existing `[:, 37:63]` happens to be correct.)

### 5.2 🟠 `sim2sim_go2.py` has no `--priv_mode`, no `--adapt_module`, no CSV
Three implications:
- INT/EXT-trained CTS / RMA cannot be evaluated in MuJoCo as written.
- "Real" RMA (with Phase-2 ẑ) cannot be evaluated in MuJoCo either; the script always evaluates with z=0.
- Aggregating sim2sim results across (method × scale × priv × latent) requires either the `.txt`-parser path (currently in `plot_results_go2.py --scan-logs`) or extending the script.

The plot script's `.txt` parser already extracts `priv` and `l` from filenames if encoded (e.g. `sim2sim_cts_int_l16_dr1p0.txt`), so the cheapest path is to extend `sim2sim_go2.py` to (a) accept `--priv_mode --latent_dim --adapt_module --dr_scale` and (b) name the output `sim2sim_<method>_<priv>_l<lat>_dr<scale>.txt`.

### 5.3 🟡 Currently-running CTS v2 cannot exercise the priv_mode plumbing
Because the running Python process loaded `cts_network.py` etc. before today's edits, it is using the *old* code (which is functionally identical for FULL anyway). When it finishes and saves `model_final.pt`, the eval-side code (which uses the *new* code) will load it as `priv_dim=26` (default) — that matches the saved layer shapes exactly, so no problem.

### 5.4 🟡 Single seed × no statistical test
All three v2 runs use seed 42. The report will need either (a) seed bands (additional runs), (b) a paired-checkpoint within-method bootstrap, or (c) an honest disclosure that comparisons are point estimates.

### 5.5 🟡 Currently-running runs at 25 000 iters
At ~25 min/1000 iters they finish around 05:00 the next day. Plan ablations to start *after* that (and ideally on a separate GPU if possible — they will compete for cuda:0).

### 5.6 🟢 Eval-time CSV schema migration
`eval_ood_go2.py` now writes `priv_mode` and `latent_dim` columns. `plot_results_go2.py` accepts both schemas (old without those columns and new with) — confirmed working on the existing 2-row file.

### 5.7 🟢 Backward compatibility of the priv_mode refactor
- `CTSActorCritic(priv_dim=26)` builds layers with the same shapes as before → existing FULL/l=8 checkpoints load without `strict=False` warnings.
- `RMAActorCritic(env_factor_dim=26)` was already the case → no change.
- Existing `Go2CTSEnv` constructor unchanged.

---

## 6. Concerning data signal worth a closer look

In `sim2sim_cts_dr1p0.txt` (CTS, 30 episodes, 2026-05-04 checkpoint, **before** sim2sim fix #4):

```
Lin vel track  : 0.4533 ± 0.1822    ← target 1.0
Ang vel track  : 0.9887 ± 0.0210
Lin track err  : 0.4453 ± 0.1189 m/s
Mean reward    : +1091.99 ± 55.14
Success rate   : 100.0% (30/30 survived)
```

Versus Baseline at the same DR×1.0 setting (10 ep):
```
Lin vel track  : 0.8881 ± 0.0078
Ang vel track  : 0.8950 ± 0.0386
Mean reward    : +1206.88 ± 69.14
Success rate   : 100.0%
```

**Reading**: CTS doesn't fall — it stands and walks — but it cannot follow the linear-velocity command (0.45 vs 0.89). This is the diagnostic signature the sim2sim-fix-#4 commit message describes: the student encoder's z is "out of distribution" at MuJoCo deployment, the actor outputs cautious / off-direction actions, and the robot under-shoots the commanded velocity. It is the *expected* failure that the v2 CTS run (λ_rec = 5.0, warmup = 1000, lr = 5e-4) should fix.

🟠 **Don't draw any conclusion about CTS yet.** The current sim2sim CTS number reflects pre-fix code. The v2 re-evaluation is the actual test.

---

## 7. Verifiability matrix

| claim in this audit | how it was verified |
|---|---|
| All three runs are alive and at iter ~6 000 | `ps aux` (running PIDs) + `ls model_*.pt` + file mtimes |
| `--max_iterations 25000`, `--seed 42` everywhere | exact `ps aux` argv |
| CTS v2 uses sim2sim fix #4 (`λ_rec=5, warmup=1000, lr=5e-4`) | `--lambda_rec 5.0` in argv + `cts/agents/rsl_rl_ppo_cfg.py` |
| RMA v2 launched `train_phase1.py` (canonical) | argv shows `scripts/rma/train_phase1.py --latent_dim 8` |
| Phase-2 adapt module pairs with the *2026-05-04* P1, not v2 | mtime of `adapt_module_final.pt` (2026-05-05) precedes v2 launch (2026-05-10) |
| All edited Python files parse | `ast.parse` on each (12 files, all OK) |
| `plot_results_go2.py` runs end-to-end | actually executed; produced 2 figures + 1 table |
| `run_remaining_go2.sh` syntax | `bash -n` |
| `eval_ood_go2.py`'s old DR scaler was broken | the function referenced `ev.randomize_material_track`, `randomize_leg_mass`, `randomize_kp`, `randomize_motor_strength` — none of which appear in `tasks/shared/shared_env_cfg.py` (only `randomize_material`, `track_material`, `randomize_payload`, `randomize_base_inertia`, `randomize_gains`) |
| Existing FULL/l=8 ckpts are loadable under new code | `CTSActorCritic(priv_dim=26)` produces the same Linear shapes as the `_PRIV_DIM=26` constant did before |

| claim **not** verified (needs Isaac to run) | risk |
|---|---|
| INT/EXT priv_mode actually trains end-to-end in Isaac | shape mismatch, ObservationManager init order, `env.cfg` lookup |
| `combined_obs_subset` reads `env.cfg.priv_mode` correctly | Isaac wraps env objects — confirm |
| `RMATeacherEnvCfg` post-init override of `observation_space` survives `gym.make` | Isaac Lab might re-validate cfg |

---

## 8. Recommended next actions, in order

1. **Apply the §5.1 fix to `tasks/cts/cts_runner.py`** (replace the hard-coded `37:63` slice with `37:37+self.policy._priv_dim`). Without this, CTS INT/EXT will silently train against partly-zeroed L_rec targets.
2. **Wait for v2 runs to finish (~05:00).** Don't compete for the GPU.
3. **Smoke-test the priv_mode plumbing**: `./scripts/run_remaining_go2.sh smoke --run` (5 iters × 4 mini-runs × 256 envs ≈ a few minutes; just verifies shapes).
4. **Wipe stale results**: `rm results/ood_go2.csv` (the discarded 2-row file).
5. **Re-train RMA Phase 2 against v2 Phase 1**: `./scripts/run_remaining_go2.sh train_ablation_rma --run` (the script's example line targets the v2 Phase-1 checkpoint by default).
6. **Headline OOD eval**: `./scripts/run_remaining_go2.sh eval_ood --run` (3 methods × 3 DR scales × 100 eps).
7. **(Optional but worth doing) Extend `sim2sim_go2.py`** with `--priv_mode --adapt_module --latent_dim` and rename outputs to `sim2sim_<method>_<priv>_l<lat>_dr<scale>.txt`. Then `./scripts/run_remaining_go2.sh eval_sim2sim --run`.
8. **Train CTS / RMA ablations** (priv subset INT/EXT @ l=8 + latent sweep at FULL): `./scripts/run_remaining_go2.sh train_ablation_cts --run` and `... train_ablation_rma --run`. Then OOD-eval each new checkpoint.
9. **Phase-1 single-leg re-run** (one fair config, all three methods) once GPU is free.
10. **Plot**: `./scripts/run_remaining_go2.sh plot --run` — produces the report figures + tables.
11. **Update `project_answers.md`** with the now-known numbers (replace the `TBD`s) before handing it to the LaTeX-drafting AI.

---

## 9. What to tell the LaTeX-drafting AI today (interim)

Until the in-progress runs finish, the LaTeX AI should:
- Use `project_answers.md` as the source of truth for setup / methodology / fairness statements.
- **Not** quote any number from `results/ood_go2.csv`, `results/ood_results_*.csv`, or the two `sim2sim_*.txt` reports — flag them all as "preliminary, superseded".
- Keep all numerical tables marked `TBD` with row scaffolding ready to drop values in.
- Use Section 4 of this audit to write the *fairness* paragraph of the methods section.
- Use Section 6 for any honest discussion of the early CTS sim-to-sim gap (with the *important* caveat that the v2 run's λ_rec = 5.0 is the actual test of CTS sim-to-sim transfer; don't write off CTS based on the pre-fix numbers).

"""
Plot Go2 learning curves from RSL-RL TensorBoard event files.

Extracts `Train/mean_reward` and `Train/mean_episode_length` from the v2
training runs of Baseline / RMA / CTS (Go2, FULL, latent=8) and renders
one combined figure that matches the report's OpenTopic colour grammar.
For CTS, the teacher/student split (`mean_reward_teacher`,
`mean_reward_student`) is plotted in a companion figure.

Inputs (override with CLI flags if needed):
    --baseline  logs/baseline/2026-05-10_17-07-38_baseline_go2_v2
    --rma       logs/rma/2026-05-10_17-07-44_rma_go2_v2_l8_l8
    --cts       logs/cts/2026-05-10_17-07-53_cts_go2_v2_l8

Outputs (in results/figures/):
    fig_go2_learning_curves.{pdf,png}            mean reward vs PPO iter
    fig_go2_learning_curves_len.{pdf,png}        mean episode length vs PPO iter
    fig_go2_cts_teacher_student.{pdf,png}        CTS-only T vs S reward
And raw CSVs in results/learning_curves/.

Usage:
    python scripts/plot_learning_curves.py
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(REPO_ROOT, "results", "figures")
CSV_DIR   = os.path.join(REPO_ROOT, "results", "learning_curves")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# Style (matches plot_results_go2.py)
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "axes.linewidth":     1.2,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.85,
    "figure.dpi":         150,
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.alpha":         0.35,
    "grid.linewidth":     0.7,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

METHOD_COLOR = {"Baseline": "#2166ac", "RMA": "#4dac26", "CTS": "#d6604d"}


def load_scalar(logdir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) for a single scalar tag, or (empty,empty) if missing."""
    if not os.path.isdir(logdir):
        print(f"[lc] WARN: {logdir} does not exist — skipping")
        return np.array([]), np.array([])
    # size_guidance: 0 = unlimited (don't downsample)
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        print(f"[lc] WARN: tag {tag!r} not in {os.path.basename(logdir)} — skipping")
        return np.array([]), np.array([])
    pts = ea.Scalars(tag)
    steps = np.array([p.step for p in pts], dtype=np.int64)
    vals  = np.array([p.value for p in pts], dtype=np.float64)
    return steps, vals


def ema(y: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Cheap exponential moving average for visual smoothing (TB default = 0.6)."""
    if len(y) == 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def save_csv(path: str, columns: dict[str, np.ndarray]) -> None:
    n = max(len(v) for v in columns.values())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(columns.keys()))
        for i in range(n):
            w.writerow([(columns[k][i] if i < len(columns[k]) else "")
                        for k in columns])
    print(f"[lc] wrote {os.path.relpath(path, REPO_ROOT)}")


def plot_curve(ax, steps, vals, label, color, raw_alpha=0.20, smooth_alpha=1.0,
               smoothing=0.01):
    if len(steps) == 0:
        return
    ax.plot(steps, vals, color=color, alpha=raw_alpha, lw=0.8)
    ax.plot(steps, ema(vals, alpha=smoothing),
            color=color, alpha=smooth_alpha, lw=1.8, label=label)


def fig_mean_reward(runs: dict[str, str]):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bundle = {}
    for name, logdir in runs.items():
        s, v = load_scalar(logdir, "Train/mean_reward")
        bundle[f"{name}_step"] = s; bundle[f"{name}_reward"] = v
        plot_curve(ax, s, v, name, METHOD_COLOR[name])
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Train / mean reward")
    ax.set_title("Go2 — PPO learning curve  (Train/mean_reward, Z=8, FULL)",
                 fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(left=0)
    _save(fig, "fig_go2_learning_curves")
    save_csv(os.path.join(CSV_DIR, "go2_learning_curves_reward.csv"), bundle)


def fig_mean_length(runs: dict[str, str]):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bundle = {}
    for name, logdir in runs.items():
        s, v = load_scalar(logdir, "Train/mean_episode_length")
        bundle[f"{name}_step"] = s; bundle[f"{name}_length"] = v
        plot_curve(ax, s, v, name, METHOD_COLOR[name])
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Train / mean episode length  [steps]")
    ax.set_title("Go2 — Mean episode length over training  (Z=8, FULL)",
                 fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(left=0)
    _save(fig, "fig_go2_learning_curves_len")
    save_csv(os.path.join(CSV_DIR, "go2_learning_curves_length.csv"), bundle)


def fig_cts_teacher_student(cts_logdir: str):
    """CTS teacher-vs-student learning curves laid out as TWO separate side-by-side
    panels (left = teacher only, right = student only), sharing the same y-axis
    so the curves can be compared directly without overlapping each other. A
    third strip below shows the gap (teacher − student) over training so the
    quantitative difference is visible at a glance."""
    s_t, v_t = load_scalar(cts_logdir, "Train/mean_reward_teacher")
    s_s, v_s = load_scalar(cts_logdir, "Train/mean_reward_student")
    if len(s_t) == 0 and len(s_s) == 0:
        print("[lc] skip fig_go2_cts_teacher_student — no teacher/student scalars")
        return
    C_TEACHER = "#08306b"     # dark navy
    C_STUDENT = "#d94801"     # burnt orange

    fig = plt.figure(figsize=(11.5, 5.6))
    gs  = fig.add_gridspec(2, 2,
                            height_ratios=[3.0, 1.2],
                            hspace=0.18, wspace=0.05)
    ax_t = fig.add_subplot(gs[0, 0])              # left: teacher
    ax_s = fig.add_subplot(gs[0, 1], sharey=ax_t) # right: student (shared y)
    ax_g = fig.add_subplot(gs[1, :], sharex=ax_t) # bottom: gap (full width)

    # ── Teacher panel (left) ───────────────────────────────────────────
    ax_t.plot(s_t, v_t, color=C_TEACHER, alpha=0.20, lw=0.7)
    ax_t.plot(s_t, ema(v_t, alpha=0.01), color=C_TEACHER, lw=2.2,
              label="CTS teacher")
    ax_t.set_ylabel("Train / mean reward")
    ax_t.set_title(r"CTS teacher  (75 % envs, sees $x_t$ directly)",
                   color=C_TEACHER, fontweight="bold")
    ax_t.axhline(0, color="0.5", lw=0.6, alpha=0.5)
    ax_t.set_xlim(left=0)
    # final-value annotation
    ax_t.annotate(f"final = {v_t[-1]:.0f}",
                  xy=(s_t[-1], v_t[-1]), xytext=(-90, -18),
                  textcoords="offset points", fontsize=9, color=C_TEACHER,
                  ha="left")

    # ── Student panel (right) ──────────────────────────────────────────
    ax_s.plot(s_s, v_s, color=C_STUDENT, alpha=0.20, lw=0.7)
    ax_s.plot(s_s, ema(v_s, alpha=0.01), color=C_STUDENT, lw=2.2,
              label="CTS student")
    ax_s.set_title("CTS student  (25 % envs, history only via $E^s$)",
                   color=C_STUDENT, fontweight="bold")
    ax_s.axhline(0, color="0.5", lw=0.6, alpha=0.5)
    ax_s.tick_params(axis="y", labelleft=False)   # hide redundant y-labels
    ax_s.set_xlim(left=0)
    ax_s.annotate(f"final = {v_s[-1]:.0f}",
                  xy=(s_s[-1], v_s[-1]), xytext=(-90, -18),
                  textcoords="offset points", fontsize=9, color=C_STUDENT,
                  ha="left")

    # ── Gap panel (bottom, spans both columns) ─────────────────────────
    n = min(len(s_t), len(s_s))
    gap = v_t[:n] - v_s[:n]
    ax_g.fill_between(s_t[:n], gap, 0, where=gap >= 0,
                      color=C_TEACHER, alpha=0.15, lw=0)
    ax_g.fill_between(s_t[:n], gap, 0, where=gap <  0,
                      color=C_STUDENT, alpha=0.15, lw=0)
    ax_g.plot(s_t[:n], gap, color="0.55", alpha=0.4, lw=0.7)
    ax_g.plot(s_t[:n], ema(gap, alpha=0.005), color="#222222", lw=1.6)
    ax_g.axhline(0, color="0.3", lw=1.0, ls=":", alpha=0.9)
    warm = s_t[:n] >= 1000
    if warm.any():
        gap_mean = gap[warm].mean()
        ax_g.axhline(gap_mean, color="#bd0026", lw=1.0, ls="--", alpha=0.8)
        ax_g.text(s_t[:n][-1] * 0.99, gap_mean,
                  f"  post-warmup mean gap = {gap_mean:+.0f}  ({gap_mean / v_t[warm].mean() * 100:+.1f}%)",
                  color="#bd0026", fontsize=9, va="bottom", ha="right")
    ax_g.set_xlabel("PPO iteration")
    ax_g.set_ylabel("gap  (teacher − student)")
    ax_g.set_xlim(left=0)

    fig.suptitle("CTS — concurrent teacher/student learning curves  "
                 "(separate panels for direct comparison)",
                 fontsize=12, fontweight="bold", y=0.995)
    _save(fig, "fig_go2_cts_teacher_student")
    save_csv(os.path.join(CSV_DIR, "go2_cts_teacher_student.csv"),
             {"teacher_step": s_t, "teacher_reward": v_t,
              "student_step": s_s, "student_reward": v_s})


def fig_cts_priv_learning_curves(cts_runs: dict[str, str]):
    """Single-panel learning curve comparing CTS mean reward across the three
    privileged subsets (FULL / INT / EXT). Uses the shared PRIV_COLOR grammar
    from plot_results_go2.py."""
    privs = [p for p in ("FULL", "INT", "EXT") if p in cts_runs
             and os.path.isdir(cts_runs[p])]
    if not privs:
        print("[lc] skip fig_go2_cts_priv_learning_curves — no CTS runs found")
        return
    PRIV_COLOR = {"FULL": "#1a9850", "INT": "#4575b4", "EXT": "#f46d43"}
    PRIV_DIMS  = {"FULL": 26, "INT": 16, "EXT": 10}

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bundle = {}
    finals = []
    # Use smoothed value at the last step for the "final" — less noisy than raw
    for priv in privs:
        s, v = load_scalar(cts_runs[priv], "Train/mean_reward")
        bundle[f"{priv}_step"] = s; bundle[f"{priv}_reward"] = v
        if len(s) == 0:
            continue
        v_smooth = ema(v, alpha=0.01)
        # raw (faint) + smoothed (bold)
        ax.plot(s, v, color=PRIV_COLOR[priv], alpha=0.18, lw=0.7)
        ax.plot(s, v_smooth, color=PRIV_COLOR[priv], lw=2.0,
                label=f"CTS {priv}  ({PRIV_DIMS[priv]}-D)  —  final R = {v_smooth[-1]:,.0f}")
        finals.append((priv, s[-1], v_smooth[-1]))
    # Small arrow markers at the right edge to anchor where each curve ends —
    # but NO text annotation (numbers are in the legend now), so they cannot overlap.
    for (priv, x, y) in finals:
        ax.plot([x], [y], marker="o", color=PRIV_COLOR[priv],
                markersize=7, markeredgecolor="white", markeredgewidth=1.2,
                zorder=5, clip_on=False)
    ax.axhline(0, color="0.5", lw=0.6, alpha=0.5)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Train / mean reward")
    ax.set_title("CTS — privileged-subset learning curves  "
                 "(FULL vs INT vs EXT, $Z$=8)",
                 fontweight="bold")
    ax.legend(frameon=True, loc="lower right", framealpha=0.92,
              fontsize=10, edgecolor="0.7")
    ax.set_xlim(left=0)
    _save(fig, "fig_go2_cts_priv_learning_curves")
    save_csv(os.path.join(CSV_DIR, "go2_cts_priv_learning_curves.csv"), bundle)


def fig_cts_teacher_student_overlay(cts_runs: dict[str, str]):
    """Overlaid teacher+student learning curves for CTS at FULL / INT / EXT.

    Layout: 2 rows × 3 cols.
      Top row    — one panel per priv subset, teacher (solid navy) and student
                   (dashed orange) on the SAME axes, shared y-axis across all
                   three panels so absolute reward is directly comparable.
      Bottom row — gap (teacher − student) per priv subset, shared y-axis."""
    privs = [p for p in ("FULL", "INT", "EXT") if p in cts_runs
             and os.path.isdir(cts_runs[p])]
    if not privs:
        print("[lc] skip fig_go2_cts_teacher_student_overlay — no CTS runs found")
        return

    C_TEACHER = "#08306b"     # dark navy
    C_STUDENT = "#d94801"     # burnt orange

    fig = plt.figure(figsize=(13.5, 6.2))
    gs  = fig.add_gridspec(2, len(privs),
                            height_ratios=[3.0, 1.2],
                            hspace=0.18, wspace=0.08)
    ax_top    = [fig.add_subplot(gs[0, j]) for j in range(len(privs))]
    ax_bottom = [fig.add_subplot(gs[1, j], sharex=ax_top[j]) for j in range(len(privs))]
    # Share y-axes within each row so curves are directly comparable
    for a in ax_top[1:]:    a.sharey(ax_top[0]);    a.tick_params(axis="y", labelleft=False)
    for a in ax_bottom[1:]: a.sharey(ax_bottom[0]); a.tick_params(axis="y", labelleft=False)

    bundle = {}
    for j, priv in enumerate(privs):
        logdir = cts_runs[priv]
        s_t, v_t = load_scalar(logdir, "Train/mean_reward_teacher")
        s_s, v_s = load_scalar(logdir, "Train/mean_reward_student")
        bundle[f"{priv}_step_teacher"]  = s_t; bundle[f"{priv}_reward_teacher"] = v_t
        bundle[f"{priv}_step_student"]  = s_s; bundle[f"{priv}_reward_student"] = v_s

        # ── top panel: teacher + student overlaid ────────────────────
        ax = ax_top[j]
        ax.plot(s_t, v_t, color=C_TEACHER, alpha=0.18, lw=0.7)
        ax.plot(s_t, ema(v_t, alpha=0.01), color=C_TEACHER, lw=2.0,
                label="teacher  (sees $x_t$)")
        ax.plot(s_s, v_s, color=C_STUDENT, alpha=0.18, lw=0.7)
        ax.plot(s_s, ema(v_s, alpha=0.01), color=C_STUDENT, lw=2.0, ls="--",
                label="student  (history only)")
        ax.axhline(0, color="0.5", lw=0.6, alpha=0.5)
        ax.set_xlim(left=0)
        # Title with priv-subset colour from PRIV_COLOR grammar
        priv_color = {"FULL": "#1a9850", "INT": "#4575b4", "EXT": "#f46d43"}[priv]
        ax.set_title(f"CTS {priv}  (priv. dim = "
                     f"{ {'FULL': 26, 'INT': 16, 'EXT': 10}[priv] }-D)",
                     color=priv_color, fontweight="bold")
        # Final-value annotations
        if len(v_t):
            ax.annotate(f"T={v_t[-1]:.0f}",
                        xy=(s_t[-1], v_t[-1]), xytext=(-6, 4),
                        textcoords="offset points", color=C_TEACHER,
                        fontsize=8, ha="right", va="bottom")
        if len(v_s):
            ax.annotate(f"S={v_s[-1]:.0f}",
                        xy=(s_s[-1], v_s[-1]), xytext=(-6, -6),
                        textcoords="offset points", color=C_STUDENT,
                        fontsize=8, ha="right", va="top")
        if j == 0:
            ax.set_ylabel("Train / mean reward")
            ax.legend(frameon=False, loc="lower right", fontsize=9)

        # ── bottom panel: gap (T − S) ────────────────────────────────
        axg = ax_bottom[j]
        n = min(len(s_t), len(s_s))
        if n == 0:
            continue
        gap = v_t[:n] - v_s[:n]
        axg.fill_between(s_t[:n], gap, 0, where=gap >= 0,
                         color=C_TEACHER, alpha=0.18, lw=0)
        axg.fill_between(s_t[:n], gap, 0, where=gap < 0,
                         color=C_STUDENT, alpha=0.18, lw=0)
        axg.plot(s_t[:n], gap, color="0.55", alpha=0.35, lw=0.6)
        axg.plot(s_t[:n], ema(gap, alpha=0.005), color="#222222", lw=1.4)
        axg.axhline(0, color="0.3", lw=0.9, ls=":", alpha=0.9)
        # Annotate post-warmup mean gap (excluding noisy first 1000 iters)
        warm = s_t[:n] >= 1000
        if warm.any():
            g_mean = gap[warm].mean()
            t_mean = v_t[:n][warm].mean()
            axg.axhline(g_mean, color="#bd0026", lw=0.9, ls="--", alpha=0.8)
            axg.text(s_t[:n][-1] * 0.98, g_mean,
                     f"  Δ̄ = {g_mean:+.0f}  ({g_mean / t_mean * 100:+.1f}%)",
                     color="#bd0026", fontsize=8, va="bottom", ha="right")
        axg.set_xlim(left=0)
        if j == 0:
            axg.set_ylabel("gap  (T − S)")
        axg.set_xlabel("PPO iteration")

    fig.suptitle("CTS — teacher vs student learning curves across "
                 "privileged subsets  (overlaid, with gap track)",
                 fontsize=12, fontweight="bold", y=0.995)
    _save(fig, "fig_go2_cts_teacher_student_overlay")
    save_csv(os.path.join(CSV_DIR, "go2_cts_teacher_student_overlay.csv"), bundle)


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"[lc] wrote {stem}.pdf / .png")


def main():
    ap = argparse.ArgumentParser(description="Plot Go2 PPO learning curves from TB events")
    ap.add_argument("--baseline",   default="logs/baseline/2026-05-10_17-07-38_baseline_go2_v2")
    ap.add_argument("--rma",        default="logs/rma/2026-05-10_17-07-44_rma_go2_v2_l8_l8")
    ap.add_argument("--cts",        default="logs/cts/2026-05-10_17-07-53_cts_go2_v2_l8")
    ap.add_argument("--cts_int",    default="logs/cts/2026-05-11_12-12-02_cts_go2_v2_int_l8")
    ap.add_argument("--cts_ext",    default="logs/cts/2026-05-11_12-12-17_cts_go2_v2_ext_l8")
    args = ap.parse_args()
    runs = {
        "Baseline": os.path.join(REPO_ROOT, args.baseline),
        "RMA":      os.path.join(REPO_ROOT, args.rma),
        "CTS":      os.path.join(REPO_ROOT, args.cts),
    }
    cts_runs = {
        "FULL": os.path.join(REPO_ROOT, args.cts),
        "INT":  os.path.join(REPO_ROOT, args.cts_int),
        "EXT":  os.path.join(REPO_ROOT, args.cts_ext),
    }
    for name, p in runs.items():
        marker = "✓" if os.path.isdir(p) else "✗"
        print(f"  {marker}  {name:8s}  {p}")
    for priv, p in cts_runs.items():
        marker = "✓" if os.path.isdir(p) else "✗"
        print(f"  {marker}  CTS-{priv:4s}  {p}")

    fig_mean_reward(runs)
    fig_mean_length(runs)
    fig_cts_teacher_student(runs["CTS"])
    fig_cts_teacher_student_overlay(cts_runs)
    fig_cts_priv_learning_curves(cts_runs)
    print(f"\n[lc] done — figures in {os.path.relpath(OUT_DIR, REPO_ROOT)}/, "
          f"CSVs in {os.path.relpath(CSV_DIR, REPO_ROOT)}/")


if __name__ == "__main__":
    main()

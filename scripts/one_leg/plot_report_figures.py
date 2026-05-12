"""
Generate the 4 report figures exactly as specified in the brief.

  fig_exp1_comparison.pdf    — Architecture comparison (Baseline / RMA ℓ=16 / CTS ℓ=128)
  fig_exp2_ablation.pdf      — Privileged knowledge ablation (INT / EXT / FULL-16 / FULL-128)
  fig_latent_ablation.pdf    — Latent dimension ablation (log2 x-axis)
  fig_sim2sim_gap.pdf        — Sim-to-sim gap placeholder

Usage:
    python scripts/one_leg/plot_report_figures.py [--results results/ood_results_all.csv]
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import NullLocator

# ── Global style (academic serif) ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        200,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "pdf.fonttype":      42,   # editable text in PDF
    "ps.fonttype":       42,
})

# ── Palette ───────────────────────────────────────────────────────────────────
C_BASE      = "#555555"
C_RMA       = "#2166ac"
C_CTS       = "#d6604d"
C_INT       = "#4575b4"   # blue  — INT
C_EXT       = "#f46d43"   # orange — EXT
C_FULL16    = "#1a9850"   # green  — FULL ℓ=16
C_FULL128   = "#a6d96a"   # light-green — FULL ℓ=128
OUT_DIR     = "results/figures"
FIG_EXT     = "pdf"       # "pdf" for LaTeX, "png" for preview


# ── Helpers ───────────────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path, keep_default_na=False)
    df.columns = df.columns.str.strip()
    df["latent_dim"] = df["latent_dim"].astype(str)
    df["dr_scale"]   = df["dr_scale"].astype(float)
    return df


def get(df, method, priv, l, dr):
    mask = (
        (df["method"]     == method) &
        (df["priv_mode"]  == priv)   &
        (df["latent_dim"] == str(l)) &
        (df["dr_scale"]   == float(dr))
    )
    rows = df[mask]
    if rows.empty:
        return np.nan, np.nan
    m, s = rows["mean_reward"].iloc[0], rows["std_reward"].iloc[0]
    if m < -500:
        return np.nan, np.nan
    return m, s


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(name)[0]
    path = os.path.join(OUT_DIR, f"{stem}.{FIG_EXT}")
    dpi  = 300 if FIG_EXT == "png" else None
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"  saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Architecture Comparison
# Baseline / RMA ℓ=16 / CTS ℓ=128  ×  3 DR scales
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_arch_comparison(df):
    dr_scales  = [1.0, 1.5, 2.0]
    xlabels    = ["In-dist.\n(DR×1.0)", "Mild OOD\n(DR×1.5)", "Hard OOD\n(DR×2.0)"]
    configs    = [
        ("BASELINE", "FULL", "N/A", C_BASE, "Baseline"),
        ("RMA2",     "FULL", "16",  C_RMA,  "RMA  $\\ell$=16"),
        ("CTS",      "FULL", "128", C_CTS,  "CTS  $\\ell$=128"),
    ]
    n_groups = len(dr_scales)
    n_bars   = len(configs)
    bar_w    = 0.22
    gap      = 0.04
    offsets  = np.linspace(-(n_bars-1)/2, (n_bars-1)/2, n_bars) * (bar_w + gap)
    x        = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    for bi, (method, priv, l, color, label) in enumerate(configs):
        means = [get(df, method, priv, l, dr)[0] for dr in dr_scales]
        stds  = [get(df, method, priv, l, dr)[1] for dr in dr_scales]
        xpos  = x + offsets[bi]
        ax.bar(xpos, means, bar_w, yerr=stds, label=label,
               color=color, edgecolor="white", linewidth=0.5,
               error_kw=dict(elinewidth=0.9, ecolor="k", capsize=3))

        # annotate % gain of CTS over Baseline above CTS bars
        if method == "CTS":
            base_means = [get(df, "BASELINE", "FULL", "N/A", dr)[0] for dr in dr_scales]
            pcts = [(m - b) / b * 100 for m, b in zip(means, base_means)]
            for xi, (xp, m, s, pct) in enumerate(zip(xpos, means, stds, pcts)):
                if not np.isnan(m):
                    ax.text(xp, m + s + 18, f"+{pct:.0f}%",
                            ha="center", va="bottom", fontsize=7,
                            color=C_CTS, fontweight="bold")

    # Baseline ID reference line
    base_id = get(df, "BASELINE", "FULL", "N/A", 1.0)[0]
    ax.axhline(base_id, color=C_BASE, lw=0.9, ls=":", alpha=0.7,
               label=f"Baseline ID = {base_id:.0f}")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_ylim(400, 1280)
    ax.set_title("Experiment 1: Architecture Comparison")
    ax.legend(loc="lower left", framealpha=0.85, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_exp1_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Privileged Knowledge Ablation
# 1×2 subplots: RMA (a) / CTS (b)  ×  4 bars each
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_priv_ablation(df):
    # bar definitions per subplot
    rma_bars = [
        ("RMA2", "INT",  "16",  C_INT,    "",     "INT $\\ell$=16"),
        ("RMA2", "EXT",  "16",  C_EXT,    "",     "EXT $\\ell$=16"),
        ("RMA2", "FULL", "16",  C_FULL16, "",     "FULL $\\ell$=16"),
        (None,   None,   None,  "none",   "////", "FULL $\\ell$=128\n(N/A — not tested)"),
    ]
    cts_bars = [
        ("CTS", "INT",  "16",  C_INT,     "",     "INT $\\ell$=16"),
        ("CTS", "EXT",  "16",  C_EXT,     "",     "EXT $\\ell$=16"),
        ("CTS", "FULL", "16",  C_FULL16,  "",     "FULL $\\ell$=16"),
        ("CTS", "FULL", "128", C_FULL128, "",     "FULL $\\ell$=128 *"),
    ]
    xlabels = ["INT\n$\\ell$=16", "EXT\n$\\ell$=16",
               "FULL\n$\\ell$=16", "FULL\n$\\ell$=128"]
    base_id = 815.7

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), sharey=True)

    for ax, bars, title in [
        (axes[0], rma_bars, "(a) RMA"),
        (axes[1], cts_bars, "(b) CTS"),
    ]:
        for xi, (method, priv, l, color, hatch, label) in enumerate(bars):
            if method is None:
                # N/A placeholder bar
                ax.bar(xi, 0, 0.55, color="white",
                       edgecolor="#888", linewidth=1.0, hatch=hatch)
                ax.text(xi, 60, "N/A", ha="center", va="bottom",
                        fontsize=8, color="#888")
                continue
            m, s = get(df, method, priv, l, 1.0)
            if np.isnan(m):
                ax.bar(xi, 0, 0.55, color="white",
                       edgecolor="#888", linewidth=1.0, hatch="////")
                ax.text(xi, 60, "N/A", ha="center", va="bottom",
                        fontsize=8, color="red")
                continue
            ax.bar(xi, m, 0.55, yerr=s, color=color, hatch=hatch,
                   edgecolor="white", linewidth=0.5,
                   error_kw=dict(elinewidth=0.9, ecolor="k", capsize=3))
            # annotate exact mean above bar
            ax.text(xi, m + s + 18, f"{m:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color="#222")

        # baseline reference
        ax.axhline(base_id, color=C_BASE, lw=0.9, ls=":",
                   alpha=0.7, label=f"Baseline = {base_id:.0f}")

        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(xlabels, fontsize=7.5)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Mean Episode Reward (DR×1.0)" if ax == axes[0] else "")
        ax.legend(loc="upper left", fontsize=6.5, framealpha=0.8)

    axes[1].text(3, 820, "* capacity\nrecovery", ha="center",
                 fontsize=7, color=C_FULL128, style="italic")

    axes[0].set_ylim(0, 1380)
    fig.suptitle("Experiment 2: Privileged Knowledge Ablation (ID, DR×1.0)", y=1.01)
    fig.tight_layout()
    save(fig, "fig_exp2_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Latent Dimension Ablation
# CTS-FULL + RMA-FULL, log2 x-axis, annotations for ℓ*
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_latent_ablation(df):
    cts_l   = [8, 16, 32, 64, 128]
    rma_l   = [8, 16, 32]
    base_id = 815.7

    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    # ── CTS-FULL ──────────────────────────────────────────────────────────────
    ms_cts = np.array([get(df, "CTS", "FULL", str(l), 1.0)[0] for l in cts_l])
    ss_cts = np.array([get(df, "CTS", "FULL", str(l), 1.0)[1] for l in cts_l])
    ax.plot(cts_l, ms_cts, color=C_CTS, marker="o", markersize=5,
            lw=1.6, label="CTS-FULL", zorder=3)
    ax.fill_between(cts_l, ms_cts - ss_cts, ms_cts + ss_cts,
                    alpha=0.15, color=C_CTS)

    # ── RMA-FULL ──────────────────────────────────────────────────────────────
    ms_rma = np.array([get(df, "RMA2", "FULL", str(l), 1.0)[0] for l in rma_l])
    ss_rma = np.array([get(df, "RMA2", "FULL", str(l), 1.0)[1] for l in rma_l])
    ax.plot(rma_l, ms_rma, color=C_RMA, marker="s", markersize=5,
            lw=1.6, ls="--", label="RMA-FULL", zorder=3)
    ax.fill_between(rma_l, ms_rma - ss_rma, ms_rma + ss_rma,
                    alpha=0.15, color=C_RMA)

    # ── Baseline reference ────────────────────────────────────────────────────
    ax.axhline(base_id, color=C_BASE, lw=0.9, ls=":", alpha=0.7,
               label=f"Baseline = {base_id:.0f}")

    # ── Annotations ──────────────────────────────────────────────────────────
    # RMA best: ℓ=16
    ax.annotate("$\\ell^*=16$\n(RMA best)",
                xy=(16, ms_rma[1]), xytext=(20, ms_rma[1] + 100),
                fontsize=7.5, color=C_RMA, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_RMA, lw=0.9))
    # CTS best: ℓ=128
    ax.annotate("$\\ell^*=128$\n(CTS best)",
                xy=(128, ms_cts[4]), xytext=(75, ms_cts[4] + 80),
                fontsize=7.5, color=C_CTS, ha="right",
                arrowprops=dict(arrowstyle="->", color=C_CTS, lw=0.9))
    # CTS dip: ℓ=16
    ax.annotate("U-shape dip",
                xy=(16, ms_cts[1]), xytext=(22, ms_cts[1] - 120),
                fontsize=7.5, color=C_CTS, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_CTS, lw=0.9))

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xscale("log", base=2)
    ax.xaxis.set_minor_locator(NullLocator())   # suppress minor ticks
    ax.set_xticks(cts_l)
    ax.set_xticklabels([str(l) for l in cts_l])
    ax.set_xlabel("Latent Dimension $\\ell$")
    ax.set_ylabel("Mean Episode Reward (DR×1.0)")
    ax.set_ylim(500, 1280)
    ax.set_title("Experiment 3: Latent Dimension Ablation")
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()
    save(fig, "fig_latent_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Sim-to-Sim Gap (placeholder)
# Isaac Lab bars filled; MuJoCo bars hatched "TBD"
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_sim2sim_gap(df):
    methods = [
        ("BASELINE", "FULL", "N/A", C_BASE, "Baseline"),
        ("RMA2",     "FULL", "16",  C_RMA,  "RMA  $\\ell$=16"),
        ("CTS",      "FULL", "128", C_CTS,  "CTS  $\\ell$=128"),
    ]
    bar_w = 0.30
    x     = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    isaac_means, isaac_stds = [], []
    for method, priv, l, color, label in methods:
        m, s = get(df, method, priv, l, 1.0)
        isaac_means.append(m); isaac_stds.append(s)

    # Isaac bars (filled)
    ax.bar(x - bar_w/2, isaac_means, bar_w,
           yerr=isaac_stds, color=[c for _, _, _, c, _ in methods],
           edgecolor="white", linewidth=0.5,
           error_kw=dict(elinewidth=0.9, ecolor="k", capsize=3),
           label="Isaac Lab (DR×1.0)")

    # MuJoCo bars (placeholder — hatched, height=0 visually shown as thin bar)
    placeholder_h = 60   # short stub so hatch pattern is visible
    for xi, (_, _, _, color, _) in enumerate(methods):
        ax.bar(xi + bar_w/2, placeholder_h, bar_w,
               color="white", edgecolor=color, linewidth=1.2,
               hatch="////", alpha=0.85)
        ax.text(xi + bar_w/2, placeholder_h + 20, "TBD",
                ha="center", va="bottom", fontsize=8,
                color=color, fontweight="bold")

    # Bidirectional gap arrows
    for xi, (isaac_m, (_, _, _, color, _)) in enumerate(
            zip(isaac_means, methods)):
        ax.annotate("", xy=(xi + bar_w/2, placeholder_h),
                    xytext=(xi + bar_w/2, isaac_m),
                    arrowprops=dict(arrowstyle="<->",
                                    color="crimson", lw=1.2))
        ax.text(xi + bar_w/2 + 0.06, (isaac_m + placeholder_h) / 2,
                "Gap?", fontsize=7, color="crimson", va="center")

    # Method labels on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, _, _, label in methods])
    ax.set_ylabel("Mean Episode Reward")
    ax.set_ylim(0, 1340)
    ax.set_title("Sim-to-Sim Generalisation Gap (Isaac Lab → MuJoCo)")

    isaac_patch  = mpatches.Patch(color="gray", label="Isaac Lab (DR×1.0)")
    mujoco_patch = mpatches.Patch(facecolor="white", edgecolor="gray",
                                  hatch="////", label="MuJoCo (pending)")
    ax.legend(handles=[isaac_patch, mujoco_patch],
              loc="upper right", framealpha=0.85, fontsize=7.5)

    ax.text(0.5, -0.14,
            "* Replace TBD values with MuJoCo results when available.",
            transform=ax.transAxes, ha="center", fontsize=7,
            color="#555", style="italic")
    fig.tight_layout()
    save(fig, "fig_sim2sim_gap.pdf")


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/ood_results_all.csv")
    args = parser.parse_args()

    print(f"Loading {args.results} ...")
    df = load(args.results)
    print(f"  {len(df)} rows loaded.\n")

    print("Generating report figures ...")
    fig1_arch_comparison(df)
    fig2_priv_ablation(df)
    fig3_latent_ablation(df)
    fig4_sim2sim_gap(df)
    print(f"\nAll figures saved to {OUT_DIR}/")

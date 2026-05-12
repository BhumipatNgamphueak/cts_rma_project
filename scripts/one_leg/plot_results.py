"""
Generate publication-quality figures for the FRA503 LaTeX report.

Figures produced (saved to results/figures/):
  fig1_exp1_arch_comparison.pdf  — Exp 1: Baseline vs RMA2 vs CTS (FULL, l=32)
  fig2_exp2_priv_ablation.pdf    — Exp 2: priv_mode ablation at l=16
  fig3_exp2_all_latents.pdf      — Exp 2 supplement: priv_mode across l=8/16/32
  fig4_exp3_latent_ablation.pdf  — Exp 3: latent-dim ablation (CTS-FULL + RMA2-FULL)
  fig5_hypothesis.pdf            — Hypothesis: CTS FULL recovers INT at large latent
  fig6_ood_profile.pdf           — OOD profile: reward vs dr_scale for best configs

Usage:
    python scripts/one_leg/plot_results.py [--results results/ood_results_all.csv]
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "errorbar.capsize": 3,
})

# ── Colours ───────────────────────────────────────────────────────────────────
C_BASE  = "#555555"   # Baseline — grey
C_RMA   = "#2166ac"   # RMA  — blue
C_CTS   = "#d6604d"   # CTS  — red-orange
C_INT   = "#4dac26"   # INT  — green
C_EXT   = "#984ea3"   # EXT  — purple
C_FULL  = "#ff7f00"   # FULL — orange

PRIV_COLOR  = {"FULL": C_FULL,  "INT": C_INT,  "EXT": C_EXT}
PRIV_HATCH  = {"FULL": "",      "INT": "///",   "EXT": "..."}
METHOD_COLOR = {"BASELINE": C_BASE, "RMA2": C_RMA, "CTS": C_CTS}

OUT_DIR = "results/figures"
FIG_EXT = "pdf"   # change to "png" for raster output (e.g. Word/slides)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load(path: str) -> pd.DataFrame:
    # keep_default_na=False prevents "N/A" being read as float NaN,
    # which would coerce the entire latent_dim column to float64
    # and turn e.g. 8 → "8.0", breaking exact-string matching.
    df = pd.read_csv(path, keep_default_na=False)
    df.columns = df.columns.str.strip()
    df["latent_dim"] = df["latent_dim"].astype(str)
    df["dr_scale"]   = df["dr_scale"].astype(float)
    return df


def get(df, method, priv, l, dr):
    """Return (mean, std) or (nan, nan) if row missing/anomalous."""
    mask = (
        (df["method"] == method) &
        (df["priv_mode"] == priv) &
        (df["latent_dim"] == str(l)) &
        (df["dr_scale"] == float(dr))
    )
    rows = df[mask]
    if rows.empty:
        return np.nan, np.nan
    m, s = rows["mean_reward"].iloc[0], rows["std_reward"].iloc[0]
    # mark obviously broken rows as nan
    if m < -500:
        return np.nan, np.nan
    return m, s


def bar_group(ax, groups, values, errors, colors, hatches,
              group_labels, bar_width=0.18, gap=0.06):
    """
    Draw grouped bar chart.
    groups       : list of group positions (x centres)
    values       : list-of-lists [group][bar]
    errors       : list-of-lists [group][bar]
    colors/hatches: list per bar within group
    """
    n_bars = len(values[0])
    offsets = np.linspace(-(n_bars-1)/2, (n_bars-1)/2, n_bars) * (bar_width + gap)
    for gi, (gx, gv, ge) in enumerate(zip(groups, values, errors)):
        for bi, (v, e, c, h) in enumerate(zip(gv, ge, colors, hatches)):
            if np.isnan(v):
                continue
            ax.bar(gx + offsets[bi], v, bar_width,
                   yerr=e, color=c, hatch=h,
                   edgecolor="white", linewidth=0.6,
                   error_kw=dict(elinewidth=0.8, ecolor="k", capsize=3))


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(name)[0]
    path = os.path.join(OUT_DIR, f"{stem}.{FIG_EXT}")
    dpi = 300 if FIG_EXT == "png" else None
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"  saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Exp 1: Architecture comparison (Baseline / RMA2 / CTS) at FULL l=32
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_exp1(df):
    configs = [
        ("BASELINE", "FULL", "N/A"),
        ("RMA2",     "FULL", "32"),
        ("CTS",      "FULL", "32"),
    ]
    labels  = ["Baseline", "RMA2-FULL\n$\\ell{=}32$", "CTS-FULL\n$\\ell{=}32$"]
    colors  = [C_BASE, C_RMA, C_CTS]
    dr_scales = [1.0, 1.5, 2.0]
    dr_labels = ["ID\n(DR×1.0)", "Mild OOD\n(DR×1.5)", "Hard OOD\n(DR×2.0)"]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    groups   = np.arange(len(dr_scales))
    n_bars   = len(configs)
    bar_w    = 0.22
    gap      = 0.03
    offsets  = np.linspace(-(n_bars-1)/2, (n_bars-1)/2, n_bars) * (bar_w + gap)

    for bi, ((method, priv, l), label, color) in enumerate(zip(configs, labels, colors)):
        means, stds = [], []
        for dr in dr_scales:
            m, s = get(df, method, priv, l, dr)
            means.append(m); stds.append(s)
        ax.bar(groups + offsets[bi], means, bar_w,
               yerr=stds, label=label, color=color,
               edgecolor="white", linewidth=0.5,
               error_kw=dict(elinewidth=0.8, ecolor="#333", capsize=3))

    ax.set_xticks(groups)
    ax.set_xticklabels(dr_labels)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_ylim(0, 1250)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.axhline(815.7, color=C_BASE, lw=0.8, ls=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_title("Experiment 1: Architecture Comparison (FULL privileged knowledge, $\\ell=32$)")
    fig.tight_layout()
    save(fig, "fig1_exp1_arch_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Exp 2: Privileged-info ablation at l=16 (ID and OOD)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_exp2_l16(df):
    privs    = ["FULL", "INT", "EXT"]
    priv_labels = ["FULL", "INT-only", "EXT-only"]
    dr_vals  = [1.0, 2.0]
    dr_names = ["ID (DR×1.0)", "Hard OOD (DR×2.0)"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), sharey=True)

    for ax, method, method_label, mcolor in [
        (axes[0], "RMA2", "RMA2 ($\\ell=16$)", C_RMA),
        (axes[1], "CTS",  "CTS  ($\\ell=16$)", C_CTS),
    ]:
        for di, (dr, dr_name) in enumerate(zip(dr_vals, dr_names)):
            means, stds = [], []
            for priv in privs:
                m, s = get(df, method, priv, "16", dr)
                means.append(m); stds.append(s)
            x = np.arange(len(privs)) + di * 0.35
            bars = ax.bar(x, means, 0.3,
                          yerr=stds, label=dr_name,
                          color=[PRIV_COLOR[p] for p in privs],
                          alpha=0.9 if di == 0 else 0.55,
                          edgecolor="white", linewidth=0.5,
                          error_kw=dict(elinewidth=0.8, ecolor="#333", capsize=3))

        ax.set_title(method_label)
        ax.set_xticks(np.arange(len(privs)) + 0.175)
        ax.set_xticklabels(priv_labels)
        ax.set_ylabel("Mean Episode Reward" if ax == axes[0] else "")
        ax.set_ylim(0, 1350)
        ax.yaxis.set_major_locator(MaxNLocator(6))

    # shared legend for priv_mode colours
    patches = [mpatches.Patch(color=PRIV_COLOR[p], label=f"{p}") for p in privs]
    patches += [mpatches.Patch(facecolor="grey", alpha=0.9, label="ID (DR×1.0)"),
                mpatches.Patch(facecolor="grey", alpha=0.55, label="OOD (DR×2.0)")]
    fig.legend(handles=patches, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.85)
    fig.suptitle("Experiment 2: Privileged Knowledge Ablation ($\\ell=16$)", y=1.08)
    fig.tight_layout()
    save(fig, "fig2_exp2_priv_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Exp 2 supplement: priv_mode across l=8/16/32 for CTS and RMA2
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_exp2_all_latents(df):
    latents = [8, 16, 32]
    privs   = ["FULL", "INT", "EXT"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), sharey=True)

    for ax, method, title in [
        (axes[0], "RMA2", "RMA2"),
        (axes[1], "CTS",  "CTS"),
    ]:
        x = np.arange(len(latents))
        bar_w = 0.25
        offsets = [-bar_w, 0, bar_w]
        for bi, priv in enumerate(privs):
            means, stds = [], []
            for l in latents:
                m, s = get(df, method, priv, str(l), 1.0)
                means.append(m); stds.append(s)
            ax.bar(x + offsets[bi], means, bar_w,
                   yerr=stds, label=priv,
                   color=PRIV_COLOR[priv],
                   hatch=PRIV_HATCH[priv],
                   edgecolor="white", linewidth=0.5,
                   error_kw=dict(elinewidth=0.8, ecolor="#333", capsize=3))

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$\\ell={l}$" for l in latents])
        ax.set_xlabel("Latent dimension $\\ell$")
        ax.set_ylabel("Mean Reward (ID, DR×1.0)" if ax == axes[0] else "")
        ax.set_ylim(0, 1350)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.legend(title="priv_mode", loc="upper left", fontsize=7)

        # mark broken bars with N/A label
        for bi, priv in enumerate(privs):
            for li, l in enumerate(latents):
                m, s = get(df, method, priv, str(l), 1.0)
                if np.isnan(m):
                    ax.text(li + offsets[bi], 30, "N/A", ha="center",
                            va="bottom", fontsize=6, color="red", rotation=90)

    fig.suptitle("Experiment 2: Privileged Info Ablation Across Latent Sizes (ID, DR×1.0)")
    fig.tight_layout()
    save(fig, "fig3_exp2_all_latents.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Exp 3: Latent dimension ablation (CTS-FULL + RMA2-FULL)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_exp3_latent(df):
    cts_latents  = [8, 16, 32, 64, 128]
    rma_latents  = [8, 16, 32]
    dr_vals      = [1.0, 2.0]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    # evenly-spaced integer positions — avoids log-scale issues
    x_pos   = np.arange(len(cts_latents))               # [0,1,2,3,4]
    rma_x   = np.array([cts_latents.index(l) for l in rma_latents])  # [0,1,2]

    for dr, ls, alpha in [(1.0, "-", 1.0), (2.0, "--", 0.65)]:
        # CTS FULL
        ms, ss = zip(*[get(df, "CTS", "FULL", str(l), dr) for l in cts_latents])
        ms, ss = np.array(ms), np.array(ss)
        ax.plot(x_pos, ms, ls=ls, color=C_CTS, marker="o",
                markersize=5, lw=1.5, label=f"CTS-FULL (DR×{dr:.1f})", alpha=alpha)
        valid = ~np.isnan(ms)
        ax.fill_between(x_pos[valid], (ms-ss)[valid], (ms+ss)[valid],
                        alpha=0.12*alpha, color=C_CTS)

        # RMA2 FULL
        mr, sr = zip(*[get(df, "RMA2", "FULL", str(l), dr) for l in rma_latents])
        mr, sr = np.array(mr), np.array(sr)
        ax.plot(rma_x, mr, ls=ls, color=C_RMA, marker="s",
                markersize=5, lw=1.5, label=f"RMA2-FULL (DR×{dr:.1f})", alpha=alpha)
        ax.fill_between(rma_x, mr-sr, mr+sr, alpha=0.12*alpha, color=C_RMA)

    # CTS INT reference lines
    m_int16, _ = get(df, "CTS", "INT", "16", 1.0)
    m_int32, _ = get(df, "CTS", "INT", "32", 1.0)
    ax.axhline(m_int16, color=C_INT, ls=":", lw=1.2, alpha=0.8,
               label="CTS-INT $\\ell$=16 (ID, ref)")
    ax.axhline(m_int32, color=C_INT, ls="-.", lw=1.0, alpha=0.6,
               label="CTS-INT $\\ell$=32 (ID, ref)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(l) for l in cts_latents])
    ax.set_xlabel("Latent dimension $\\ell$")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_ylim(500, 1300)
    ax.set_title("Experiment 3: Latent Dimension Ablation (FULL privileged knowledge)")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.85)

    # annotate with x_pos indices
    ax.annotate("U-shaped trough\n(CTS-FULL)", xy=(1, 724), xytext=(1.7, 570),
                fontsize=7, color=C_CTS,
                arrowprops=dict(arrowstyle="->", color=C_CTS, lw=0.8))
    ax.annotate("Recovery\nat $\\ell$=128", xy=(4, 1092), xytext=(3.0, 940),
                fontsize=7, color=C_CTS,
                arrowprops=dict(arrowstyle="->", color=C_CTS, lw=0.8))

    fig.tight_layout()
    save(fig, "fig4_exp3_latent_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Hypothesis: CTS-FULL recovers INT at large latent
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_hypothesis(df):
    configs = [
        ("CTS", "FULL",  "8",   C_CTS,  "",      "CTS-FULL $\\ell$=8"),
        ("CTS", "FULL",  "16",  C_CTS,  "///",   "CTS-FULL $\\ell$=16"),
        ("CTS", "FULL",  "32",  C_CTS,  "xxx",   "CTS-FULL $\\ell$=32"),
        ("CTS", "FULL",  "64",  C_CTS,  "...",   "CTS-FULL $\\ell$=64"),
        ("CTS", "FULL",  "128", C_CTS,  "---",   "CTS-FULL $\\ell$=128 (best)"),
        (None,  None,    None,  "none", "",       ""),   # spacer
        ("CTS", "INT",   "8",   C_INT,  "",       "CTS-INT  $\\ell$=8"),
        ("CTS", "INT",   "16",  C_INT,  "///",    "CTS-INT  $\\ell$=16"),
        ("CTS", "INT",   "32",  C_INT,  "xxx",    "CTS-INT  $\\ell$=32"),
    ]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))

    xpos = 0
    xticks, xticklabels = [], []
    x128 = None   # track position of CTS FULL l=128 bar
    for method, priv, l, color, hatch, label in configs:
        if method is None:
            xpos += 0.3
            continue
        m, s = get(df, method, priv, l, 1.0)
        if np.isnan(m):
            xpos += 0.6
            continue
        ax.bar(xpos, m, 0.5, yerr=s, color=color, hatch=hatch,
               edgecolor="white", linewidth=0.5,
               error_kw=dict(elinewidth=0.8, ecolor="#333", capsize=3))
        xticks.append(xpos)
        short = label.replace("CTS-", "").replace("$\\ell$=", "").replace(" ★", "★")
        xticklabels.append(short)
        if method == "CTS" and priv == "FULL" and l == "128":
            x128 = xpos
        xpos += 0.6

    # Horizontal bands showing INT l=16 and l=32
    m16, _ = get(df, "CTS", "INT", "16", 1.0)
    m32, _ = get(df, "CTS", "INT", "32", 1.0)
    ax.axhspan(min(m16, m32), max(m16, m32), alpha=0.10, color=C_INT,
               label="CTS-INT l=16–32 band")
    ax.axhline(m16, color=C_INT, ls="--", lw=1.0, alpha=0.7)
    ax.axhline(m32, color=C_INT, ls=":",  lw=1.0, alpha=0.7)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("Mean Episode Reward (ID, DR×1.0)")
    ax.set_ylim(500, 1300)
    ax.set_title("Hypothesis: CTS-FULL with $\\ell=128$ Recovers INT Performance")

    # annotation on l=128 bar
    m128, _ = get(df, "CTS", "FULL", "128", 1.0)
    if x128 is not None:
        ax.annotate("FULL $\\ell$=128\n$\\approx$ INT $\\ell$=16,32",
                    xy=(x128, m128),
                    xytext=(x128 - 0.9, m128 - 150),
                    fontsize=7.5, color=C_CTS,
                    arrowprops=dict(arrowstyle="->", color=C_CTS, lw=0.8))

    green_patch = mpatches.Patch(color=C_INT, alpha=0.4, label="CTS-INT $\\ell$=16–32 reference band")
    ax.legend(handles=[green_patch], loc="lower right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig5_hypothesis.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — OOD profile: mean reward vs dr_scale for key configurations
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_ood_profile(df):
    key_configs = [
        ("BASELINE", "FULL",  "N/A", C_BASE, "-",  "o", "Baseline"),
        ("RMA2",     "FULL",  "16",  C_RMA,  "-",  "s", "RMA2-FULL $\\ell$=16"),
        ("RMA2",     "INT",   "16",  C_RMA,  "--", "^", "RMA2-INT $\\ell$=16"),
        ("CTS",      "FULL",  "128", C_CTS,  "-",  "o", "CTS-FULL $\\ell$=128"),
        ("CTS",      "INT",   "32",  C_INT,  "-",  "s", "CTS-INT $\\ell$=32"),
        ("CTS",      "INT",   "16",  C_INT,  "--", "^", "CTS-INT $\\ell$=16"),
    ]
    dr_scales = [1.0, 1.5, 2.0]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    for method, priv, l, color, ls, marker, label in key_configs:
        ms, ss = zip(*[get(df, method, priv, l, dr) for dr in dr_scales])
        ms, ss = np.array(ms), np.array(ss)
        valid = ~np.isnan(ms)
        ax.plot(np.array(dr_scales)[valid], ms[valid],
                ls=ls, color=color, marker=marker,
                markersize=5, lw=1.5, label=label)
        ax.fill_between(np.array(dr_scales)[valid],
                        (ms-ss)[valid], (ms+ss)[valid],
                        alpha=0.10, color=color)

    ax.set_xticks(dr_scales)
    ax.set_xticklabels(["ID\n(DR×1.0)", "Mild OOD\n(DR×1.5)", "Hard OOD\n(DR×2.0)"])
    ax.set_xlabel("DR scale")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_ylim(600, 1300)
    ax.set_title("OOD Robustness Profile — Key Configurations")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85, ncol=2)
    fig.tight_layout()
    save(fig, "fig6_ood_profile.pdf")


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/ood_results_all.csv")
    args = parser.parse_args()

    print(f"Loading {args.results} ...")
    df = load(args.results)
    print(f"  {len(df)} rows loaded.\n")

    print("Generating figures ...")
    fig1_exp1(df)
    fig2_exp2_l16(df)
    fig3_exp2_all_latents(df)
    fig4_exp3_latent(df)
    fig5_hypothesis(df)
    fig6_ood_profile(df)
    print(f"\nAll figures saved to {OUT_DIR}/")

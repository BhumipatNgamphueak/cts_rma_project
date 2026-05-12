"""scripts/plot_results_one_leg.py — Plot one-leg (Phase 1) Isaac-only OOD results.

Reads:  results/ood_one_leg.csv  (written by scripts/one_leg/eval_ood.py)
Writes: results/figures/one_leg/fig_one_leg_*.{pdf,png}
        results/one_leg_results_table.md / .tex

Uses the same colour grammar as scripts/plot_results_go2.py so that one-leg and
Go2 figures share a single visual convention:
    Baseline = blue (#2166ac)
    RMA      = green (#4dac26)
    CTS      = red   (#d6604d)
    DR ×2 (OOD) = hatched "//" + alpha 0.70   ;   DR ×1 = solid alpha 1.00

Usage:
    python scripts/plot_results_one_leg.py
    python scripts/plot_results_one_leg.py --ood results/ood_one_leg.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Style — matches scripts/plot_results_go2.py / OpenTopic palette
# ─────────────────────────────────────────────────────────────────────────────
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(REPO_ROOT, "results", "figures", "one_leg")

METHOD_ORDER  = ["BASELINE", "RMA", "CTS"]
METHOD_LABEL  = {"BASELINE": "Baseline", "RMA": "RMA", "CTS": "CTS"}
METHOD_COLOR  = {"BASELINE": "#2166ac", "RMA": "#4dac26", "CTS": "#d6604d"}
COND_HATCH    = {1.0: "",   2.0: "//"}
COND_ALPHA    = {1.0: 1.00, 2.0: 0.70}


# ─────────────────────────────────────────────────────────────────────────────
def _load(path):
    if not os.path.exists(path):
        print(f"[plot] {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["method"] = df["method"].astype(str).str.strip().str.upper()
    df["method"] = df["method"].replace({"RMA2": "RMA", "BASE": "BASELINE"})
    for c in ("dr_scale", "mean_reward", "std_reward",
              "mean_length", "std_length", "success_rate"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote one_leg/{name}.pdf / .png")


def _select(df, m, s, priv="FULL"):
    sub = df[(df["method"] == m) & (np.isclose(df["dr_scale"], s))]
    if "priv_mode" in df.columns:
        sub = sub[sub["priv_mode"].astype(str).str.upper().isin({priv, "BASE"})]
    if not len(sub):
        return np.nan, np.nan, np.nan
    r  = float(sub["mean_reward"].iloc[0])
    se = float(sub["std_reward"].iloc[0])
    sc = float(sub["success_rate"].iloc[0]) if "success_rate" in sub.columns and pd.notna(sub["success_rate"].iloc[0]) else np.nan
    return r, se, sc


def _figure_legend(fig, methods, y=0.02):
    handles = [mpatches.Patch(color=METHOD_COLOR[m], label=METHOD_LABEL[m]) for m in methods]
    handles += [
        mpatches.Patch(facecolor="white", edgecolor="0.3", hatch="",
                       label="DR ×1 (training)"),
        mpatches.Patch(facecolor="white", edgecolor="0.3", hatch="//",
                       label="DR ×2 (OOD)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, y),
               ncol=len(handles), frameon=False, fontsize=10)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: headline — single panel, two metrics side-by-side per method
# ─────────────────────────────────────────────────────────────────────────────
def fig_headline(df):
    if df.empty: return
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    if not methods: return
    dr_lo, dr_hi = sorted(df["dr_scale"].unique())[0], sorted(df["dr_scale"].unique())[-1]

    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(11.0, 5.0))
    x = np.arange(len(methods)); w = 0.38

    # (A) reward
    for i, m in enumerate(methods):
        r_lo, e_lo, _ = _select(df, m, dr_lo)
        r_hi, e_hi, _ = _select(df, m, dr_hi)
        ax_r.bar(x[i] - w/2, r_lo, w, yerr=e_lo, capsize=2.5,
                 color=METHOD_COLOR[m], alpha=COND_ALPHA[dr_lo],
                 hatch=COND_HATCH[dr_lo], edgecolor="white", linewidth=0.5)
        ax_r.bar(x[i] + w/2, r_hi, w, yerr=e_hi, capsize=2.5,
                 color=METHOD_COLOR[m], alpha=COND_ALPHA[dr_hi],
                 hatch=COND_HATCH[dr_hi], edgecolor="white", linewidth=0.5)
        # annotate retention% above the right bar
        if r_lo and r_lo > 1e-6 and not np.isnan(r_hi):
            ret = 100.0 * r_hi / r_lo
            ax_r.annotate(f"{ret:.0f}%", xy=(x[i] + w/2, max(r_hi, 0) + max(e_hi, 1) * 1.1 + 50),
                          ha="center", va="bottom", fontsize=10, fontweight="bold", color="0.20")
    ax_r.set_xticks(x); ax_r.set_xticklabels([METHOD_LABEL[m] for m in methods])
    ax_r.set_ylabel("episode return (mean $\\pm$ std)")
    ax_r.set_title("(A)  Mean reward  (DR×1 vs DR×2 in Isaac)", fontweight="bold")
    ax_r.text(0.02, 0.97, f"solid: DR×{dr_lo:g}\nhatched: DR×{dr_hi:g}",
              transform=ax_r.transAxes, ha="left", va="top", fontsize=9, color="0.25",
              bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.7", alpha=0.85))

    # (B) success rate
    for i, m in enumerate(methods):
        _, _, s_lo = _select(df, m, dr_lo)
        _, _, s_hi = _select(df, m, dr_hi)
        ax_s.bar(x[i] - w/2, s_lo, w,
                 color=METHOD_COLOR[m], alpha=COND_ALPHA[dr_lo],
                 hatch=COND_HATCH[dr_lo], edgecolor="white", linewidth=0.5)
        ax_s.bar(x[i] + w/2, s_hi, w,
                 color=METHOD_COLOR[m], alpha=COND_ALPHA[dr_hi],
                 hatch=COND_HATCH[dr_hi], edgecolor="white", linewidth=0.5)
        for xc, v in [(x[i] - w/2, s_lo), (x[i] + w/2, s_hi)]:
            if not np.isnan(v):
                ax_s.annotate(f"{v:.0f}%", xy=(xc, v + 2),
                              ha="center", va="bottom",
                              fontsize=10, fontweight="bold", color="0.15")
    ax_s.axhline(80, color="0.2", lw=1.4, ls="--", alpha=0.8, zorder=1)
    ax_s.text(-0.55, 80, " 80% spec ", va="center", ha="right",
              fontsize=9, fontstyle="italic",
              bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5"))
    ax_s.set_xticks(x); ax_s.set_xticklabels([METHOD_LABEL[m] for m in methods])
    ax_s.set_ylabel("success rate  [%]")
    ax_s.set_ylim(0, 115)
    ax_s.set_xlim(-0.7, len(methods) - 0.3)
    ax_s.set_title("(B)  Success rate  (DR×1 vs DR×2 in Isaac)", fontweight="bold")
    ax_s.text(0.02, 0.97, f"solid: DR×{dr_lo:g}\nhatched: DR×{dr_hi:g}",
              transform=ax_s.transAxes, ha="left", va="top", fontsize=9, color="0.25",
              bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.7", alpha=0.85))

    _figure_legend(fig, methods, y=0.02)
    fig.suptitle("One-leg hexapod (Phase 1) — Isaac-Lab OOD test    "
                 "Baseline / RMA / CTS  ·  FULL · $Z$=8  ·  DR×1 vs DR×2",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    _save(fig, "fig_one_leg_headline")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: OOD retention (single panel, % retained at DR×2 vs DR×1)
# ─────────────────────────────────────────────────────────────────────────────
def fig_ood_retention(df):
    if df.empty: return
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    if not methods: return
    dr_scales = sorted(df["dr_scale"].unique())
    if len(dr_scales) < 2: return
    dr_lo, dr_hi = dr_scales[0], dr_scales[-1]

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    x = np.arange(len(methods)); w = 0.55
    threshold = 70.0
    for i, m in enumerate(methods):
        r_lo, _, _ = _select(df, m, dr_lo)
        r_hi, _, _ = _select(df, m, dr_hi)
        ret = 100.0 * r_hi / r_lo if (r_lo and r_lo > 1e-6 and not np.isnan(r_hi)) else np.nan
        passed = (not np.isnan(ret)) and ret >= threshold
        edge_col = "#1a9850" if passed else "#b2182b"
        if np.isnan(ret): continue
        ax.bar(x[i], ret, w, color=METHOD_COLOR[m],
               hatch=COND_HATCH[dr_hi], alpha=COND_ALPHA[dr_hi],
               edgecolor=edge_col, linewidth=2.8, zorder=2)
        ax.annotate(f"✓ PASS" if passed else "✗ FAIL",
                    xy=(x[i], ret + 2),
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.30", facecolor=edge_col,
                              edgecolor="none"))
        ax.annotate(f"{ret:.0f}%",
                    xy=(x[i], ret + 13),
                    ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.axhline(threshold, color="0.2", lw=1.6, ls="--", alpha=0.85, zorder=1)
    ax.text(-0.55, threshold, f" {threshold:.0f}% spec ",
            va="center", ha="right", fontsize=10, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.4"))
    ax.axhline(100.0, color="0.5", lw=0.8, ls=":", alpha=0.4, zorder=1)
    ax.text(-0.55, 100.0, " perfect ", va="center", ha="right",
            fontsize=9, fontstyle="italic", color="0.4",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="0.7"))
    ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in methods], fontsize=14)
    ax.set_ylabel(f"R_Isaac,{dr_hi:g}× / R_Isaac,{dr_lo:g}×  [%]", fontsize=12)
    ax.set_ylim(0, 135)
    ax.set_xlim(-0.65, len(methods) - 0.35)
    ax.set_title("One-leg hexapod — Isaac OOD retention", fontsize=16, fontweight="bold", pad=14)
    _figure_legend(fig, methods, y=0.05)
    fig.text(0.5, 0.01,
             "OOD retention = reward at DR×2 as a fraction of reward at DR×1 (Isaac only).   "
             "Higher = more OOD-robust.",
             ha="center", fontsize=10, fontstyle="italic", color="0.3")
    fig.tight_layout(rect=[0, 0.14, 1, 0.96])
    _save(fig, "fig_one_leg_ood_retention")


# ─────────────────────────────────────────────────────────────────────────────
def write_table(df):
    if df.empty: return
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    dr_scales = sorted(df["dr_scale"].unique())
    out = ["# One-leg hexapod (Phase 1) — Isaac-Lab OOD results", "",
           "_Auto-generated by scripts/plot_results_one_leg.py. "
           "Numbers come from results/ood_one_leg.csv._", ""]
    out.append("| method | priv | Z | dr | mean reward | success % | mean length |")
    out.append("|---|---|---|---|---|---|---|")
    for m in methods:
        for s in dr_scales:
            sub = df[(df["method"] == m) & (np.isclose(df["dr_scale"], s))]
            if not len(sub): continue
            r  = sub["mean_reward"].iloc[0]; se = sub["std_reward"].iloc[0]
            sc = sub["success_rate"].iloc[0] if "success_rate" in sub.columns else np.nan
            ln = sub["mean_length"].iloc[0]
            priv = sub.get("priv_mode", pd.Series(["FULL"])).iloc[0]
            Z = sub.get("latent_dim", pd.Series(["—"])).iloc[0]
            out.append(f"| {METHOD_LABEL[m]} | {priv} | {Z} | {s:g} | "
                       f"{r:.1f} ± {se:.1f} | {sc:.0f} | {ln:.0f} |")
    # OOD retention table
    dr_lo, dr_hi = dr_scales[0], dr_scales[-1]
    out += ["", f"## OOD retention  (DR×{dr_hi:g} / DR×{dr_lo:g}, spec ≥ 70 %)", ""]
    out.append("| method | R_DR×{lo} | R_DR×{hi} | OOD retention | spec |".format(lo=dr_lo, hi=dr_hi))
    out.append("|---|---|---|---|---|")
    for m in methods:
        r_lo, _, _ = _select(df, m, dr_lo); r_hi, _, _ = _select(df, m, dr_hi)
        if r_lo and r_lo > 1e-6 and not np.isnan(r_hi):
            ret = 100.0 * r_hi / r_lo
            flag = "PASS" if ret >= 70 else "FAIL"
            out.append(f"| {METHOD_LABEL[m]} | {r_lo:.1f} | {r_hi:.1f} | {ret:.1f}% | {flag} |")

    md_path = os.path.join(REPO_ROOT, "results", "one_leg_results_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"[plot] wrote {os.path.relpath(md_path, REPO_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Plot one-leg (Phase 1) Isaac OOD results")
    ap.add_argument("--ood", default=os.path.join(REPO_ROOT, "results", "ood_one_leg.csv"))
    args = ap.parse_args()

    df = _load(args.ood)
    if df.empty:
        print("[plot] no input data. Run scripts/one_leg/eval_ood.py first "
              "(see scripts/run_one_leg_eval.sh).")
        return
    fig_headline(df)
    fig_ood_retention(df)
    write_table(df)
    print(f"\n[plot] done — figures in {os.path.relpath(OUT_DIR, REPO_ROOT)}/")


if __name__ == "__main__":
    main()

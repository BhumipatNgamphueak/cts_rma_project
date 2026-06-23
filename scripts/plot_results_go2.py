"""
Generate the Go2 (Phase-2) report figures and result tables from evaluation CSVs.

This script is *defensive*: it plots whatever data is present and skips (with a
message) any figure whose data is missing — so it can be re-run continuously as
the training/eval jobs finish.

Inputs (all optional — give what you have):
  --ood      results/ood_go2.csv         Isaac-Lab OOD eval
  --sim2sim  results/sim2sim_go2.csv      Isaac->MuJoCo sim-to-sim eval

Expected (flexible) columns:
  ood csv     : method, dr_scale, mean_reward, std_reward, [success_rate],
                [priv_mode], [latent_dim], [mean_length], [std_length], ...
  sim2sim csv : method, dr_scale, mean_reward, std_reward, [success_rate],
                [lin_vel_track], [ang_vel_track], [priv_mode], [latent_dim], ...
  (Missing priv_mode -> assumed "FULL"; missing latent_dim -> assumed "8".
   Method names are upper-cased; "RMA2" is treated the same as "RMA".)

If you only have the human-readable sim2sim *.txt reports under logs/*/ood_eval/,
run with --scan-logs and they will be parsed into a sim2sim dataframe.

Outputs (written to results/figures/):
  fig_go2_ood_profile.{pdf,png}        mean reward vs DR scale, per method
  fig_go2_sim2sim_transfer.{pdf,png}   Isaac vs MuJoCo reward (+ tracking), per method
  fig_go2_latent_ablation.{pdf,png}    reward vs latent dim (CTS / RMA)        [if data]
  fig_go2_priv_ablation.{pdf,png}      FULL/INT/EXT (+ BASE), per arch         [if data]
  go2_results_table.md / .tex          combined results table

Usage:
  python scripts/plot_results_go2.py \
      --ood results/ood_go2.csv --sim2sim results/sim2sim_go2.csv
  python scripts/plot_results_go2.py --scan-logs            # parse logs/*/ood_eval/*.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style (matches /home/drl-68/OpenTopic plot_results.py + visualize_results.py) ──
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
    "axes.grid.axis":     "y",       # y-only grid (matches OpenTopic)
    "grid.alpha":         0.35,
    "grid.linewidth":     0.7,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(REPO_ROOT, "results", "figures")

# Canonical method order / colours / markers (matches OpenTopic palette)
METHOD_ORDER  = ["BASELINE", "RMA_TEACHER", "RMA", "CTS"]
METHOD_LABEL  = {"BASELINE":    "Baseline",
                 "RMA_TEACHER": "RMA Teacher",
                 "RMA":         "RMA Student",
                 "CTS":         "CTS"}
METHOD_COLOR  = {"BASELINE":    "#2166ac",   # blue
                 "RMA_TEACHER": "#762a83",   # purple  (oracle upper-bound)
                 "RMA":         "#4dac26",   # green
                 "CTS":         "#d6604d"}   # red
METHOD_MARKER = {"BASELINE": "o", "RMA_TEACHER": "D", "RMA": "s", "CTS": "^"}
PRIV_ORDER    = ["BASE", "INT", "EXT", "FULL"]
PRIV_COLOR    = {"BASE": "#999999", "INT": "#4575b4", "EXT": "#f46d43", "FULL": "#1a9850"}
SIM_COLOR     = {"isaac": "#2166ac", "mujoco": "#d6604d"}

# OOD condition encoding — matches OpenTopic (no hatch + alpha=1 for 1×, hatch + alpha 0.75 for 2×)
COND_HATCH = {1.0: "",   2.0: "//"}
COND_ALPHA = {1.0: 1.00, 2.0: 0.70}


# ─────────────────────────────────────────────────────────────────────────────
# Loading / normalisation
# ─────────────────────────────────────────────────────────────────────────────
def _normalise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Lower/upper-case keys, fill optional columns, coerce dtypes."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # method
    df["method"] = df["method"].astype(str).str.strip().str.upper()
    df["method"] = df["method"].replace({"RMA2": "RMA", "BASE": "BASELINE", "BASELINE_GO2": "BASELINE"})
    # dr_scale
    if "dr_scale" not in df.columns:
        df["dr_scale"] = 1.0
    df["dr_scale"] = pd.to_numeric(df["dr_scale"], errors="coerce")
    # reward
    for c in ("mean_reward", "std_reward"):
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Column aliases — accept both the OpenTopic-style names (lin_vel_track) and
    # the unified CSV-writer names (mean_lin_track / mean_ang_track / mean_track_err).
    _alias = {
        "mean_lin_track": "lin_vel_track",
        "mean_ang_track": "ang_vel_track",
        "mean_track_err": "lin_track_err",
    }
    for src, dst in _alias.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    # Gait-metric columns (8 metrics × {mean, std}) — see scripts/gait_metrics.py.
    _gait_cols = []
    for c in ("gait_adh", "clear_err", "slip_rate", "smoothness",
              "base_z_var", "contact_sym", "stride_var", "jtorque_var"):
        _gait_cols += [c, f"{c}_std"]
    # optional metrics
    for c in ("success_rate", "partial_rate", "fall_rate", "survival_rate",
              "episode_length_s",
              "mean_length", "std_length",
              "lin_vel_track", "ang_vel_track", "lin_track_err",
              "std_lin_track", "std_ang_track", "std_track_err",
              "mean_fwd_disp", "std_fwd_disp",
              *_gait_cols):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # priv_mode / latent_dim
    if "priv_mode" not in df.columns:
        df["priv_mode"] = "FULL"
    df["priv_mode"] = df["priv_mode"].astype(str).str.strip().str.upper()
    df.loc[df["priv_mode"].isin(["", "NAN", "N/A", "NONE"]), "priv_mode"] = "FULL"
    df.loc[df["method"] == "BASELINE", "priv_mode"] = "BASE"
    if "latent_dim" not in df.columns:
        df["latent_dim"] = "8"
    df["latent_dim"] = df["latent_dim"].astype(str).str.strip()
    # pandas infers an all-numeric latent column as float -> astype(str) gives
    # "8.0"; an object column (mixed with "N/A") gives "8". Canonicalise so the
    # (method, priv, latent) join matches across the ood and sim2sim frames.
    df["latent_dim"] = df["latent_dim"].str.replace(r"\.0$", "", regex=True)
    df.loc[df["latent_dim"].isin(["", "nan", "N/A", "NaN", "None"]), "latent_dim"] = "—"
    df["source"] = source
    df = df.dropna(subset=["mean_reward", "dr_scale"])
    # Deduplicate: when the same (method, priv, latent, dr_scale) appears more
    # than once (e.g. old pre-fix run + new v2fix run), keep only the newest row
    # so that retrained checkpoints supersede earlier eval results.
    if "timestamp" in df.columns:
        df = df.copy()
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("_ts", ascending=False)
        dedup_keys = ["method", "priv_mode", "latent_dim", "dr_scale"]
        df = df.groupby(dedup_keys, as_index=False).first().reset_index(drop=True)
        df = df.drop(columns=["_ts"])
    return df


def load_csv(path: str, source: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        import io
        with open(path, "r") as f:
            lines = f.readlines()
        if lines:
            n_header = len(lines[0].split(","))
            fixed = [lines[0]]
            for ln in lines[1:]:
                cols = ln.rstrip("\n").split(",")
                if len(cols) > n_header:
                    # Newer eval runs accidentally inserted extra string fields
                    # (e.g. "terrain", "dist") between dr_scale and episode_length_s.
                    # Drop excess columns right after the 5th field (dr_scale).
                    excess = len(cols) - n_header
                    del cols[5:5 + excess]
                fixed.append(",".join(cols) + "\n")
            df = pd.read_csv(io.StringIO("".join(fixed)),
                             keep_default_na=False, na_values=[""])
        else:
            df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    except Exception as e:                                   # noqa: BLE001
        print(f"[plot] WARN could not read {path}: {e}")
        return pd.DataFrame()
    out = _normalise(df, source)
    print(f"[plot] loaded {len(out):3d} rows from {os.path.relpath(path, REPO_ROOT)} ({source})")
    return out


# ── Parse human-readable sim2sim_*.txt reports (fallback) ────────────────────
_TXT_PATTERNS = {
    "dr_scale":      r"DR[x×]\s*([0-9.]+)",
    "method":        r"Method\s*:\s*([A-Za-z0-9_]+)",
    "checkpoint":    r"Checkpoint\s*:\s*([^\s]+)",
    "mean_reward":   r"Mean reward\s*:\s*([+\-0-9.]+)\s*[±+\-]\s*([0-9.]+)",
    "success_rate":  r"Success rate\s*:\s*([0-9.]+)\s*%",
    "lin_vel_track": r"Lin vel track\s*:\s*([0-9.]+)",
    "ang_vel_track": r"Ang vel track\s*:\s*([0-9.]+)",
    "lin_track_err": r"Lin track err\s*:\s*([0-9.]+)",
}


def scan_log_txt_reports() -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(REPO_ROOT, "logs", "*", "*", "ood_eval", "*.txt")):
        txt = open(p, "r", errors="ignore").read()
        rec: dict = {}
        for key, pat in _TXT_PATTERNS.items():
            m = re.search(pat, txt)
            if not m:
                continue
            if key == "mean_reward":
                rec["mean_reward"] = float(m.group(1))
                rec["std_reward"]  = float(m.group(2))
            elif key in ("dr_scale", "success_rate", "lin_vel_track", "ang_vel_track", "lin_track_err"):
                rec[key] = float(m.group(1))
            else:
                rec[key] = m.group(1)
        # infer priv/latent from path/filename if encoded (e.g. _int_l16)
        fn = os.path.basename(p).lower()
        mm = re.search(r"_(full|int|ext)\b", fn)
        if mm:
            rec["priv_mode"] = mm.group(1).upper()
        ml = re.search(r"_l(\d+)\b", fn)
        if ml:
            rec["latent_dim"] = ml.group(1)
        if "method" in rec and "mean_reward" in rec:
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(f"[plot] parsed {len(df)} sim2sim report(s) from logs/*/ood_eval/*.txt")
    return _normalise(df, "mujoco")


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        try:
            fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight")
        except Exception as e:
            print(f"[plot] WARN could not save {name}.{ext}: {e}")
    plt.close(fig)
    print(f"[plot] wrote {name}.pdf / .png")


def _figure_legend(fig, methods_present, include_dr=True, include_sim=False,
                   y=0.02, fontsize=10):
    """Single figure-level legend placed BELOW all panels (no in-axes overlap).
    Combines method colours + DR-scale hatches (+ Isaac/MuJoCo marks if requested)."""
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=METHOD_COLOR[m], label=METHOD_LABEL[m])
               for m in methods_present]
    if include_dr:
        handles += [
            mpatches.Patch(facecolor="white", edgecolor="0.3", hatch="",
                           label="DR ×1 (training)"),
            mpatches.Patch(facecolor="white", edgecolor="0.3", hatch="//",
                           label="DR ×2 (OOD)"),
        ]
    if include_sim:
        handles += [
            mpatches.Patch(facecolor="0.6", alpha=1.00, label="Isaac"),
            mpatches.Patch(facecolor="0.6", alpha=0.75, hatch="//", label="MuJoCo"),
        ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, y),
               ncol=len(handles), frameon=False, fontsize=fontsize)


def _headline(df: pd.DataFrame) -> pd.DataFrame:
    """Headline config = the FULL / l=8 (or 'best available') row per method.
    When duplicate (method, dr_scale, priv, latent) rows exist, the most recent
    timestamp is preferred so v2fix checkpoint results supersede earlier runs."""
    if df.empty:
        return df
    d = df.copy()
    d["_priv_rank"] = d["priv_mode"].map({"BASE": 0, "FULL": 1, "INT": 2, "EXT": 3}).fillna(9)
    d["_lat"]       = pd.to_numeric(d["latent_dim"], errors="coerce")
    d["_lat_rank"]  = (d["_lat"] - 8).abs().fillna(99)   # prefer latent 8
    if "timestamp" in d.columns:
        d["_ts"] = pd.to_datetime(d["timestamp"], errors="coerce")
        d = d.sort_values(["method", "dr_scale", "_priv_rank", "_lat_rank", "_ts"],
                           ascending=[True, True, True, True, False])
    else:
        d = d.sort_values(["method", "dr_scale", "_priv_rank", "_lat_rank"])
    return d.groupby(["method", "dr_scale"], as_index=False).first()


# ── Fig 0: HEADLINE — single-panel "elevator pitch" of the project result ────
def fig_headline(ood: pd.DataFrame, sim: pd.DataFrame):
    """One-glance figure for a title slide / abstract figure / poster header.

    Shows the *single* most important number: Sim2Sim retention G(π,s) per
    method at DR×1 vs DR×2, with the 60 % spec line and PASS/FAIL stamps.
    All other figures are deep-dives off of this one.
    """
    if ood.empty or sim.empty:
        print("[plot] skip fig_go2_headline — need both Isaac and MuJoCo data")
        return
    dr_scales = sorted(set(ood["dr_scale"]).intersection(set(sim["dr_scale"])))
    methods_present = [m for m in METHOD_ORDER
                       if m in set(ood["method"]) and m in set(sim["method"])]
    if not (dr_scales and methods_present):
        print("[plot] skip fig_go2_headline — no overlapping data")
        return

    def _r(df, m, s):
        sub = df[(df["method"] == m) & (np.isclose(df["dr_scale"], s))]
        return float(sub["mean_reward"].iloc[0]) if len(sub) else np.nan

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    x = np.arange(len(methods_present)); w = 0.80 / len(dr_scales)
    threshold = 60.0
    pass_col   = "#1a9850"   # green
    fail_col   = "#b2182b"   # red
    import matplotlib.patches as mpatches

    # All bars, with annotations ABOVE the bars (no overlap with bar interior).
    for k, dr in enumerate(dr_scales):
        offset = (k - (len(dr_scales) - 1) / 2) * w
        for i, m in enumerate(methods_present):
            r_iso = _r(ood, m, dr); r_muj = _r(sim, m, dr)
            ret   = 100.0 * r_muj / r_iso if r_iso and r_iso > 1e-6 and not np.isnan(r_muj) else np.nan
            if np.isnan(ret): continue
            passed   = ret >= threshold
            edge_col = pass_col if passed else fail_col
            mark     = "✓" if passed else "✗"
            stamp    = "PASS" if passed else "FAIL"

            if ret >= 0:
                ax.bar(x[i] + offset, ret, w,
                       color=METHOD_COLOR[m],
                       hatch=COND_HATCH.get(dr, ""), alpha=COND_ALPHA.get(dr, 1.0),
                       edgecolor=edge_col, linewidth=2.8, zorder=2)
                # PASS/FAIL chip placed clearly ABOVE the bar.
                ax.annotate(f"{mark} {stamp}",
                            xy=(x[i] + offset, ret + 2.5),
                            ha="center", va="bottom",
                            fontsize=10, fontweight="bold",
                            color="white",
                            bbox=dict(boxstyle="round,pad=0.30",
                                      facecolor=edge_col, edgecolor="none"),
                            zorder=4)
                # Big % number above the chip.
                ax.annotate(f"{ret:.0f}%",
                            xy=(x[i] + offset, ret + 13),
                            ha="center", va="bottom",
                            fontsize=14, fontweight="bold", color="0.15",
                            zorder=4)
            else:
                # Negative retention (catastrophic failure): draw a downward
                # arrow from the x-axis with the actual value annotated.
                ax.annotate("",
                            xy=(x[i] + offset, 0),
                            xytext=(x[i] + offset, 8),
                            arrowprops=dict(arrowstyle="-|>", color=fail_col,
                                            lw=2.5), zorder=4)
                ax.annotate(f"✗ FAIL\n{ret:.0f}%",
                            xy=(x[i] + offset, 9),
                            ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color="white",
                            bbox=dict(boxstyle="round,pad=0.30",
                                      facecolor=fail_col, edgecolor="none"),
                            zorder=4)

    # Spec threshold line + label placed in the LEFT margin (no bar there).
    ax.axhline(threshold, color="0.2", lw=1.8, ls="--", alpha=0.85, zorder=1)
    ax.text(-0.50, threshold,
            f" 60 % spec ",
            va="center", ha="right", fontsize=10, fontstyle="italic",
            color="0.15",
            bbox=dict(boxstyle="round,pad=0.30",
                      facecolor="white", edgecolor="0.4"))
    ax.axhline(100.0, color="0.5", lw=0.8, ls=":", alpha=0.4, zorder=1)
    ax.text(-0.50, 100.0,
            " perfect ",
            va="center", ha="right", fontsize=9, fontstyle="italic",
            color="0.4",
            bbox=dict(boxstyle="round,pad=0.20",
                      facecolor="white", edgecolor="0.7"))

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods_present], fontsize=14)
    ax.set_ylabel("Sim-to-Sim retention   G($\\pi$, $s$)  [%]", fontsize=12)
    ax.set_ylim(0, 135)
    ax.set_xlim(-0.65, len(methods_present) - 0.35)        # room for left-margin labels
    ax.set_title("Go2 — Sim-to-Sim Transfer Performance",
                 fontsize=16, fontweight="bold", pad=16)

    # Single figure-level legend BELOW the plot (no overlap with bars).
    _figure_legend(fig, methods_present, include_dr=True, y=0.05, fontsize=11)
    # Subtitle / caption — one-sentence elevator pitch.
    fig.text(0.5, 0.01,
             "G($\\pi$, $s$) = reward retained in MuJoCo as a fraction of Isaac at the same DR scale $s$.   "
             "Higher = better sim-to-sim transfer.",
             ha="center", fontsize=10, fontstyle="italic", color="0.3")
    fig.tight_layout(rect=[0, 0.14, 1, 0.96])
    _save(fig, "fig_go2_headline")


# ── Fig 1: OOD test in Isaac (DR×1 vs DR×2 per method) ──────────────────────
def fig_ood_profile(ood: pd.DataFrame):
    """OOD test (Isaac only) — answers "how does the policy degrade from
    DR×1 to DR×2 per method?". Saved as both .pdf + .png.

    Layout:
      (A) OOD-gap retention  R_Isaac,2× / R_Isaac,1× × 100 %  per method
      (B) Absolute Isaac reward at DR×1 vs DR×2 per method
    """
    if ood.empty:
        print("[plot] skip fig_go2_ood_profile — no OOD data")
        return
    hl = _headline(ood)
    dr_scales = sorted(hl["dr_scale"].unique())
    methods_present = [m for m in METHOD_ORDER if m in set(hl["method"])]
    if not methods_present:
        print("[plot] skip fig_go2_ood_profile — no recognised methods"); return

    def _r(m, s):
        sub = hl[(hl["method"] == m) & (np.isclose(hl["dr_scale"], s))]
        if not len(sub): return np.nan, np.nan
        return float(sub["mean_reward"].iloc[0]), float(sub["std_reward"].iloc[0])

    # If only one DR scale → fall back to a single-panel bar chart.
    if len(dr_scales) < 2:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        x = np.arange(len(methods_present))
        for i, m in enumerate(methods_present):
            v, e = _r(m, dr_scales[0])
            ax.bar(x[i], v, yerr=e, capsize=3,
                   color=METHOD_COLOR[m], width=0.6, label=METHOD_LABEL[m])
        ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in methods_present])
        ax.set_xlabel(f"method  (Isaac OOD, $s$ = {dr_scales[0]:g})")
        ax.set_ylabel("episode return  (mean $\\pm$ std)")
        ax.set_title("Go2 — Isaac-Lab OOD robustness", fontweight="bold")
        _save(fig, "fig_go2_ood_profile"); return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    import matplotlib.patches as mpatches

    # ── (A) OOD-gap retention bars ─────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(methods_present)); w = 0.55
    dr_lo, dr_hi = dr_scales[0], dr_scales[-1]
    # Threshold + reference lines (LEFT-margin labels — no bar overlap).
    ax.axhline(70.0, color="0.2", lw=1.4, ls="--", alpha=0.8, zorder=1)
    ax.text(-0.55, 70.0, " 70% spec ",
            va="center", ha="right", fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5"))
    ax.axhline(100.0, color="0.5", lw=0.7, ls=":", alpha=0.5, zorder=1)
    rets = []
    for i, m in enumerate(methods_present):
        r_lo, _ = _r(m, dr_lo); r_hi, _ = _r(m, dr_hi)
        ret = 100.0 * r_hi / r_lo if r_lo and r_lo > 1e-6 and not np.isnan(r_hi) else np.nan
        rets.append(ret)
        if not np.isnan(ret):
            ax.bar(x[i], ret, w, color=METHOD_COLOR[m],
                   edgecolor="white", linewidth=0.6, alpha=COND_ALPHA.get(dr_hi, 0.7),
                   hatch=COND_HATCH.get(dr_hi, ""), zorder=2)
            ax.annotate(f"{ret:.0f}%", xy=(x[i], ret + 3),
                        ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color="0.15", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in methods_present])
    ax.set_ylabel(f"R_Isaac,{dr_hi:g}× / R_Isaac,{dr_lo:g}× [%]")
    ax.set_title("(A)  Isaac OOD retention  (DR×%g → DR×%g)" % (dr_lo, dr_hi),
                 fontweight="bold")
    ax.set_ylim(0, 125)
    ax.set_xlim(-0.7, len(methods_present) - 0.3)

    # ── (B) Absolute Isaac reward at each DR scale ─────────────────────────
    ax = axes[1]
    w2 = 0.85 / len(dr_scales)
    for k, dr in enumerate(dr_scales):
        offset = (k - (len(dr_scales) - 1) / 2) * w2
        for i, m in enumerate(methods_present):
            v, e = _r(m, dr)
            ax.bar(x[i] + offset, v, w2, yerr=e, capsize=2.5,
                   color=METHOD_COLOR[m],
                   hatch=COND_HATCH.get(dr, ""), alpha=COND_ALPHA.get(dr, 1.0),
                   edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in methods_present])
    ax.set_ylabel("episode return (mean $\\pm$ std)")
    ax.set_title("(B)  Isaac reward at DR×%g vs DR×%g" % (dr_lo, dr_hi),
                 fontweight="bold")
    # Figure-level legend BELOW both panels — no in-axes overlap.
    _figure_legend(fig, methods_present, include_dr=True, y=0.02, fontsize=10)
    fig.suptitle("Go2 — Isaac-Lab OOD test  (DR×%g → DR×%g per method)" % (dr_lo, dr_hi),
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    _save(fig, "fig_go2_ood_profile")


# ── Fig 2: sim-to-sim gap (Isaac vs MuJoCo at DR×1 AND DR×2) ────────────────
def fig_sim2sim_transfer(ood: pd.DataFrame, sim: pd.DataFrame):
    """Dedicated Sim2Sim gap figure — answers "how does the Isaac→MuJoCo gap
    change between DR×1 (in-distribution) and DR×2 (OOD), per method?"

    Layout (saved as both .pdf + .png):
      (A) Reward retention G(π,s) = R_MuJoCo,s / R_Isaac,s × 100 %, per method, per DR
      (B) Paired Isaac/MuJoCo absolute reward bars, per method, per DR
    Annotations make the gap legible without zooming.
    """
    if ood.empty or sim.empty:
        print("[plot] skip fig_go2_sim2sim_transfer — need both Isaac and MuJoCo data")
        return

    dr_scales = sorted(set(ood["dr_scale"]).intersection(set(sim["dr_scale"])))
    if not dr_scales:
        print("[plot] skip fig_go2_sim2sim_transfer — no overlapping DR scales")
        return

    methods_present = [m for m in METHOD_ORDER
                       if m in set(ood["method"]) and m in set(sim["method"])]
    if not methods_present:
        print("[plot] skip fig_go2_sim2sim_transfer — no overlapping methods")
        return

    # Build retention table.
    def _r(df, m, s):
        sub = df[(df["method"] == m) & (np.isclose(df["dr_scale"], s))]
        if not len(sub): return np.nan, np.nan
        return float(sub["mean_reward"].iloc[0]), float(sub["std_reward"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    import matplotlib.patches as mpatches

    # ── (A) Reward retention G(π) at each DR scale ────────────────────────
    ax = axes[0]
    x = np.arange(len(methods_present)); w = 0.80 / len(dr_scales)
    fail_col = "#b2182b"
    # Pre-compute all retention values to set a sensible y range.
    all_rets = []
    for k, dr in enumerate(dr_scales):
        for m in methods_present:
            r_iso, _ = _r(ood, m, dr); r_muj, _ = _r(sim, m, dr)
            if r_iso and r_iso > 1e-6 and not np.isnan(r_muj):
                all_rets.append(100.0 * r_muj / r_iso)
    y_min_A = min(0, min(all_rets) * 1.10) if all_rets else 0
    y_max_A = 130
    ax.axhline(60.0, color="0.2", lw=1.4, ls="--", alpha=0.8, zorder=1)
    ax.text(-0.70, 60.0, " 60% spec ",
            va="center", ha="right", fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5"))
    ax.axhline(100.0, color="0.5", lw=0.7, ls=":", alpha=0.5, zorder=1)
    ax.axhline(0.0, color="0.3", lw=0.8, alpha=0.6, zorder=1)
    for k, dr in enumerate(dr_scales):
        offset = (k - (len(dr_scales) - 1) / 2) * w
        for i, m in enumerate(methods_present):
            r_iso, _ = _r(ood, m, dr)
            r_muj, _ = _r(sim, m, dr)
            if not (r_iso and r_iso > 1e-6 and not np.isnan(r_muj)):
                continue
            ret = 100.0 * r_muj / r_iso
            if ret >= 0:
                ax.bar(x[i] + offset, ret, w,
                       color=METHOD_COLOR[m], hatch=COND_HATCH.get(dr, ""),
                       alpha=COND_ALPHA.get(dr, 1.0),
                       edgecolor="white", linewidth=0.6, zorder=2)
                ax.annotate(f"{ret:.0f}%",
                            xy=(x[i] + offset, ret + 3),
                            ha="center", va="bottom",
                            fontsize=10, fontweight="bold", color="0.15", zorder=3)
            else:
                # Negative retention: bar below zero, FAIL chip at axis crossing.
                ax.bar(x[i] + offset, ret, w,
                       color=METHOD_COLOR[m], hatch=COND_HATCH.get(dr, ""),
                       alpha=COND_ALPHA.get(dr, 1.0),
                       edgecolor=fail_col, linewidth=1.5, zorder=2)
                ax.annotate(f"✗ {ret:.0f}%",
                            xy=(x[i] + offset, -4),
                            ha="center", va="top",
                            fontsize=9, fontweight="bold", color="white",
                            bbox=dict(boxstyle="round,pad=0.25",
                                      facecolor=fail_col, edgecolor="none"),
                            zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods_present], rotation=15, ha="right")
    ax.set_ylabel("MuJoCo reward / Isaac reward  [%]")
    ax.set_title("(A)  Sim-to-sim retention  G($\\pi$, $s$)", fontweight="bold")
    ax.set_ylim(y_min_A, y_max_A)
    ax.set_xlim(-0.7, len(methods_present) - 0.3)
    # (legend placed at figure-level — see _figure_legend call near the end)

    # ── (B) Absolute reward — Isaac vs MuJoCo, per (method, DR) ──────────
    ax = axes[1]
    n_conds = 2 * len(dr_scales)        # Isaac + MuJoCo per DR scale
    w2      = 0.85 / n_conds
    cond_seq = [("Isaac", dr) for dr in dr_scales] + [("MuJoCo", dr) for dr in dr_scales]
    cond_seq.sort(key=lambda t: (t[1], 0 if t[0] == "Isaac" else 1))   # I1, M1, I2, M2
    for i, m in enumerate(methods_present):
        for j, (sim_lab, dr) in enumerate(cond_seq):
            df = ood if sim_lab == "Isaac" else sim
            v, e = _r(df, m, dr)
            offset = (j - (n_conds - 1) / 2) * w2
            # Isaac = full opacity, MuJoCo = lighter + hatch
            alpha = 1.0 if sim_lab == "Isaac" else 0.75
            hatch = "" if sim_lab == "Isaac" else "//"
            ax.bar(x[i] + offset, v, w2, yerr=e, capsize=1.8,
                   color=METHOD_COLOR[m], alpha=alpha,
                   hatch=hatch, edgecolor="white", linewidth=0.5)
        # Annotate per-DR sim2sim drop above the Isaac bar pairs.
        y_top = 0
        for dr in dr_scales:
            r_iso, _ = _r(ood, m, dr); r_muj, _ = _r(sim, m, dr)
            if not np.isnan(r_iso) and not np.isnan(r_muj):
                drop = r_iso - r_muj
                # x position of midpoint of this (Isaac, MuJoCo) DR pair
                j_iso = cond_seq.index(("Isaac",  dr))
                j_muj = cond_seq.index(("MuJoCo", dr))
                xm = x[i] + ((j_iso + j_muj) / 2 - (n_conds - 1) / 2) * w2
                ym = max(r_iso, r_muj) * 1.04
                ax.annotate(f"$\\Delta${drop:+.0f}",
                            xy=(xm, ym), ha="center", va="bottom",
                            fontsize=8, color="0.25", fontstyle="italic")
                y_top = max(y_top, ym)
    ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in methods_present])
    ax.set_ylabel("episode return (mean $\\pm$ std)")
    ax.set_title("(B)  Absolute reward — Isaac vs MuJoCo", fontweight="bold")

    # Single figure-level legend BELOW both panels — methods + DR + simulator.
    _figure_legend(fig, methods_present, include_dr=True, include_sim=True,
                   y=0.02, fontsize=10)
    fig.suptitle("Go2 — Sim2Sim gap at DR×1 vs DR×2  "
                 "($\\Delta$ = Isaac $-$ MuJoCo)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    _save(fig, "fig_go2_sim2sim_transfer")


# ── Fig 3: latent dimension ablation ─────────────────────────────────────────
def fig_latent_ablation(ood: pd.DataFrame, sim: pd.DataFrame):
    frames = []
    for df, src in [(ood, "Isaac OOD"), (sim, "MuJoCo")]:
        if df.empty:
            continue
        d = df.copy()
        d["_lat"] = pd.to_numeric(d["latent_dim"], errors="coerce")
        d = d[(d["priv_mode"] == "FULL") & d["_lat"].notna()]
        if d["_lat"].nunique() < 2:
            continue
        # take the smallest dr_scale (in-distribution headline) per (method, latent)
        s = d["dr_scale"].min()
        d = d[d["dr_scale"] == s]
        d["src"] = f"{src} (s={s:g})"
        frames.append(d)
    if not frames:
        print("[plot] skip fig_go2_latent_ablation — need >=2 latent dims with priv=FULL")
        return
    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    styles = {}
    for (method, src), sub in data.groupby(["method", "src"]):
        sub = sub.sort_values("_lat")
        ls = "-" if "Isaac" in src else "--"
        ax.errorbar(sub["_lat"], sub["mean_reward"], yerr=sub["std_reward"],
                    marker=METHOD_MARKER.get(method, "o"), color=METHOD_COLOR.get(method, "#333"),
                    ls=ls, capsize=2.5, lw=1.5, ms=5,
                    label=f"{METHOD_LABEL.get(method, method)} · {src}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(data["_lat"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("latent dimension  $Z$")
    ax.set_ylabel("episode return  (mean $\\pm$ std)")
    ax.set_title("Go2 — latent-dimension ablation (priv = FULL)")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, "fig_go2_latent_ablation")


# ── Fig: gait-quality panels (8 metrics x methods) ──────────────────────────
_GAIT_METRICS = [
    ("gait_adh",    "gait adherence",      "higher better, [0,1]"),
    ("contact_sym", "contact symmetry",    "higher better, [0,1]"),
    ("clear_err",   "swing clearance err.", "lower better"),
    ("slip_rate",   "foot slip rate",      "lower better"),
    ("smoothness",  "action smoothness",   "lower better"),
    ("base_z_var",  "base-height variance", "lower better"),
    ("stride_var",  "stride variance",     "lower better"),
    ("jtorque_var", "joint-torque variance", "lower better"),
]


def fig_gait_quality(ood: pd.DataFrame, sim: pd.DataFrame):
    """2×4 grid of bar charts comparing the 8 gait metrics across methods.
    Shows both DR×1 and DR×2 for Isaac and MuJoCo (4 bars per method group).
    Encoding: colour = method, hatch = simulator, alpha = DR scale."""
    rows = []
    for df, src_label in [(ood, "Isaac"), (sim, "MuJoCo")]:
        if df.empty or not all(c in df.columns for c, _, _ in _GAIT_METRICS):
            continue
        hl = _headline(df)
        for _, r in hl.iterrows():
            entry = {"method": r["method"], "sim": src_label,
                     "dr": float(r["dr_scale"])}
            for col, _, _ in _GAIT_METRICS:
                entry[col] = float(r[col]) if pd.notna(r.get(col)) else np.nan
            rows.append(entry)
    if not rows:
        print("[plot] skip fig_go2_gait_quality — gait columns absent or empty")
        return
    data = pd.DataFrame(rows)
    methods_present = [m for m in METHOD_ORDER if m in set(data["method"])]
    if not methods_present:
        print("[plot] skip fig_go2_gait_quality — no recognised methods")
        return
    sims_present = [s for s in ("Isaac", "MuJoCo") if s in set(data["sim"])]
    dr_scales    = sorted(data["dr"].unique())

    # 4 conditions per method: (Isaac DR×1, Isaac DR×2, MuJoCo DR×1, MuJoCo DR×2)
    conditions = [(s, dr) for s in sims_present for dr in dr_scales]
    sim_hatch  = {"Isaac": "",   "MuJoCo": "//"}
    dr_alpha   = {1.0: 1.00,    2.0: 0.50}
    dr_edge    = {1.0: "white", 2.0: "black"}
    dr_lw      = {1.0: 0.4,     2.0: 0.8}

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 5.5))
    for k, (col, title, direction) in enumerate(_GAIT_METRICS):
        ax = axes[k // 4, k % 4]
        x = np.arange(len(methods_present))
        w = 0.80 / max(1, len(conditions))
        for j, (simlabel, dr) in enumerate(conditions):
            offset = (j - (len(conditions) - 1) / 2) * w
            for i, m in enumerate(methods_present):
                sub = data[(data["method"] == m) & (data["sim"] == simlabel)
                           & np.isclose(data["dr"], dr)]
                v = float(sub[col].iloc[0]) if len(sub) and pd.notna(sub[col].iloc[0]) else np.nan
                ax.bar(x[i] + offset, v, w,
                       color=METHOD_COLOR[m],
                       hatch=sim_hatch[simlabel],
                       alpha=dr_alpha[dr],
                       edgecolor=dr_edge[dr],
                       linewidth=dr_lw[dr])
        short = [METHOD_LABEL[m].replace("RMA Teacher", "RMA\nTeacher")
                                 .replace("RMA Student", "RMA\nStudent") for m in methods_present]
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7)
        ax.set_title(f"{title}\n({direction})", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    import matplotlib.patches as mpatches
    method_handles = [mpatches.Patch(color=METHOD_COLOR[m], label=METHOD_LABEL[m])
                      for m in methods_present]
    sim_handles = [
        mpatches.Patch(facecolor="0.6", alpha=1.00, hatch="",   label="Isaac"),
        mpatches.Patch(facecolor="0.6", alpha=0.70, hatch="//", label="MuJoCo"),
    ]
    dr_handles = [
        mpatches.Patch(facecolor="0.6", alpha=1.00, edgecolor="white", lw=0.4, label="DR×1"),
        mpatches.Patch(facecolor="0.6", alpha=0.50, edgecolor="black", lw=0.8, label="DR×2"),
    ]
    fig.legend(handles=method_handles + sim_handles + dr_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.00),
               ncol=len(method_handles) + len(sim_handles) + len(dr_handles),
               frameon=False, fontsize=9)
    fig.suptitle("Go2 — Gait-quality metrics  (FULL / $Z$=8 / DR×1 and DR×2)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.09, 1, 0.94])
    _save(fig, "fig_go2_gait_quality")


# ── Fig 4: privileged-knowledge ablation (FULL / INT / EXT, + BASE) ──────────
def fig_priv_ablation(ood: pd.DataFrame, sim: pd.DataFrame):
    base = ood if not ood.empty else sim
    src_name = "Isaac OOD" if not ood.empty else "MuJoCo"
    if base.empty:
        print("[plot] skip fig_go2_priv_ablation — no data")
        return
    d = base.copy()
    s = d["dr_scale"].min()
    d = d[d["dr_scale"] == s]
    # restrict to the headline latent (8 if available, else the only one present)
    d["_lat"] = pd.to_numeric(d["latent_dim"], errors="coerce")
    lat_pref = 8 if (d["_lat"] == 8).any() else (d["_lat"].dropna().iloc[0] if d["_lat"].notna().any() else None)
    arch_present = [m for m in ("RMA", "CTS") if m in set(d["method"])]
    priv_present = [p for p in PRIV_ORDER if p in set(d["priv_mode"])]
    if len(arch_present) == 0 or len([p for p in priv_present if p != "BASE"]) < 2:
        print("[plot] skip fig_go2_priv_ablation — need >=2 privileged subsets for RMA/CTS")
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(len(arch_present)); w = 0.8 / max(1, len(priv_present))
    for j, p in enumerate(priv_present):
        vals, errs = [], []
        for m in arch_present:
            sub = d[(d["method"] == m) & (d["priv_mode"] == p)]
            if p != "BASE" and lat_pref is not None and (sub["_lat"] == lat_pref).any():
                sub = sub[sub["_lat"] == lat_pref]
            vals.append(float(sub["mean_reward"].iloc[0]) if len(sub) else np.nan)
            errs.append(float(sub["std_reward"].iloc[0]) if len(sub) else 0.0)
        ax.bar(x + (j - (len(priv_present) - 1) / 2) * w, vals, w, yerr=errs, capsize=2,
               color=PRIV_COLOR.get(p, "#777"), label=p)
    # BASE (Baseline) reference line
    bsub = base[(base["method"] == "BASELINE") & (base["dr_scale"] == s)]
    if len(bsub):
        ax.axhline(float(bsub["mean_reward"].iloc[0]), color=PRIV_COLOR["BASE"],
                   lw=1.2, ls=":", label="Baseline")
    ax.set_xticks(x); ax.set_xticklabels(arch_present)
    ax.set_ylabel("episode return  (mean $\\pm$ std)")
    lat_txt = f", $Z$={int(lat_pref)}" if lat_pref is not None else ""
    ax.set_title(f"Go2 — privileged-knowledge ablation ({src_name}, $s$={s:g}{lat_txt})")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    _save(fig, "fig_go2_priv_ablation")


# ─────────────────────────────────────────────────────────────────────────────
# Detailed CTS privileged-subset ablation (FULL vs INT vs EXT) — 4 views × 2 metrics
# ─────────────────────────────────────────────────────────────────────────────
def _cts_priv_pick(df: pd.DataFrame, priv: str, dr: float, col: str,
                   sim_tag: str | None = None) -> tuple[float, float]:
    """Return (mean, std) from a normalised CSV for CTS at the given priv/dr.
    Picks latent_dim=8 if present, otherwise the only latent_dim available.
    Std column is auto-inferred as "std_"+col if present, else 0."""
    if df.empty:
        return (np.nan, 0.0)
    d = df[(df["method"] == "CTS") & (df["priv_mode"] == priv)
           & (df["dr_scale"] == dr)]
    if sim_tag is not None and "sim" in d.columns:
        d = d[d["sim"] == sim_tag]
    if d.empty:
        return (np.nan, 0.0)
    lat = pd.to_numeric(d["latent_dim"], errors="coerce")
    if (lat == 8).any():
        d = d[lat == 8]
    elif lat.notna().any():
        d = d[lat == lat.dropna().iloc[0]]
    if col not in d.columns or d.empty:
        return (np.nan, 0.0)
    val = float(d[col].iloc[0]) if pd.notna(d[col].iloc[0]) else np.nan
    std_col = "std_" + col
    err = float(d[std_col].iloc[0]) if (std_col in d.columns
                                         and pd.notna(d[std_col].iloc[0])) else 0.0
    return (val, err)


def fig_cts_priv_ablation(ood: pd.DataFrame, sim: pd.DataFrame):
    """CTS-only FULL/INT/EXT ablation across 4 (sim × DR) views, two metric rows:
    top = episode reward, bottom = success rate. Skipped if INT or EXT are missing."""
    if ood.empty and sim.empty:
        print("[plot] skip fig_go2_cts_priv_ablation — no data"); return
    privs = ["FULL", "INT", "EXT"]
    # Need both INT and EXT data for at least one (sim,dr) cell to make this figure useful.
    have = False
    for src in (ood, sim):
        if src.empty:
            continue
        sub = src[src["method"] == "CTS"]
        if {"INT", "EXT"}.issubset(set(sub["priv_mode"])):
            have = True; break
    if not have:
        print("[plot] skip fig_go2_cts_priv_ablation — CTS INT/EXT not present in CSVs"); return
    # Tag rows by sim if not already tagged (sim2sim CSV has "sim"; OOD does not).
    if "sim" not in ood.columns:
        ood = ood.copy(); ood["sim"] = "isaac"
    if "sim" not in sim.columns:
        sim = sim.copy(); sim["sim"] = "mujoco"
    views = [
        ("isaac",  1.0, "Isaac  DR×1"),
        ("isaac",  2.0, "Isaac  DR×2"),
        ("mujoco", 1.0, "MuJoCo DR×1"),
        ("mujoco", 2.0, "MuJoCo DR×2"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.0), sharey="row")
    x = np.arange(len(privs))
    w = 0.65
    src_for_sim = {"isaac": ood, "mujoco": sim}
    for col, (sim_tag, dr, title) in enumerate(views):
        df_src = src_for_sim[sim_tag]
        # ── reward (top row) ────────────────────────────────────────
        ax_r = axes[0, col]
        rew  = [_cts_priv_pick(df_src, p, dr, "mean_reward")    for p in privs]
        for i, (p, (v, e)) in enumerate(zip(privs, rew)):
            ax_r.bar(i, v, w, yerr=e, capsize=2.5,
                     color=PRIV_COLOR[p], edgecolor="black", linewidth=0.6,
                     alpha=COND_ALPHA[dr],
                     hatch=COND_HATCH[dr] if dr != 1.0 else "")
            if not np.isnan(v):
                ax_r.text(i, v + (e if not np.isnan(e) else 0) + 25,
                          f"{v:.0f}", ha="center", va="bottom", fontsize=9)
        ax_r.set_xticks(x); ax_r.set_xticklabels(privs)
        ax_r.set_title(title, fontsize=11, pad=6)
        if col == 0:
            ax_r.set_ylabel("episode return\n(mean $\\pm$ std)")
        ax_r.set_ylim(bottom=0)
        # ── success rate (bottom row) ───────────────────────────────
        ax_s = axes[1, col]
        succ = [_cts_priv_pick(df_src, p, dr, "success_rate")[0] for p in privs]
        for i, (p, v) in enumerate(zip(privs, succ)):
            ax_s.bar(i, v, w,
                     color=PRIV_COLOR[p], edgecolor="black", linewidth=0.6,
                     alpha=COND_ALPHA[dr],
                     hatch=COND_HATCH[dr] if dr != 1.0 else "")
            if not np.isnan(v):
                ax_s.text(i, v + 1.5, f"{v:.0f}%",
                          ha="center", va="bottom", fontsize=9)
        ax_s.axhline(80, color="#444", lw=0.8, ls="--", alpha=0.7)
        ax_s.set_xticks(x); ax_s.set_xticklabels(privs)
        if col == 0:
            ax_s.set_ylabel("success rate  [%]")
            ax_s.text(-0.18, 80, "80% spec", color="#444", fontsize=8,
                      va="center", ha="right", transform=ax_s.get_yaxis_transform())
        ax_s.set_ylim(0, 110)
    fig.suptitle("Go2 — CTS privileged-subset ablation  (FULL vs INT vs EXT, $Z$=8)",
                 fontsize=13, y=0.995)
    # Figure-level legend (priv subsets only)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PRIV_COLOR[p],
                              edgecolor="black", linewidth=0.6, label=p) for p in privs]
    fig.legend(handles=handles, ncol=3, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _save(fig, "fig_go2_cts_priv_ablation")


# ─────────────────────────────────────────────────────────────────────────────
# CTS priv-ablation — behaviour (gait-quality) metrics, mirrors fig_gait_quality
# but on the FULL/INT/EXT axis instead of the Baseline/RMA/CTS method axis.
# ─────────────────────────────────────────────────────────────────────────────
def _fig_cts_priv_ablation_gait_one(df: pd.DataFrame, privs_present, sims_present,
                                    dr_use: float, file_stem: str,
                                    title_suffix: str):
    """Render one 2×4 gait-metric panel for CTS priv subsets at a given DR scale."""
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.2))
    sim_style = {"isaac": dict(hatch="", alpha=1.00),
                 "mujoco": dict(hatch="//", alpha=0.70)}
    for k, (col, title, direction) in enumerate(_GAIT_METRICS):
        ax = axes[k // 4, k % 4]
        x = np.arange(len(privs_present))
        w = 0.8 / max(1, len(sims_present))
        for j, (sim_tag, sim_label) in enumerate(sims_present):
            offset = (j - (len(sims_present) - 1) / 2) * w
            for i, p in enumerate(privs_present):
                sub = df[(df["priv_mode"] == p) & (df["sim"] == sim_tag)]
                v = float(sub[col].iloc[0]) if len(sub) and pd.notna(sub[col].iloc[0]) else np.nan
                ax.bar(x[i] + offset, v, w,
                       color=PRIV_COLOR[p],
                       hatch=sim_style[sim_tag]["hatch"],
                       alpha=sim_style[sim_tag]["alpha"],
                       edgecolor="white", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(privs_present, fontsize=8, rotation=0)
        ax.set_title(f"{title}\n({direction})", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    import matplotlib.patches as mpatches
    priv_handles = [mpatches.Patch(color=PRIV_COLOR[p], label=p) for p in privs_present]
    sim_handles  = [
        mpatches.Patch(facecolor="0.6", alpha=1.00, label="Isaac (solid)"),
        mpatches.Patch(facecolor="0.6", alpha=0.70, hatch="//", label="MuJoCo (hatched)"),
    ]
    fig.legend(handles=priv_handles + sim_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.01),
               ncol=len(priv_handles) + len(sim_handles),
               frameon=False, fontsize=10)
    fig.suptitle(f"Go2 — CTS priv-ablation: Gait-quality metrics  "
                 f"(CTS / $Z$=8 / DR×{dr_use:g}){title_suffix}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    _save(fig, file_stem)


def fig_cts_priv_ablation_gait(ood: pd.DataFrame, sim: pd.DataFrame):
    """Two 2×4 grids of bar charts comparing the 8 gait metrics across CTS priv
    subsets (FULL/INT/EXT) — one figure per DR scale.

    Outputs:
        fig_go2_cts_priv_ablation_gait_dr1.{pdf,png}    DR×1 (training distribution)
        fig_go2_cts_priv_ablation_gait_dr2.{pdf,png}    DR×2 (OOD = "worst-DR axis")

    Skipped if INT/EXT are absent."""
    if ood.empty and sim.empty:
        print("[plot] skip fig_go2_cts_priv_ablation_gait — no data"); return
    if "sim" not in ood.columns:
        ood = ood.copy(); ood["sim"] = "isaac"
    if "sim" not in sim.columns:
        sim = sim.copy(); sim["sim"] = "mujoco"
    df_all = pd.concat([ood, sim], ignore_index=True)
    df_all = df_all[(df_all["method"] == "CTS")]
    df_all["_lat"] = pd.to_numeric(df_all["latent_dim"], errors="coerce")
    if (df_all["_lat"] == 8).any():
        df_all = df_all[df_all["_lat"] == 8]
    privs_present = [p for p in ("FULL", "INT", "EXT") if p in set(df_all["priv_mode"])]
    if not {"INT", "EXT"}.issubset(set(privs_present)):
        print("[plot] skip fig_go2_cts_priv_ablation_gait — CTS INT/EXT not present"); return
    if not all(c in df_all.columns for c, _, _ in _GAIT_METRICS):
        print("[plot] skip fig_go2_cts_priv_ablation_gait — gait columns missing"); return
    sims_present = [("isaac", "Isaac"), ("mujoco", "MuJoCo")]
    sims_present = [(t, l) for (t, l) in sims_present if t in set(df_all["sim"])]

    drs_available = sorted(set(float(v) for v in df_all["dr_scale"].unique()))
    # Always produce DR×1 (headline) and DR×2 (OOD) if both are present.
    for dr_use in (1.0, 2.0):
        if dr_use not in drs_available:
            continue
        df = df_all[df_all["dr_scale"] == dr_use]
        suffix = "" if dr_use == 1.0 else "  —  OOD (worst-DR axis)"
        stem   = f"fig_go2_cts_priv_ablation_gait_dr{int(dr_use)}"
        _fig_cts_priv_ablation_gait_one(df, privs_present, sims_present,
                                        dr_use, stem, suffix)


# ─────────────────────────────────────────────────────────────────────────────
# Spec-sheet sim2sim transfer ratios:
#   G(π)         = R_MuJoCo,1× / R_Isaac,1× × 100   (≥ 60% — Da et al.)
#   OOD gap      = R_Isaac ,2× / R_Isaac,1× × 100   (≥ 70% — LocoFormer)
#   Combined gap = R_MuJoCo,2× / R_Isaac,1× × 100   (≥ 40% — Da et al.)
# ─────────────────────────────────────────────────────────────────────────────
def compute_transfer_gaps(ood: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """Combine Isaac OOD and MuJoCo sim2sim CSV rows to compute G(π), OOD gap,
    and Combined gap per (method, priv_mode, latent_dim).
    Returns an empty DataFrame if either source is missing the required rows."""
    rows = []
    if ood.empty and sim.empty:
        return pd.DataFrame()
    keys = pd.concat([
        ood[["method", "priv_mode", "latent_dim"]] if not ood.empty else pd.DataFrame(),
        sim[["method", "priv_mode", "latent_dim"]] if not sim.empty else pd.DataFrame(),
    ], ignore_index=True).drop_duplicates()

    def _r(df, method, priv, lat, s):
        sub = df[(df["method"] == method) & (df["priv_mode"] == priv)
                 & (df["latent_dim"] == lat) & (df["dr_scale"] == s)]
        return float(sub["mean_reward"].iloc[0]) if len(sub) else np.nan

    for _, k in keys.iterrows():
        m, p, l = k["method"], k["priv_mode"], k["latent_dim"]
        r_iso_1  = _r(ood, m, p, l, 1.0) if not ood.empty else np.nan
        r_iso_2  = _r(ood, m, p, l, 2.0) if not ood.empty else np.nan
        r_muj_1  = _r(sim, m, p, l, 1.0) if not sim.empty else np.nan
        r_muj_2  = _r(sim, m, p, l, 2.0) if not sim.empty else np.nan
        def _ratio(num, den):
            if np.isnan(num) or np.isnan(den) or abs(den) < 1e-6:
                return np.nan
            return 100.0 * num / den
        rows.append({
            "method":       METHOD_LABEL.get(m, m),
            "priv_mode":    p,
            "latent_dim":   l,
            "R_isaac_1x":   r_iso_1,
            "R_isaac_2x":   r_iso_2,
            "R_mujoco_1x":  r_muj_1,
            "R_mujoco_2x":  r_muj_2,
            "G_pi_pct":     _ratio(r_muj_1, r_iso_1),
            "OOD_gap_pct":  _ratio(r_iso_2, r_iso_1),
            "combined_pct": _ratio(r_muj_2, r_iso_1),
        })
    return pd.DataFrame(rows)


def fig_rma_phase2_loss(lc_path: str | None = None):
    """Plot RMA Phase-2 MSE loss (adaptation module, teacher→student gap).

    Reads results/learning_curves_go2.csv (or the path given) for method==RMA_Phase2.
    Shows the full training curve with a rolling mean and annotates convergence level.
    """
    import csv as _csv, io as _io

    if lc_path is None:
        lc_path = os.path.join(REPO_ROOT, "results", "learning_curves_go2.csv")
    if not os.path.exists(lc_path):
        print(f"[plot] skip fig_rma_phase2_loss — {lc_path} not found")
        return

    # Load only Phase-2 rows
    iters, losses = [], []
    with open(lc_path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row.get("method") == "RMA_Phase2" and row.get("mse_loss", ""):
                try:
                    iters.append(int(row["iteration"]))
                    losses.append(float(row["mse_loss"]))
                except ValueError:
                    pass

    if not iters:
        print("[plot] skip fig_rma_phase2_loss — no RMA_Phase2 rows in learning_curves_go2.csv")
        return

    iters  = np.array(iters)
    losses = np.array(losses)

    # Separate iteration-0 pre-training point from the training curve
    has_init = iters[0] == 0 and len(iters) > 1 and iters[1] > 1
    if has_init:
        init_iter, init_loss = iters[0], losses[0]
        train_iters, train_losses = iters[1:], losses[1:]
    else:
        train_iters, train_losses = iters, losses
        init_iter = init_loss = None

    # Rolling mean over training curve only (window=50)
    win = 50
    padded = np.pad(train_losses, (win // 2, win - 1 - win // 2), mode="edge")
    roll   = np.convolve(padded, np.ones(win) / win, mode="valid")

    plateau_mean = float(np.mean(train_losses[-100:]))
    first_loss   = float(losses[0])

    fig, ax = plt.subplots(figsize=(8, 4))

    # Raw training curve (faded)
    ax.plot(train_iters, train_losses, color="#aaaaaa", lw=0.6, alpha=0.55,
            label="Per-iteration MSE")
    # Rolling mean
    ax.plot(train_iters, roll, color=METHOD_COLOR["RMA"], lw=2.0,
            label=f"Rolling mean (w={win})")

    # Pre-training initial point
    if has_init:
        ax.axvline(train_iters[0], color="#888888", lw=1.0, ls=":", alpha=0.6)
        ax.scatter([init_iter], [init_loss], color="black", zorder=5, s=40,
                   label=f"Pre-training loss = {init_loss:.3f}")
        ax.annotate(f"warmup ends →\ntraining starts",
                    xy=(train_iters[0], init_loss),
                    xytext=(train_iters[0] + len(train_iters) * 0.05, init_loss * 1.04),
                    fontsize=8, color="#555555",
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

    # Horizontal reference lines
    ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.5,
               label="Teacher (oracle, MSE = 0)")
    ax.axhline(plateau_mean, color="#d6604d", lw=1.2, ls=":",
               label=f"Student plateau ≈ {plateau_mean:.3f}")

    # Gap annotation
    mid_iter = train_iters[len(train_iters) // 2]
    ax.annotate("", xy=(mid_iter, 0), xytext=(mid_iter, plateau_mean),
                arrowprops=dict(arrowstyle="<->", color="#762a83", lw=1.8))
    ax.text(mid_iter + len(train_iters) * 0.01, plateau_mean / 2,
            f"Gap ≈ {plateau_mean:.3f}", va="center", color="#762a83", fontsize=9)

    ax.set_xlabel("Phase-2 Iteration")
    ax.set_ylabel("MSE Loss  (teacher z vs student ẑ)")
    ax.set_title("RMA Phase-2: Adaptation Module Training\n"
                 "(teacher = oracle encoder, student = history-based encoder)")
    ax.set_ylim(-0.05, max(losses) * 1.15)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_go2_rma_phase2_loss")
    print(f"[plot] fig_go2_rma_phase2_loss  plateau={plateau_mean:.4f}  "
          f"init={first_loss:.4f}  iters={len(iters)}")


def _pass(value, threshold):
    """Return a "PASS"/"FAIL"/"—" marker for a numeric value against a >= threshold."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return "PASS" if value >= threshold else "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# Fig: 4-view comparison matrix
#   2×2 grid in one figure (PDF + PNG), one panel per question:
#     (A) Isaac OOD       — DR×1 vs DR×2 in Isaac        (OOD test inside Isaac)
#     (B) MuJoCo OOD      — DR×1 vs DR×2 in MuJoCo        (OOD test inside MuJoCo)
#     (C) Sim2Sim @ DR×1  — Isaac vs MuJoCo at DR×1       (cross-sim in-distribution)
#     (D) Sim2Sim @ DR×2  — Isaac vs MuJoCo at DR×2       (cross-sim out-of-distribution)
#   All 4 panels share the same y-axis so visual comparison across them is fair.
# ─────────────────────────────────────────────────────────────────────────────
def fig_comparison_matrix(ood: pd.DataFrame, sim: pd.DataFrame):
    if ood.empty or sim.empty:
        print("[plot] skip fig_go2_comparison — need both Isaac and MuJoCo data")
        return
    dr_scales = sorted(set(ood["dr_scale"]).intersection(set(sim["dr_scale"])))
    methods_present = [m for m in METHOD_ORDER
                       if m in set(ood["method"]) and m in set(sim["method"])]
    if not (dr_scales and methods_present):
        print("[plot] skip fig_go2_comparison — no overlap"); return
    if len(dr_scales) < 2:
        print("[plot] skip fig_go2_comparison — need ≥2 DR scales"); return
    dr_lo, dr_hi = dr_scales[0], dr_scales[-1]

    def _r(df, m, s):
        sub = df[(df["method"] == m) & (np.isclose(df["dr_scale"], s))]
        if not len(sub): return np.nan, np.nan
        return float(sub["mean_reward"].iloc[0]), float(sub["std_reward"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), sharey=True)
    x = np.arange(len(methods_present)); w = 0.34

    # Common y-axis limits — handle negative rewards (e.g. RMA Student in MuJoCo).
    all_means, all_tops = [], []
    for m in methods_present:
        for df in (ood, sim):
            for s in dr_scales:
                v, e = _r(df, m, s)
                if not np.isnan(v):
                    all_means.append(v)
                    all_tops.append(v + e)
    y_top = max(all_tops) * 1.25 if all_tops else 1500
    y_min = min(0, min(all_means) * 1.15) if all_means else 0

    def _panel(ax, title_prefix, title_q, left, right, left_lbl, right_lbl,
               annot_prefix="Δ"):
        """Generic 2-bar-per-method panel.

        left[i], right[i]    : (mean, std) tuples per method
        left_lbl, right_lbl  : strings for the in-panel mini-legend
        """
        for i, m in enumerate(methods_present):
            v_l, e_l = left[i]; v_r, e_r = right[i]
            ax.bar(x[i] - w/2, v_l, w, yerr=e_l, capsize=2.5,
                   color=METHOD_COLOR[m], alpha=1.0,
                   edgecolor="white", linewidth=0.5, zorder=2)
            ax.bar(x[i] + w/2, v_r, w, yerr=e_r, capsize=2.5,
                   color=METHOD_COLOR[m], alpha=0.70, hatch="//",
                   edgecolor="white", linewidth=0.5, zorder=2)
            # Δ annotation BELOW the x-axis label to avoid any overlap.
            if not np.isnan(v_l) and not np.isnan(v_r) and v_l > 1e-6:
                drop_abs = v_l - v_r
                drop_pct = 100.0 * (v_r / v_l)
                ax.annotate(f"{annot_prefix}{drop_abs:+.0f}\n({drop_pct:.0f}%)",
                            xy=(x[i], y_top * 0.97),
                            ha="center", va="top",
                            fontsize=9, fontweight="bold",
                            color="0.20", zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in methods_present])
        ax.set_title(title_prefix + "  " + title_q, fontweight="bold")
        # Tiny in-panel mini-legend that does NOT overlap bars: put it in the
        # axes title area instead, using a text box at the top-left corner.
        ax.text(0.02, 0.97, f"solid: {left_lbl}\nhatched: {right_lbl}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color="0.25",
                bbox=dict(boxstyle="round,pad=0.30",
                          facecolor="white", edgecolor="0.7", alpha=0.85))
        ax.set_ylim(y_min, y_top)
        ax.set_ylabel("episode return (mean $\\pm$ std)")

    # (A) Isaac OOD — DR×1 vs DR×2 in Isaac
    left_A  = [_r(ood, m, dr_lo) for m in methods_present]
    right_A = [_r(ood, m, dr_hi) for m in methods_present]
    _panel(axes[0, 0], "(A)",
           f"Isaac OOD:  DR×{dr_lo:g} vs DR×{dr_hi:g}",
           left_A, right_A, f"Isaac DR×{dr_lo:g}", f"Isaac DR×{dr_hi:g}")

    # (B) MuJoCo OOD — DR×1 vs DR×2 in MuJoCo
    left_B  = [_r(sim, m, dr_lo) for m in methods_present]
    right_B = [_r(sim, m, dr_hi) for m in methods_present]
    _panel(axes[0, 1], "(B)",
           f"MuJoCo OOD:  DR×{dr_lo:g} vs DR×{dr_hi:g}",
           left_B, right_B, f"MuJoCo DR×{dr_lo:g}", f"MuJoCo DR×{dr_hi:g}")

    # (C) Sim2Sim @ DR×1 — Isaac vs MuJoCo at DR×1
    left_C  = [_r(ood, m, dr_lo) for m in methods_present]
    right_C = [_r(sim, m, dr_lo) for m in methods_present]
    _panel(axes[1, 0], "(C)",
           f"Sim2Sim @ DR×{dr_lo:g}:  Isaac vs MuJoCo",
           left_C, right_C, f"Isaac DR×{dr_lo:g}", f"MuJoCo DR×{dr_lo:g}")

    # (D) Sim2Sim @ DR×2 — Isaac vs MuJoCo at DR×2
    left_D  = [_r(ood, m, dr_hi) for m in methods_present]
    right_D = [_r(sim, m, dr_hi) for m in methods_present]
    _panel(axes[1, 1], "(D)",
           f"Sim2Sim @ DR×{dr_hi:g}:  Isaac vs MuJoCo  (worst-case)",
           left_D, right_D, f"Isaac DR×{dr_hi:g}", f"MuJoCo DR×{dr_hi:g}")

    # Method-only legend at figure level (DR / Sim encoding is shown per panel).
    import matplotlib.patches as mpatches
    method_handles = [mpatches.Patch(color=METHOD_COLOR[m], label=METHOD_LABEL[m])
                      for m in methods_present]
    fig.legend(handles=method_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=len(method_handles),
               frameon=False, fontsize=11, title="Method")

    fig.suptitle(
        "Go2 — Four-view performance comparison    "
        "(rows: OOD inside sim ↑   |   sim-to-sim transfer ↓)",
        fontsize=12, fontweight="bold")
    fig.text(0.5, 0.05,
             "Δ annotation = (right_bar − left_bar);   "
             "(N%) = right_bar / left_bar × 100  (retention)",
             ha="center", fontsize=9, fontstyle="italic", color="0.35")
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    _save(fig, "fig_go2_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Helper used by the per-result-metric 4-view figures (survival / vel-RMSE / outcome).
def _4view_pairs(ood, sim, methods_present, dr_lo, dr_hi, value_col, std_col=None):
    """Return four (left, right) lists of (mean, std) tuples, one per 4-view panel."""
    def _v(df, m, s):
        sub = df[(df["method"] == m) & (np.isclose(df["dr_scale"], s))]
        if not len(sub): return np.nan, np.nan
        mean = float(sub[value_col].iloc[0]) if value_col in sub.columns and pd.notna(sub[value_col].iloc[0]) else np.nan
        std  = float(sub[std_col].iloc[0])   if std_col and std_col in sub.columns and pd.notna(sub[std_col].iloc[0]) else 0.0
        return mean, std

    return {
        "A_isaac_ood":   ([_v(ood, m, dr_lo) for m in methods_present], [_v(ood, m, dr_hi) for m in methods_present],
                          f"Isaac DR×{dr_lo:g}",   f"Isaac DR×{dr_hi:g}"),
        "B_mujoco_ood":  ([_v(sim, m, dr_lo) for m in methods_present], [_v(sim, m, dr_hi) for m in methods_present],
                          f"MuJoCo DR×{dr_lo:g}",  f"MuJoCo DR×{dr_hi:g}"),
        "C_sim2sim_lo":  ([_v(ood, m, dr_lo) for m in methods_present], [_v(sim, m, dr_lo) for m in methods_present],
                          f"Isaac DR×{dr_lo:g}",   f"MuJoCo DR×{dr_lo:g}"),
        "D_sim2sim_hi":  ([_v(ood, m, dr_hi) for m in methods_present], [_v(sim, m, dr_hi) for m in methods_present],
                          f"Isaac DR×{dr_hi:g}",   f"MuJoCo DR×{dr_hi:g}"),
    }


def _4view_bar_panel(ax, methods_present, left_pairs, right_pairs,
                     title, left_lbl, right_lbl, ylabel,
                     threshold=None, threshold_label=None, threshold_cmp=">=",
                     value_fmt="{:.0f}", y_max=None, error_bars=False):
    """Generic 4-view bar panel: methods × 2 bars (left, right). Used by survival /
    vel-RMSE / generic single-value comparisons."""
    x = np.arange(len(methods_present)); w = 0.34
    if threshold is not None:
        ax.axhline(threshold, color="0.2", lw=1.4, ls="--", alpha=0.8, zorder=1)
        ax.text(-0.55, threshold, f" {threshold_label} ",
                va="center", ha="right", fontsize=9, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5"))
    for i, m in enumerate(methods_present):
        v_l, e_l = left_pairs[i]; v_r, e_r = right_pairs[i]
        ax.bar(x[i] - w/2, v_l, w,
               yerr=(e_l if error_bars else None), capsize=2.5,
               color=METHOD_COLOR[m], alpha=1.0,
               edgecolor="white", linewidth=0.5, zorder=2)
        ax.bar(x[i] + w/2, v_r, w,
               yerr=(e_r if error_bars else None), capsize=2.5,
               color=METHOD_COLOR[m], alpha=0.70, hatch="//",
               edgecolor="white", linewidth=0.5, zorder=2)
        # Numeric annotations above each bar.
        for xc, v in [(x[i] - w/2, v_l), (x[i] + w/2, v_r)]:
            if not np.isnan(v):
                ax.annotate(value_fmt.format(v),
                            xy=(xc, v + (y_max * 0.02 if y_max else 1.5)),
                            ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color="0.15", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in methods_present])
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.7, len(methods_present) - 0.3)
    if y_max is not None:
        ax.set_ylim(0, y_max)
    # Small inset that says what the two bar styles mean in THIS panel.
    ax.text(0.02, 0.97, f"solid: {left_lbl}\nhatched: {right_lbl}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="0.25",
            bbox=dict(boxstyle="round,pad=0.30",
                      facecolor="white", edgecolor="0.7", alpha=0.85))


def _fig_metric_4view(ood, sim, value_col, std_col, title_metric, ylabel,
                      file_stem, threshold=None, threshold_label=None,
                      value_fmt="{:.0f}", y_max=None, error_bars=False):
    """Render a 2×2 4-view figure for a single result metric."""
    if ood.empty or sim.empty:
        print(f"[plot] skip {file_stem} — need both Isaac and MuJoCo"); return
    dr_scales = sorted(set(ood["dr_scale"]).intersection(set(sim["dr_scale"])))
    if len(dr_scales) < 2:
        print(f"[plot] skip {file_stem} — need >= 2 DR scales"); return
    if value_col not in ood.columns or value_col not in sim.columns:
        print(f"[plot] skip {file_stem} — column '{value_col}' missing"); return
    methods_present = [m for m in METHOD_ORDER
                       if m in set(ood["method"]) and m in set(sim["method"])]
    if not methods_present:
        print(f"[plot] skip {file_stem} — no overlapping methods"); return

    dr_lo, dr_hi = dr_scales[0], dr_scales[-1]
    panels = _4view_pairs(ood, sim, methods_present, dr_lo, dr_hi, value_col, std_col)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), sharey=True)
    titles = {
        "A_isaac_ood":  f"(A)  Isaac OOD:  DR×{dr_lo:g} vs DR×{dr_hi:g}",
        "B_mujoco_ood": f"(B)  MuJoCo OOD:  DR×{dr_lo:g} vs DR×{dr_hi:g}",
        "C_sim2sim_lo": f"(C)  Sim2Sim @ DR×{dr_lo:g}:  Isaac vs MuJoCo",
        "D_sim2sim_hi": f"(D)  Sim2Sim @ DR×{dr_hi:g}:  Isaac vs MuJoCo  (worst-case)",
    }
    keys = ["A_isaac_ood", "B_mujoco_ood", "C_sim2sim_lo", "D_sim2sim_hi"]
    for ax, k in zip(axes.flat, keys):
        l, r, l_lbl, r_lbl = panels[k]
        _4view_bar_panel(ax, methods_present, l, r,
                         title=titles[k], left_lbl=l_lbl, right_lbl=r_lbl,
                         ylabel=ylabel, threshold=threshold,
                         threshold_label=threshold_label, value_fmt=value_fmt,
                         y_max=y_max, error_bars=error_bars)

    # One method legend below all panels.
    import matplotlib.patches as mpatches
    method_handles = [mpatches.Patch(color=METHOD_COLOR[m], label=METHOD_LABEL[m])
                      for m in methods_present]
    fig.legend(handles=method_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=len(method_handles),
               frameon=False, fontsize=11, title="Method")
    fig.suptitle(f"Go2 — Four-view comparison:  {title_metric}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    _save(fig, file_stem)


def fig_comparison_survival(ood, sim):
    """4-view survival rate (spec sheet ≥ 80%)."""
    _fig_metric_4view(
        ood, sim,
        value_col="survival_rate", std_col=None,
        title_metric="Survival rate  (success + partial; spec ≥ 80%)",
        ylabel="survival rate  [%]",
        file_stem="fig_go2_comparison_survival",
        threshold=80.0, threshold_label="80% spec",
        value_fmt="{:.0f}%", y_max=115,
    )


def fig_comparison_rmse(ood, sim):
    """4-view velocity-tracking RMSE (spec sheet < 0.3 m/s)."""
    # Vel-RMSE auto-scales to the data; smaller values are better, so a 0.3 line is
    # the spec-sheet upper bound (failure boundary).
    all_vals = []
    for df in (ood, sim):
        if "mean_track_err" in df.columns:
            all_vals += [v for v in df["mean_track_err"].dropna()]
    y_max = max(all_vals) * 1.45 if all_vals else 1.0
    _fig_metric_4view(
        ood, sim,
        value_col="mean_track_err", std_col="std_track_err",
        title_metric="Velocity-tracking RMSE  (spec < 0.3 m/s; lower is better)",
        ylabel="vel-RMSE  [m/s]",
        file_stem="fig_go2_comparison_rmse",
        threshold=0.3, threshold_label="0.3 m/s spec",
        value_fmt="{:.2f}", y_max=y_max, error_bars=True,
    )


def fig_comparison_outcome(ood, sim):
    """4-view outcome breakdown — stacked success / partial / fail per (method, condition)."""
    if ood.empty or sim.empty:
        print("[plot] skip fig_go2_comparison_outcome — need both Isaac and MuJoCo"); return
    dr_scales = sorted(set(ood["dr_scale"]).intersection(set(sim["dr_scale"])))
    if len(dr_scales) < 2:
        print("[plot] skip fig_go2_comparison_outcome — need >= 2 DR scales"); return
    methods_present = [m for m in METHOD_ORDER
                       if m in set(ood["method"]) and m in set(sim["method"])]
    if not methods_present:
        print("[plot] skip fig_go2_comparison_outcome — no methods"); return
    needed = ["success_rate", "partial_rate", "fall_rate"]
    if not all(c in ood.columns and c in sim.columns for c in needed):
        print("[plot] skip fig_go2_comparison_outcome — outcome columns missing"); return
    dr_lo, dr_hi = dr_scales[0], dr_scales[-1]

    def _v(df, m, s, col):
        # Headline priv_mode per method: Baseline = BASE, RMA/CTS = FULL.
        # Without this filter the alphabetical first row (EXT) leaks into the
        # CTS bars and the "CTS" column silently shows the EXT-ablation result.
        priv = "BASE" if m == "BASELINE" else "FULL"
        sub = df[(df["method"] == m) & (np.isclose(df["dr_scale"], s))
                 & (df["priv_mode"] == priv)]
        if not len(sub) or pd.isna(sub[col].iloc[0]): return 0.0
        return float(sub[col].iloc[0])

    def _stacked_panel(ax, left_get, right_get, title, left_lbl, right_lbl):
        x = np.arange(len(methods_present)); w = 0.34
        # Left bar = first condition; right bar = second condition.
        for i, m in enumerate(methods_present):
            for side, getfn in [(-w/2, left_get), (+w/2, right_get)]:
                s_v = getfn(m, "success_rate")
                p_v = getfn(m, "partial_rate")
                f_v = getfn(m, "fall_rate")
                ax.bar(x[i] + side, s_v, w, color="#1a9850",
                       edgecolor="white", linewidth=0.4, zorder=2)
                ax.bar(x[i] + side, p_v, w, bottom=s_v, color="#fdae61",
                       edgecolor="white", linewidth=0.4, zorder=2)
                ax.bar(x[i] + side, f_v, w, bottom=s_v + p_v, color="#b2182b",
                       edgecolor="white", linewidth=0.4, zorder=2)
        ax.axhline(80.0, color="0.2", lw=1.2, ls="--", alpha=0.7, zorder=1)
        ax.text(-0.55, 80.0, " 80% spec ",
                va="center", ha="right", fontsize=9, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5"))
        ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in methods_present])
        ax.set_ylim(0, 110); ax.set_xlim(-0.7, len(methods_present) - 0.3)
        ax.set_ylabel("episodes  [%]")
        ax.set_title(title, fontweight="bold")
        # Per-panel hint: left bar / right bar identity.
        ax.text(0.02, 0.97, f"left bar:  {left_lbl}\nright bar:  {right_lbl}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color="0.25",
                bbox=dict(boxstyle="round,pad=0.30",
                          facecolor="white", edgecolor="0.7", alpha=0.85))

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), sharey=True)
    _stacked_panel(axes[0, 0],
                   lambda m, c: _v(ood, m, dr_lo, c), lambda m, c: _v(ood, m, dr_hi, c),
                   f"(A)  Isaac OOD:  DR×{dr_lo:g} vs DR×{dr_hi:g}",
                   f"Isaac DR×{dr_lo:g}", f"Isaac DR×{dr_hi:g}")
    _stacked_panel(axes[0, 1],
                   lambda m, c: _v(sim, m, dr_lo, c), lambda m, c: _v(sim, m, dr_hi, c),
                   f"(B)  MuJoCo OOD:  DR×{dr_lo:g} vs DR×{dr_hi:g}",
                   f"MuJoCo DR×{dr_lo:g}", f"MuJoCo DR×{dr_hi:g}")
    _stacked_panel(axes[1, 0],
                   lambda m, c: _v(ood, m, dr_lo, c), lambda m, c: _v(sim, m, dr_lo, c),
                   f"(C)  Sim2Sim @ DR×{dr_lo:g}:  Isaac vs MuJoCo",
                   f"Isaac DR×{dr_lo:g}", f"MuJoCo DR×{dr_lo:g}")
    _stacked_panel(axes[1, 1],
                   lambda m, c: _v(ood, m, dr_hi, c), lambda m, c: _v(sim, m, dr_hi, c),
                   f"(D)  Sim2Sim @ DR×{dr_hi:g}:  Isaac vs MuJoCo  (worst-case)",
                   f"Isaac DR×{dr_hi:g}", f"MuJoCo DR×{dr_hi:g}")

    # Methods are implicit via x-axis; outcome legend below.
    import matplotlib.patches as mpatches
    outcome_handles = [
        mpatches.Patch(color="#1a9850", label="success  (time-out · vel_rmse < 0.3)"),
        mpatches.Patch(color="#fdae61", label="partial  (time-out · vel_rmse ≥ 0.3)"),
        mpatches.Patch(color="#b2182b", label="fail  (fell)"),
    ]
    fig.legend(handles=outcome_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False, fontsize=11,
               title="Outcome class")
    fig.suptitle("Go2 — Four-view comparison:  Episode outcome  (success / partial / fail)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    _save(fig, "fig_go2_comparison_outcome")


# ─────────────────────────────────────────────────────────────────────────────
# Fig: spec-sheet summary dashboard
#   2×2 grid in one figure (saved as both PDF and PNG):
#     (a) episode return per method × (sim, DR)
#     (b) outcome breakdown — stacked bars success/partial/fail
#     (c) velocity-tracking RMSE with the 0.3 m/s threshold line
#     (d) PASS/FAIL grid against the 5 spec-sheet thresholds
# ─────────────────────────────────────────────────────────────────────────────
def _conditions_data(ood: pd.DataFrame, sim: pd.DataFrame):
    """Build a wide-form DataFrame indexed by method, with columns per condition."""
    rows = []
    for df, label in [(ood, "iso"), (sim, "muj")]:
        if df is None or df.empty:
            continue
        hl = _headline(df).copy()
        for _, r in hl.iterrows():
            rows.append({
                "method":     r["method"],
                "sim":        label,
                "dr":         float(r["dr_scale"]),
                "mean_reward": float(r["mean_reward"]) if pd.notna(r["mean_reward"]) else np.nan,
                "std_reward":  float(r["std_reward"])  if pd.notna(r.get("std_reward"))  else 0.0,
                "success":     float(r["success_rate"]) if "success_rate" in r and pd.notna(r["success_rate"]) else np.nan,
                "partial":     float(r["partial_rate"]) if "partial_rate" in r and pd.notna(r["partial_rate"]) else np.nan,
                "fall":        float(r["fall_rate"])    if "fall_rate"    in r and pd.notna(r["fall_rate"])    else np.nan,
                "vel_rmse":    float(r["lin_track_err"]) if "lin_track_err" in r and pd.notna(r.get("lin_track_err")) else np.nan,
                "vel_rmse_std":float(r["std_track_err"]) if "std_track_err" in r and pd.notna(r.get("std_track_err")) else 0.0,
            })
    return pd.DataFrame(rows)


def _grouped_bar(ax, data: pd.DataFrame, value_col: str, err_col: str | None = None,
                 ylabel: str = "", title: str = "", ylim=None, hline=None, hline_label="",
                 show_method_legend: bool = True):
    """OpenTopic-style grouped bars: x-axis = condition (Isaac 1×/2× | MuJoCo 1×/2×),
    bars within each group coloured by method. DR×2 conditions get hatch + alpha 0.7."""
    conds = [("iso", 1.0), ("iso", 2.0), ("muj", 1.0), ("muj", 2.0)]
    cond_labels = ["Isaac\n1×", "Isaac\n2×", "MuJoCo\n1×", "MuJoCo\n2×"]
    methods_present = [m for m in METHOD_ORDER if m in set(data["method"])]
    if not methods_present:
        ax.set_visible(False); return
    x = np.arange(len(conds)); w = 0.85 / len(methods_present)
    legend_handles_set = set()
    for k, m in enumerate(methods_present):
        offset = (k - (len(methods_present) - 1) / 2) * w
        for j, (sim_lab, dr_v) in enumerate(conds):
            row = data[(data["method"] == m) & (data["sim"] == sim_lab)
                       & (np.isclose(data["dr"], dr_v))]
            v = float(row[value_col].iloc[0]) if len(row) and pd.notna(row[value_col].iloc[0]) else np.nan
            e = float(row[err_col].iloc[0]) if (err_col and len(row)
                  and pd.notna(row[err_col].iloc[0])) else 0.0
            ax.bar(x[j] + offset, v, w,
                   yerr=e if err_col else None, capsize=2,
                   color=METHOD_COLOR[m], hatch=COND_HATCH[dr_v],
                   alpha=COND_ALPHA[dr_v], edgecolor="white", linewidth=0.5,
                   label=(METHOD_LABEL[m] if (show_method_legend and m not in legend_handles_set) else None))
            legend_handles_set.add(m)
    # Visually group Isaac vs MuJoCo with a thin vertical divider.
    ax.axvline(1.5, color="0.7", lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(cond_labels)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    if ylim is not None: ax.set_ylim(*ylim)
    if hline is not None:
        ax.axhline(hline, color="k", lw=1.0, ls="--", alpha=0.6)
        ax.text(ax.get_xlim()[1], hline, f"  {hline_label}",
                va="center", ha="left", fontsize=8, color="k")


def _outcome_stacked(ax, data: pd.DataFrame):
    """Stacked bars (success / partial / fail) per (method, sim, DR).

    Bars are grouped by method (visual gap between groups). X-axis shows the
    short condition tag (I1× / I2× / M1× / M2×) directly under each bar, and a
    bold method name (Baseline / RMA / CTS) is placed below the tick row as a
    group label. Legend is hoisted to the top of the panel — above the bars —
    so it never overlaps the data.
    """
    conds = [("iso", 1.0, "I1×"), ("iso", 2.0, "I2×"),
             ("muj", 1.0, "M1×"), ("muj", 2.0, "M2×")]
    methods_present = [m for m in METHOD_ORDER if m in set(data["method"])]
    if not methods_present:
        ax.set_visible(False); return

    n_conds = len(conds)
    group_gap = 0.8      # gap (in bar-width units) between method groups
    bar_positions, cond_tags, group_centers = [], [], []
    for k in range(len(methods_present)):
        start = k * (n_conds + group_gap)
        for j, (_, _, ctag) in enumerate(conds):
            bar_positions.append(start + j)
            cond_tags.append(ctag)
        group_centers.append(start + (n_conds - 1) / 2)
    bar_positions = np.array(bar_positions, dtype=float)

    succ_v, part_v, fail_v = [], [], []
    for k, m in enumerate(methods_present):
        for sim_lab, dr_v, _ in conds:
            row = data[(data["method"] == m) & (data["sim"] == sim_lab)
                       & (np.isclose(data["dr"], dr_v))]
            if not len(row):
                succ_v.append(0.0); part_v.append(0.0); fail_v.append(0.0); continue
            succ_v.append(float(row["success"].iloc[0]) if pd.notna(row["success"].iloc[0]) else 0.0)
            p = row.get("partial", pd.Series([np.nan])).iloc[0]
            f = row.get("fall",    pd.Series([np.nan])).iloc[0]
            part_v.append(float(p) if pd.notna(p) else 0.0)
            fail_v.append(float(f) if pd.notna(f) else 0.0)

    ax.bar(bar_positions, succ_v, color="#1a9850", label="success",
           edgecolor="white", linewidth=0.4, zorder=2)
    ax.bar(bar_positions, part_v, bottom=succ_v, color="#fdae61", label="partial",
           edgecolor="white", linewidth=0.4, zorder=2)
    bot2 = [(s or 0) + (p or 0) for s, p in zip(succ_v, part_v)]
    ax.bar(bar_positions, fail_v, bottom=bot2, color="#b2182b", label="fail",
           edgecolor="white", linewidth=0.4, zorder=2)

    # 80% survival line + label in the LEFT margin (no overlap with rightmost bars).
    ax.axhline(80, color="0.2", lw=1.2, ls="--", alpha=0.7, zorder=1)
    ax.text(bar_positions[0] - 1.1, 80, " 80% spec ",
            va="center", ha="right", fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5"))

    # X-tick labels: short condition tags only (no rotation needed).
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(cond_tags, fontsize=9, rotation=0)
    # Method group names BELOW the x-tick row.
    for centre, m in zip(group_centers, methods_present):
        ax.annotate(METHOD_LABEL[m],
                    xy=(centre, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -28), textcoords="offset points",
                    ha="center", va="top", fontsize=11, fontweight="bold")

    # Headroom + legend at the TOP of the panel (between bars and title).
    ax.set_ylim(0, 130)
    ax.set_xlim(bar_positions[0] - 1.3, bar_positions[-1] + 0.7)
    ax.set_ylabel("episodes [%]")
    ax.set_title("(B)  Outcome breakdown (success / partial / fail)",
                 fontweight="bold", pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00),
              ncol=3, frameon=False, fontsize=10)


def _passfail_grid(ax, ood: pd.DataFrame, sim: pd.DataFrame, data: pd.DataFrame):
    """5-column PASS/FAIL grid: Survival≥80, vel_rmse<0.3, G(π)≥60, OOD≥70, Combined≥40."""
    methods_present = [m for m in METHOD_ORDER if m in set(data["method"])]
    if not methods_present:
        ax.set_visible(False); return
    gaps = compute_transfer_gaps(ood, sim)
    cols = [
        ("Survival\nIsaac 1×",  "iso", 1.0, "survival", 80.0, ">="),
        ("Survival\nMuJoCo 1×", "muj", 1.0, "survival", 80.0, ">="),
        ("vel_rmse\nMuJoCo 1×", "muj", 1.0, "vel_rmse",  0.3, "<="),
        ("G(π)\n≥60%",     "ratio", None, "G_pi_pct",       60.0, ">="),
        ("OOD gap\n≥70%",  "ratio", None, "OOD_gap_pct",   70.0, ">="),
        ("Comb gap\n≥40%", "ratio", None, "combined_pct",  40.0, ">="),
    ]
    n_m, n_c = len(methods_present), len(cols)
    grid = np.zeros((n_m, n_c)); annot = np.empty((n_m, n_c), dtype=object)
    for i, m in enumerate(methods_present):
        for j, (_lbl, src, dr, key, thr, cmp_op) in enumerate(cols):
            if src == "ratio":
                row = gaps[gaps["method"] == METHOD_LABEL.get(m, m)]
                v = float(row[key].iloc[0]) if len(row) and pd.notna(row[key].iloc[0]) else np.nan
                tag = f"{v:.0f}%" if not np.isnan(v) else "—"
            else:
                row = data[(data["method"] == m) & (data["sim"] == src) & (np.isclose(data["dr"], dr))]
                if not len(row):
                    v = np.nan; tag = "—"
                elif key == "survival":
                    v = (row["success"].iloc[0] + (row.get("partial", pd.Series([0.0])).iloc[0] or 0.0))
                    tag = f"{v:.0f}%" if not np.isnan(v) else "—"
                else:
                    v = float(row[key].iloc[0]) if pd.notna(row[key].iloc[0]) else np.nan
                    tag = f"{v:.2f}" if not np.isnan(v) else "—"
            if np.isnan(v):
                grid[i, j] = 0.0; annot[i, j] = "—"
            else:
                ok = (v >= thr) if cmp_op == ">=" else (v <= thr)
                grid[i, j] = 1.0 if ok else -1.0
                annot[i, j] = f"{tag}\n{'PASS' if ok else 'FAIL'}"
    cmap = matplotlib.colors.ListedColormap(["#fdd0a2", "#f0f0f0", "#a6d96a"])
    norm = matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(n_c)); ax.set_xticklabels([c[0] for c in cols], fontsize=8)
    ax.set_yticks(range(n_m)); ax.set_yticklabels([METHOD_LABEL[m] for m in methods_present])
    for i in range(n_m):
        for j in range(n_c):
            txt = annot[i, j] if annot[i, j] is not None else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_title("(D)  Spec-sheet thresholds", fontweight="bold")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)


def fig_summary_dashboard(ood: pd.DataFrame, sim: pd.DataFrame):
    data = _conditions_data(ood, sim)
    if data.empty:
        print("[plot] skip fig_go2_summary — no data")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

    # (A) cumulative episode reward — method colours (legend at figure level)
    _grouped_bar(axes[0, 0], data, "mean_reward", "std_reward",
                 ylabel="episode return (mean $\\pm$ std)",
                 title="(A)  Cumulative episode reward",
                 show_method_legend=False)

    # (B) survival rate — method colours (same colour grammar as A and C).
    #     Outcome 3-class breakdown lives in fig_go2_comparison_outcome — a
    #     dedicated figure where success/partial/fail green/orange/red colours
    #     don't compete with the method-colour scheme used everywhere else.
    if "survival" not in data.columns:
        data = data.copy()
        data["survival"] = data["success"].fillna(0.0) + data["partial"].fillna(0.0)
    _grouped_bar(axes[0, 1], data, "survival",
                 ylabel="survival rate  [%]",
                 title="(B)  Survival rate  (success + partial)",
                 hline=80.0, hline_label="80% (spec)", ylim=(0, 110),
                 show_method_legend=False)

    # (C) velocity-tracking RMSE — method colours
    _grouped_bar(axes[1, 0], data, "vel_rmse", "vel_rmse_std",
                 ylabel="vel_rmse [m/s]",
                 title="(C)  Velocity-tracking RMSE",
                 hline=0.3, hline_label="0.3 m/s (spec)",
                 show_method_legend=False)

    # (D) spec-sheet PASS/FAIL grid
    _passfail_grid(axes[1, 1], ood, sim, data)

    # Figure-level legend BELOW all 4 panels — methods + DR scale.
    methods_in_data = [m for m in METHOD_ORDER if m in set(data["method"])]
    _figure_legend(fig, methods_in_data, include_dr=True, y=0.01, fontsize=10)
    fig.suptitle("Go2 — Spec-sheet summary    "
                 "Baseline / RMA Teacher / RMA Student / CTS  ·  FULL · $Z$=8  ·  DR×{1, 2} × {Isaac, MuJoCo}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    _save(fig, "fig_go2_summary")


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────
def write_tables(ood: pd.DataFrame, sim: pd.DataFrame):
    rows = []
    for df, src in [(ood, "Isaac"), (sim, "MuJoCo")]:
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({
                "sim": src,
                "method": METHOD_LABEL.get(r["method"], r["method"]),
                "priv": r["priv_mode"],
                "Z": r["latent_dim"],
                "s": f"{r['dr_scale']:g}",
                "return": f"{r['mean_reward']:.1f} ± {r['std_reward']:.1f}"
                          if pd.notna(r["std_reward"]) else f"{r['mean_reward']:.1f}",
                "success%": (f"{r['success_rate']:.0f}" if "success_rate" in df.columns
                             and pd.notna(r.get("success_rate")) else "—"),
            })
    if not rows:
        print("[plot] skip results table — no data")
        return
    tab = pd.DataFrame(rows).sort_values(["sim", "method", "priv", "Z", "s"]).reset_index(drop=True)

    md_path = os.path.join(REPO_ROOT, "results", "go2_results_table.md")
    with open(md_path, "w") as f:
        f.write("# Go2 (Phase 2) — evaluation results\n\n")
        f.write("_Auto-generated by scripts/plot_results_go2.py. "
                "Numbers come from results/ood_go2.csv and results/sim2sim_go2.csv "
                "(or parsed logs). Rows are TBD until the corresponding eval job has run._\n\n")
        f.write("## Per-row evaluation results\n\n")
        f.write("| sim | method | priv | Z | s | episode return | success % |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, r in tab.iterrows():
            f.write(f"| {r['sim']} | {r['method']} | {r['priv']} | {r['Z']} | "
                    f"{r['s']} | {r['return']} | {r['success%']} |\n")
        # ── Spec-sheet transfer ratios ────────────────────────────────────────
        gaps = compute_transfer_gaps(ood, sim)
        if not gaps.empty:
            f.write("\n## Sim2Sim transfer ratios (spec sheet, page 2)\n\n")
            f.write("Reward-based ratios over (method × priv × latent), with the spec-sheet "
                    "threshold check applied:\n")
            f.write("- **G(π)** = R_MuJoCo,1× / R_Isaac,1× × 100% (target ≥ 60%)\n")
            f.write("- **OOD gap** = R_Isaac,2× / R_Isaac,1× × 100% (target ≥ 70%)\n")
            f.write("- **Combined gap** = R_MuJoCo,2× / R_Isaac,1× × 100% (target ≥ 40%)\n\n")
            f.write("| method | priv | Z | R_iso,1× | R_iso,2× | R_muj,1× | R_muj,2× | "
                    "G(π) | OOD gap | Combined |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for _, r in gaps.iterrows():
                def _fr(x): return f"{x:.1f}" if not np.isnan(x) else "—"
                def _fp(x, thr): return (f"{x:.1f}% [{_pass(x, thr)}]"
                                          if not np.isnan(x) else "—")
                f.write(f"| {r['method']} | {r['priv_mode']} | {r['latent_dim']} | "
                        f"{_fr(r['R_isaac_1x'])} | {_fr(r['R_isaac_2x'])} | "
                        f"{_fr(r['R_mujoco_1x'])} | {_fr(r['R_mujoco_2x'])} | "
                        f"{_fp(r['G_pi_pct'], 60)} | {_fp(r['OOD_gap_pct'], 70)} | "
                        f"{_fp(r['combined_pct'], 40)} |\n")
    print(f"[plot] wrote {os.path.relpath(md_path, REPO_ROOT)}")

    tex_path = os.path.join(REPO_ROOT, "results", "go2_results_table.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by scripts/plot_results_go2.py\n")
        f.write("\\begin{tabular}{llllrr}\n\\toprule\n")
        f.write("Sim & Method & Priv & $Z$ & $s$ & Episode return & Succ.\\% \\\\\n\\midrule\n")
        for _, r in tab.iterrows():
            ret = r["return"].replace("±", "$\\pm$")
            f.write(f"{r['sim']} & {r['method']} & {r['priv']} & {r['Z']} & "
                    f"{r['s']} & {ret} & {r['success%']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[plot] wrote {os.path.relpath(tex_path, REPO_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# CTS-only privileged-subset ablation table (FULL vs INT vs EXT)
# ─────────────────────────────────────────────────────────────────────────────
def write_cts_priv_ablation_table(ood: pd.DataFrame, sim: pd.DataFrame):
    """Write results/go2_cts_priv_ablation_table.md — one row per (sim × DR × priv)
    with reward, success, vel_rmse, and Δ vs FULL columns. Skipped if INT/EXT are
    not in the CSVs."""
    if ood.empty and sim.empty:
        return
    if "sim" not in ood.columns:
        ood = ood.copy(); ood["sim"] = "isaac"
    if "sim" not in sim.columns:
        sim = sim.copy(); sim["sim"] = "mujoco"
    df = pd.concat([ood, sim], ignore_index=True)
    df = df[df["method"] == "CTS"]
    if df.empty:
        return
    privs_present = set(df["priv_mode"])
    if not {"INT", "EXT"}.issubset(privs_present):
        print("[plot] skip go2_cts_priv_ablation_table — need INT and EXT rows for CTS")
        return
    privs = [p for p in ("FULL", "INT", "EXT") if p in privs_present]
    # Pick latent=8 if present
    df["_lat"] = pd.to_numeric(df["latent_dim"], errors="coerce")
    if (df["_lat"] == 8).any():
        df = df[df["_lat"] == 8]

    def _row(sim_tag, dr, priv):
        d = df[(df["sim"] == sim_tag) & (df["dr_scale"] == dr)
               & (df["priv_mode"] == priv)]
        if d.empty:
            return None
        r = d.iloc[0]
        return {
            "reward":   float(r["mean_reward"]) if pd.notna(r.get("mean_reward")) else np.nan,
            "rew_std":  float(r["std_reward"])  if pd.notna(r.get("std_reward"))  else np.nan,
            "success":  float(r["success_rate"]) if pd.notna(r.get("success_rate")) else np.nan,
            "survival": float(r["survival_rate"]) if pd.notna(r.get("survival_rate")) else np.nan,
            "rmse":     float(r["mean_track_err"]) if pd.notna(r.get("mean_track_err")) else np.nan,
        }

    cells = []
    for sim_tag, sim_label in [("isaac", "Isaac"), ("mujoco", "MuJoCo")]:
        for dr in (1.0, 2.0):
            full = _row(sim_tag, dr, "FULL")
            for priv in privs:
                r = _row(sim_tag, dr, priv)
                if r is None:
                    continue
                if full is not None and not np.isnan(full["reward"]) and priv != "FULL":
                    delta_r = (r["reward"] - full["reward"]) / full["reward"] * 100.0
                else:
                    delta_r = np.nan
                cells.append({
                    "sim": sim_label, "DR×s": f"{dr:g}", "priv": priv,
                    "reward":  r["reward"], "rew_std": r["rew_std"],
                    "success": r["success"], "survival": r["survival"],
                    "rmse": r["rmse"],
                    "delta_r": delta_r,
                })

    md_path = os.path.join(REPO_ROOT, "results", "go2_cts_priv_ablation_table.md")
    with open(md_path, "w") as f:
        f.write("# Go2 — CTS privileged-subset ablation  (FULL vs INT vs EXT, $Z$=8)\n\n")
        f.write("_Auto-generated by scripts/plot_results_go2.py from "
                "`results/ood_go2.csv` + `results/sim2sim_go2.csv`._\n\n")
        f.write("## Per-cell results (CTS only)\n\n")
        f.write("| sim | DR×s | priv | reward (mean ± std) | success % | survival % | "
                "vel-RMSE [m/s] | Δreward vs FULL |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|\n")
        for c in cells:
            rew = (f"{c['reward']:.1f} ± {c['rew_std']:.1f}"
                   if not np.isnan(c['reward']) else "—")
            succ = f"{c['success']:.0f}"   if not np.isnan(c['success']) else "—"
            surv = f"{c['survival']:.0f}"  if not np.isnan(c['survival']) else "—"
            rmse = f"{c['rmse']:.3f}"      if not np.isnan(c['rmse'])    else "—"
            dr_  = (f"{c['delta_r']:+.1f} %"
                    if not np.isnan(c['delta_r']) else "—")
            f.write(f"| {c['sim']} | {c['DR×s']} | {c['priv']} | "
                    f"{rew} | {succ} | {surv} | {rmse} | {dr_} |\n")

        # Behaviour (gait-quality) metrics per (sim × DR × priv)
        if all(c in df.columns for c, _, _ in _GAIT_METRICS):
            f.write("\n## Behaviour (gait-quality) metrics at DR×1 and DR×2\n\n")
            f.write("_Lower is better except `gait_adh` and `contact_sym` "
                    "(higher = closer to a textbook trot). DR×2 rows are the "
                    "OOD condition (each scaled-DR range widened ×2 about its "
                    "midpoint)._\n\n")
            headers = ["sim", "DR×s", "priv"] + [m[1] for m in _GAIT_METRICS]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
            for dr_use in (1.0, 2.0):
                for sim_tag, sim_label in [("isaac", "Isaac"), ("mujoco", "MuJoCo")]:
                    for priv in privs:
                        d = df[(df["sim"] == sim_tag) & (df["dr_scale"] == dr_use)
                               & (df["priv_mode"] == priv)]
                        if d.empty:
                            continue
                        r = d.iloc[0]
                        cells_v = [sim_label, f"{dr_use:g}", priv]
                        for col, _, _ in _GAIT_METRICS:
                            v = r.get(col)
                            if pd.isna(v):
                                cells_v.append("—")
                            elif abs(v) >= 1.0:
                                cells_v.append(f"{v:.2f}")
                            elif abs(v) >= 0.01:
                                cells_v.append(f"{v:.3f}")
                            else:
                                cells_v.append(f"{v:.4f}")
                        f.write("| " + " | ".join(cells_v) + " |\n")

        # Transfer-ratio table specific to CTS×privilege
        f.write("\n## Spec-sheet transfer ratios per privileged subset\n\n")
        f.write("- **G(π)** = R_MuJoCo,1× / R_Isaac,1× × 100 % (target ≥ 60 %)\n")
        f.write("- **OOD gap** = R_Isaac,2× / R_Isaac,1× × 100 % (target ≥ 70 %)\n")
        f.write("- **Combined** = R_MuJoCo,2× / R_Isaac,1× × 100 % (target ≥ 40 %)\n\n")
        f.write("| priv | R_iso,1× | R_iso,2× | R_muj,1× | R_muj,2× | "
                "G(π) | OOD gap | Combined |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for priv in privs:
            riso1 = _row("isaac",  1.0, priv); riso2 = _row("isaac",  2.0, priv)
            rmuj1 = _row("mujoco", 1.0, priv); rmuj2 = _row("mujoco", 2.0, priv)
            def _v(c): return c["reward"] if c is not None else np.nan
            R_i1, R_i2, R_m1, R_m2 = _v(riso1), _v(riso2), _v(rmuj1), _v(rmuj2)
            def _fr(x): return f"{x:.1f}" if not np.isnan(x) else "—"
            def _fp(x, thr): return (f"{x:.1f}% [{_pass(x, thr)}]"
                                     if not np.isnan(x) else "—")
            G_pi    = (R_m1 / R_i1 * 100) if (R_i1 and not np.isnan(R_i1) and not np.isnan(R_m1)) else np.nan
            OOD_gap = (R_i2 / R_i1 * 100) if (R_i1 and not np.isnan(R_i1) and not np.isnan(R_i2)) else np.nan
            comb    = (R_m2 / R_i1 * 100) if (R_i1 and not np.isnan(R_i1) and not np.isnan(R_m2)) else np.nan
            f.write(f"| {priv} | {_fr(R_i1)} | {_fr(R_i2)} | {_fr(R_m1)} | {_fr(R_m2)} | "
                    f"{_fp(G_pi, 60)} | {_fp(OOD_gap, 70)} | {_fp(comb, 40)} |\n")
    print(f"[plot] wrote {os.path.relpath(md_path, REPO_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Plot Go2 (Phase-2) results")
    ap.add_argument("--ood",     default=os.path.join(REPO_ROOT, "results", "ood_go2.csv"))
    ap.add_argument("--sim2sim", default=os.path.join(REPO_ROOT, "results", "sim2sim_go2.csv"))
    ap.add_argument("--scan-logs", action="store_true",
                    help="also parse logs/*/ood_eval/*.txt sim2sim reports")
    args = ap.parse_args()

    ood = load_csv(args.ood, "isaac")
    sim = load_csv(args.sim2sim, "mujoco")
    if args.scan_logs:
        parsed = scan_log_txt_reports()
        if not parsed.empty:
            sim = pd.concat([sim, parsed], ignore_index=True) if not sim.empty else parsed

    if ood.empty and sim.empty:
        print("[plot] no input data found. Run the eval jobs first "
              "(see scripts/run_remaining_go2.sh) — nothing to plot yet.")
        return

    fig_headline(ood, sim)                  # one-glance "elevator pitch"
    fig_comparison_matrix(ood, sim)         # 4-view comparison — reward
    fig_comparison_survival(ood, sim)       # 4-view comparison — survival rate
    fig_comparison_rmse(ood, sim)           # 4-view comparison — velocity-tracking RMSE
    fig_comparison_outcome(ood, sim)        # 4-view comparison — outcome breakdown
    fig_summary_dashboard(ood, sim)         # all-in-one spec-sheet dashboard
    fig_ood_profile(ood)
    fig_sim2sim_transfer(ood, sim)
    fig_latent_ablation(ood, sim)
    # NOTE: fig_priv_ablation (the older RMA+CTS overview panel) is intentionally
    # NOT called: with only CTS-INT/EXT trained (RMA INT/EXT was scoped out), it
    # rendered an asymmetric single-bar RMA group next to three CTS bars, which
    # was visually misleading. Use the CTS-only ablation figures below instead.
    fig_cts_priv_ablation(ood, sim)          # CTS-only 4-view ablation (NEW)
    fig_cts_priv_ablation_gait(ood, sim)     # CTS-only gait-quality ablation (NEW)
    fig_gait_quality(ood, sim)
    fig_rma_phase2_loss()                     # teacher-student gap via Phase-2 MSE
    write_tables(ood, sim)
    write_cts_priv_ablation_table(ood, sim)  # NEW
    print(f"\n[plot] done — figures in {os.path.relpath(OUT_DIR, REPO_ROOT)}/")


if __name__ == "__main__":
    main()

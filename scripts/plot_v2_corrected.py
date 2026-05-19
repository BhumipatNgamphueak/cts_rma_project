"""Standalone corrected Go2 comparison figure from the post-bugfix v2 datasets.

Renders results/figures/fig_go2_v2_corrected.{png,pdf}: a 1x3 panel
(survival %, reward, velocity-tracking error) for Baseline/CTS/RMA at DR x1/x2
across the three authoritative conditions:
  - Isaac training-faithful (flat + push, no impulse)   [canonical]
  - Isaac no-push (flat)                                 [sim2sim companion]
  - MuJoCo no-push (flat)                                [sim2sim target]
All: 20 s, 30 episodes, v2 checkpoints.
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_isaac(p):
    d = {}
    for r in csv.DictReader(open(os.path.join(ROOT, p))):
        d[(r["method"].upper(), r["dr_scale"])] = dict(
            surv=float(r["survival_rate"]),
            rew=float(r["mean_reward"]),
            trk=float(r["mean_track_err"]),
        )
    return d


TF = load_isaac("results/isaac_v2_trainfaithful_20s.csv")
NP = load_isaac("results/isaac_v2_matched_20s.csv")
mj = json.load(open(os.path.join(ROOT, "results/sim2sim_report_v2_matched.json")))["conditions"]
MJ = {}
for m in ("baseline", "cts", "rma"):
    for k, s in (("1x", "1.0"), ("2x", "2.0")):
        c = mj[f"{m}_{k}"]
        MJ[(m.upper(), s)] = dict(surv=c["survival_rate"] * 100.0,
                                  rew=c["reward"]["mean"],
                                  trk=c["vel_rmse"]["mean"])

METHODS = ["BASELINE", "CTS", "RMA"]
COND = [("Isaac\n(push, faithful)", TF), ("Isaac\n(no-push)", NP), ("MuJoCo\n(no-push)", MJ)]
CMETH = {"BASELINE": "#4C72B0", "CTS": "#C44E52", "RMA": "#55A868"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = [("surv", "Survival rate [%]", (0, 105), 80.0, "80% spec"),
           ("rew", "Mean episode reward", None, None, None),
           ("trk", "Velocity-tracking error [m/s]", (0, 0.55), 0.30, "0.3 m/s spec")]

for ax, (key, ylab, ylim, spec, speclab) in zip(axes, metrics):
    x = np.arange(len(COND))
    w = 0.12
    for mi, M in enumerate(METHODS):
        for di, s in enumerate(("1.0", "2.0")):
            off = (mi * 2 + di - 2.5) * w
            vals = [src.get((M, s), {}).get(key, np.nan) for _, src in COND]
            hatch = "" if s == "1.0" else "//"
            ax.bar(x + off, vals, w, color=CMETH[M], hatch=hatch,
                   edgecolor="black", linewidth=0.4,
                   label=f"{M} DR×{int(float(s))}" if di in (0, 1) else None)
    if spec is not None:
        ax.axhline(spec, ls="--", c="gray", lw=1)
        ax.text(2.55, spec, speclab, fontsize=8, va="bottom", ha="right", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([c for c, _ in COND])
    ax.set_ylabel(ylab)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", ls=":", alpha=0.4)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=8,
           bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Go2 v2 — corrected post-bugfix comparison (20 s, 30 ep, FULL, Z=8). "
             "Solid = DR×1, hatched = DR×2.", y=0.97, fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])

for ext in ("png", "pdf"):
    out = os.path.join(ROOT, f"results/figures/fig_go2_v2_corrected.{ext}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

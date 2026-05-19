"""
Extract per-iteration learning-curve scalars from tfevents logs into a single CSV.

Covers Phase-1 runs (Baseline v2, RMA v2fix, CTS v2) — all scalars logged by
RSL-RL's OnPolicyRunner.  Phase-2 loss is written live to
  logs/rma/phase2/<run>/phase2_loss.csv
by RMAPhase2Runner; run this script after Phase-2 to append that too.

Output: results/learning_curves_go2.csv
Columns: method, iteration, mean_reward, mean_episode_length,
         surrogate_loss, value_loss, entropy_loss,
         track_lin_vel_xy, track_ang_vel_z, alive
"""

import csv
import glob
import os
import sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("Install tensorboard:  pip install tensorboard")

# ── Run directories (edit if your log paths differ) ──────────────────────────
RUNS = {
    "Baseline":      "logs/baseline/2026-05-10_17-07-38_baseline_go2_v2",
    "RMA_Phase1":    "logs/rma/2026-05-18_09-44-46_rma_go2_v2fix_full_l8",
    "CTS":           "logs/cts/2026-05-10_17-07-53_cts_go2_v2_l8",
}

# Phase-2 CSV written live by RMAPhase2Runner — pick latest run automatically
_p2_dirs = sorted(glob.glob("logs/rma/phase2/*/phase2_loss.csv"))
PHASE2_CSV = _p2_dirs[-1] if _p2_dirs else None

TAGS = {
    "mean_reward":         "Train/mean_reward",
    "mean_episode_length": "Train/mean_episode_length",
    "surrogate_loss":      "Loss/surrogate",
    "value_loss":          "Loss/value_function",
    "entropy_loss":        "Loss/entropy",
    "track_lin_vel_xy":    "Episode_Reward/track_lin_vel_xy",
    "track_ang_vel_z":     "Episode_Reward/track_ang_vel_z",
    "alive":               "Episode_Reward/alive",
}

OUT_DIR = "results"
OUT_PATH = os.path.join(OUT_DIR, "learning_curves_go2.csv")
os.makedirs(OUT_DIR, exist_ok=True)


def load_events(run_dir: str):
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        return None
    ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def extract(ea, tag: str):
    available = ea.Tags().get("scalars", [])
    if tag not in available:
        return {}
    return {s.step: s.value for s in ea.Scalars(tag)}


def main():
    rows = []

    for method, run_dir in RUNS.items():
        print(f"Loading {method} from {run_dir} ...", end=" ", flush=True)
        ea = load_events(run_dir)
        if ea is None:
            print("NO EVENTS FOUND — skipping")
            continue

        # Build per-iteration dict: {iteration: {col: val}}
        data = {}
        for col, tag in TAGS.items():
            for step, val in extract(ea, tag).items():
                data.setdefault(step, {})[col] = val

        for iteration in sorted(data):
            row = {"method": method, "iteration": iteration}
            row.update({col: data[iteration].get(col, "") for col in TAGS})
            rows.append(row)
        print(f"{len(data)} iterations")

    # Phase-2 MSE loss
    if PHASE2_CSV and os.path.exists(PHASE2_CSV):
        print(f"Loading Phase-2 loss from {PHASE2_CSV} ...", end=" ", flush=True)
        with open(PHASE2_CSV) as f:
            reader = csv.DictReader(f)
            p2_rows = list(reader)
        for r in p2_rows:
            rows.append({
                "method": "RMA_Phase2",
                "iteration": int(r["iteration"]),
                "mean_reward": "",
                "mean_episode_length": "",
                "surrogate_loss": "",
                "value_loss": "",
                "entropy_loss": "",
                "track_lin_vel_xy": "",
                "track_ang_vel_z": "",
                "alive": "",
                "mse_loss": r["mse_loss"],
            })
        print(f"{len(p2_rows)} iterations")
    else:
        print("No Phase-2 CSV found yet (run Phase-2 first, or it's still training).")

    if not rows:
        print("No data extracted.")
        return

    # Collect all columns in order
    base_cols = ["method", "iteration"] + list(TAGS.keys())
    all_cols  = base_cols + (["mse_loss"] if any("mse_loss" in r for r in rows) else [])

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {OUT_PATH}")


if __name__ == "__main__":
    main()

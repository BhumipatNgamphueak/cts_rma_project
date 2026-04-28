"""
Print a formatted summary table from the OOD results CSV.

Usage:
    python scripts/one_leg/format_results.py results/ood_results_cts_l8.csv
    python scripts/one_leg/format_results.py results/ood_results_cts_l8.csv --save_md
"""
import csv, sys, os, argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("csv_file", type=str, help="Path to the results CSV")
parser.add_argument("--save_md", action="store_true",
                    help="Also save a Markdown table alongside the CSV")
args = parser.parse_args()

if not os.path.exists(args.csv_file):
    print(f"[format] File not found: {args.csv_file}")
    sys.exit(1)

with open(args.csv_file, newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("[format] CSV is empty.")
    sys.exit(0)

# ── Group rows by (method, priv_mode, latent_dim) ───────────────────────────
groups = defaultdict(list)
for r in rows:
    key = (r["method"], r["priv_mode"], r["latent_dim"])
    groups[key].append(r)

# ── Build table ──────────────────────────────────────────────────────────────
SCALE_LABELS = {"1.0": "Train (1.0x)", "1.5": "OOD-Mild (1.5x)", "2.0": "OOD-Hard (2.0x)"}

sep_wide  = "═" * 90
sep_mid   = "─" * 90
col_w     = 20

def fmt_cell(r):
    rew  = float(r["mean_reward"])
    srew = float(r["std_reward"])
    suc  = float(r["success_rate"])
    return f"{rew:+.1f}±{srew:.1f}  ({suc:.0f}%)"

lines = []
lines.append(sep_wide)
lines.append(f"  OOD Robustness Summary")
lines.append(sep_wide)

header = f"  {'Config':<22}" + "".join(f"{SCALE_LABELS[s]:>{col_w}}" for s in ["1.0","1.5","2.0"])
lines.append(header)
lines.append(sep_mid)

for (method, priv, latent), run_rows in sorted(groups.items()):
    scale_map = {r["dr_scale"]: r for r in run_rows}
    label     = f"{method} {priv} l={latent}"
    cells     = []
    for s in ["1.0", "1.5", "2.0"]:
        if s in scale_map:
            cells.append(f"{fmt_cell(scale_map[s]):>{col_w}}")
        else:
            cells.append(f"{'—':>{col_w}}")
    lines.append(f"  {label:<22}" + "".join(cells))

lines.append(sep_wide)
lines.append("  Columns: mean_reward ± std  (success%)")
lines.append("")

output = "\n".join(lines)
print(output)

# ── Save Markdown ─────────────────────────────────────────────────────────────
if args.save_md:
    md_path = args.csv_file.replace(".csv", "_table.md")
    with open(md_path, "w") as f:
        f.write("# OOD Robustness Results\n\n")
        f.write("| Config | Train (1.0x) | OOD-Mild (1.5x) | OOD-Hard (2.0x) |\n")
        f.write("|--------|-------------|-----------------|----------------|\n")
        for (method, priv, latent), run_rows in sorted(groups.items()):
            scale_map = {r["dr_scale"]: r for r in run_rows}
            label     = f"{method} {priv} l={latent}"
            cells     = []
            for s in ["1.0", "1.5", "2.0"]:
                if s in scale_map:
                    cells.append(fmt_cell(scale_map[s]))
                else:
                    cells.append("—")
            f.write(f"| {label} | {' | '.join(cells)} |\n")
        f.write("\n_reward ± std (success %)_\n")
    print(f"[format] Markdown table saved: {md_path}")

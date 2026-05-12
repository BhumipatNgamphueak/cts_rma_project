"""Format ood_results CSV into a readable text report.

Usage:
    python results/format_results.py                          # reads ood_results.csv
    python results/format_results.py --input ood_results_v2.csv
"""
import argparse, csv, os

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=None,
                    help="CSV file to read (default: results/ood_results.csv)")
args = parser.parse_args()

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
if args.input:
    CSV = args.input if os.path.isabs(args.input) else os.path.join(os.getcwd(), args.input)
else:
    CSV = os.path.join(RESULTS_DIR, "ood_results.csv")

base = os.path.splitext(os.path.basename(CSV))[0]
OUT  = os.path.join(RESULTS_DIR, base + "_table.md")

rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows.append({
            "method": r["method"],
            "priv":   r["priv_mode"],
            "l":      r["latent_dim"],
            "dr":     float(r["dr_scale"]),
            "rew":    float(r["mean_reward"]),
            "std":    float(r["std_reward"]),
            "succ":   float(r["success_rate"]),
        })

# Detect which DR scales are present in the data
DR_SCALES = sorted(set(r["dr"] for r in rows))


def get(method, priv, l, dr):
    for r in rows:
        if (r["method"] == method and r["priv"] == priv
                and r["l"] == str(l) and r["dr"] == dr):
            return r["rew"], r["std"], r["succ"]
    return None, None, None


def drop(r_lo, r_hi):
    if r_lo and r_hi:
        return (r_lo - r_hi) / r_lo * 100
    return None


def fmt(v, s, su=None):
    if v is None:
        return "—"
    base = f"{v:+.1f}±{s:.1f}"
    if su is not None and su < 100.0:
        base += f" ({su:.0f}%)"
    return base


def pct(v):
    return f"{v:.1f}%" if v is not None else "—"


lines = []


def sep():  lines.append("")
def h1(t):  lines.append(f"## {t}")
def row(*cells, widths=None):
    if widths:
        lines.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |")
    else:
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")


def divider(n):
    lines.append("|" + "|".join(["---"] * n) + "|")


dr_headers = [f"DR×{d:.1f}" for d in DR_SCALES]
dr_max = max(DR_SCALES)


# ── Table 6: Architecture Comparison ──────────────────────────────────────────
sep()
h1("TABLE 6 — Architecture Comparison (FULL priv, best ℓ)")
lines.append(f"_reward ± std (success%), 100 eps. OOD drop = (R×1.0 − R×{dr_max:.1f}) / R×1.0_")
sep()

cols = ["Config", "ℓ"] + dr_headers + [f"OOD drop (×{dr_max:.1f})", "vs Baseline"]
row(*cols)
divider(len(cols))

base_10, base_s10, _ = get("BASELINE", "FULL", "N/A", 1.0)
base_hi, _, _        = get("BASELINE", "FULL", "N/A", dr_max)
dr_vals_base = [fmt(*get("BASELINE", "FULL", "N/A", d)) for d in DR_SCALES]
row("Baseline", "—", *dr_vals_base, pct(drop(base_10, base_hi)), "—")

for method, priv, l, label in [
    ("RMA2", "FULL", 8,   "RMA2-FULL ℓ=8"),
    ("RMA2", "FULL", 16,  "RMA2-FULL ℓ=16"),
    ("RMA2", "FULL", 32,  "RMA2-FULL ℓ=32"),
    ("CTS",  "FULL", 8,   "CTS-FULL ℓ=8"),
    ("CTS",  "FULL", 16,  "CTS-FULL ℓ=16"),
    ("CTS",  "FULL", 32,  "CTS-FULL ℓ=32"),
    ("CTS",  "FULL", 64,  "CTS-FULL ℓ=64"),
    ("CTS",  "FULL", 128, "CTS-FULL ℓ=128"),
]:
    r10, s10, _ = get(method, priv, l, 1.0)
    r_hi, _, _  = get(method, priv, l, dr_max)
    dr_vals     = [fmt(*get(method, priv, l, d)) for d in DR_SCALES]
    d = pct(drop(r10, r_hi))
    vs = f"+{(r10 - base_10) / base_10 * 100:.1f}%" if r10 and base_10 else "—"
    row(label, str(l), *dr_vals, d, vs)

sep()


# ── Table 8: Privileged Knowledge Ablation ────────────────────────────────────
sep()
h1("TABLE 8 — Privileged Knowledge Ablation (ℓ=16)")
sep()

cols = ["Arch", "Cond"] + dr_headers + [f"OOD drop (×{dr_max:.1f})"]
row(*cols)
divider(len(cols))

dr_vals_base = [fmt(*get("BASELINE", "FULL", "N/A", d)) for d in DR_SCALES]
row("BASE", "BASE", *dr_vals_base, pct(drop(base_10, base_hi)))

for method, label in [("RMA2", "RMA2"), ("CTS", "CTS")]:
    for priv, cond in [("INT", "INT"), ("EXT", "EXT"), ("FULL", "FULL")]:
        r10, s10, _ = get(method, priv, 16, 1.0)
        r_hi, _, _  = get(method, priv, 16, dr_max)
        dr_vals     = [fmt(*get(method, priv, 16, d)) for d in DR_SCALES]
        row(label, cond, *dr_vals, pct(drop(r10, r_hi)))

sep()


# ── Table: Latent Dimension Sweep ─────────────────────────────────────────────
sep()
h1("TABLE — Latent Dimension Sweep (FULL, all DR)")
sep()

cols = ["ℓ", "Method"] + dr_headers + [f"OOD drop (×{dr_max:.1f})"]
row(*cols)
divider(len(cols))

for l in [8, 16, 32, 64, 128]:
    for method in ["RMA2", "CTS"]:
        r10, s10, _ = get(method, "FULL", l, 1.0)
        r_hi, _, _  = get(method, "FULL", l, dr_max)
        dr_vals     = [fmt(*get(method, "FULL", l, d)) for d in DR_SCALES]
        row(str(l), method, *dr_vals, pct(drop(r10, r_hi)))

dr_vals_base = [fmt(*get("BASELINE", "FULL", "N/A", d)) for d in DR_SCALES]
row("—", "Baseline", *dr_vals_base, pct(drop(base_10, base_hi)))

sep()

txt = "\n".join(lines)
print(txt)
with open(OUT, "w") as f:
    f.write(txt + "\n")
print(f"\n[saved] {OUT}")

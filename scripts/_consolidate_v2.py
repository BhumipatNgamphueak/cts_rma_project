"""Build the two consolidated CSVs that plot_results_go2.py consumes, from the
corrected post-bugfix v2 datasets.

ood_go2_v2.csv      (Isaac, CANONICAL = training-faithful flat+push):
    isaac_v2_trainfaithful_20s.csv  (Baseline, CTS-FULL, RMA)
  + isaac_cts_intext_trainfaithful_20s.csv  (CTS-INT, CTS-EXT)

sim2sim_go2_v2.csv  (MuJoCo, no-push — only mode the harness supports):
    Baseline + RMA  : converted from sim2sim_report_v2_matched.json
  + CTS FULL/INT/EXT: mujoco_cts_priv_20s.csv (clean per-ckpt sim2sim_go2.py)
"""
import csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda p: os.path.join(ROOT, p)

HEADER = ("sim,method,priv_mode,latent_dim,dr_scale,terrain,disturbance,"
          "episode_length_s,mean_reward,std_reward,mean_length,std_length,"
          "success_rate,partial_rate,fall_rate,survival_rate,mean_lin_track,"
          "std_lin_track,mean_ang_track,std_ang_track,mean_track_err,"
          "std_track_err,mean_fwd_disp,std_fwd_disp,gait_adh,gait_adh_std,"
          "clear_err,clear_err_std,slip_rate,slip_rate_std,smoothness,"
          "smoothness_std,base_z_var,base_z_var_std,contact_sym,"
          "contact_sym_std,stride_var,stride_var_std,jtorque_var,"
          "jtorque_var_std,episodes,checkpoint,timestamp").split(",")


def _norm_latent(v):
    """'8.0' -> '8' so the (method,priv,Z) join matches across ood/sim2sim."""
    v = (v or "").strip()
    if v in ("", "—", "N/A", "NAN", "None"):
        return v
    try:
        return str(int(float(v)))
    except ValueError:
        return v


def cat_csv(paths, out):
    rows = []
    for p in paths:
        with open(R(p)) as f:
            rows += list(csv.DictReader(f))
    with open(R(out), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            o = {k: r.get(k, "") for k in HEADER}
            o["latent_dim"] = _norm_latent(o.get("latent_dim"))
            w.writerow(o)
    print(f"[ood]   {out}: {len(rows)} rows")


def json_to_rows(jpath):
    d = json.load(open(R(jpath)))["conditions"]
    out = []
    for m in ("baseline", "rma"):                 # CTS comes from the clean csv
        for k, s in (("1x", "1.0"), ("2x", "2.0")):
            c = d[f"{m}_{k}"]
            g = lambda key, sub="mean": c[key][sub]
            row = {h: "" for h in HEADER}
            row.update({
                "sim": "mujoco", "method": m.upper(),
                "priv_mode": "BASE" if m == "baseline" else "FULL",
                "latent_dim": "" if m == "baseline" else "8",
                "dr_scale": s, "terrain": "flat", "disturbance": "no_dist",
                "episode_length_s": "20.0",
                "mean_reward": f"{g('reward'):.4f}",
                "std_reward": f"{g('reward','std'):.4f}",
                "success_rate": f"{c['survival_rate']*100:.4f}",
                "survival_rate": f"{c['survival_rate']*100:.4f}",
                "mean_track_err": f"{g('vel_rmse'):.6f}",
                "std_track_err": f"{g('vel_rmse','std'):.6f}",
                "mean_fwd_disp": f"{g('fwd_disp'):.4f}",
                "std_fwd_disp": f"{g('fwd_disp','std'):.4f}",
                "gait_adh": f"{g('gait_adh'):.6f}",
                "gait_adh_std": f"{g('gait_adh','std'):.6f}",
                "clear_err": f"{g('clear_err'):.6f}",
                "slip_rate": f"{g('slip_rate'):.6f}",
                "smoothness": f"{g('smoothness'):.6f}",
                "base_z_var": f"{g('base_z_var'):.6f}",
                "contact_sym": f"{g('contact_sym'):.6f}",
                "stride_var": f"{g('stride_var'):.6f}",
                "jtorque_var": f"{g('jtorque_var'):.6f}",
                "episodes": str(c["n_episodes"]),
                "checkpoint": "v2", "timestamp": "v2_corrected",
            })
            out.append(row)
    return out


def build_sim2sim(out):
    rows = json_to_rows("results/sim2sim_report_v2_matched.json")
    cts_csv = R("results/mujoco_cts_priv_20s.csv")
    n_cts = 0
    if os.path.exists(cts_csv):
        for r in csv.DictReader(open(cts_csv)):
            o = {k: r.get(k, "") for k in HEADER}
            o["latent_dim"] = _norm_latent(o.get("latent_dim"))
            rows.append(o); n_cts += 1
    with open(R(out), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[s2s]   {out}: {len(rows)} rows (baseline/rma from json, "
          f"{n_cts} CTS rows from mujoco_cts_priv_20s.csv)")
    if n_cts == 0:
        print("  [warn] mujoco_cts_priv_20s.csv missing — CTS MuJoCo rows absent")


if __name__ == "__main__":
    cat_csv(["results/isaac_v2_trainfaithful_20s.csv",
             "results/isaac_cts_intext_trainfaithful_20s.csv"],
            "results/ood_go2_v2.csv")
    build_sim2sim("results/sim2sim_go2_v2.csv")
    print("done")

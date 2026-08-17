#!/usr/bin/env python3
"""M-SPLIT study analysis (cluster login node, numpy only).

Discovers all arms in models/meansplit_study/ (<arm>_s<seed> dirs: base, ms,
ms_t0, ms_tik, ms_tik_d10, ms_wall, ...) and compares them:
  1. headline per-run metrics (test NRMSE per field, train/val loss, BFGS stop);
  2. static/dynamic x region error decomposition of the full-grid reconstruction
     dumps in results/ms_{arm}_s{seed}_rom_{a025,cc060}/ — same conventions as
     hmetric_static_dynamic.py (regions from analysis_hmetric);
  3. seed-aggregated arm summary with the improvement factor vs the 'base' arm.

Writes analysis_meansplit/{meansplit_runs.csv, meansplit_static_dynamic.csv,
meansplit_arm_summary.csv} and prints the vx focus table (near + surface).
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path("/work/u10677113/NACA2312/recon")
RES = BASE / "results"
HM = BASE / "analysis_hmetric"
STUDY = BASE / "models" / "meansplit_study"
OUT = BASE / "analysis_meansplit"
OUT.mkdir(exist_ok=True)

CASES = {"sim_A_025": "ms_{arm}_s{seed}_rom_a025",
         "sim_Cc_060": "ms_{arm}_s{seed}_rom_cc060"}
FIELDS = ["vx", "vy", "p"]
REF_ARM = "base"

region_label = np.load(HM / "region_labels.npy")
air = np.load(HM / "airfoil_nodes.npy")
N = len(region_label)
REGIONS = {"all": np.ones(N, bool), "near": region_label == 0,
           "wake": region_label == 1, "far": region_label == 2,
           "surface": np.isin(np.arange(N), air)}


def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


# --- discover (arm, seed) runs ---------------------------------------------
runs_found = []
for d in sorted(STUDY.iterdir()):
    m = re.fullmatch(r"(.+)_s(\d+)", d.name)
    if m and d.is_dir():
        runs_found.append((m.group(1), int(m.group(2)), d))
arms = sorted({a for a, _, _ in runs_found})
print(f"found {len(runs_found)} runs, arms: {arms}")

rows, runs = [], []
for arm, seed, d in runs_found:
    lat_dirs = sorted(d.glob("latent_*"))
    md = lat_dirs[0] if lat_dirs else None
    mt = jload(md / "metrics.json") if md else None
    ri = jload(md / "run_info.json") if md else None
    if mt and ri:
        runs.append(dict(
            arm=arm, seed=seed, d_s=mt.get("num_latent_states"),
            NRMSE=mt["NRMSE"], NRMSE_vx=mt["NRMSE_vx"],
            NRMSE_vy=mt["NRMSE_vy"], NRMSE_p=mt["NRMSE_p"],
            rho_vx=mt.get("rho_vx"),
            train=ri.get("final_train_loss"), valid=ri.get("final_valid_loss"),
            bfgs_msg=(ri.get("bfgs_result") or {}).get("message"),
            bfgs_grad=(ri.get("bfgs_result") or {}).get("grad_inf_norm"),
            wall_h=(ri.get("total_wall_s") or 0) / 3600))
    else:
        print(f"[warn] {arm} s{seed}: metrics/run_info missing (not finished?)")
    for sim, pat in CASES.items():
        rd = RES / pat.format(arm=arm, seed=seed)
        if not (rd / f"rom_{sim}.npy").exists():
            print(f"[warn] missing {rd} — skipped")
            continue
        fom = np.load(rd / f"fom_{sim}.npy").astype(np.float64)
        rom = np.load(rd / f"rom_{sim}.npy").astype(np.float64)
        err = rom - fom
        e_static = err.mean(0)                       # [N,3]
        e_dyn = err - e_static[None]                 # [T,N,3]
        fom_fluct = fom - fom.mean(0)[None]
        for k, fname in enumerate(FIELDS):
            rng_full = fom[:, :, k].max() - fom[:, :, k].min()
            rng_fluct = fom_fluct[:, :, k].max() - fom_fluct[:, :, k].min()
            for rname, msk in REGIONS.items():
                rms_s = np.sqrt((e_static[msk, k] ** 2).mean())
                rms_d = np.sqrt((e_dyn[:, msk, k] ** 2).mean())
                rms_t = np.sqrt((err[:, msk, k] ** 2).mean())
                rows.append(dict(
                    arm=arm, seed=seed, sim=sim, field=fname, region=rname,
                    rmse_total=rms_t, rmse_static=rms_s, rmse_dynamic=rms_d,
                    static_share=(rms_s ** 2) / max(rms_t ** 2, 1e-300),
                    nrmse_total_fullrange=rms_t / rng_full,
                    nrmse_static_fullrange=rms_s / rng_full,
                    nrmse_dyn_fluctrange=rms_d / rng_fluct,
                    n_nodes=int(msk.sum())))
        print(f"{arm} s{seed} {sim}: decomposed")

if runs:
    with open(OUT / "meansplit_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0].keys()))
        w.writeheader()
        for r in runs:
            w.writerow(r)

if rows:
    with open(OUT / "meansplit_static_dynamic.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r = dict(r)
            for k, v in r.items():
                if isinstance(v, float):
                    r[k] = f"{v:.6e}"
            w.writerow(r)

# --- seed-aggregated arm summary (factor vs REF_ARM) ------------------------
summ = []
for sim in CASES:
    for fname in FIELDS:
        for rname in REGIONS:
            per_arm = {}
            for arm in arms:
                v = [r["nrmse_total_fullrange"] for r in rows
                     if r["arm"] == arm and r["sim"] == sim
                     and r["field"] == fname and r["region"] == rname]
                s = [r["static_share"] for r in rows
                     if r["arm"] == arm and r["sim"] == sim
                     and r["field"] == fname and r["region"] == rname]
                dyn = [r["nrmse_dyn_fluctrange"] for r in rows
                       if r["arm"] == arm and r["sim"] == sim
                       and r["field"] == fname and r["region"] == rname]
                if v:
                    per_arm[arm] = dict(mean=float(np.mean(v)), std=float(np.std(v)),
                                        static=float(np.mean(s)),
                                        dyn=float(np.mean(dyn)), n=len(v))
            ref = per_arm.get(REF_ARM)
            for arm, pa in per_arm.items():
                summ.append(dict(sim=sim, field=fname, region=rname, arm=arm,
                                 nrmse_mean=pa["mean"], nrmse_std=pa["std"],
                                 vs_base_x=(ref["mean"] / max(pa["mean"], 1e-300))
                                 if ref else float("nan"),
                                 static_share=pa["static"],
                                 dyn_nrmse_fluctrange=pa["dyn"], n_seeds=pa["n"]))
if summ:
    with open(OUT / "meansplit_arm_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        for r in summ:
            r = dict(r)
            for k, v in r.items():
                if isinstance(v, float):
                    r[k] = f"{v:.4e}"
            w.writerow(r)
    for reg in ("near", "surface"):
        print(f"\n=== FOCUS: vx {reg} (NRMSE fullrange, mean +/- std over seeds) ===")
        print(f"{'sim':<11} {'arm':<11} {'nrmse':<22} {'vs base':<8} "
              f"{'static%':<8} {'dyn(fluct)':<10}")
        for r in summ:
            if r["field"] != "vx" or r["region"] != reg:
                continue
            print(f"{r['sim']:<11} {r['arm']:<11} "
                  f"{r['nrmse_mean']:.3e}+/-{r['nrmse_std']:.1e}   "
                  f"{r['vs_base_x']:<8.2f} {100*r['static_share']:<8.0f} "
                  f"{r['dyn_nrmse_fluctrange']:<10.3e}")
print(f"\nwrote {OUT}/meansplit_runs.csv, meansplit_static_dynamic.csv, "
      f"meansplit_arm_summary.csv ({len(rows)} decomposition rows, {len(runs)} runs)")

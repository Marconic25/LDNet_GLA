#!/usr/bin/env python3
"""H-METRIC follow-up: split reconstruction error into STATIC (time-mean bias) and
DYNAMIC (fluctuation) components, per field x region.

Motivation: per-window NRMSE is nearly flat in time (quiescent ~ peak), so the error
may be a persistent field bias. The paper metric normalizes by the FULL field range
(vx range includes the ~80 m/s mean flow), which can hide whether higher d_s tracks
the gust/flap RESPONSE better. Here:
    err_static  = time-mean(rom - fom)               [N,3]  (steady offset)
    err_dyn     = (rom - fom) - err_static           [T,N,3] (dynamics mismatch)
    dyn NRMSE   = rms(err_dyn) / range(fom - time-mean(fom))   <- fluctuation range
Writes static_vs_dynamic.csv -> analysis_hmetric/.
"""
import csv
import numpy as np
from pathlib import Path

BASE = Path("/work/u10677113/NACA2312")
RES = BASE / "recon" / "results"
OUT = BASE / "recon" / "analysis_hmetric"

CASES = {"sim_A_025": "div_rom_a_l{ds}", "sim_Cc_060": "div_rom_cc_l{ds}"}
DS_LIST = [1, 5, 10]
FIELDS = ["vx", "vy", "p"]

region_label = np.load(OUT / "region_labels.npy")
air = np.load(OUT / "airfoil_nodes.npy")
N = len(region_label)
REGIONS = {"all": np.ones(N, bool), "near": region_label == 0,
           "wake": region_label == 1, "far": region_label == 2,
           "surface": np.isin(np.arange(N), air)}

rows = []
for sim, pat in CASES.items():
    for ds in DS_LIST:
        d = RES / pat.format(ds=ds)
        fom = np.load(d / f"fom_{sim}.npy").astype(np.float64)
        rom = np.load(d / f"rom_{sim}.npy").astype(np.float64)
        err = rom - fom
        e_static = err.mean(0)                      # [N,3]
        e_dyn = err - e_static[None]                # [T,N,3]
        fom_fluct = fom - fom.mean(0)[None]
        for k, fname in enumerate(FIELDS):
            rng_full = fom[:, :, k].max() - fom[:, :, k].min()
            rng_fluct = fom_fluct[:, :, k].max() - fom_fluct[:, :, k].min()
            for rname, m in REGIONS.items():
                rms_s = np.sqrt((e_static[m, k] ** 2).mean())
                rms_d = np.sqrt((e_dyn[:, m, k] ** 2).mean())
                # fom fluctuation RMS in this region (signal actually to track)
                sig_d = np.sqrt((fom_fluct[:, m, k] ** 2).mean())
                rows.append(dict(sim=sim, d_s=ds, field=fname, region=rname,
                                 component="static", rmse=rms_s,
                                 nrmse_fullrange=rms_s / rng_full,
                                 nrmse_fluctrange=rms_s / rng_fluct,
                                 rel_to_fluct_rms=rms_s / sig_d,
                                 n_nodes=int(m.sum())))
                rows.append(dict(sim=sim, d_s=ds, field=fname, region=rname,
                                 component="dynamic", rmse=rms_d,
                                 nrmse_fullrange=rms_d / rng_full,
                                 nrmse_fluctrange=rms_d / rng_fluct,
                                 rel_to_fluct_rms=rms_d / sig_d,
                                 n_nodes=int(m.sum())))
        print(f"{sim} d_s={ds}: done")

cols = ["sim", "d_s", "field", "region", "component", "rmse",
        "nrmse_fullrange", "nrmse_fluctrange", "rel_to_fluct_rms", "n_nodes"]
with open(OUT / "static_vs_dynamic.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        r = dict(r)
        for k in ("rmse", "nrmse_fullrange", "nrmse_fluctrange", "rel_to_fluct_rms"):
            r[k] = f"{r[k]:.6e}"
        w.writerow(r)
print(f"wrote {OUT}/static_vs_dynamic.csv ({len(rows)} rows)")

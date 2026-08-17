#!/usr/bin/env python3
"""Collect the loads depth-sweep results into one table: for each (depth, seed, d_s),
read config.json (architecture) + metrics.json (NRMSE F_y, M_z, combined). Print and
save CSV. Runs anywhere with plain python (json only)."""
import json, pathlib, csv

import os
ROOT = pathlib.Path(os.environ.get("LD_ROOT",
                    "/work/u10677113/NACA2312/recon/models/loads_depth_reg"))
rows = []
for md in sorted(ROOT.glob("L*_s*/in6/latent_*")):
    mj = md / "metrics.json"
    cj = md / "config.json"
    if not mj.exists():
        continue
    m = json.load(open(mj))
    cfg = json.load(open(cj)) if cj.exists() else {}
    # dir name L{L}_s{S}
    top = md.parents[1].name          # L24_s100
    L = int(top.split("_")[0][1:])
    seed = int(top.split("_s")[1])
    ds = m.get("num_latent_states")
    rows.append({
        "depth_L": L, "seed": seed, "d_s": ds,
        "NRMSE_Fy": m.get("NRMSE_F_y"), "NRMSE_Mz": m.get("NRMSE_M_z"),
        "NRMSE_combined": m.get("NRMSE"),
    })

rows.sort(key=lambda r: (r["d_s"], r["depth_L"], r["seed"]))
print(f"\n{'depth':>5} {'seed':>4} {'d_s':>3} {'NRMSE_Fy':>10} {'NRMSE_Mz':>10} {'combined':>10}")
for r in rows:
    print(f"{r['depth_L']:5d} {r['seed']:4d} {r['d_s']:3d} "
          f"{r['NRMSE_Fy']:10.4e} {r['NRMSE_Mz']:10.4e} {r['NRMSE_combined']:10.4e}")

out = pathlib.Path("/work/u10677113/NACA2312/recon/analysis") / (ROOT.name + "_table.csv")
out.parent.mkdir(exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["depth_L", "seed", "d_s",
                                      "NRMSE_Fy", "NRMSE_Mz", "NRMSE_combined"])
    w.writeheader(); w.writerows(rows)
print(f"\nwrote {out}  ({len(rows)} rows)")

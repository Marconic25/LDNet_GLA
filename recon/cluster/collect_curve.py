#!/usr/bin/env python3
"""Collect the full learning-curve + depth-study table from all model dirs into one CSV."""
import json, pathlib, csv, sys

ROOT = pathlib.Path("/work/u10677113/NACA2312/recon/models")

def m(path):
    p = ROOT / path / "metrics.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d.get("NRMSE"), d.get("NRMSE_vx"), d.get("NRMSE_vy"), d.get("NRMSE_p")

rows = []
# learning curve: N x d_s, standard arch
sets = {"N15": "final_div", "N30": "lc_N30", "N60": "lc_N60", "N100": "lc_N100"}
for tag, d in sets.items():
    for ds in (1, 3, 5, 10, 20):
        r = m(f"{d}/latent_{ds}")
        if r:
            rows.append((tag, "L6", ds, *r))
# L12 arms
for tag, d in {"N15": "final", "N60": "lc_N60_L12", "N100": "lc_N100_L12"}.items():
    for ds in (1, 10):
        r = m(f"{d}/latent_{ds}")
        if r:
            rows.append((tag, "L12", ds, *r))

print(f"{'set':5s} {'arch':4s} {'ds':>3s} {'test_NRMSE':>11s} {'vx':>8s} {'vy':>8s} {'p':>8s}")
for row in rows:
    tag, arch, ds, nr, vx, vy, pp = row
    print(f"{tag:5s} {arch:4s} {ds:3d} {nr:11.4e} {vx:8.4f} {vy:8.4f} {pp:8.4f}")

out = ROOT / "learning_curve_summary.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["set", "arch", "d_s", "test_NRMSE", "NRMSE_vx", "NRMSE_vy", "NRMSE_p"])
    w.writerows(rows)
print(f"\nwrote {out} ({len(rows)} rows)")

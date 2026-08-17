#!/usr/bin/env python3
"""Re-check: (1) input_signals global ranges of the CURRENT N100 h5, (2) per-sim
field_times length for ALL train sims to catch any still-broken (3-snapshot) dir."""
import h5py, numpy as np, pathlib
SIG = ["h","hd","a","ad","delta","W_gust"]

p = "/work/u10677113/NACA2312/recon/data/FIELDS_lc_N100_train.h5"
with h5py.File(p, "r") as f:
    si = f["input_signals"][:]
    print(f"=== {p} : input_signals {si.shape} ===")
    for c in range(6):
        print(f"  {SIG[c]:7s} min {si[...,c].min():12.4e}  max {si[...,c].max():12.4e}")

print("\n=== train-sim field_times lengths (flag < 800) ===")
rf = pathlib.Path("/work/u10677113/NACA2312/recon_fields")
bad = []
for d in sorted(rf.glob("sim_*_train")):
    ft = d / "field_times.npy"
    if not ft.exists():
        bad.append(d.name + " (no times)"); continue
    n = len(np.load(ft))
    if n < 800:
        bad.append(f"{d.name} (n={n})")
print("SHORT:", bad if bad else "none — all >=800 snapshots")

#!/usr/bin/env python3
"""Diagnose the corrupted N=100 build: per-sim field ranges in the train h5,
flagging sims whose |values| blow past the sane envelope (vx ~ +-150, p ~ +-1e4).
Run on the login node inside the TF container (needs h5py only)."""
import h5py
import numpy as np

H5 = "/work/u10677113/NACA2312/recon/data/FIELDS_lc_N100_train.h5"
with h5py.File(H5, "r") as f:
    names = [n.decode() if isinstance(n, bytes) else str(n)
             for n in f["sim_names"][:]] if "sim_names" in f else None
    of = f["output_fields"]          # (N, T, P, 3)
    N = of.shape[0]
    print(f"h5 {H5}: output_fields {of.shape}")
    bad = []
    for i in range(N):
        x = of[i]                    # (T, P, 3)
        nm = names[i] if names else f"sim#{i}"
        nan = int(np.isnan(x).sum())
        amax = np.abs(x).max(axis=(0, 1))   # per field
        line = (f"{i:3d} {nm:28s} nan={nan:8d} "
                f"|vx|max={amax[0]:12.3e} |vy|max={amax[1]:12.3e} |p|max={amax[2]:12.3e}")
        suspicious = nan > 0 or amax[0] > 500 or amax[1] > 500 or amax[2] > 1e6
        if suspicious:
            bad.append(nm)
            line += "   <<< SUSPECT"
        print(line)
    print("\nSUSPECTS:", bad if bad else "none — corruption is elsewhere")

#!/usr/bin/env python3
"""N=100 corruption hunt, round 2: input_signals ranges per sim (vs the healthy
N=60 h5), plus global range comparison of every dataset key."""
import h5py
import numpy as np

SIG = ["h", "hd", "a", "ad", "delta", "W_gust"]

def scan(path, deep=False):
    print(f"\n=== {path} ===")
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print("keys:", keys)
        names = None
        if "sim_names" in f:
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["sim_names"][:]]
        si = f["input_signals"][:]     # (N, T, 6)
        print("input_signals", si.shape, "nan:", int(np.isnan(si).sum()))
        for c in range(si.shape[2]):
            print(f"  {SIG[c]:7s} global min {si[...,c].min():12.4e}  max {si[...,c].max():12.4e}")
        if deep:
            gmin = si.min(axis=(0,1)); gmax = si.max(axis=(0,1))
            span = np.maximum(gmax - gmin, 1e-12)
            for i in range(si.shape[0]):
                m = si[i].min(axis=0); M = si[i].max(axis=0)
                odd = (M - m) > 0.98 * span   # this sim alone spans ~the whole range
                if odd.any():
                    nm = names[i] if names else f"#{i}"
                    ch = [SIG[c] for c in range(6) if odd[c]]
                    print(f"  RANGE-DOMINATING sim {i:3d} {nm:28s} channels {ch}")

scan("/work/u10677113/NACA2312/recon/data/FIELDS_lc_N60_train.h5")
scan("/work/u10677113/NACA2312/recon/data/FIELDS_lc_N100_train.h5", deep=True)

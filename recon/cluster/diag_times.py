#!/usr/bin/env python3
"""Print the field_times window of the recently re-extracted sims vs old ones —
hunting the sim whose window collapsed the N=100 common time grid into the trim."""
import numpy as np

SIMS = ["sim_A_000_train", "sim_Cc_005_train", "sim_B_016_train", "sim_Cc_030_train",
        "sim_Cc_039_train", "sim_B_026_train", "sim_A_018_train", "sim_Cc_049_train",
        "sim_Cc_006_train", "sim_B_004_train", "sim_Cc_014_train",
        "sim_Cc_038_train", "sim_A_016_train"]
for s in SIMS:
    try:
        t = np.load(f"/work/u10677113/NACA2312/recon_fields/{s}/field_times.npy")
        print(f"{s:24s} n={len(t):5d}  t=[{t[0]:.5f}, {t[-1]:.5f}]  dt~{np.diff(t).mean():.5f}")
    except Exception as e:
        print(f"{s:24s} ERR {e}")

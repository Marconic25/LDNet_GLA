#!/usr/bin/env python3
"""N99's reproduced reversed-flow fraction came out EXACTLY equal to the FOM's
(0.2335 both), while its sign-flip rate is the worst of the ladder and its
near/surface dynamic errors are the best. Those three facts cannot all be
taken at face value: check whether the N99 reconstruction is degenerate.
"""
import numpy as np
from pathlib import Path
import decomp_rates as base

AN = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
region = np.load("region_labels.npy")
air = np.load("airfoil_nodes.npy")
near = region == 0

for tag, d in [("champion", "ms_coral_o10_s0_rom_cc060"),
               ("N60 s100", "ms_coral_o10_N60_s100_rom_cc060"),
               ("N99 s0", "ms_coral_o10_N100_s0_rom_cc060")]:
    fom, rom, pts = base.load(d)
    d2 = ((pts[:, None, :] - pts[air][None, :, :]) ** 2).sum(-1)
    nn = d2.argmin(1)
    flap = near & (nn >= 292)

    vref = fom[0, :, :2]
    vn = vref / (np.linalg.norm(vref, axis=1, keepdims=True) + 1e-12)
    pf = (fom[:, :, :2] * vn[None]).sum(-1)
    pr = (rom[:, :, :2] * vn[None]).sum(-1)
    revf, revr = pf < 0, pr < 0
    tpk = int(revf[:, flap].mean(1).argmax())

    ff = revf[tpk, flap].mean()
    rf = revr[tpk, flap].mean()
    print(f"\n=== {tag} ===")
    print(f"  peak t index {tpk}: FOM rev frac={ff:.6f}  ROM rev frac={rf:.6f}")
    print(f"  n flap nodes={int(flap.sum())}, FOM rev count={int(revf[tpk,flap].sum())}, "
          f"ROM rev count={int(revr[tpk,flap].sum())}")

    # is the ROM field sane at all, or has it blown up / gone flat?
    rv = rom[tpk][flap]
    fv = fom[tpk][flap]
    print(f"  vx at peak, flap nodes: FOM [{fv[:,0].min():9.2f},{fv[:,0].max():9.2f}] "
          f"mean={fv[:,0].mean():8.2f} std={fv[:,0].std():7.2f}")
    print(f"                          ROM [{rv[:,0].min():9.2f},{rv[:,0].max():9.2f}] "
          f"mean={rv[:,0].mean():8.2f} std={rv[:,0].std():7.2f}")
    print(f"  |ROM-FOM| vx at flap: mean={np.abs(rv[:,0]-fv[:,0]).mean():.3f} "
          f"max={np.abs(rv[:,0]-fv[:,0]).max():.3f}")
    # whole-field sanity
    print(f"  whole field: ROM finite={np.isfinite(rom).all()} "
          f"vx range [{rom[...,0].min():.1f},{rom[...,0].max():.1f}] "
          f"(FOM [{fom[...,0].min():.1f},{fom[...,0].max():.1f}])")

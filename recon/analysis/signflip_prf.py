#!/usr/bin/env python3
"""Precision/recall decomposition of the near-flap reversal readout.

The plain sign-flip rate (fraction of near-flap nodes where FOM and ROM
disagree on the sign) conflates two very different errors, and that hid a
real effect: a surrogate that predicts almost no reversal scores WELL on
sign-flip simply by never risking a false positive. The champion reproduces
37 of the 403 truly reversed nodes (9% recall) yet posts the best sign-flip
of the early ladder; N99 reproduces exactly 403 and posts the worst.

Reported here per arm, at the gust peak, over near-flap nodes:
  recall    = fraction of TRULY reversed nodes the surrogate also reverses
  precision = fraction of the surrogate's reversed nodes that are truly reversed
  F1        = harmonic mean, the honest single number
  count ratio = ROM reversed count / FOM reversed count (calibration of amount)
"""
import numpy as np
import decomp_rates as base

ARMS = [
    ("champion (15)", "ms_coral_o10_s0_rom_cc060"),
    ("N30 s0", "ms_coral_o10_N30_s0_rom_cc060"),
    ("N60 s0", "ms_coral_o10_N60_s0_rom_cc060"),
    ("N60 s100", "ms_coral_o10_N60_s100_rom_cc060"),
    ("N99 s0", "ms_coral_o10_N100_s0_rom_cc060"),
]

region = np.load("region_labels.npy")
air = np.load("airfoil_nodes.npy")
near = region == 0

print(f"{'arm':16s} {'recall':>8s} {'precis':>8s} {'F1':>8s} "
      f"{'cnt ROM/FOM':>12s} {'signflip':>9s}")
for name, d in ARMS:
    fom, rom, pts = base.load(d)
    d2 = ((pts[:, None, :] - pts[air][None, :, :]) ** 2).sum(-1)
    flap = near & (d2.argmin(1) >= 292)

    vref = fom[0, :, :2]
    vn = vref / (np.linalg.norm(vref, axis=1, keepdims=True) + 1e-12)
    pf = (fom[:, :, :2] * vn[None]).sum(-1)
    pr = (rom[:, :, :2] * vn[None]).sum(-1)
    revf, revr = pf < 0, pr < 0
    tpk = int(revf[:, flap].mean(1).argmax())

    f = revf[tpk, flap]
    r = revr[tpk, flap]
    tp = int((f & r).sum())
    fp = int((~f & r).sum())
    fn = int((f & ~r).sum())
    recall = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * recall * prec / max(recall + prec, 1e-12)
    ratio = r.sum() / max(f.sum(), 1)
    flip = (f != r).mean()
    print(f"{name:16s} {recall:8.3f} {prec:8.3f} {f1:8.3f} "
          f"{ratio:12.3f} {100*flip:8.2f}%")

print("\nrecall = how much of the separation event is represented at all;")
print("precision = how much of what it calls separated really is;")
print("count ratio 1.0 = the right AMOUNT of reversal, regardless of placement.")

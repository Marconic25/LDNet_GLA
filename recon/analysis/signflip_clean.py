#!/usr/bin/env python3
"""Corrected near-flap reversal readout.

The established indicator (decomp_stall.py, reused by every decomp_*.py in
this campaign) projects each node's velocity onto its OWN t=0 velocity
direction and calls a negative projection reversed flow. At t=0 the flow is
quiescent, so the no-slip nodes ON the surface have |v(0)| = 0 exactly: 446
of the 1726 near-flap nodes, 25.8%. For those the reference direction is
undefined, the projection is ~0, and its sign is decided by round-off. They
sit in the denominator of every fraction reported so far, and they contribute
arbitrary counts to the ROM side -- two different but numerically identical
copies of the same reference field gave ROM counts of 403 and 173 for the
same model, purely from that noise.

The reference field itself is unaffected: none of its 403 reversed near-flap
nodes is degenerate. So the separation event is real; only the metric needed
fixing. This script excludes the degenerate nodes and reports precision and
recall alongside the flip rate, since a model that predicts almost no
reversal scores well on flip rate by never risking a false positive.
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
DEG_TOL = 1e-9

region = np.load("region_labels.npy")
air = np.load("airfoil_nodes.npy")
near = region == 0

print(f"{'arm':16s} {'FOM rev':>8s} {'ROM rev':>8s} {'recall':>7s} "
      f"{'precis':>7s} {'F1':>7s} {'flip%':>7s}")
for name, d in ARMS:
    fom, rom, pts = base.load(d)
    d2 = ((pts[:, None, :] - pts[air][None, :, :]) ** 2).sum(-1)
    flap = near & (d2.argmin(1) >= 292)

    vref = fom[0, :, :2]
    nrm = np.linalg.norm(vref, axis=1)
    valid = flap & (nrm > DEG_TOL)          # drop no-slip / zero-reference nodes

    vn = vref / (nrm[:, None] + 1e-30)
    pf = (fom[:, :, :2] * vn[None]).sum(-1)
    pr = (rom[:, :, :2] * vn[None]).sum(-1)
    revf, revr = pf < 0, pr < 0
    tpk = int(revf[:, valid].mean(1).argmax())

    f, r = revf[tpk, valid], revr[tpk, valid]
    tp = int((f & r).sum()); fp = int((~f & r).sum()); fn = int((f & ~r).sum())
    rec = tp / max(tp + fn, 1)
    pre = tp / max(tp + fp, 1)
    f1 = 2 * rec * pre / max(rec + pre, 1e-12)
    print(f"{name:16s} {f.mean():8.4f} {r.mean():8.4f} {rec:7.3f} "
          f"{pre:7.3f} {f1:7.3f} {100*(f != r).mean():6.2f}%")

print(f"\nnon-degenerate near-flap nodes: {int(valid.sum())} of {int(flap.sum())} "
      f"({int(flap.sum()) - int(valid.sum())} dropped as |v(t=0)| < {DEG_TOL:g})")

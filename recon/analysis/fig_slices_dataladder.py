#!/usr/bin/env python3
"""vx / vy slices and reconstruction error at the gust peak, across the data
ladder, zoomed on the flap region where the separation event lives.

Row 1: vx   Row 2: vy
Col 1: reference (FOM)
Col 2-4: |error| for the 15-trajectory champion, N60 s100 (best model) and
         N99 s0, on a common colour scale per row so the panels compare.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
import decomp_rates as base

AN = Path(__file__).resolve().parent
RES = AN.parent / "results"
OUT = AN / "figs_dataladder"
OUT.mkdir(exist_ok=True)

ARMS = [
    ("15 trajectories", "ms_coral_o10_s0_rom_cc060"),
    ("60 trajectories", "ms_coral_o10_N60_s100_rom_cc060"),
    ("99 trajectories", "ms_coral_o10_N100_s0_rom_cc060"),
]
tri = np.load(RES / "mesh_triangles.npy")
region = np.load(AN / "region_labels.npy")
air = np.load(AN / "airfoil_nodes.npy")

# gust-peak index from the reference, same convention as decomp_stall.py
fom, _, pts = base.load(ARMS[0][1])
d2 = ((pts[:, None, :] - pts[air][None, :, :]) ** 2).sum(-1)
flap = (region == 0) & (d2.argmin(1) >= 292)
vref = fom[0, :, :2]
vn = vref / (np.linalg.norm(vref, axis=1, keepdims=True) + 1e-12)
revf = ((fom[:, :, :2] * vn[None]).sum(-1)) < 0
tpk = int(revf[:, flap].mean(1).argmax())

roms = []
for label, d in ARMS:
    f, r, _ = base.load(d)
    roms.append((label, r[tpk]))
fom_pk = fom[tpk]

triang = Triangulation(pts[:, 0], pts[:, 1], tri)
XLIM, YLIM = (0.55, 1.55), (-0.32, 0.28)

fig, axs = plt.subplots(2, 4, figsize=(19, 7.2), constrained_layout=True)
for row, (comp, name) in enumerate([(0, "$v_x$"), (1, "$v_y$")]):
    errs = [np.abs(r[:, comp] - fom_pk[:, comp]) for _, r in roms]
    emax = float(np.percentile(np.concatenate(errs), 99.5))
    vlim = float(np.percentile(np.abs(fom_pk[:, comp]), 99.5))

    ax = axs[row, 0]
    tc = ax.tripcolor(triang, fom_pk[:, comp], shading="gouraud",
                      cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_title(f"reference {name}  [m/s]", fontsize=11)
    fig.colorbar(tc, ax=ax, fraction=0.046)

    for k, ((label, _), e) in enumerate(zip(roms, errs)):
        ax = axs[row, k + 1]
        tc = ax.tripcolor(triang, e, shading="gouraud", cmap="inferno",
                          vmin=0, vmax=emax)
        ax.set_title(f"|error| {name} — {label}", fontsize=11)
        fig.colorbar(tc, ax=ax, fraction=0.046)

    for ax in axs[row]:
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(f"Gust peak (t index {tpk}), flap region — error on a common "
             f"scale per row", fontsize=13)
p = OUT / "fig_slices_dataladder.png"
fig.savefig(p, dpi=130)
print(f"saved {p}")

# --- second figure: where the flow is actually reversed ---
fig2, axs2 = plt.subplots(1, 4, figsize=(19, 4.0), constrained_layout=True)
proj_f = (fom_pk[:, :2] * vn).sum(-1)
panels = [("reference", proj_f < 0)]
for label, r in roms:
    panels.append((label, ((r[:, :2] * vn).sum(-1)) < 0))
for ax, (label, mask) in zip(axs2, panels):
    c = np.where(flap, np.where(mask, 1.0, 0.0), np.nan)
    ax.tripcolor(triang, np.nan_to_num(c, nan=0.0), shading="gouraud",
                 cmap="coolwarm", vmin=0, vmax=1)
    n = int((mask & flap).sum())
    ax.set_title(f"reversed flow — {label}\n{n} of {int(flap.sum())} near-flap nodes",
                 fontsize=11)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
p2 = OUT / "fig_reversal_dataladder.png"
fig2.savefig(p2, dpi=130)
print(f"saved {p2}")

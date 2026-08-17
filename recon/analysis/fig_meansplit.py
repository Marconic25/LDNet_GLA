#!/usr/bin/env python3
"""Thesis figures for the mean-split near-wall study (cluster login node, numpy+mpl).

Reads the M-SPLIT reconstruction dumps and the arm-summary CSV and produces two
publication figures + a LaTeX-ready numbers file:

  fig_meansplit_decomp.png  static/dynamic vx-error decomposition by region,
                            base vs mean-split, for both test sims (the headline:
                            the static near-wall bias is annihilated).
  fig_meansplit_fields.png  near-airfoil vx snapshot at the gust peak: FOM |
                            |error| base | |error| mean-split (shared error scale)
                            — the visual proof of the near-wall improvement.
  meansplit_numbers.tex     \\newcommand macros with the headline figures so the
                            text never hardcodes a number.

Run: python3 analysis/fig_meansplit.py   (from recon/, on the login node)
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

BASE = Path("/work/u10677113/NACA2312/recon")
RES = BASE / "results"
HM = BASE / "analysis_hmetric"
MS = BASE / "analysis_meansplit"
OUT = MS / "figs"
OUT.mkdir(exist_ok=True)

TRI = np.load(BASE.parent / "recon_fields" / "sim_A_025_test_T3p0" / "mesh_triangles.npy")
region_label = np.load(HM / "region_labels.npy")
air = np.load(HM / "airfoil_nodes.npy")
N = len(region_label)

BASE_C, MS_C = "#c44e52", "#4c72b0"      # base = red, mean-split = blue
SIMS = {"sim_A_025": "gust only ($W_0=11.5$ m/s)",
        "sim_Cc_060": "gust + flap ($W_0=37.5$ m/s)"}
DIR = {("base", "sim_A_025"): "ms_base_s0_rom_a025",
       ("base", "sim_Cc_060"): "ms_base_s0_rom_cc060",
       ("ms", "sim_A_025"): "ms_ms_s0_rom_a025",
       ("ms", "sim_Cc_060"): "ms_ms_s0_rom_cc060"}
REGIONS = [("near", "near\nairfoil"), ("wake", "near\nwake"),
           ("far", "far\nfield"), ("surface", "airfoil\nsurface")]
RMASK = {"near": region_label == 0, "wake": region_label == 1,
         "far": region_label == 2, "surface": np.isin(np.arange(N), air)}


def load(arm, sim):
    d = RES / DIR[(arm, sim)]
    fom = np.load(d / f"fom_{sim}.npy").astype(np.float64)
    rom = np.load(d / f"rom_{sim}.npy").astype(np.float64)
    return fom, rom


def decomp(fom, rom, mask, k=0):
    """static / dynamic vx NRMSE (full-range denominator), region-masked."""
    err = rom - fom
    e_s = err.mean(0)                    # [N,3] time-mean bias
    e_d = err - e_s[None]
    rng = fom[:, :, k].max() - fom[:, :, k].min()
    ns = np.sqrt((e_s[mask, k] ** 2).mean()) / rng
    nd = np.sqrt((e_d[:, mask, k] ** 2).mean()) / rng
    return ns, nd


# ============================================================ FIG 1: decomp bars
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (sim, sub) in zip(axs, SIMS.items()):
    fb, rb = load("base", sim)
    fm, rm = load("ms", sim)
    x = np.arange(len(REGIONS))
    w = 0.38
    for j, (rk, _) in enumerate(REGIONS):
        m = RMASK[rk]
        bs, bd = decomp(fb, rb, m)
        ms, md = decomp(fm, rm, m)
        # base bar (left), mean-split bar (right); static solid, dynamic hatched
        ax.bar(x[j] - w / 2, bs, w, color=BASE_C, edgecolor="k", lw=.4,
               label="baseline — static" if j == 0 else None)
        ax.bar(x[j] - w / 2, bd, w, bottom=bs, color=BASE_C, edgecolor="k", lw=.4,
               alpha=.45, hatch="///",
               label="baseline — dynamic" if j == 0 else None)
        ax.bar(x[j] + w / 2, ms, w, color=MS_C, edgecolor="k", lw=.4,
               label="mean-split — static" if j == 0 else None)
        ax.bar(x[j] + w / 2, md, w, bottom=ms, color=MS_C, edgecolor="k", lw=.4,
               alpha=.45, hatch="///",
               label="mean-split — dynamic" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in REGIONS], fontsize=9)
    ax.set_title(sub, fontsize=10)
    ax.grid(axis="y", alpha=.3)
axs[0].set_ylabel("$v_x$ NRMSE (field range)")
axs[0].legend(fontsize=8, loc="upper right", framealpha=.95)
fig.suptitle("Static vs dynamic reconstruction error, per region "
             "($d_s{=}1$, single seed)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_meansplit_decomp.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote fig_meansplit_decomp.png")

# ============================================================ FIG 2: field maps
# near-airfoil vx at the gust peak, |error| base vs mean-split (shared scale).
SIM = "sim_Cc_060"
fb, rb = load("base", SIM)
fm, rm = load("ms", SIM)
pts = np.load((RES / DIR[("base", SIM)]) / "rom_points.npy").astype(np.float64)
# gust peak = time of max |vx fluctuation| RMS over the near region
near = RMASK["near"]
fl = fb[:, near, 0] - fb[:, near, 0].mean(0)
tpk = int(np.argmax((fl ** 2).mean(1)))
triang = Triangulation(pts[:, 0], pts[:, 1], TRI)
eb = np.abs(rb[tpk, :, 0] - fb[tpk, :, 0])
em = np.abs(rm[tpk, :, 0] - fm[tpk, :, 0])
emax = np.percentile(np.concatenate([eb, em]), 99)
fmax = np.percentile(np.abs(fb[tpk, :, 0]), 99)

fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
data = [(fb[tpk, :, 0], "FOM  $v_x$ [m/s]", "RdBu_r", (-fmax, fmax)),
        (eb, "$|v_x$ error$|$  baseline", "inferno", (0, emax)),
        (em, "$|v_x$ error$|$  mean-split", "inferno", (0, emax))]
for ax, (v, ttl, cmap, vlim) in zip(axs, data):
    tpc = ax.tripcolor(triang, v, shading="gouraud", cmap=cmap,
                       vmin=vlim[0], vmax=vlim[1])
    ax.set_title(ttl, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 1.6); ax.set_ylim(-0.6, 0.6)   # near-airfoil crop
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.02)
fig.suptitle(f"{SIMS[SIM]}, gust peak ($t$ index {tpk})  —  error shares a common "
             f"scale (max {emax:.1f} m/s)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig_meansplit_fields.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote fig_meansplit_fields.png")

# ============================================================ numbers -> tex
def arm_agg(csvpath):
    rows = list(csv.DictReader(open(csvpath)))
    def g(arm, sim, reg, field="vx"):
        vs = [float(r["nrmse_mean"]) for r in rows if r["arm"] == arm
              and r["sim"] == sim and r["region"] == reg and r["field"] == field]
        return vs[0] if vs else float("nan")
    return g

g = arm_agg(MS / "meansplit_arm_summary.csv")
macros = {
    "msCcBaseNear": f"{g('base','sim_Cc_060','near'):.3f}",
    "msCcMsNear": f"{g('ms','sim_Cc_060','near'):.4f}",
    "msCcFactor": f"{g('base','sim_Cc_060','near')/g('ms','sim_Cc_060','near'):.1f}",
    "msABaseNear": f"{g('base','sim_A_025','near'):.3f}",
    "msAMsNear": f"{g('ms','sim_A_025','near'):.4f}",
    "msAFactor": f"{g('base','sim_A_025','near')/g('ms','sim_A_025','near'):.0f}",
}
with open(OUT / "meansplit_numbers.tex", "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("wrote meansplit_numbers.tex:", macros)

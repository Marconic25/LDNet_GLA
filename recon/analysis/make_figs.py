#!/usr/bin/env python3
"""H-METRIC figures (local, WSL python): error-vs-time with excitation overlay,
per-field/per-region NRMSE bars, and combined-metric composition."""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DS = [1, 5, 10]
COLORS = {1: "#1f77b4", 5: "#ff7f0e", 10: "#2ca02c"}
SIMS = ["sim_A_025", "sim_Cc_060"]
TITLES = {"sim_A_025": "sim_A_025 (gust only, W0=11.5 m/s, Tg=1.12 s)",
          "sim_Cc_060": "sim_Cc_060 (gust+flap, W0=37.5 m/s, Tg=0.67 s)"}

rows = list(csv.DictReader(open(HERE / "error_decomposition.csv")))
shares = list(csv.DictReader(open(HERE / "mse_shares.csv")))


def dec(sim, f, r, w):
    return {int(x["d_s"]): float(x["nrmse"]) for x in rows
            if x["sim"] == sim and x["field"] == f and x["region"] == r
            and x["time_window"] == w}


# ---------------------------------------------------------------- err vs time
for sim in SIMS:
    ts = np.genfromtxt(HERE / f"timeseries_{sim}.csv", delimiter=",", names=True)
    t, W, delta = ts["t"], ts["W_gust"], ts["delta"]
    exc = np.abs(W) >= 0.1 * np.abs(W).max()
    if np.abs(delta).max() > 1e-9:
        exc |= np.abs(delta) >= 0.1 * np.abs(delta).max()

    fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    ax = axs[0]
    ax.plot(t, W, "k-", lw=1.5, label="gust W(t) [m/s]")
    ax.set_ylabel("W [m/s]")
    if np.abs(delta).max() > 1e-9:
        ax2 = ax.twinx()
        ax2.plot(t, delta, "r--", lw=1.2, label="flap $\\delta$(t) [deg]")
        ax2.set_ylabel("$\\delta$ [deg]", color="r")
        ax2.tick_params(axis="y", colors="r")
    ax.set_title(TITLES[sim] + "\nexcitation window shaded", fontsize=11)

    panels = [("combined", "combined NRMSE\n(repo convention,\nglobal mixed range)"),
              ("vx", "$v_x$ NRMSE\n(field range)"),
              ("p_surf", "surface-$p$ NRMSE\n(surface range,\nloads proxy)")]
    for ax, (key, lab) in zip(axs[1:], panels):
        for ds in DS:
            ax.plot(t, ts[f"l{ds}_{key}"], color=COLORS[ds], lw=1.4,
                    label=f"$d_s$={ds}")
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axs[1].legend(loc="upper right", fontsize=9, ncol=3)
    for ax in axs:
        yl = ax.get_ylim()
        ax.fill_between(t, 0, 1, where=exc, transform=ax.get_xaxis_transform(),
                        color="0.85", zorder=0)
        ax.set_ylim(yl)
    axs[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(HERE / f"err_vs_time_{sim}.png", dpi=150)
    plt.close(fig)
    print(f"wrote err_vs_time_{sim}.png")

# ------------------------------------------------- per-field / per-region bars
n_nodes = {x["region"]: int(x["n_nodes"]) for x in rows if x["sim"] == "sim_A_025"
           and x["d_s"] == "1" and x["field"] == "vx" and x["time_window"] == "all"}
regions = ["near", "wake", "far", "surface"]
fields = ["vx", "vy", "p"]
fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), sharey="row")
for i, sim in enumerate(SIMS):
    for j, f in enumerate(fields):
        ax = axs[i, j]
        xpos = np.arange(len(regions))
        for k, ds in enumerate(DS):
            vals = [dec(sim, f, r, "all")[ds] for r in regions]
            ax.bar(xpos + (k - 1) * 0.25, vals, 0.25, color=COLORS[ds],
                   label=f"$d_s$={ds}" if (i == 0 and j == 0) else None)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{r}\n({n_nodes[r]})" for r in regions], fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        if i == 0:
            ax.set_title(f, fontsize=11)
        if j == 0:
            ax.set_ylabel(f"{sim}\nNRMSE (per-field global range)", fontsize=9)
fig.legend(loc="upper right", fontsize=9, ncol=3)
fig.suptitle("Per-field / per-region NRMSE by latent dim (window=all; node counts in axis labels)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(HERE / "nrmse_by_field_region.png", dpi=150)
plt.close(fig)
print("wrote nrmse_by_field_region.png")

# ------------------------------------------------- combined-metric composition
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, kind, title in [(axs[0], "field", "SSE share by FIELD"),
                        (axs[1], "region", "SSE share by REGION")]:
    names = fields if kind == "field" else ["near", "wake", "far"]
    xpos = np.arange(len(SIMS) * len(DS))
    bottoms = np.zeros(len(xpos))
    cmap = plt.get_cmap("tab20")
    for ni, name in enumerate(names):
        vals = []
        for sim in SIMS:
            for ds in DS:
                vals.append([float(x["share"]) for x in shares
                             if x["sim"] == sim and int(x["d_s"]) == ds
                             and x["kind"] == kind and x["name"] == name][0])
        vals = np.array(vals)
        ax.bar(xpos, vals, 0.7, bottom=bottoms, label=name, color=cmap(ni * 2))
        bottoms += vals
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{s.split('_')[1]}{s.split('_')[2]}\n$d_s$={d}"
                        for s in SIMS for d in DS], fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
fig.suptitle("What the reported combined NRMSE is made of (share of total squared error)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(HERE / "combined_metric_composition.png", dpi=150)
plt.close(fig)
print("wrote combined_metric_composition.png")

#!/usr/bin/env python3
"""Parse train.log and plot loss curves for all 3 latent sizes."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = "/home/marco/LDNet_OF/recon/models/train.log"

def parse_log(path):
    runs = {}
    cur_nls = None
    cur_phase = None
    with open(path) as f:
        for line in f:
            m = re.search(r"num_latent_states\s*=\s*(\d+)", line)
            if m:
                cur_nls = int(m.group(1))
                runs[cur_nls] = {"adam": [], "bfgs": []}
                cur_phase = None
                continue
            if "Adam..." in line:
                cur_phase = "adam"
            if "BFGS..." in line:
                cur_phase = "bfgs"
            m = re.match(r"\s*epoch\s+(\d+)\s+-\s+training loss:\s+([0-9.e+-]+)\s+-\s+validation loss\s+([0-9.e+-]+)", line)
            if m and cur_nls is not None and cur_phase is not None:
                ep = int(m.group(1))
                tr = float(m.group(2))
                va = float(m.group(3))
                runs[cur_nls][cur_phase].append((ep, tr, va))
    return runs

runs = parse_log(LOG)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
colors = {"adam": "#e07b39", "bfgs": "#3a7fc1"}
titles = {1: "$d_s = 1$", 5: "$d_s = 5$", 10: "$d_s = 10$"}

for ax, nls in zip(axes, [1, 5, 10]):
    data = runs[nls]
    for phase in ["adam", "bfgs"]:
        pts = data[phase]
        if not pts:
            continue
        ep = np.array([p[0] for p in pts])
        tr = np.array([p[1] for p in pts])
        va = np.array([p[2] for p in pts])
        ax.semilogy(ep, tr, color=colors[phase], lw=1.5, label=f"{phase.upper()} train")
        ax.semilogy(ep, va, color=colors[phase], lw=1.5, ls="--", label=f"{phase.upper()} val")
    # mark Adam→BFGS boundary
    if data["bfgs"]:
        boundary = data["bfgs"][0][0]
        ax.axvline(boundary, color="gray", lw=0.8, ls=":")
        ax.text(boundary + 20, ax.get_ylim()[0] * 1.2, "BFGS", fontsize=8, color="gray")
    ax.set_title(titles[nls], fontsize=12)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (normalized)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

fig.suptitle("LDNet field reconstruction — training curves", fontsize=13)
fig.tight_layout()
out = "/home/marco/LDNet_OF/recon/results/loss_curves.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")

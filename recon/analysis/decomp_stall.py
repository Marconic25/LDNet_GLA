#!/usr/bin/env python3
"""STALL DIAGNOSTIC: does the FOM data near the flap actually show flow separation
(local velocity reversal relative to its own attached/quiescent baseline), and does
the champion ROM (mean-split + CORAL o10, d_s=1) get the sign wrong there too?

Runs locally (no cluster needed) on the already-synced champion FOM/ROM dump for the
hardest test case (sim_Cc_060, gust+flap). Reuses region_labels.npy/airfoil_nodes.npy
(local copies at recon/analysis/, NOT the cluster-only analysis_hmetric/ path).

Separation indicator: for each near-airfoil domain node, take its own FOM velocity
vector at t=0 (quiescent baseline, flow attached) as a per-node reference direction.
At every later time, project the local velocity onto that fixed reference direction.
A negative projection means the local flow has reversed relative to its own attached
baseline -- this sidesteps needing to reason about surface-tangent sign conventions
across the closed airfoil+flap contour (upper/lower surface loop-winding flips sign
in a way that's easy to get backwards).

Outputs: printed verdict + recon/analysis/figs_stall/stall_diagnostic.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path

AN = Path(__file__).resolve().parent
RES = AN.parent / "results"
D = RES / "ms_coral_o10_s0_rom_cc060"
OUT = AN / "figs_stall"
OUT.mkdir(exist_ok=True)

region = np.load(AN / "region_labels.npy")     # 0 near, 1 wake, 2 far
air = np.load(AN / "airfoil_nodes.npy")        # main[0:292) + flap[292:387) perimeter-ordered

fom = np.load(D / "fom_sim_Cc_060.npy").astype(np.float64)   # [T,N,3] vx,vy,p
rom = np.load(D / "rom_sim_Cc_060.npy").astype(np.float64)
pts = np.load(D / "rom_points.npy").astype(np.float64)
times = np.load(D / "rom_times.npy").astype(np.float64)
tri = np.load(RES / "mesh_triangles.npy")
t_rel = times - times[0]
T, N, _ = fom.shape
print(f"T={T} snapshots, N={N} nodes, dt~{np.mean(np.diff(t_rel)):.4f}s, "
      f"near={int((region==0).sum())} wake={int((region==1).sum())} "
      f"far={int((region==2).sum())} surface={len(air)}")

near = region == 0

# nearest airfoil-surface node per domain node -> which element (main vs flap)
d2 = ((pts[:, None, :] - pts[air][None, :, :]) ** 2).sum(-1)
nn = d2.argmin(1)
is_flap_near = near & (nn >= 292)
is_main_near = near & (nn < 292)
print(f"near-flap nodes: {int(is_flap_near.sum())}, near-main nodes: {int(is_main_near.sum())}")

# reference direction = each node's own FOM velocity at t=0 (quiescent, attached)
vref = fom[0, :, :2]
vref_norm = np.linalg.norm(vref, axis=1) + 1e-6
proj_fom = (fom[:, :, 0] * vref[None, :, 0] + fom[:, :, 1] * vref[None, :, 1]) / vref_norm[None, :]
proj_rom = (rom[:, :, 0] * vref[None, :, 0] + rom[:, :, 1] * vref[None, :, 1]) / vref_norm[None, :]
reversed_fom = proj_fom < 0
reversed_rom = proj_rom < 0

frac_flap_fom = reversed_fom[:, is_flap_near].mean(1)
frac_main_fom = reversed_fom[:, is_main_near].mean(1)
frac_flap_rom = reversed_rom[:, is_flap_near].mean(1)
frac_main_rom = reversed_rom[:, is_main_near].mean(1)

tpk = int(frac_flap_fom.argmax())
print(f"\nFOM peak flap-region reversed-flow fraction: {frac_flap_fom[tpk]:.3f} "
      f"at t={t_rel[tpk]:.3f}s (idx {tpk})")
print(f"FOM main-element reversed-flow fraction at same time: {frac_main_fom[tpk]:.3f}")
print(f"FOM baseline (t=0..2) flap frac: {frac_flap_fom[:3].mean():.3f}, "
      f"main frac: {frac_main_fom[:3].mean():.3f}")
print(f"ROM flap-region reversed-flow fraction at t={t_rel[tpk]:.3f}s: {frac_flap_rom[tpk]:.3f} "
      f"(FOM says {frac_flap_fom[tpk]:.3f})")
elevated = frac_flap_fom > 0.25
if elevated.any():
    print(f"FOM flap frac > 0.25 window: t={t_rel[elevated].min():.3f}s .. "
          f"t={t_rel[elevated].max():.3f}s ({elevated.sum()} of {T} snapshots)")

# worst-error near-flap points at the peak time: what does each model say?
flap_idx = np.where(is_flap_near)[0]
err_pk = np.abs(rom[tpk, :, 0] - fom[tpk, :, 0])
worst = flap_idx[np.argsort(-err_pk[flap_idx])[:15]]
print(f"\nworst 15 near-flap points at t={t_rel[tpk]:.3f}s "
      f"(x,y | fom_vx rom_vx | fom_proj rom_proj):")
for i in worst:
    print(f"  ({pts[i,0]:5.2f},{pts[i,1]:6.3f})  fom_vx={fom[tpk,i,0]:7.2f} "
          f"rom_vx={rom[tpk,i,0]:7.2f}  fom_proj={proj_fom[tpk,i]:7.2f} "
          f"rom_proj={proj_rom[tpk,i]:7.2f}")
n_sign_flip = int(((proj_fom[tpk, flap_idx] < 0) != (proj_rom[tpk, flap_idx] < 0)).sum())
print(f"\nof {len(flap_idx)} near-flap points at peak time, ROM gets the "
      f"attached/reversed SIGN wrong at {n_sign_flip} ({100*n_sign_flip/len(flap_idx):.1f}%)")

# --- figure: time series + spatial snapshot ---
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})

ax = axs[0]
ax.plot(t_rel, frac_flap_fom, "o-", ms=3, label="FOM, near-flap", color="C3")
ax.plot(t_rel, frac_main_fom, "o-", ms=3, label="FOM, near-main", color="C0")
ax.plot(t_rel, frac_flap_rom, "--", lw=1.3, label="ROM, near-flap", color="C3", alpha=0.6)
ax.axvline(t_rel[tpk], color="k", ls=":", lw=1, label=f"peak t={t_rel[tpk]:.3f}s")
ax.set_xlabel("t [s]"); ax.set_ylabel("fraction reversed vs t=0 baseline")
ax.set_title("Local flow reversal fraction over time")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axs[1]
triang = Triangulation(pts[:, 0], pts[:, 1], tri)
c = np.where(is_flap_near, np.where(reversed_fom[tpk], 2, 1), np.where(is_main_near, 0.3, np.nan))
sc = ax.tripcolor(triang, np.nan_to_num(c, nan=0.0), shading="flat", cmap="RdYlBu_r",
                   vmin=0, vmax=2, alpha=0.9)
ax.set_xlim(0.5, 1.15); ax.set_ylim(-0.25, 0.15); ax.set_aspect("equal")
ax.set_title(f"near-flap FOM reversed-flow mask, t={t_rel[tpk]:.3f}s\n"
             f"(red=reversed, blue=attached, near-flap band only)")
fig.suptitle("Stall/separation diagnostic -- champion (mean-split+CORAL o10), sim_Cc_060")
fig.tight_layout()
fig.savefig(OUT / "stall_diagnostic.png", dpi=150, bbox_inches="tight")
print(f"\nsaved {OUT / 'stall_diagnostic.png'}")

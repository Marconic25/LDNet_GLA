#!/usr/bin/env python3
"""Velocity animation, Cc_050 (validation, held out): CFD (left) vs LDNet (right).

Same layout as anim_W20Tg07_velocity.py (that script is the format spec): two
field panels side by side, a small centred panel underneath showing W_gust(t)
with a marker tracking the current instant. The difference from that script is
what the two panels compare -- there it was two different runs (open vs closed
loop); here it is two SOURCES of the same run (CFD reference vs the LDNet
reconstruction from ms_coral_o10_N100_s0), so both panels share one mesh and
one time axis (rom_times.npy, already absolute -- no checkpoint offset).

Reuses fig_Cc050_recon_slices.py for everything shared: data loading, the
u_x/2*U_inf colour window, the exaggerated-motion outline and the grid warp
that keeps the frozen reference grid's body-shaped hole locked to the drawn
outline (see that module, and fig_W20Tg07_slices.py, for why the warp exists).

Run:  python3 recon/analysis/anim_Cc050_velocity.py [--stride N] [--fps N]
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.tri import Triangulation

import fig_Cc050_recon_slices as F

OUT = F.OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth snapshot")
    ap.add_argument("--fps", type=float, default=15,
                    help="150 frames @ 20.13 ms native spacing; 15 fps plays "
                         "back at roughly 1x")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--out", default="anim_Cc050_velocity.mp4")
    ap.add_argument("--tmax", type=float, default=None,
                    help="stop at this physical simulation time [s] "
                         "(gust time axis), not video duration")
    # fig_Cc050_recon_slices.py uses EXAG_WING=5, EXAG_FLAP=1 for its 4 static
    # columns -- chosen because THIS case's flap motion is real and already
    # visible (peak 5.3 deg) so amplifying it would misrepresent the point of
    # the figure, while the wing's rigid-body motion is millimetric and
    # invisible unaided. That reasoning does not depend on frame count, so it
    # carries over unchanged to every frame of the animation.
    ap.add_argument("--exag-wing", type=float, default=F.EXAG_WING)
    ap.add_argument("--exag-flap", type=float, default=F.EXAG_FLAP)
    args = ap.parse_args()

    F.EXAG_WING = args.exag_wing
    F.EXAG_FLAP = args.exag_flap
    print(f"amplification: wing x{F.EXAG_WING:g}, flap x{F.EXAG_FLAP:g}")

    fom, rom, pts, times, traj, tri = F.load_case()
    wing_idx, flap_idx = F.wing_flap_indices(pts, tri)
    case = dict(
        pts=pts, tri=tri, traj=traj,
        wing_idx=wing_idx, flap_idx=flap_idx,
        wing0=pts[wing_idx].copy(), flap0=pts[flap_idx].copy(),
        t_ref_pose=F.traj_at(traj, times[0]),
        d_wing=F._dist_to_loop(pts, pts[wing_idx]),
        d_flap=F._dist_to_loop(pts, pts[flap_idx]))
    case["d_body"] = np.minimum(case["d_wing"], case["d_flap"])
    print(f"{F.SIM}: {len(times)} snapshots, t=[{times[0]:.3f}, {times[-1]:.3f}] s")

    frame_idx = list(range(0, len(times), args.stride))
    if args.tmax is not None:
        frame_idx = [i for i in frame_idx if times[i] <= args.tmax]
    print(f"{len(frame_idx)} frames @ {args.fps} fps "
          f"-> {len(frame_idx)/args.fps:.1f} s of video, "
          f"sim time [{times[frame_idx[0]]:.3f}, {times[frame_idx[-1]]:.3f}] s")

    sources = {"CFD": (fom, F.C_FOM), "LDNet": (rom, F.C_ROM)}

    # ── figure: two field panels + a small centred gust panel underneath ──
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.028],
                          height_ratios=[1, 0.42],
                          hspace=0.30, wspace=0.06,
                          left=0.015, right=0.945, top=0.94, bottom=0.09)
    ax_fom = fig.add_subplot(gs[0, 0])
    ax_rom = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    ax_g = fig.add_axes([0.375, 0.10, 0.25, 0.24])

    axes = {"CFD": ax_fom, "LDNet": ax_rom}

    sm = plt.cm.ScalarMappable(
        cmap=F.CMAP_VEL,
        norm=matplotlib.colors.BoundaryNorm(F.UMAG_LEVELS, 256))
    fig.colorbar(sm, cax=cax, label=F.LBL_UMAG, extend="both")

    # ── gust panel (static background drawn once) ──
    # fig_Cc050_recon_slices.py's traj dict carries only t/h/alpha/delta (what
    # the outline needs), not W_gust -- read it separately here.
    import csv
    gt2, gw2 = [], []
    with open(F.CASE / "structural_trajectory.csv") as fh:
        for row in csv.DictReader(fh):
            gt2.append(float(row["t"])); gw2.append(float(row["W_gust"]))
    gt, gw = np.array(gt2), np.array(gw2)
    ax_g.plot(gt, gw, color="0.25", lw=1.2)
    ax_g.set_xlim(0, times[-1])
    ax_g.set_ylim(-1, max(gw.max(), 1.0) * 1.15)
    ax_g.set_xlabel("t [s]", fontsize=8, labelpad=1)
    ax_g.set_ylabel("$W_g$ [m/s]", fontsize=8, labelpad=2)
    ax_g.tick_params(labelsize=7, length=2, pad=1)
    ax_g.grid(alpha=0.25, lw=0.5)
    for s in ax_g.spines.values():
        s.set_linewidth(0.6)
    marker_line = ax_g.axvline(0.0, color="#CC3311", lw=1.4)
    marker_dot, = ax_g.plot([], [], "o", color="#CC3311", ms=4)

    for label, ax in axes.items():
        ax.set_title(label, fontsize=12, fontweight="bold",
                     color=sources[label][1], pad=6)

    def draw_frame(k):
        c_idx = frame_idx[k]
        t = float(times[c_idx])
        wpts = F.warped_points(case, t)
        inv = F.inverted_mask(case["pts"], wpts, case["tri"])
        triang = Triangulation(wpts[:, 0], wpts[:, 1], case["tri"])
        triang.set_mask(inv)

        for label, ax in axes.items():
            src, color = sources[label]
            ax.clear()
            f = src[c_idx]
            vel = np.sqrt(f[:, 0] ** 2 + f[:, 1] ** 2)
            ax.tricontourf(triang, vel, levels=F.UMAG_LEVELS,
                           cmap=F.CMAP_VEL, extend="both")
            F.draw_outline(ax, case, t)
            F.style_ax(ax)
            ax.set_title(label, fontsize=12, fontweight="bold", color=color, pad=6)

        marker_line.set_xdata([t, t])
        j = int(np.argmin(np.abs(gt - t)))
        marker_dot.set_data([t], [gw[j]])
        ax_g.set_title(f"t = {t:.3f} s", fontsize=8.5, pad=3)
        if k % 20 == 0:
            print(f"  frame {k}/{len(frame_idx)} (t={t:.3f}s)", flush=True)
        return []

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(frame_idx),
                                   blit=False)
    path = OUT / args.out
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=6000,
                                    metadata=dict(artist="LDNet_OF"))
    anim.save(str(path), writer=writer, dpi=args.dpi)
    print(f"saved {path}")


if __name__ == "__main__":
    main()

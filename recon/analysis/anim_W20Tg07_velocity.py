#!/usr/bin/env python3
"""Velocity animation, W20/Tg0.70: open loop (left) vs closed loop (right).

Every extracted snapshot becomes a frame. A small centred panel shows the gust
W_gust(t) with a marker tracking the current instant, so the flow response can
be read against the gust phase.

Reuses fig_W20Tg07_slices.py for everything shared -- colour scale, banded
levels, the exaggerated body outline and, crucially, the grid warp that keeps
the field's body-shaped hole locked to the drawn outline (see that module for
why the stored grid is frozen and why the warp is required).

Run:  python3 recon/analysis/anim_W20Tg07_velocity.py [--stride N] [--fps N]
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.tri import Triangulation

import fig_W20Tg07_slices as F

OUT = Path(__file__).resolve().parent / "figs_W20Tg07"
OUT.mkdir(exist_ok=True)

CASES = [("Open loop", "viz_W20Tg0.70_OL", F.C_OPEN),
         ("Closed loop", "viz_W20Tg0.70_CL", F.C_CLOSED)]


def load_gust(traj):
    return traj["t"], traj["w"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth snapshot")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--out", default="anim_W20Tg07_velocity.mp4")
    # The static figure amplifies motion x5 because its four sampled instants
    # happen to sit where the flap has barely moved (~2.5 deg). Over the FULL
    # trajectory the closed-loop flap actually sweeps -13.8..+10.3 deg, which
    # needs no amplification to read -- and amplifying it tears the warped grid
    # (x5 would be -69..+52 deg). Hence x1 here: true positions, no distortion.
    # The warp itself is still required: the stored grid is frozen at the
    # reference pose, so even true motion must be carried by the coordinates.
    ap.add_argument("--exag-wing", type=float, default=1.0)
    ap.add_argument("--exag-flap", type=float, default=1.0)
    args = ap.parse_args()

    F.EXAG_WING = args.exag_wing
    F.EXAG_FLAP = args.exag_flap
    print(f"amplification: wing x{F.EXAG_WING:g}, flap x{F.EXAG_FLAP:g}")

    cases = {}
    for label, name, color in CASES:
        pts, tri, times, fields, traj = F.load_case(name)
        wing0, flap0 = F.wing_flap_outlines(pts, tri)
        d_wing = F._dist_to_loop(pts, wing0)
        d_flap = F._dist_to_loop(pts, flap0)
        cases[label] = dict(
            pts=pts, tri=tri, times=times, fields=fields, traj=traj, color=color,
            wing0=wing0, flap0=flap0, d_wing=d_wing, d_flap=d_flap,
            d_body=np.minimum(d_wing, d_flap), t0_ref=F.traj_at(traj, 0.0))
        print(f"{label}: {len(times)} snapshots, "
              f"t=[{times[0]-F.CHECKPOINT_T0:.3f}, {times[-1]-F.CHECKPOINT_T0:.3f}] s")

    ref = cases["Open loop"]
    frame_idx = list(range(0, len(ref["times"]), args.stride))
    t_phys = ref["times"][frame_idx] - F.CHECKPOINT_T0
    print(f"{len(frame_idx)} frames @ {args.fps} fps "
          f"-> {len(frame_idx)/args.fps:.1f} s of video")

    # ── figure: two field panels + a small centred gust panel underneath ──
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.028],
                          height_ratios=[1, 0.42],
                          hspace=0.30, wspace=0.06,
                          left=0.015, right=0.945, top=0.94, bottom=0.09)
    ax_ol = fig.add_subplot(gs[0, 0])
    ax_cl = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    # centred and small: sits under the seam between the two field panels
    ax_g = fig.add_axes([0.375, 0.10, 0.25, 0.24])

    axes = {"Open loop": ax_ol, "Closed loop": ax_cl}

    # static colorbar (scale is fixed, so build it once from a throwaway mapping)
    sm = plt.cm.ScalarMappable(
        cmap=F.CMAP_VEL,
        norm=matplotlib.colors.BoundaryNorm(F.VEL_LEVELS, 256))
    fig.colorbar(sm, cax=cax, label=F.LBL_VEL, extend="both")

    # ── gust panel (static background drawn once) ──
    gt, gw = ref["traj"]["t"], ref["traj"]["w"]
    ax_g.plot(gt, gw, color="0.25", lw=1.2)
    ax_g.set_xlim(0, t_phys[-1])
    ax_g.set_ylim(-1, max(gw) * 1.15)
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
                     color=cases[label]["color"], pad=6)

    def draw_frame(k):
        i = frame_idx[k]
        t = float(ref["times"][i] - F.CHECKPOINT_T0)
        for label, ax in axes.items():
            c = cases[label]
            ax.clear()
            f = c["fields"][i]
            vel = np.sqrt(f[:, 0] ** 2 + f[:, 1] ** 2)
            wpts = F.warped_points(c, t)
            inv = F.inverted_mask(c["pts"], wpts, c["tri"])
            triang = F.make_triang(wpts, c["tri"], inv)
            ax.tricontourf(triang, vel, levels=F.VEL_LEVELS,
                           cmap=F.CMAP_VEL, extend="both")
            F.draw_outline(ax, c, t)
            F.style_ax(ax)
            ax.set_title(label, fontsize=12, fontweight="bold",
                         color=c["color"], pad=6)

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

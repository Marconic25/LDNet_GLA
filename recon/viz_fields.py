#!/usr/bin/env python3
"""Field visualization & FOM-vs-LDNet comparison for the reconstruction model.

Renders the extracted CFD fields (and, once available, LDNet reconstructions) using
the native mesh triangulation, so the airfoil hole and wake are respected (no
interpolation artifacts). Channels: 0=Ux (vx), 1=Uy (vy), 2=p (pressure).

Phase-4 deliverables this supports:
  * snapshot       — single-time multi-panel FOM field (paper panels c/d style)
  * compare        — side-by-side FOM vs LDNet at a snapshot (added when a model exists)
  * video          — time animation (added when a model exists)
  * nrmse-curve    — NRMSE vs num latent states (added after the latent sweep)

For now `snapshot` works on FOM data alone (validates the extraction + renderer).

Usage:
  python viz_fields.py snapshot --data recon/data/smoke --name sim_A_025_test \
      --idx 8 --out recon/results/fom_smoke.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

CHANNELS = ["vx", "vy", "p"]
CMAPS = {"vx": "RdBu_r", "vy": "RdBu_r", "p": "viridis", "speed": "magma"}


def load_fields(data_dir, name):
    """Load fields_<name>.npy + mesh + times from a directory."""
    data_dir = Path(data_dir)
    fpath = data_dir / f"fields_{name}.npy"
    if not fpath.exists():  # fall back to any single fields_*.npy
        cands = sorted(data_dir.glob("fields_*.npy"))
        if not cands:
            raise FileNotFoundError(f"no fields_*.npy in {data_dir}")
        fpath = cands[0]
    fields = np.load(fpath)                              # [T, Npts, 3]
    points = np.load(data_dir / "mesh_points.npy")        # [Npts, 2]
    tri = np.load(data_dir / "mesh_triangles.npy")        # [Ntri, 3]
    times = np.load(data_dir / "field_times.npy")         # [T]
    return fields, points, tri, times


def make_tri(points, tri):
    return Triangulation(points[:, 0], points[:, 1], tri)


def panel(ax, triang, vals, title, cmap, vlim=None):
    if vlim is None:
        vmax = np.percentile(np.abs(vals), 99)
        vlim = (-vmax, vmax) if cmap == "RdBu_r" else (np.percentile(vals, 1),
                                                        np.percentile(vals, 99))
    tpc = ax.tripcolor(triang, vals, shading="gouraud", cmap=cmap,
                       vmin=vlim[0], vmax=vlim[1])
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.02)


def cmd_snapshot(args):
    fields, points, tri, times = load_fields(args.data, args.name)
    T = fields.shape[0]
    idx = args.idx if args.idx >= 0 else T // 2
    idx = max(0, min(idx, T - 1))
    triang = make_tri(points, tri)

    vx, vy, p = fields[idx, :, 0], fields[idx, :, 1], fields[idx, :, 2]
    speed = np.hypot(vx, vy)

    fig, axs = plt.subplots(2, 2, figsize=(11, 6))
    fig.suptitle(f"FOM (OpenFOAM)  {args.name}  snapshot {idx}/{T-1}  t={times[idx]:.4f}s",
                 fontsize=12)
    panel(axs[0, 0], triang, vx,    "$v_x$ [m/s]",  CMAPS["vx"])
    panel(axs[0, 1], triang, vy,    "$v_y$ [m/s]",  CMAPS["vy"])
    panel(axs[1, 0], triang, speed, "$|v|$ [m/s]",  CMAPS["speed"])
    panel(axs[1, 1], triang, p,     "$p$ [Pa]",     CMAPS["p"])
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}  (fields {fields.shape}, {len(points)} pts, {len(tri)} tri)")


def load_recon(recon_dir, name, tri_path):
    """Load FOM + ROM fields produced by reconstruct_fields.py + a triangulation."""
    d = Path(recon_dir)
    rom = np.load(d / f"rom_{name}.npy")          # [T,Npts,3]
    fom = np.load(d / f"fom_{name}.npy")          # [T,Npts,3]
    points = np.load(d / "rom_points.npy")
    times = np.load(d / "rom_times.npy")
    tri = np.load(tri_path)
    return fom, rom, points, tri, times


def cmd_compare(args):
    fom, rom, points, tri, times = load_recon(args.recon, args.name, args.tri)
    T = fom.shape[0]
    idx = args.idx if args.idx >= 0 else T // 2
    idx = max(0, min(idx, T - 1))
    triang = make_tri(points, tri)

    labels = ["$v_x$ [m/s]", "$v_y$ [m/s]", "$p$ [Pa]"]
    fig, axs = plt.subplots(3, 3, figsize=(13, 7.5))
    fig.suptitle(f"FOM vs LDNet ($d_s$={args.latent})  {args.name}  "
                 f"snapshot {idx}/{T-1}  t={times[idx]-times[0]:.3f}s", fontsize=12)
    for r in range(3):
        f, m = fom[idx, :, r], rom[idx, :, r]
        cmap = CMAPS["p"] if r == 2 else "RdBu_r"
        if cmap == "RdBu_r":
            vmax = np.percentile(np.abs(f), 99); vlim = (-vmax, vmax)
        else:
            vlim = (np.percentile(f, 1), np.percentile(f, 99))
        panel(axs[r, 0], triang, f, f"FOM  {labels[r]}", cmap, vlim)
        panel(axs[r, 1], triang, m, f"LDNet  {labels[r]}", cmap, vlim)
        emax = np.percentile(np.abs(m - f), 99) or 1.0
        panel(axs[r, 2], triang, np.abs(m - f), f"|error|  {labels[r]}", "inferno", (0, emax))
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    nrmse = float(np.sqrt(np.mean((rom - fom) ** 2)) / (fom.max() - fom.min()))
    print(f"saved {out}  combined NRMSE {nrmse:.3e}")


def cmd_compare_video(args):
    import matplotlib.animation as animation
    fom, rom, points, tri, times = load_recon(args.recon, args.name, args.tri)
    if args.stride > 1:
        fom, rom, times = fom[::args.stride], rom[::args.stride], times[::args.stride]
    T = fom.shape[0]
    triang = make_tri(points, tri)
    ch = {"vx": 0, "vy": 1, "p": 2}[args.channel]
    F, M = fom[:, :, ch], rom[:, :, ch]
    cmap = CMAPS["p"] if args.channel == "p" else "RdBu_r"
    if cmap == "RdBu_r":
        vmax = np.percentile(np.abs(F), 99); vlim = (-vmax, vmax)
    else:
        vlim = (np.percentile(F, 1), np.percentile(F, 99))

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    def update(i):
        for ax, arr, ttl in zip(axs, [F[i], M[i]], ["FOM", "LDNet"]):
            ax.clear(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.tripcolor(triang, arr, shading="gouraud", cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            ax.set_title(f"{ttl} {args.channel}  t={times[i]-times[0]:.3f}s", fontsize=9)
    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 / args.fps)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".mp4":
        anim.save(out, writer="ffmpeg", fps=args.fps, dpi=130)
    else:
        anim.save(out, writer="pillow", fps=args.fps, dpi=100)
    plt.close(fig)
    print(f"saved {out}  ({T} frames)")


def cmd_video(args):
    import matplotlib.animation as animation
    fields, points, tri, times = load_fields(args.data, args.name)
    if args.stride > 1:
        fields = fields[::args.stride]
        times = times[::args.stride]
    T = fields.shape[0]
    triang = make_tri(points, tri)

    ch = {"vx": 0, "vy": 1, "p": 2}
    if args.channel == "speed":
        series = np.hypot(fields[:, :, 0], fields[:, :, 1])
        cmap, label = CMAPS["speed"], "$|v|$ [m/s]"
    else:
        series = fields[:, :, ch[args.channel]]
        cmap = CMAPS[args.channel]
        label = {"vx": "$v_x$ [m/s]", "vy": "$v_y$ [m/s]", "p": "$p$ [Pa]"}[args.channel]

    # fixed color scale across frames
    if cmap == "RdBu_r":
        vmax = np.percentile(np.abs(series), 99); vlim = (-vmax, vmax)
    else:
        vlim = (np.percentile(series, 1), np.percentile(series, 99))

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    tpc = ax.tripcolor(triang, series[0], shading="gouraud", cmap=cmap,
                       vmin=vlim[0], vmax=vlim[1])
    plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.02, label=label)
    ttl = ax.set_title("")

    def update(i):
        ax.clear()
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.tripcolor(triang, series[i], shading="gouraud", cmap=cmap,
                     vmin=vlim[0], vmax=vlim[1])
        ax.set_title(f"FOM {args.name}  {args.channel}  frame {i}/{T-1}  t={times[i]-times[0]:.3f}s",
                     fontsize=9)

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 / args.fps)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".mp4":
        anim.save(out, writer="ffmpeg", fps=args.fps, dpi=130)
    else:
        anim.save(out, writer="pillow", fps=args.fps, dpi=110)
    plt.close(fig)
    print(f"saved {out}  ({T} frames, channel={args.channel})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("snapshot", help="single-time multi-panel FOM field")
    sp.add_argument("--data", required=True, help="dir with fields_*.npy + mesh + times")
    sp.add_argument("--name", default="sim", help="sim name for fields_<name>.npy")
    sp.add_argument("--idx", type=int, default=-1, help="snapshot index (-1 = middle)")
    sp.add_argument("--out", required=True, help="output png path")
    sp.set_defaults(func=cmd_snapshot)

    vp = sub.add_parser("video", help="time animation of one FOM channel")
    vp.add_argument("--data", required=True)
    vp.add_argument("--name", default="sim")
    vp.add_argument("--channel", default="vx", choices=["vx", "vy", "p", "speed"])
    vp.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    vp.add_argument("--fps", type=int, default=15)
    vp.add_argument("--out", required=True, help="output .gif or .mp4")
    vp.set_defaults(func=cmd_video)

    cp = sub.add_parser("compare", help="FOM vs LDNet panels at one snapshot")
    cp.add_argument("--recon", required=True, help="dir from reconstruct_fields.py")
    cp.add_argument("--name", default="sim")
    cp.add_argument("--tri", required=True, help="mesh_triangles.npy")
    cp.add_argument("--latent", default="?", help="d_s label for the title")
    cp.add_argument("--idx", type=int, default=-1)
    cp.add_argument("--out", required=True)
    cp.set_defaults(func=cmd_compare)

    cv = sub.add_parser("compare-video", help="FOM vs LDNet animation of one channel")
    cv.add_argument("--recon", required=True)
    cv.add_argument("--name", default="sim")
    cv.add_argument("--tri", required=True)
    cv.add_argument("--channel", default="vx", choices=["vx", "vy", "p"])
    cv.add_argument("--stride", type=int, default=1)
    cv.add_argument("--fps", type=int, default=15)
    cv.add_argument("--out", required=True)
    cv.set_defaults(func=cmd_compare_video)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

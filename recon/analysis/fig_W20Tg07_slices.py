#!/usr/bin/env python3
"""Velocity + Cp slice grid: open-loop vs closed-loop, W20/Tg0.70.

4 time instants (t=0s, 1st C_L peak, 2nd C_L peak, t=1.00s) x 2 runs
(open loop / closed loop). For each of |U| and Cp: an "absolute" block,
immediately followed by a "delta relative to t=0" block (deadbanded to
white near zero -- so the t=0 column, and any near-steady column, reads
as blank without needing a text label).

The airfoil+flap outline is drawn as a THIN LINE ONLY (no fill) on every
panel, using the true hinge kinematics from
recon/cluster/cosim_driver_extract.py::compute_motion_tables (elastic axis
at x=0.40, flap hinge at x=0.779), with the deviation from the t=0 pose
amplified (EXAG_WING for the wing, EXAG_FLAP for the flap) so the motion is visible. Unfilled on purpose: the CFD
field is real everywhere, including the narrow flap-well gap, and must
never be painted over.

Reads recon_fields/viz_W20Tg0.70_{OL,CL}/ downloaded from the cluster
(mesh_points.npy, mesh_triangles.npy, field_times.npy, fields_<name>.npy,
structural_trajectory.csv). Field "times" are raw OpenFOAM case time
(checkpoint replay continues from the frozen baseline at t=3.000s), so the
logical replay time is field_time - CHECKPOINT_T0.

Style follows light/latex/AGENTS.md ("Linee guida per i plot"): serif +
cm mathtext, no in-figure title, 300 dpi.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
})

AN = Path(__file__).resolve().parent
RES = AN.parent / "recon_fields"
OUT = AN / "figs_W20Tg07"
OUT.mkdir(exist_ok=True)

CHECKPOINT_T0 = 3.000  # frozen baseline checkpoint time (cosim_main/checkpoint_W0_baseline)
U_INF = 80.0
Q_INF = 0.5 * U_INF ** 2  # p is kinematic (p/rho); farfield p_ref = 0

# Rigid-body kinematics (must match recon/cluster/cosim_driver_extract.py)
EA_X, EA_Y = 0.40, 0.0         # elastic axis (wing heave/pitch CoR)
HINGE_X, HINGE_Y = 0.779, 0.0  # flap hinge, initial mesh

# Motion amplification for the outline (and the co-moving grid). Wing rigid-body
# motion and flap deflection carry separate factors -- the flap angle is a
# RELATIVE degree of freedom, so it can be scaled independently while the flap
# still translates with the wing (its hinge rides along). Kept equal so a single
# amplification factor describes the whole figure: with different factors the
# caption must state both, or the visual wing/flap ratio misleads.
EXAG_WING = 5.0
EXAG_FLAP = 5.0

# house palette (light/tests/cs25_thesis_figs.py): open-loop grey, closed-loop blue
C_OPEN, C_CLOSED = "0.35", "#4477AA"

TIMES = [0.00, 0.39, 0.76, 1.00]
TIME_LABELS = ["t = 0 s", "t = 0.39 s", "t = 0.76 s", "t = 1.00 s"]

# 0-120 spans the data (panel max 130 m/s saturates in the "over" colour) and
# wastes none of the colormap. Widening to 160 to centre the 80 m/s freestream
# was tried and read WORSE: the useful variation is a narrow band around 80
# (median 77, p95 103) -- near-zero speeds occur only at the stagnation points --
# so a wider range compresses that band into fewer colour steps and the field
# flattens, while the top 19% of the scale is never reached.
VEL_LEVELS = np.linspace(0, 120, 13)      # step 10 m/s
CP_LEVELS = np.linspace(-1.2, 1.0, 23)    # step 0.1
DVEL_LEVELS = np.linspace(-10, 10, 21)    # step 1 m/s
DCP_LEVELS = np.linspace(-0.4, 0.4, 33)   # step 0.025

# Near-zero delta band painted pure white. Aligned to a level boundary so the
# white region ends on a contour line instead of cutting across mesh edges
# (a triangle-level mask produced visible stair-stepping).
DVEL_WHITE = 1.0
DCP_WHITE = 0.025

# Laplacian smoothing passes for the delta fields: the far-field grid is coarse
# and p/U arrive as cell->point averages, so raw difference contours come out
# chunky. Mild smoothing only; peaks move by well under one contour band.
DELTA_SMOOTH_ITERS = 6
DELTA_SMOOTH_W = 0.5

# The gust shifts pressure over the whole domain, so Delta Cp carries a diffuse
# offset that tints the entire panel and hides the structure near the body.
# Referencing each Delta Cp panel to its own far field removes it. Measured
# offsets are -0.072 / +0.055 / +0.036 with an interquartile spread of only
# 0.003-0.011, i.e. genuinely uniform, so one subtraction is meaningful.
# NOT applied to Delta |U|: there the far field is structured (IQR 0.85-1.31,
# larger than the offset itself), so a single number would misrepresent it.
DCP_FARFIELD_REF = True
DELTA_FARFIELD_R = 1.0  # chords from the body

# Symbols follow the thesis text: chapter1 defines the velocity vector as
# \mathbf{u} and its components as u_x, u_y (output table, chapter1.tex:224).
# NOTE: chapter3.tex and meansplit_thesis.tex still call the same components
# v_x, v_y -- that text is the outlier and should be migrated to u.
LBL_VEL = r"$|\mathbf{u}|$ [m/s]"
LBL_DVEL = r"$\Delta|\mathbf{u}|$ [m/s]"
LBL_CP = r"$C_p$ [-]"
LBL_DCP = r"$\Delta C_p$ [-]"

# Colormaps, all available as ParaView presets so the figure can be
# reproduced there: "Rainbow Desaturated", "Turbo", "Cool to Warm".
# Rainbow Desaturated has no matplotlib equivalent, so it is rebuilt below
# from the preset's own control points (ParaView ColorMaps.xml).
PARAVIEW_RAINBOW_DESATURATED = [
    (0.000, (0.278431, 0.278431, 0.858824)),
    (0.143, (0.000000, 0.000000, 0.360784)),
    (0.285, (0.000000, 1.000000, 1.000000)),
    (0.429, (0.000000, 0.501961, 0.000000)),
    (0.571, (1.000000, 1.000000, 0.000000)),
    (0.714, (1.000000, 0.380392, 0.000000)),
    (0.857, (0.419608, 0.000000, 0.000000)),
    (1.000, (0.878431, 0.301961, 0.301961)),
]
RAINBOW_DESATURATED = LinearSegmentedColormap.from_list(
    "rainbow_desaturated", PARAVIEW_RAINBOW_DESATURATED)
CMAP_VEL = "turbo"
CMAP_CP = RAINBOW_DESATURATED
CMAP_DVEL = "coolwarm"
CMAP_DCP = "coolwarm"

# Mesh-warp blend lengths [chords]: how far the exaggerated rigid-body motion
# is felt in the fluid grid before decaying to zero (mimics the OpenFOAM
# displacement motion solver, which is rigid at the wall and decays outward).
WARP_L_WING = 0.55
WARP_L_FLAP = 0.35

XLIM, YLIM = (-0.3, 1.7), (-0.5, 0.5)


# ─────────────────────────── data loading ────────────────────────────────

def load_case(name):
    d = RES / name
    pts = np.load(d / "mesh_points.npy")
    tri = np.load(d / "mesh_triangles.npy")
    times = np.load(d / "field_times.npy")
    fields = np.load(d / f"fields_{name}.npy")  # [T, N, 3] = Ux, Uy, p
    traj_t, traj_h, traj_a, traj_d, traj_w = [], [], [], [], []
    with open(d / "structural_trajectory.csv") as f:
        for row in csv.DictReader(f):
            traj_t.append(float(row["t"])); traj_h.append(float(row["h"]))
            traj_a.append(float(row["alpha"])); traj_d.append(float(row["delta"]))
            traj_w.append(float(row["W_gust"]))
    traj = dict(t=np.array(traj_t), h=np.array(traj_h),
               a=np.array(traj_a), d=np.array(traj_d), w=np.array(traj_w))
    return pts, tri, times, fields, traj


def nearest_snapshots(times, targets, t0=CHECKPOINT_T0):
    idxs = []
    for t in targets:
        i = int(np.argmin(np.abs(times - (t + t0))))
        idxs.append(i)
        print(f"    target t={t:.3f}s -> snapshot t={times[i]-t0:.4f}s "
              f"(|dt|={abs(times[i]-t0-t)*1e3:.1f} ms)")
    return idxs


def traj_at(traj, t):
    i = int(np.argmin(np.abs(traj["t"] - t)))
    return traj["h"][i], traj["a"][i], traj["d"][i]


# ─────────────────────────── outline geometry ─────────────────────────────

def boundary_loops(pts, tri):
    """Closed boundary-edge loops of the triangulation (crop box + holes)."""
    edges = {}
    for a, b, c in tri:
        for e in [(a, b), (b, c), (c, a)]:
            key = (int(min(e)), int(max(e)))
            edges[key] = edges.get(key, 0) + 1
    boundary = [e for e, cnt in edges.items() if cnt == 1]
    adj = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b); adj[b].append(a)

    visited, loops = set(), []
    for start in adj:
        if start in visited:
            continue
        comp, stack = set(), [start]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            for v in adj[u]:
                if v not in comp:
                    stack.append(v)
        visited |= comp
        loops.append(comp)

    def order_loop(comp):
        start = next(iter(comp))
        order, prev, cur = [start], None, start
        while True:
            nbrs = [n for n in adj[cur] if n in comp and n != prev]
            if not nbrs or nbrs[0] == start:
                break
            nxt = nbrs[0]
            order.append(nxt)
            prev, cur = cur, nxt
        return order

    return [order_loop(c) for c in loops]


def wing_flap_outlines(pts, tri):
    loops = boundary_loops(pts, tri)
    spans = [pts[l][:, 0].max() - pts[l][:, 0].min() for l in loops]
    holes = [l for l, s in zip(loops, spans) if s < 2.0]
    holes.sort(key=lambda l: pts[l][:, 0].mean())
    assert len(holes) == 2, f"expected wing+flap holes, found {len(holes)}"
    wing_loop, flap_loop = holes
    return pts[wing_loop].copy(), pts[flap_loop].copy()


def rotate(p, center, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (p - center) @ R.T + center


def pose_increment(case, t_now, exag_wing=None, exag_flap=None):
    """Amplified motion of the body BETWEEN t=0 and t_now.

    Anchored incrementally on the t=0 pose, not on absolute (h, alpha, delta):
    the stored grid is one frozen reference mesh (extract_fields.py saves
    mesh_points once, from the first snapshot), so the only self-consistent
    anchor is "zero increment at t=0". This makes t=0 coincide exactly and
    every later time move grid and outline by the same amount.

    Wing rigid-body motion and flap deflection carry independent factors --
    the flap angle is relative to the wing, so the flap still translates with
    the wing at exag_wing while its own rotation is scaled by exag_flap.
    """
    # resolved at call time, not bound as defaults, so callers (e.g. the
    # animation) can override the module-level factors after import
    if exag_wing is None:
        exag_wing = EXAG_WING
    if exag_flap is None:
        exag_flap = EXAG_FLAP
    h0, a0, d0 = case["t0_ref"]
    h_t, a_t, d_t = traj_at(case["traj"], t_now)
    d_alpha = exag_wing * (a_t - a0)                       # rad
    d_h = exag_wing * (h_t - h0)                           # m
    d_delta = -np.radians(exag_flap * (d_t - d0))          # rad, driver sign convention
    return d_alpha, d_h, d_delta


def apply_wing_motion(p, d_alpha, d_h):
    q = rotate(p, np.array([EA_X, EA_Y]), d_alpha)
    q[:, 1] += d_h
    return q


def apply_flap_motion(p, d_alpha, d_h, d_delta):
    q = apply_wing_motion(p, d_alpha, d_h)
    hinge = apply_wing_motion(np.array([[HINGE_X, HINGE_Y]]), d_alpha, d_h)[0]
    return rotate(q, hinge, d_delta)


def transform_outline(case, t_now):
    d_alpha, d_h, d_delta = pose_increment(case, t_now)
    wing = apply_wing_motion(case["wing0"].copy(), d_alpha, d_h)
    flap = apply_flap_motion(case["flap0"].copy(), d_alpha, d_h, d_delta)
    return wing, flap


def _dist_to_loop(pts, poly):
    """Min distance from each point to a closed polygon's edges."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    denom = np.maximum((ab ** 2).sum(-1), 1e-12)
    out = np.empty(len(pts))
    step = 2000  # chunked: (N, M, 2) would otherwise be a large temporary
    for i in range(0, len(pts), step):
        chunk = pts[i:i + step]
        ap = chunk[:, None, :] - a[None]
        t = np.clip((ap * ab[None]).sum(-1) / denom[None], 0.0, 1.0)
        proj = a[None] + t[..., None] * ab[None]
        out[i:i + step] = np.linalg.norm(chunk[:, None, :] - proj, axis=-1).min(1)
    return out


def _decay(d, L):
    """1 at the wall, smoothly 0 beyond L (smoothstep)."""
    u = np.clip(d / L, 0.0, 1.0)
    return 1.0 - (3 * u ** 2 - 2 * u ** 3)


def warped_points(case, t_now):
    """Grid coordinates carrying the same amplified motion as the outline.

    The stored field lives on a frozen reference grid whose hole cannot move,
    so drawing a displaced body over it leaves a ghost of the reference
    position. Warping the plotting coordinates by the same rigid-body motion
    (blended into the fluid, as the real mesh solver does) keeps hole and
    outline locked together. Field VALUES are untouched.
    """
    d_alpha, d_h, d_delta = pose_increment(case, t_now)
    if abs(d_alpha) < 1e-12 and abs(d_h) < 1e-12 and abs(d_delta) < 1e-12:
        return case["pts"]

    pts = case["pts"]
    d_wing, d_flap = case["d_wing"], case["d_flap"]

    # Rigid-body weight of the WHOLE body (wing + flap): the flap is rigidly
    # attached, so its nodes must carry the full wing motion even where they
    # are far from the wing surface -- measuring only d_wing would decay it
    # and leave the flap trailing edge a few mm behind the drawn outline.
    w_wing = _decay(np.minimum(d_wing, d_flap), WARP_L_WING)
    # flap rotation acts only on the flap side of the wing/flap medial line,
    # so the wing surface is never dragged along by the flap deflection
    w_flap = _decay(d_flap, WARP_L_FLAP) * (d_flap < d_wing)

    disp = np.zeros_like(pts)
    disp += w_wing[:, None] * (apply_wing_motion(pts.copy(), d_alpha, d_h) - pts)
    disp += w_flap[:, None] * (
        apply_flap_motion(pts.copy(), d_alpha, d_h, d_delta)
        - apply_wing_motion(pts.copy(), d_alpha, d_h))
    return pts + disp


def draw_outline(ax, case, t_now):
    """Body at the amplified pose. Filled white to match the (co-moving,
    warped) grid hole exactly -- see warped_points for why the grid moves."""
    wing, flap = transform_outline(case, t_now)
    for poly in (wing, flap):
        ax.fill(poly[:, 0], poly[:, 1], facecolor="white", edgecolor="black",
               linewidth=0.9, zorder=5)


def style_ax(ax):
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def inverted_mask(pts0, wpts, tri):
    """Triangles whose orientation flips under the warp (tangled cells in the
    high-shear wing/flap gap): a handful out of ~18k, masked so they cannot
    render as specks."""
    def cross(p):
        a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    return np.sign(cross(wpts)) != np.sign(cross(pts0))


def make_triang(pts, tri, base_mask=None):
    triang = Triangulation(pts[:, 0], pts[:, 1], tri)
    if base_mask is not None:
        triang.set_mask(base_mask)
    return triang


def smooth_on_mesh(val, tri, n_iter=DELTA_SMOOTH_ITERS, w=DELTA_SMOOTH_W):
    """Laplacian (neighbour-average) smoothing over the triangulation."""
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    i, j = e[:, 0], e[:, 1]
    n = len(val)
    deg = np.zeros(n)
    np.add.at(deg, i, 1.0); np.add.at(deg, j, 1.0)
    deg = np.maximum(deg, 1.0)
    out = val.astype(float).copy()
    for _ in range(n_iter):
        s = np.zeros(n)
        np.add.at(s, i, out[j]); np.add.at(s, j, out[i])
        out = (1.0 - w) * out + w * (s / deg)
    return out


def white_centre_cmap(base, levels, white_halfwidth):
    """Banded colormap whose bands within +/-white_halfwidth are pure white.

    Replaces the previous triangle-level deadband mask: blanking whole cells
    cut the white region along mesh edges and produced stair-stepped blocks,
    whereas here the white region ends exactly on a contour level."""
    cmap = plt.get_cmap(base)
    mids = 0.5 * (levels[:-1] + levels[1:])
    span = levels[-1] - levels[0]
    colors = [(1.0, 1.0, 1.0, 1.0) if abs(m) < white_halfwidth
              else cmap((m - levels[0]) / span) for m in mids]
    listed = ListedColormap(colors)
    listed.set_under(cmap(0.0)); listed.set_over(cmap(1.0))
    return listed, BoundaryNorm(levels, len(colors))


def hide_ax(ax):
    ax.set_visible(False)


# ─────────────────────────────── main ──────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quantities", default="cp",
                    help="comma-separated blocks to draw, in order: cp, vel "
                         "(default: cp only)")
    ap.add_argument("--out", default=None,
                    help="output filename (default derived from --quantities)")
    args = ap.parse_args()
    keys = [q.strip() for q in args.quantities.split(",") if q.strip()]
    unknown = [q for q in keys if q not in ("cp", "vel")]
    if unknown:
        raise SystemExit(f"unknown quantity {unknown}; choose from cp, vel")

    case_defs = [("Open loop", "viz_W20Tg0.70_OL", C_OPEN),
                ("Closed loop", "viz_W20Tg0.70_CL", C_CLOSED)]
    cases = {}
    for label, name, color in case_defs:
        print(f"=== {label} ({name}) ===")
        pts, tri, times, fields, traj = load_case(name)
        idxs = nearest_snapshots(times, TIMES)
        wing0, flap0 = wing_flap_outlines(pts, tri)
        cases[label] = dict(
            pts=pts, tri=tri, color=color,
            snaps=[fields[i] for i in idxs], traj=traj,
            wing0=wing0, flap0=flap0, t0_ref=traj_at(traj, 0.0),
            d_wing=_dist_to_loop(pts, wing0), d_flap=_dist_to_loop(pts, flap0))
        cases[label]["d_body"] = np.minimum(cases[label]["d_wing"],
                                            cases[label]["d_flap"])

    labels = [c[0] for c in case_defs]
    n_cols = len(TIMES)

    # Each quantity block is 4 rows -- OL-abs, OL-delta, CL-abs, CL-delta, so an
    # absolute row is immediately followed by its own delta. Blocks are stacked
    # with a spacer row between them.
    SPACER = 0.32
    n_blocks = len(keys)
    height_ratios, block_rows, r = [], [], 0
    for b in range(n_blocks):
        if b:
            height_ratios.append(SPACER); r += 1
        block_rows.append([r, r + 1, r + 2, r + 3])
        height_ratios += [1, 1, 1, 1]
        r += 4

    fig_h = 7.9 * n_blocks + SPACER * 1.7 * (n_blocks - 1)
    fig = plt.figure(figsize=(15, fig_h))
    gs = fig.add_gridspec(
        len(height_ratios), n_cols + 1, width_ratios=[1, 1, 1, 1, 0.06],
        height_ratios=height_ratios,
        hspace=0.08, wspace=0.05, left=0.05, right=0.925,
        top=1 - 0.30 / fig_h, bottom=0.23 / fig_h)
    ROW_LABELS = ["Open loop", "$\\Delta$ to t = 0 s",
                  "Closed loop", "$\\Delta$ to t = 0 s"]

    def quantity_block(rows, key):
        for ri, row in enumerate(rows):
            label = labels[ri // 2]
            case = cases[label]
            is_delta = ri % 2 == 1
            ref = case["snaps"][0]
            mesh = None
            for col, t in enumerate(TIMES):
                ax = fig.add_subplot(gs[row, col])
                # delta at the reference time is identically zero -- no panel
                if is_delta and col == 0:
                    hide_ax(ax)
                    continue

                field = case["snaps"][col]
                if key == "vel":
                    val = np.sqrt(field[:, 0] ** 2 + field[:, 1] ** 2)
                    if is_delta:
                        val = val - np.sqrt(ref[:, 0] ** 2 + ref[:, 1] ** 2)
                else:
                    val = field[:, 2] / Q_INF
                    if is_delta:
                        val = val - ref[:, 2] / Q_INF

                wpts = warped_points(case, t)
                inv = inverted_mask(case["pts"], wpts, case["tri"])
                triang = make_triang(wpts, case["tri"], inv)
                if not is_delta:
                    levels = VEL_LEVELS if key == "vel" else CP_LEVELS
                    cmap = CMAP_VEL if key == "vel" else CMAP_CP
                    mesh = ax.tricontourf(triang, val, levels=levels,
                                          cmap=cmap, extend="both")
                else:
                    levels, white, base = ((DVEL_LEVELS, DVEL_WHITE, CMAP_DVEL) if key == "vel"
                                           else (DCP_LEVELS, DCP_WHITE, CMAP_DCP))
                    if key == "cp" and DCP_FARFIELD_REF:
                        far = case["d_body"] > DELTA_FARFIELD_R
                        if far.any():
                            val = val - np.median(val[far])
                    val = smooth_on_mesh(val, case["tri"])
                    cmap, norm = white_centre_cmap(base, levels, white)
                    mesh = ax.tricontourf(triang, val, levels=levels,
                                          cmap=cmap, norm=norm, extend="both")

                draw_outline(ax, case, t)
                style_ax(ax)
                if row == rows[0]:
                    ax.set_title(TIME_LABELS[col], fontsize=9.5)
                if col == (1 if is_delta else 0):  # first visible panel of the row
                    ax.set_ylabel(ROW_LABELS[ri], fontsize=9.5, fontweight="bold",
                                 color=case["color"])
            cax = fig.add_subplot(gs[row, n_cols])
            if key == "vel":
                cbar_label = LBL_DVEL if is_delta else LBL_VEL
            else:
                cbar_label = LBL_DCP if is_delta else LBL_CP
            fig.colorbar(mesh, cax=cax, label=cbar_label, extend="both")

    for rows, key in zip(block_rows, keys):
        quantity_block(rows, key)

    p = OUT / (args.out or
               f"fig_W20Tg07_OLvsCL_{'_'.join(keys)}.png")
    fig.savefig(p)
    print(f"saved {p}")


if __name__ == "__main__":
    main()

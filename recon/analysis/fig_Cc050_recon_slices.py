#!/usr/bin/env python3
"""Field-reconstruction slice grid on the LOW-SEPARATION case Cc_050.

Same figure format as fig_W20Tg07_slices.py (that file is the authoritative
spec for the layout and for the frozen-grid warp); the content differs in what
the two "runs" are, and this figure carries ABSOLUTE FIELDS ONLY -- no delta
rows:

  run A = CFD reference (FOM)      run B = LDNet reconstruction (ROM)

Blocks are u_x then u_y, each with a CFD row and an LDNet row, so the
reconstruction is read by comparing the two rows column by column. Colour
scale, step and view window follow the reference so the two figures can be
read side by side; see UX_LEVELS/UY_LEVELS for how the reference's "undisturbed
flow at mid-colormap" rule carries over to a component that is ~0 far away.

Case choice: Cc_050 is the mildest of the Cc family on all three excitation
axes (gust 32.1 m/s vs 37.6 for Cc_060, flap 5.27 deg vs 6.61, heave amplitude
half), and its peak near-flap reversed-flow fraction is 0.16% against Cc_060's
31.48% -- i.e. effectively attached throughout, which is the regime this figure
is meant to show. It is a VALIDATION trajectory, so it is held out of training.

Reads recon/results/ms_coral_o10_s0_rom_cc050/ (fom/rom/points + the sim's
structural_trajectory.csv) and recon/results/mesh_triangles.npy. The ROM there
was produced by reconstruct_fields.py from the champion checkpoint
(recon/models/coral_o10_s0_synced/latent_1, mean-split + CORAL omega0=10,
d_s=1) on recon/data/FIELDS_Cc050val.h5 -- no new CFD.

Style follows light/latex/AGENTS.md: serif + cm mathtext, no in-figure title,
300 dpi. The description belongs in the LaTeX caption.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

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
RES = AN.parent / "results"
CASE = RES / "ms_coral_o10_s0_rom_cc050"
OUT = AN / "figs_Cc050"
OUT.mkdir(exist_ok=True)

SIM = "sim_Cc_050"

# Rigid-body kinematics (must match recon/cluster/cosim_driver_extract.py)
EA_X, EA_Y = 0.40, 0.0         # elastic axis (wing heave/pitch CoR)
HINGE_X, HINGE_Y = 0.779, 0.0  # flap hinge, initial mesh

# Motion amplification for the outline and the co-moving grid. UNEQUAL here, on
# purpose, and the caption must say so: the wing's elastic response is
# millimetric (heave amplitude 5.8e-3 c, pitch 0.2 deg) and invisible unaided,
# while the flap is an actuated surface already deflecting 5.3 deg. This case
# was chosen for its SMALL flap motion, so amplifying the flap would contradict
# the very point of the figure -- the flap is drawn at its true deflection.
EXAG_WING = 5.0
EXAG_FLAP = 1.0

# house palette (light/tests/cs25_thesis_figs.py): reference grey, model blue
C_FOM, C_ROM = "0.35", "#4477AA"

# Snapshot indices into the 150-frame window (dt = 20.13 ms, t = index * dt):
# quiescent reference, gust peak (W_gust = 32.1 m/s), peak flap deflection
# (-5.26 deg), and the settled recovery.
COLS = [0, 28, 40, 75]
COL_NOTES = ["quiescent", "gust peak", "peak flap", "recovery"]

U_INF = 80.0

# Banded (stepped) contour levels, not smooth shading.
#
# u_x takes the reference's window unchanged (fig_W20Tg07_slices.py): top of
# scale at 2*U_inf so the undisturbed flow lands mid-colormap instead of
# crowding the top third.
#
# u_y does NOT inherit that window: its undisturbed value is 0, not U_inf, so
# it keeps its own asymmetric range at a finer 5 m/s step (the quantity spans
# about a third as much as u_x).
#
# The u_y top is carried to +120 rather than the +60 the bulk of the field
# needs, and that choice is about the LEADING-EDGE STAGNATION PEAK. There u_y
# reaches 153.6 m/s in the CFD against 173.5 in the reconstruction: a real
# 20 m/s overshoot by the model. At a +60 top BOTH saturate (13.9% and 11.7%
# of the nodes within 0.15c of the leading edge), so the panels show two solid
# patches of different size and the eye reads the difference as much larger
# than it is. At +120 only 1.7% and 2.6% saturate -- the peak is resolved
# instead of clipped, and the 20 m/s gap reads at its true proportion of the
# scale. This is the LESS clipped option, not a cosmetic one, but the caption
# must still state that the stagnation peak is off the bulk-field scale.
UX_LEVELS = np.linspace(0.0, 2 * U_INF, 17)        # step 10 m/s, as reference
UY_LEVELS = np.linspace(-40.0, 120.0, 33)          # step 5 m/s, u_y's own range

# Velocity-magnitude figure: the reference's own quantity and window, unchanged.
UMAG_LEVELS = np.linspace(0.0, 2 * U_INF, 17)      # step 10 m/s
LBL_UMAG = r"$|\mathbf{u}|$ [m/s]"

# Symbols follow the thesis text (chapter1.tex:224 defines u_x, u_y); see the
# note in fig_W20Tg07_slices.py -- chapter3's v_x/v_y is the outlier.
LBL_UX = r"$u_x$ [m/s]"
LBL_UY = r"$u_y$ [m/s]"

# Colormap, a ParaView preset so the figure is reproducible there: "Turbo".
CMAP_VEL = "turbo"

# Mesh-warp blend lengths [chords]: how far the amplified rigid-body motion is
# felt in the fluid grid before decaying to zero (mimics the OpenFOAM
# displacement solver, rigid at the wall and decaying outward).
WARP_L_WING = 0.55
WARP_L_FLAP = 0.35

XLIM, YLIM = (-0.3, 1.7), (-0.5, 0.5)
ZOOM_XLIM, ZOOM_YLIM = (0.62, 1.12), (-0.20, 0.10)   # warp verification crop


# ─────────────────────────── data loading ────────────────────────────────

def load_case():
    fom = np.load(CASE / f"fom_{SIM}.npy").astype(np.float64)
    rom = np.load(CASE / f"rom_{SIM}.npy").astype(np.float64)
    pts = np.load(CASE / "rom_points.npy").astype(np.float64)
    times = np.load(CASE / "rom_times.npy").astype(np.float64)
    tt, hh, aa, dd = [], [], [], []
    with open(CASE / "structural_trajectory.csv") as fh:
        for row in csv.DictReader(fh):
            tt.append(float(row["t"])); hh.append(float(row["h"]))
            aa.append(float(row["alpha"])); dd.append(float(row["delta"]))
    traj = dict(t=np.array(tt), h=np.array(hh), a=np.array(aa), d=np.array(dd))
    tri = np.load(RES / "mesh_triangles.npy")
    return fom, rom, pts, times, traj, tri


def traj_at(traj, t):
    i = int(np.argmin(np.abs(traj["t"] - t)))
    return traj["h"][i], traj["a"][i], traj["d"][i]


# ─────────────────────────── outline geometry ─────────────────────────────

def boundary_loops(pts, tri):
    """Closed boundary-edge loops of the triangulation (crop box + holes),
    returned as ORDERED index lists. Ordered indices (not coordinates) are what
    the warp check needs: an unordered comparison silently pairs mismatched
    vertices and reports a false gap."""
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


def wing_flap_indices(pts, tri):
    loops = boundary_loops(pts, tri)
    spans = [pts[l][:, 0].max() - pts[l][:, 0].min() for l in loops]
    holes = [l for l, s in zip(loops, spans) if s < 2.0]
    holes.sort(key=lambda l: pts[l][:, 0].mean())
    assert len(holes) == 2, f"expected wing+flap holes, found {len(holes)}"
    return np.array(holes[0]), np.array(holes[1])


def rotate(p, center, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (p - center) @ R.T + center


def pose_increment(case, t_now):
    """Amplified motion of the body BETWEEN the reference time and t_now.

    Anchored incrementally on the reference-time pose, not on absolute
    (h, alpha, delta): the stored grid is ONE frozen reference mesh
    (extract_fields.py saves mesh_points once, from the first snapshot), so the
    only self-consistent anchor is "zero increment at t_ref". This makes the
    reference column coincide exactly and every later time move grid and
    outline by the same amount.
    """
    h0, a0, d0 = case["t_ref_pose"]
    h_t, a_t, d_t = traj_at(case["traj"], t_now)
    d_alpha = EXAG_WING * (a_t - a0)                     # rad
    d_h = EXAG_WING * (h_t - h0)                         # m
    d_delta = -np.radians(EXAG_FLAP * (d_t - d0))        # rad, driver sign
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
    so drawing a displaced body over it would leave a ghost of the reference
    position and send the wake off at the wrong angle. Warping the plotting
    coordinates by the same rigid-body motion (blended into the fluid as the
    real mesh solver does) keeps hole and outline locked. Field VALUES are
    never touched.
    """
    d_alpha, d_h, d_delta = pose_increment(case, t_now)
    if abs(d_alpha) < 1e-12 and abs(d_h) < 1e-12 and abs(d_delta) < 1e-12:
        return case["pts"]

    pts = case["pts"]
    d_wing, d_flap = case["d_wing"], case["d_flap"]

    # Rigid-body weight of the WHOLE body (wing + flap): the flap is rigidly
    # attached, so its nodes must carry the full wing motion even where they
    # are far from the wing surface -- measuring only d_wing would decay it and
    # leave the flap trailing edge trailing behind the drawn outline.
    w_wing = _decay(np.minimum(d_wing, d_flap), WARP_L_WING)
    # flap rotation acts only on the flap side of the wing/flap medial line, so
    # the wing surface is never dragged along by the flap deflection
    w_flap = _decay(d_flap, WARP_L_FLAP) * (d_flap < d_wing)

    disp = np.zeros_like(pts)
    disp += w_wing[:, None] * (apply_wing_motion(pts.copy(), d_alpha, d_h) - pts)
    disp += w_flap[:, None] * (
        apply_flap_motion(pts.copy(), d_alpha, d_h, d_delta)
        - apply_wing_motion(pts.copy(), d_alpha, d_h))
    return pts + disp


def draw_outline(ax, case, t_now):
    """Body at the amplified pose. Filled white to match the co-moving warped
    grid hole exactly -- see warped_points for why the grid moves."""
    wing, flap = transform_outline(case, t_now)
    for poly in (wing, flap):
        ax.fill(poly[:, 0], poly[:, 1], facecolor="white", edgecolor="black",
                linewidth=0.9, zorder=5)


def style_ax(ax, xlim=XLIM, ylim=YLIM):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def inverted_mask(pts0, wpts, tri):
    """Triangles whose orientation flips under the warp (tangled cells in the
    high-shear wing/flap gap): masked so they cannot render as specks."""
    def cross(p):
        a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    return np.sign(cross(wpts)) != np.sign(cross(pts0))


# ─────────────────────────── verification ─────────────────────────────────

def _verify_warp(case, times):
    """The warped grid hole must coincide with the drawn outline: compare
    ORDERED boundary indices against the transformed outline, at every column."""
    print("  warp check (max |warped[hole] - outline|):")
    ok = True
    for c in COLS:
        t = times[c]
        w = warped_points(case, t)
        wing, flap = transform_outline(case, t)
        e_w = np.abs(w[case["wing_idx"]] - wing).max()
        e_f = np.abs(w[case["flap_idx"]] - flap).max()
        inv = int(inverted_mask(case["pts"], w, case["tri"]).sum())
        tag = ""
        if c == COLS[0] and max(e_w, e_f) != 0.0:
            tag = "   <-- FAIL (must be exactly 0 at the reference time)"
            ok = False
        elif max(e_w, e_f) >= 1e-7:
            tag = "   <-- FAIL"
            ok = False
        print(f"    t={t:6.3f}s  wing {e_w:.2e}  flap {e_f:.2e}  "
              f"inverted triangles {inv:4d} / {len(case['tri'])}{tag}")
    assert ok, "warp verification failed"


def _zoom_crop(case, times):
    """Before/after crop where the body moves most. Numeric agreement alone is
    not proof the figure reads right -- this is the panel to look at."""
    t = times[COLS[2]]                                  # peak flap deflection
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0), constrained_layout=True)
    for ax, warp in zip(axs, [False, True]):
        pts = warped_points(case, t) if warp else case["pts"]
        mask = inverted_mask(case["pts"], pts, case["tri"]) if warp else None
        triang = Triangulation(pts[:, 0], pts[:, 1], case["tri"])
        if mask is not None:
            triang.set_mask(mask)
        ax.triplot(triang, color="0.75", linewidth=0.25)
        wing, flap = transform_outline(case, t)
        for poly in (wing, flap):
            ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
                    color="crimson", linewidth=1.2)
        ax.set_title("frozen grid + drawn outline" if not warp
                     else "co-moving warped grid", fontsize=9.5)
        style_ax(ax, ZOOM_XLIM, ZOOM_YLIM)
    p = OUT / "fig_Cc050_warp_check.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  saved warp crop {p}")


# ─────────────────────────────── main ──────────────────────────────────

def main():
    fom, rom, pts, times, traj, tri = load_case()
    wing_idx, flap_idx = wing_flap_indices(pts, tri)
    case = dict(
        pts=pts, tri=tri, traj=traj,
        wing_idx=wing_idx, flap_idx=flap_idx,
        wing0=pts[wing_idx].copy(), flap0=pts[flap_idx].copy(),
        t_ref_pose=traj_at(traj, times[COLS[0]]),
        d_wing=_dist_to_loop(pts, pts[wing_idx]),
        d_flap=_dist_to_loop(pts, pts[flap_idx]))
    case["d_body"] = np.minimum(case["d_wing"], case["d_flap"])

    print(f"=== {SIM} (validation, held out) ===")
    print(f"  grid {len(pts)} nodes, {len(tri)} triangles; "
          f"wing hole {len(wing_idx)}, flap hole {len(flap_idx)}")
    for c, note in zip(COLS, COL_NOTES):
        h, a, d = traj_at(traj, times[c])
        print(f"  col t={times[c]:6.3f}s ({note:9s}) h={h:+.5f} m  "
              f"alpha={np.degrees(a):+.3f} deg  delta={d:+.3f} deg")
    _verify_warp(case, times)
    _zoom_crop(case, times)

    for k, nm in [(0, "v_x"), (1, "v_y")]:
        rng = fom[:, :, k].max() - fom[:, :, k].min()
        nr = np.sqrt(((rom[:, :, k] - fom[:, :, k]) ** 2).mean()) / rng
        print(f"  {nm} NRMSE over the whole window: {nr:.3e}")

    time_labels = [f"$t$ = {times[c]:.2f} s" for c in COLS]
    n_cols = len(COLS)
    ROW_LABELS = ["CFD", "LDNet"]
    ROW_COLORS = [C_FOM, C_ROM]

    def quantity_block(fig, gs, rows, getter, levels, cbar_label):
        """One CFD row + one LDNet row of the same quantity, columns = times."""
        for ri, row in enumerate(rows):
            src = fom if ri == 0 else rom
            mesh = None
            for col, c in enumerate(COLS):
                ax = fig.add_subplot(gs[row, col])
                t = times[c]
                wpts = warped_points(case, t)
                triang = Triangulation(wpts[:, 0], wpts[:, 1], tri)
                triang.set_mask(inverted_mask(pts, wpts, tri))
                mesh = ax.tricontourf(triang, getter(src, c), levels=levels,
                                      cmap=CMAP_VEL, extend="both")
                draw_outline(ax, case, t)
                style_ax(ax)
                if row == rows[0]:
                    ax.set_title(time_labels[col], fontsize=9.5)
                if col == 0:
                    ax.set_ylabel(ROW_LABELS[ri], fontsize=9.5, fontweight="bold",
                                  color=ROW_COLORS[ri])
            cax = fig.add_subplot(gs[row, n_cols])
            fig.colorbar(mesh, cax=cax, label=cbar_label, extend="both")

    def component(k):
        return lambda src, c: src[c, :, k]

    def magnitude(src, c):
        return np.sqrt(src[c, :, 0] ** 2 + src[c, :, 1] ** 2)

    # ---- figure 1: the two components, one block each -----------------------
    fig = plt.figure(figsize=(15, 7.9))
    gs = fig.add_gridspec(
        5, n_cols + 1, width_ratios=[1, 1, 1, 1, 0.06],
        height_ratios=[1, 1, 0.32, 1, 1],
        hspace=0.08, wspace=0.05, left=0.05, right=0.925, top=0.97, bottom=0.02)
    quantity_block(fig, gs, [0, 1], component(0), UX_LEVELS, LBL_UX)
    quantity_block(fig, gs, [3, 4], component(1), UY_LEVELS, LBL_UY)
    p = OUT / "fig_Cc050_recon_slices.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"saved {p}")

    # ---- figure 2: velocity magnitude only, the reference's own quantity ----
    fig = plt.figure(figsize=(15, 4.0))
    gs = fig.add_gridspec(
        2, n_cols + 1, width_ratios=[1, 1, 1, 1, 0.06],
        height_ratios=[1, 1],
        hspace=0.08, wspace=0.05, left=0.05, right=0.925, top=0.94, bottom=0.03)
    quantity_block(fig, gs, [0, 1], magnitude, UMAG_LEVELS, LBL_UMAG)
    p = OUT / "fig_Cc050_recon_umag.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"saved {p}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""H-METRIC: decompose the LDNet field-reconstruction error (cluster login node, numpy only).

Reads the already-computed ROM/FOM reconstructions in
  /work/u10677113/NACA2312/recon/results/div_rom_{a,cc}_l{1,5,10}/
and decomposes the reported "combined NRMSE" per field (vx, vy, p), per region
(near-airfoil / near-wake / far-field / airfoil surface), and per time window
(excitation / peak / decay / quiescent), plus a linear-FE vorticity error and the
airfoil-surface pressure error (loads proxy).

Convention matched EXACTLY to reconstruct_fields.py / viz_fields.py:
    combined NRMSE = sqrt(mean((rom - fom)**2)) / (fom.max() - fom.min())
over the full [T, Npts, 3] array in PHYSICAL units (vx, vy [m/s] and p [Pa] mixed,
single global min/max across all three channels).

Outputs (small CSVs) -> /work/u10677113/NACA2312/recon/analysis_hmetric/
    error_decomposition.csv   tidy: sim, d_s, field, region, time_window, nrmse, n_nodes, ...
    mse_shares.csv            fraction of the combined SSE from each field / region
    timeseries_<sim>.csv      per-time error curves + gust W(t) + flap delta(t)
    region_labels.npy         int8 per-node region id (0 near, 1 wake, 2 far)
    airfoil_nodes.npy         indices of airfoil-surface nodes
    geometry.txt              chord, region definitions, node counts
"""
import csv
import numpy as np
from pathlib import Path

BASE = Path("/work/u10677113/NACA2312")
RES = BASE / "recon" / "results"
OUT = BASE / "recon" / "analysis_hmetric"
OUT.mkdir(exist_ok=True)

CASES = {
    "sim_A_025": dict(dirpat="div_rom_a_l{ds}", dataset="sim_A_025_test"),
    "sim_Cc_060": dict(dirpat="div_rom_cc_l{ds}", dataset="sim_Cc_060_test"),
}
DS_LIST = [1, 5, 10]
FIELDS = ["vx", "vy", "p"]

# ----------------------------------------------------------------------------
# Geometry: fixed reference grid shared by every recon dir
# ----------------------------------------------------------------------------
points = np.load(RES / "div_rom_a_l1" / "rom_points.npy").astype(np.float64)
tri = np.load(BASE / "recon_fields" / "sim_A_025_test_T3p0" / "mesh_triangles.npy")
N = points.shape[0]
assert tri.max() < N, "triangulation does not match rom_points"

# consistency across all six dirs
for sim, case in CASES.items():
    for ds in DS_LIST:
        p2 = np.load(RES / case["dirpat"].format(ds=ds) / "rom_points.npy")
        assert p2.shape == points.shape and np.allclose(p2, points), \
            f"points mismatch in {case['dirpat'].format(ds=ds)}"

# boundary edges = edges used by exactly one triangle
edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
uniq, cnt = np.unique(edges, axis=0, return_counts=True)
bedges = uniq[cnt == 1]
bnodes = np.unique(bedges)

x, y = points[:, 0], points[:, 1]
xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()

# split boundary into connected loops: the loop with the largest extent is the
# ragged outer crop boundary; the interior loops are the airfoil (main + flap)
from collections import defaultdict
adj = defaultdict(list)
for a_, b_ in bedges:
    adj[int(a_)].append(int(b_))
    adj[int(b_)].append(int(a_))
unvisited = set(int(v) for v in bnodes)
loops = []
while unvisited:
    start = unvisited.pop()
    comp, stack = [start], [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in unvisited:
                unvisited.remove(v)
                stack.append(v)
                comp.append(v)
    loops.append(np.array(comp))
extent = [max(np.ptp(points[lp, 0]), np.ptp(points[lp, 1])) for lp in loops]
outer_i = int(np.argmax(extent))
loop_info = []
for li, lp in enumerate(loops):
    tag = "OUTER(crop)" if li == outer_i else "airfoil element"
    loop_info.append(f"loop {li}: {len(lp)} nodes, x [{points[lp,0].min():.4f},"
                     f"{points[lp,0].max():.4f}] y [{points[lp,1].min():.4f},"
                     f"{points[lp,1].max():.4f}]  -> {tag}")
air = np.concatenate([lp for li, lp in enumerate(loops) if li != outer_i])
ax, ay = points[air, 0], points[air, 1]
le_x, te_x = ax.min(), ax.max()
chord = te_x - le_x
te_y = ay[np.argmax(ax)]

# per-node distance to nearest airfoil-surface node
dist = np.sqrt(((points[:, None, :] - points[air][None, :, :]) ** 2).sum(-1)).min(1)

# regions (mutually exclusive, cover all nodes)
near = dist < 0.5 * chord
wake = (~near) & (x > te_x) & (x < te_x + 3.0 * chord) & (np.abs(y - te_y) < 0.75 * chord)
far = ~near & ~wake
region_label = np.full(N, 2, dtype=np.int8)
region_label[near] = 0
region_label[wake] = 1
np.save(OUT / "region_labels.npy", region_label)
np.save(OUT / "airfoil_nodes.npy", air)

REGIONS = {"all": np.ones(N, bool), "near": near, "wake": wake, "far": far,
           "surface": np.isin(np.arange(N), air)}

geo_lines = loop_info + [
    f"nodes {N}, triangles {len(tri)}, boundary nodes {len(bnodes)}, airfoil nodes {len(air)}",
    f"domain x [{xmin:.3f},{xmax:.3f}] y [{ymin:.3f},{ymax:.3f}]",
    f"airfoil x [{le_x:.4f},{te_x:.4f}] y [{ay.min():.4f},{ay.max():.4f}]  chord={chord:.4f} m",
    "region defs: near = dist(surface) < 0.5c;"
    " wake = ~near & TE_x < x < TE_x+3c & |y-TE_y| < 0.75c; far = rest",
    f"node counts: near {near.sum()} ({100*near.mean():.1f}%), wake {wake.sum()}"
    f" ({100*wake.mean():.1f}%), far {far.sum()} ({100*far.mean():.1f}%), surface {len(air)}",
]
(OUT / "geometry.txt").write_text("\n".join(geo_lines) + "\n")
print("\n".join(geo_lines))

# triangle shape-function coefficients for linear-FE vorticity
i0, i1, i2 = tri[:, 0], tri[:, 1], tri[:, 2]
x0, y0 = points[i0].T
x1, y1 = points[i1].T
x2, y2 = points[i2].T
twoA = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)     # signed 2*area
b = np.stack([y1 - y2, y2 - y0, y0 - y1], 1) / twoA[:, None]   # d/dx weights
c = np.stack([x2 - x1, x0 - x2, x1 - x0], 1) / twoA[:, None]   # d/dy weights
cent = (points[i0] + points[i1] + points[i2]) / 3.0
tri_near = np.sqrt(((cent[:, None, :] - points[air][None, :, :]) ** 2).sum(-1)).min(1) < 0.5 * chord
tri_wake = (~tri_near) & (cent[:, 0] > te_x) & (cent[:, 0] < te_x + 3 * chord) & \
           (np.abs(cent[:, 1] - te_y) < 0.75 * chord)
tri_far = ~tri_near & ~tri_wake
TRI_REGIONS = {"all": np.ones(len(tri), bool), "near": tri_near, "wake": tri_wake, "far": tri_far}


def vorticity(f):
    """f [T,N,3] -> per-triangle omega_z [T,Ntri] via linear shape functions."""
    vx_n, vy_n = f[:, :, 0], f[:, :, 1]
    dvy_dx = (vy_n[:, i0] * b[:, 0] + vy_n[:, i1] * b[:, 1] + vy_n[:, i2] * b[:, 2])
    dvx_dy = (vx_n[:, i0] * c[:, 0] + vx_n[:, i1] * c[:, 1] + vx_n[:, i2] * c[:, 2])
    return dvy_dx - dvx_dy


# ----------------------------------------------------------------------------
# Input signals -> time windows
# ----------------------------------------------------------------------------
def load_signals(dataset, times_rel):
    traj = np.genfromtxt(BASE / "dataset_v5" / dataset / "structural_trajectory.csv",
                         delimiter=",", names=True)
    t_rel = traj["t"] - traj["t"][0]
    W = np.interp(times_rel, t_rel, traj["W_gust"])
    delta = np.interp(times_rel, t_rel, traj["delta"])
    return W, delta


def windows(times_rel, W, delta):
    exc = np.abs(W) >= 0.1 * np.abs(W).max()
    peak = np.abs(W) >= 0.5 * np.abs(W).max()
    if np.abs(delta).max() > 1e-9:
        exc |= np.abs(delta) >= 0.1 * np.abs(delta).max()
        peak |= np.abs(delta) >= 0.5 * np.abs(delta).max()
    t_exc_end = times_rel[exc].max()
    decay = ~exc & (times_rel <= t_exc_end + 1.0)
    quiet = ~exc & ~decay
    return {"all": np.ones(len(times_rel), bool), "excitation": exc, "peak": peak,
            "decay": decay, "quiescent": quiet}


# ----------------------------------------------------------------------------
# Main decomposition
# ----------------------------------------------------------------------------
rows = []       # error_decomposition.csv
shares = []     # mse_shares.csv

for sim, case in CASES.items():
    d0 = RES / case["dirpat"].format(ds=DS_LIST[0])
    times = np.load(d0 / "rom_times.npy").astype(np.float64)
    times_rel = times - times[0]
    T = len(times)
    W, delta = load_signals(case["dataset"], times_rel)
    WIN = windows(times_rel, W, delta)
    print(f"\n=== {sim}: T={T}, t_rel [0,{times_rel[-1]:.2f}]s, "
          f"max|W|={np.abs(W).max():.2f} m/s, max|delta|={np.abs(delta).max():.3f}")
    for wname, m in WIN.items():
        print(f"  window {wname:10s}: {m.sum()} steps")

    ts_cols = {"t": times_rel, "W_gust": W, "delta": delta}
    fom_ref = None

    for ds in DS_LIST:
        d = RES / case["dirpat"].format(ds=ds)
        fom = np.load(d / f"fom_{sim}.npy").astype(np.float64)
        rom = np.load(d / f"rom_{sim}.npy").astype(np.float64)
        if fom_ref is None:
            fom_ref = fom
        else:
            print(f"  fom(ds={ds}) vs fom(ds=1) max|diff| = {np.abs(fom - fom_ref).max():.2e}")
        err = rom - fom
        rng_glob = fom.max() - fom.min()
        comb_nrmse = np.sqrt(np.mean(err ** 2)) / rng_glob
        print(f"  d_s={ds:2d}: combined NRMSE (repo convention) = {comb_nrmse:.4e} "
              f"(global range {rng_glob:.1f})")
        rng_field = {f: fom[:, :, k].max() - fom[:, :, k].min() for k, f in enumerate(FIELDS)}
        print(f"          field ranges: " +
              ", ".join(f"{f}={rng_field[f]:.2f}" for f in FIELDS))

        # ---- tidy decomposition rows ----
        for fname in FIELDS + ["combined"]:
            if fname == "combined":
                e2 = (err ** 2).mean(2)          # [T,N] mean over channels
                denom, dtype_ = rng_glob, "global_range_all_fields"
            else:
                k = FIELDS.index(fname)
                e2 = err[:, :, k] ** 2
                denom, dtype_ = rng_field[fname], f"global_range_{fname}"
            for rname, rmask in REGIONS.items():
                for wname, wmask in WIN.items():
                    if wmask.sum() == 0 or rmask.sum() == 0:
                        continue
                    sel = e2[np.ix_(wmask, rmask)]
                    rmse = np.sqrt(sel.mean())
                    rows.append(dict(sim=sim, d_s=ds, field=fname, region=rname,
                                     time_window=wname, nrmse=rmse / denom,
                                     n_nodes=int(rmask.sum()), n_times=int(wmask.sum()),
                                     rmse=rmse, denom=denom, denom_type=dtype_))

        # ---- airfoil-surface pressure, own normalization (loads proxy) ----
        smask = REGIONS["surface"]
        p_f, p_r = fom[:, smask, 2], rom[:, smask, 2]
        rng_ps = p_f.max() - p_f.min()
        for wname, wmask in WIN.items():
            if wmask.sum() == 0:
                continue
            rmse = np.sqrt(((p_r - p_f)[wmask] ** 2).mean())
            rows.append(dict(sim=sim, d_s=ds, field="p_surface", region="surface",
                             time_window=wname, nrmse=rmse / rng_ps,
                             n_nodes=int(smask.sum()), n_times=int(wmask.sum()),
                             rmse=rmse, denom=rng_ps, denom_type="surface_p_range"))

        # ---- vorticity (linear-FE gradient on triangulation; near-wall caveat) ----
        w_f, w_r = vorticity(fom), vorticity(rom)
        w_e2 = (w_r - w_f) ** 2
        for rname, rmask in TRI_REGIONS.items():
            for wname, wmask in WIN.items():
                if wmask.sum() == 0 or rmask.sum() == 0:
                    continue
                rmse = np.sqrt(w_e2[np.ix_(wmask, rmask)].mean())
                denom = np.sqrt((w_f[np.ix_(wmask, rmask)] ** 2).mean())
                rows.append(dict(sim=sim, d_s=ds, field="vorticity", region=rname,
                                 time_window=wname, nrmse=rmse / denom,
                                 n_nodes=int(rmask.sum()), n_times=int(wmask.sum()),
                                 rmse=rmse, denom=denom, denom_type="fom_vorticity_rms"))

        # ---- SSE shares of the combined metric ----
        sse_tot = (err ** 2).sum()
        for k, fname in enumerate(FIELDS):
            shares.append(dict(sim=sim, d_s=ds, kind="field", name=fname,
                               share=(err[:, :, k] ** 2).sum() / sse_tot,
                               weight_frac=1.0 / 3.0))
        for rname in ["near", "wake", "far"]:
            rmask = REGIONS[rname]
            shares.append(dict(sim=sim, d_s=ds, kind="region", name=rname,
                               share=(err[:, rmask, :] ** 2).sum() / sse_tot,
                               weight_frac=rmask.sum() / N))
        # field share within the near region (is near-field error vx-dominated?)
        sse_near = (err[:, REGIONS["near"], :] ** 2).sum()
        for k, fname in enumerate(FIELDS):
            shares.append(dict(sim=sim, d_s=ds, kind="field_in_near", name=fname,
                               share=(err[:, REGIONS["near"], k] ** 2).sum() / sse_near,
                               weight_frac=1.0 / 3.0))

        # ---- per-time curves ----
        for k, fname in enumerate(FIELDS):
            ts_cols[f"l{ds}_{fname}"] = np.sqrt((err[:, :, k] ** 2).mean(1)) / rng_field[fname]
        ts_cols[f"l{ds}_vx_near"] = np.sqrt((err[:, near, 0] ** 2).mean(1)) / rng_field["vx"]
        ts_cols[f"l{ds}_p_surf"] = np.sqrt(((p_r - p_f) ** 2).mean(1)) / rng_ps
        ts_cols[f"l{ds}_combined"] = np.sqrt((err ** 2).mean((1, 2))) / rng_glob

    # write per-sim timeseries csv
    keys = list(ts_cols.keys())
    with open(OUT / f"timeseries_{sim}.csv", "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(keys)
        for irow in range(T):
            wtr.writerow([f"{ts_cols[k][irow]:.6e}" for k in keys])

# ----------------------------------------------------------------------------
# Reconcile: recombine decomposed pieces -> reported combined NRMSE
# ----------------------------------------------------------------------------
print("\n=== reconciliation: recombine (field x region, window=all) -> combined ===")
for sim, case in CASES.items():
    for ds in DS_LIST:
        sse, n = 0.0, 0
        denom = None
        for r in rows:
            if r["sim"] == sim and r["d_s"] == ds and r["time_window"] == "all" \
                    and r["field"] in FIELDS and r["region"] in ("near", "wake", "far"):
                sse += r["rmse"] ** 2 * r["n_nodes"] * r["n_times"]
                n += r["n_nodes"] * r["n_times"]
        comb = [r for r in rows if r["sim"] == sim and r["d_s"] == ds
                and r["field"] == "combined" and r["region"] == "all"
                and r["time_window"] == "all"][0]
        recombined = np.sqrt(sse / n) / comb["denom"]
        print(f"  {sim} d_s={ds:2d}: recombined {recombined:.4e}  vs  direct {comb['nrmse']:.4e}")

# ----------------------------------------------------------------------------
# Write CSVs
# ----------------------------------------------------------------------------
cols = ["sim", "d_s", "field", "region", "time_window", "nrmse", "n_nodes",
        "n_times", "rmse", "denom", "denom_type"]
with open(OUT / "error_decomposition.csv", "w", newline="") as fh:
    wtr = csv.DictWriter(fh, fieldnames=cols)
    wtr.writeheader()
    for r in rows:
        r = dict(r)
        for k in ("nrmse", "rmse", "denom"):
            r[k] = f"{r[k]:.6e}"
        wtr.writerow(r)

with open(OUT / "mse_shares.csv", "w", newline="") as fh:
    wtr = csv.DictWriter(fh, fieldnames=["sim", "d_s", "kind", "name", "share", "weight_frac"])
    wtr.writeheader()
    for s in shares:
        s = dict(s)
        s["share"] = f"{s['share']:.6f}"
        s["weight_frac"] = f"{s['weight_frac']:.6f}"
        wtr.writerow(s)

print(f"\nwrote {OUT}/error_decomposition.csv ({len(rows)} rows), mse_shares.csv, "
      f"timeseries_*.csv, region_labels.npy, airfoil_nodes.npy, geometry.txt")

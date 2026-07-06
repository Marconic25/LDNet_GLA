#!/usr/bin/env python3
"""Assemble extracted field npy + structural trajectories into a FIELDS_*.h5 dataset.

Mirrors the schema consumed by the LDNet field pipeline (src/utils.process_dataset +
the training driver), but with REAL fields instead of the scalar-only GLA dataset:

  points           (Npts, 2)         mesh grid (x,y)
  times            (T,)              common time grid [s], starting at 0
  input_parameters (N, 1)            U_inf
  input_signals    (N, T, 6)         [h, hd, alpha, ad, delta, W_gust]
  output_fields    (N, T, Npts, 3)   [vx, vy, p]
  output_signals   (N, T, 1, 2)      [Fy, Mz] at a probe point (compat; not the field target)
  sim_families     (N,)              family byte-strings

Each input sim directory must contain (produced by recon/extract_fields.py + the PBS job):
  fields_<name>.npy [T_i, Npts, 3], mesh_points.npy [Npts,2], field_times.npy [T_i],
  structural_trajectory.csv  (cols: t,h,hd,alpha,ad,Fy,Mz,W_gust,delta)

All sims share the case mesh (Npts must match). Per-sim time grids differ slightly (window
writes), so fields AND signals are linearly resampled onto a common uniform grid of n_times
points over [0, T_sim]; this also bounds memory.

Usage:
  python build_fields_h5.py --out recon/data/FIELDS_train.h5 --n-times 150 \
      --sim recon/data/sim_A_025_test:A  [--sim path2:family ...]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import h5py
from scipy.interpolate import interp1d

# structural_trajectory.csv column indices (== data/preprocess_GLA.py)
C_T, C_H, C_HD, C_A, C_AD, C_FY, C_MZ, C_WG, C_DELTA = range(9)
U_INF = 80.0


def load_sim(sim_dir):
    sim_dir = Path(sim_dir)
    fcands = sorted(sim_dir.glob("fields_*.npy"))
    if not fcands:
        raise FileNotFoundError(f"no fields_*.npy in {sim_dir}")
    fields = np.load(fcands[0]).astype(np.float64)        # [Ti, Npts, 3]
    points = np.load(sim_dir / "mesh_points.npy").astype(np.float64)  # [Npts,2]
    ftimes = np.load(sim_dir / "field_times.npy").astype(np.float64)  # [Ti]
    ftimes = ftimes - ftimes[0]                            # -> start at 0
    csv_data = np.loadtxt(sim_dir / "structural_trajectory.csv",
                          delimiter=",", skiprows=1)
    return fields, points, ftimes, csv_data


def resample_axis0(arr, t_src, t_dst):
    """Linear interpolation of arr (axis 0 = time) from t_src onto t_dst.

    t_dst is clipped to the source range (no real extrapolation); fill_value is
    'extrapolate' only so out-of-range clipping is exact at the endpoints.
    """
    f = interp1d(t_src, arr, axis=0, bounds_error=False, fill_value="extrapolate")
    return f(np.clip(t_dst, t_src[0], t_src[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim", action="append", required=True,
                    help="path[:family]  (repeatable)")
    ap.add_argument("--n-times", type=int, default=150)
    ap.add_argument("--t-sim", type=float, default=None,
                    help="end time [s]; default = min over sims of last field time")
    args = ap.parse_args()

    specs = []
    for s in args.sim:
        if ":" in s and not s[1:3] == ":\\":
            path, fam = s.rsplit(":", 1)
        else:
            path, fam = s, "A"
        specs.append((Path(path), fam))

    loaded = [(p.name, fam, *load_sim(p)) for p, fam in specs]

    # common mesh (must match)
    points = loaded[0][3]
    npts = points.shape[0]
    for name, fam, fields, pts, ft, cd in loaded:
        if pts.shape[0] != npts:
            raise ValueError(f"{name}: Npts {pts.shape[0]} != {npts} (mesh mismatch)")

    # common time grid
    # loaded tuple = (name, fam, fields, points, ftimes, csv); ftimes is index 4
    t_end = args.t_sim if args.t_sim else min(float(l[4][-1]) for l in loaded)
    t_common = np.linspace(0.0, t_end, args.n_times)

    N, T = len(loaded), args.n_times
    input_signals  = np.zeros((N, T, 6))
    output_fields  = np.zeros((N, T, npts, 3))
    output_signals = np.zeros((N, T, 1, 2))
    families = []

    for i, (name, fam, fields, pts, ft, cd) in enumerate(loaded):
        # fields onto common grid
        output_fields[i] = resample_axis0(fields, ft, t_common)
        # signals from structural csv onto common grid
        tc = cd[:, C_T]
        sig = np.column_stack([cd[:, C_H], cd[:, C_HD], cd[:, C_A],
                               cd[:, C_AD], cd[:, C_DELTA], cd[:, C_WG]])
        input_signals[i] = resample_axis0(sig, tc, t_common)
        fymz = np.column_stack([cd[:, C_FY], cd[:, C_MZ]])
        output_signals[i, :, 0, :] = resample_axis0(fymz, tc, t_common)
        families.append(fam.encode())
        print(f"  [{name}] fam={fam}  Ti={len(ft)}  csv={cd.shape}  -> T={T}")

    input_parameters = np.full((N, 1), U_INF)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("points",           data=points)
        f.create_dataset("times",            data=t_common)
        f.create_dataset("input_parameters", data=input_parameters)
        f.create_dataset("input_signals",    data=input_signals)
        f.create_dataset("output_fields",    data=output_fields)
        f.create_dataset("output_signals",   data=output_signals)
        f.create_dataset("sim_families",     data=np.array(families))
    print(f"saved {out}  [N={N}, T={T}, Npts={npts}]  "
          f"output_fields {output_fields.shape} ({output_fields.nbytes/1e6:.0f} MB)")


if __name__ == "__main__":
    main()

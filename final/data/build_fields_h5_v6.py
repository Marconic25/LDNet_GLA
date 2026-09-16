#!/usr/bin/env python3
"""v6 adaptation of recon/build_fields_h5.py: assembles each sim's
extract_fields_step.py output (fields_<name>.npy + mesh_*.npy + field_times.npy)
plus its structural_trajectory.csv into FIELDS_{train,valid,test}.h5, one file per
split, run/split selection driven by the same run_matrix.csv as
preprocess_GLA_v6.py.

Output schema is unchanged from recon/build_fields_h5.py:
  points           (Npts, 2)         mesh grid (x,y), shared by all sims
  times            (T,)              common time grid [s], starting at 0
  input_parameters (N, 1)            U_inf
  input_signals    (N, T, 6)         [h, hd, alpha, ad, delta, W_gust]
  output_fields    (N, T, Npts, 3)   [vx, vy, p]
  output_signals   (N, T, 1, 2)      [Fy, Mz] at a probe point (compat; not the field target)
  sim_families     (N,)              family byte-strings

RAGGED t_end -- must agree with preprocess_GLA_v6.py, see that file's docstring for
the full argument. Short version: T_MAX = max(t_end) over the WHOLE run_matrix
(every split, not just the one being written), so every FIELDS_*.h5 and every
GLA_*.h5 shares one absolute time origin and span. Queries past a run's own last
recorded field/csv sample hold that sample's value (resample_hold below) instead of
extrapolating -- every row's t_end was chosen as its gust/flap event plus a settling
buffer, so the state at t_end is already past the transient and holding it flat adds
no synthetic dynamics. Truncating everyone to the shortest run was rejected because
it would discard most of the 57 Cc/MPC rows, the campaign's highest-value data.

This script intentionally does NOT force its n_times (default 150, same as
recon/build_fields_h5.py) to match preprocess_GLA_v6.py's (which defaults to the
native per-window resolution instead). Only the time axis (T_MAX, t=0 origin, hold
padding) needs to agree between the two files -- they already had different T in
v5, since output_signals here is explicitly "compat; not the field target".

Validates every matrix row (directory present, fields_*.npy / mesh_points.npy /
field_times.npy present and shape-consistent, structural_trajectory.csv well
formed) and reports skipped/corrupt runs explicitly instead of silently dropping
them.

Usage:
  python3 build_fields_h5_v6.py --matrix final/design/run_matrix.csv \\
      --root /work/u10677113/NACA2312/dataset_v6 --out-dir final/data --n-times 150
"""
import argparse
import csv as csvmod
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError as e:
    sys.exit(f"h5py is required: {e}")
try:
    from scipy.interpolate import interp1d
except ImportError as e:
    sys.exit(f"scipy is required: {e}")

U_INF = 80.0

# structural_trajectory.csv columns -- identical to preprocess_GLA_v6.py / recon/build_fields_h5.py
C_T, C_H, C_HD, C_A, C_AD, C_FY, C_MZ, C_WG, C_DELTA = range(9)
EXPECTED_CSV_COLS = 9

SPLIT_MAP = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
REQUIRED_MATRIX_COLS = {"sim_id", "family", "split", "t_end"}


def read_matrix(matrix_path: Path):
    with open(matrix_path, newline="") as f:
        rows = list(csvmod.DictReader(f))
    if not rows:
        sys.exit(f"run_matrix {matrix_path} is empty")
    missing = REQUIRED_MATRIX_COLS - set(rows[0].keys())
    if missing:
        sys.exit(f"run_matrix {matrix_path} missing required column(s): {sorted(missing)}")
    return rows


def find_sim_dir(root: Path, sim_id: str):
    """Same convention as preprocess_GLA_v6.py::find_sim_dir."""
    cand = root / sim_id
    if cand.is_dir():
        return cand
    cand2 = root / f"sim_{sim_id}"
    if cand2.is_dir():
        return cand2
    return None


def load_sim(sim_dir: Path):
    """Load one sim's extracted fields + structural csv, with shape validation.

    fields_*.npy is produced by extract_fields_step.py --finalize; matching by
    glob (not by the exact --name the driver happened to pass) is what makes this
    robust, same as recon/build_fields_h5.py::load_sim.
    """
    fcands = sorted(sim_dir.glob("fields_*.npy"))
    if not fcands:
        raise FileNotFoundError("no fields_*.npy")
    fields = np.load(fcands[0]).astype(np.float64)              # [Ti, Npts, 3]
    points = np.load(sim_dir / "mesh_points.npy").astype(np.float64)  # [Npts, 2]
    ftimes = np.load(sim_dir / "field_times.npy").astype(np.float64)  # [Ti]
    ftimes = ftimes - ftimes[0]

    if fields.ndim != 3 or fields.shape[2] != 3:
        raise ValueError(f"fields_*.npy has shape {fields.shape}, expected (T,Npts,3)")
    if fields.shape[1] != points.shape[0]:
        raise ValueError(f"fields Npts={fields.shape[1]} != mesh_points Npts={points.shape[0]}")
    if fields.shape[0] != ftimes.shape[0]:
        raise ValueError(f"fields T={fields.shape[0]} != field_times T={ftimes.shape[0]}")
    if ftimes.shape[0] < 2:
        raise ValueError(f"only {ftimes.shape[0]} field snapshot(s), need >= 2 to resample")

    csv_path = sim_dir / "structural_trajectory.csv"
    if not csv_path.exists():
        raise FileNotFoundError("missing structural_trajectory.csv")
    csv_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if csv_data.ndim != 2 or csv_data.shape[1] != EXPECTED_CSV_COLS:
        raise ValueError(f"structural_trajectory.csv has shape {csv_data.shape}, expected (T,{EXPECTED_CSV_COLS})")
    if np.any(np.isnan(csv_data)) or np.any(np.isnan(fields)):
        raise ValueError("NaN found in csv or fields data")

    return fields, points, ftimes, csv_data


def resample_hold(t_src, arr, t_dst):
    """Same clip-before-eval "hold the boundary" strategy as
    recon/build_fields_h5.py::resample_axis0 / preprocess_GLA_v6.py::resample_hold."""
    f = interp1d(t_src, arr, axis=0, bounds_error=False, fill_value="extrapolate")
    return f(np.clip(t_dst, t_src[0], t_src[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default="final/design/run_matrix.csv")
    ap.add_argument("--root", default="/work/u10677113/NACA2312/dataset_v6")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--n-times", type=int, default=150)
    ap.add_argument("--t-end", type=float, default=None,
                     help="override T_MAX (grid span, seconds); "
                          "default = max(t_end) over the WHOLE matrix")
    args = ap.parse_args()

    matrix_path = Path(args.matrix)
    root = Path(args.root)
    rows = read_matrix(matrix_path)

    t_end_all = []
    for r in rows:
        try:
            t_end_all.append(float(r["t_end"]))
        except (KeyError, ValueError, TypeError):
            pass
    if not t_end_all:
        sys.exit(f"run_matrix {matrix_path} has no parseable t_end values")
    t_max = args.t_end if args.t_end is not None else max(t_end_all)
    print(f"shared time grid span T_MAX = {t_max:.4f} s (max t_end over {len(t_end_all)} matrix rows)")

    loaded = {"train": [], "valid": [], "test": []}
    skipped = []
    npts_ref = None
    points_ref = None

    for r in rows:
        sim_id = (r.get("sim_id") or "").strip()
        family = (r.get("family") or "?").strip()
        split_raw = (r.get("split") or "").strip()
        dest = SPLIT_MAP.get(split_raw)

        if not sim_id:
            skipped.append(("<blank sim_id>", "matrix row missing sim_id"))
            continue
        if dest is None:
            skipped.append((sim_id, f"unknown split {split_raw!r}"))
            continue

        sim_dir = find_sim_dir(root, sim_id)
        if sim_dir is None:
            skipped.append((sim_id, f"no directory found under {root} (tried '{sim_id}' and 'sim_{sim_id}')"))
            continue

        try:
            fields, points, ftimes, csv_data = load_sim(sim_dir)
        except Exception as e:
            skipped.append((sim_id, str(e)))
            continue

        if npts_ref is None:
            npts_ref = points.shape[0]
            points_ref = points
        elif points.shape[0] != npts_ref:
            skipped.append((sim_id, f"mesh Npts {points.shape[0]} != {npts_ref} (reference set by an earlier sim)"))
            continue

        loaded[dest].append((sim_id, family, fields, ftimes, csv_data))

    if skipped:
        print(f"\n{len(skipped)} run(s) skipped:")
        for sim_id, reason in skipped:
            print(f"  - {sim_id}: {reason}")
    else:
        print("\nall matrix rows validated cleanly.")

    if npts_ref is None:
        sys.exit("no valid runs loaded -- cannot determine mesh size")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_common = np.linspace(0.0, t_max, args.n_times)
    dt = t_common[1] - t_common[0] if args.n_times > 1 else float("nan")
    print(f"common grid: n_times={args.n_times}, dt={dt:.3e} s, Npts={npts_ref}\n")

    any_written = False
    for split_name, out_name in (("train", "FIELDS_train.h5"), ("valid", "FIELDS_valid.h5"), ("test", "FIELDS_test.h5")):
        sims = loaded[split_name]
        if not sims:
            print(f"no simulations for {split_name}, {out_name} not created.")
            continue

        N, T, npts = len(sims), args.n_times, npts_ref
        input_signals = np.zeros((N, T, 6))
        output_fields = np.zeros((N, T, npts, 3))
        output_signals = np.zeros((N, T, 1, 2))
        # Same validity mask as preprocess_GLA_v6.py, and it must stay the same:
        # the loads and fields datasets share a time origin and span, so a sample
        # masked in one has to be masked in the other. Padding here is held-last-
        # value on BOTH the fields and the signals; a run is valid up to the
        # earlier of its last field snapshot and its last CSV row.
        valid_mask = np.zeros((N, T), dtype=bool)
        families = []

        for i, (sim_id, family, fields, ftimes, cd) in enumerate(sims):
            output_fields[i] = resample_hold(ftimes, fields, t_common)
            t_src = cd[:, C_T]
            valid_mask[i] = t_common <= min(float(ftimes[-1]), float(t_src[-1]))
            sig = np.column_stack([cd[:, C_H], cd[:, C_HD], cd[:, C_A], cd[:, C_AD], cd[:, C_DELTA], cd[:, C_WG]])
            input_signals[i] = resample_hold(t_src, sig, t_common)
            fymz = np.column_stack([cd[:, C_FY], cd[:, C_MZ]])
            output_signals[i, :, 0, :] = resample_hold(t_src, fymz, t_common)
            families.append(family.encode())
            print(f"  [{sim_id}] fam={family}  Ti_field={len(ftimes)}  csv={cd.shape}  -> T={T}")

        input_parameters = np.full((N, 1), U_INF)
        out_path = out_dir / out_name
        with h5py.File(out_path, "w") as f:
            f.create_dataset("points", data=points_ref)
            f.create_dataset("times", data=t_common)
            f.create_dataset("input_parameters", data=input_parameters)
            f.create_dataset("input_signals", data=input_signals)
            f.create_dataset("output_fields", data=output_fields)
            f.create_dataset("output_signals", data=output_signals)
            f.create_dataset("sim_families", data=np.array(families))
            f.create_dataset("valid_mask", data=valid_mask)
        any_written = True
        pad = 100.0 * (1.0 - valid_mask.mean())
        print(f"saved {out_path}  [N={N}, T={T}, Npts={npts}]  "
              f"output_fields {output_fields.shape} ({output_fields.nbytes / 1e6:.0f} MB), "
              f"{pad:.1f}% padded")
        if pad > 0:
            print(f"    NOTE: {pad:.1f}% of this split's grid is held-last-value padding. "
                  f"Weight the loss by valid_mask, or the model learns a flat tail.")

    if not any_written:
        sys.exit("no FIELDS_*.h5 file written -- every split was empty after validation")


if __name__ == "__main__":
    main()

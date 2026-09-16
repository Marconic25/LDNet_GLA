#!/usr/bin/env python3
"""v6 adaptation of data/preprocess_GLA.py: turns each dataset_v6 sim's
structural_trajectory.csv into GLA_{train,valid,test}.h5, same schema as v5.

Two real differences from data/preprocess_GLA.py:

1. RUN/SPLIT SELECTION. v5 parsed `sim_{family}_{idx}_{split}`-style directory
   names to recover family/split. v6 does not promise that naming convention --
   final/design/gen_matrix.py's run_matrix.csv is the single source of truth for
   which sim_id belongs to which family/split (see final/README.md). This script
   takes --matrix and looks up each sim's directory as <root>/<sim_id> (falling
   back to <root>/sim_<sim_id> in case the id is stored bare).

2. RAGGED t_end. v5 ran every sim for a flat 3.0 s, so every structural_trajectory
   .csv had the same row count at the same sample times, and data/preprocess_GLA.py
   could reuse sim 0's raw time column verbatim as the shared "times" dataset. v6's
   run_matrix gives each row its OWN t_end (Tg-dependent -- final/ESTIMATE.md ยง3:
   A ~1.65 s avg, B ~1.80 s, Cc ~1.95 s, vs. a flat 3.0 s in v5), so raw per-sim row
   counts differ and cannot be np.stack-ed directly.

   DECISION: resample every run's signals onto ONE shared, absolute time grid
   t_common = linspace(0, T_MAX, n_times), where T_MAX = max(t_end) over the WHOLE
   matrix (every split, not just the file being written right now) -- so
   GLA_train/valid/test.h5 all share an identical time axis and dt. This is the
   same T_MAX build_fields_h5_v6.py uses (there it is spelled out in more detail);
   the two scripts must agree there since a downstream joint loss could otherwise
   silently compare loads at time t against a field frame at a different real time.
   They do NOT need to share n_times: this script defaults n_times to the largest
   raw row count found among the runs that loaded cleanly (i.e. keeps v5's "native
   per-coupling-window resolution, no downsampling" behaviour for the loads-only
   file), while build_fields_h5_v6.py downsamples to 150 as it always has --
   output_signals in that file is already documented there as "compat; not the
   field target", so the two were never required to share T.

   For query times beyond a given run's OWN last recorded sample, resample_hold()
   clips the query to that run's [t0, t_last] before interpolating -- i.e. HOLDS
   the last recorded value, and never linearly extrapolates. Considered and
   rejected:
     - truncating every run down to the shortest run's t_end: would throw away
       most of the 57 Cc/MPC rows, the highest-value data in the campaign
       (ESTIMATE.md puts them at ~1.95 s average vs ~1.2 s for the shortest A/B
       rows -- up to ~40% of a long run's timeline would be discarded);
     - linear extrapolation past a short run's last state: risks manufacturing
       drift in exactly the free-running regime already flagged as unreliable
       for this model family (project memory: ldnet-latent-instability). Every
       row's t_end is deliberately Tg (or the flap-motion end) plus a settling
       buffer, so the state at t_end is already past the transient -- holding it
       flat for the remaining padding adds no new dynamics, it just repeats a
       real, already-settled sample.

Validates every matrix row (CSV present, right column count, no NaN/Inf, monotonic
time, |delta| in range) and reports skipped/corrupt runs explicitly -- it does not
silently drop them.

Usage:
  python3 preprocess_GLA_v6.py --matrix final/design/run_matrix.csv \\
      --root /work/u10677113/NACA2312/dataset_v6 --out-dir final/data
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

# structural_trajectory.csv columns -- identical to data/preprocess_GLA.py and
# recon/build_fields_h5.py: t,h,hd,alpha,ad,Fy,Mz,W_gust,delta
COL_T, COL_H, COL_HD, COL_A, COL_AD, COL_FY, COL_MZ, COL_WGUST, COL_DELTA = range(9)
EXPECTED_COLS = 9

# run_matrix.csv "split" values -> GLA_{train,valid,test}.h5 (mirrors
# data/preprocess_GLA.py's split_map: matrix spells it "val", the file is "valid").
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
    """v6 sim directories are expected to be named exactly after sim_id; fall back
    to a 'sim_<id>' prefix in case the matrix stores the bare id (v5 used a 'sim_'
    prefix) so this keeps working either way."""
    cand = root / sim_id
    if cand.is_dir():
        return cand
    cand2 = root / f"sim_{sim_id}"
    if cand2.is_dir():
        return cand2
    return None


def validate_csv(csv_path: Path):
    """Returns (problems, data). data is None if the file could not be used at
    all; a non-empty `problems` with data != None means the run is kept but
    flagged (mirrors data/preprocess_GLA.py's behaviour)."""
    try:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    except Exception as e:
        return [f"read error: {e}"], None

    if data.ndim != 2 or data.shape[1] != EXPECTED_COLS:
        return [f"unexpected shape {getattr(data, 'shape', None)} (expected Tx{EXPECTED_COLS})"], None
    if data.shape[0] < 2:
        return [f"only {data.shape[0]} row(s), need >= 2 to resample"], None

    problems = []
    n_nan = int(np.sum(np.isnan(data)))
    if n_nan:
        problems.append(f"{n_nan} NaN value(s)")
    n_inf = int(np.sum(np.isinf(data)))
    if n_inf:
        problems.append(f"{n_inf} Inf value(s)")
    if n_nan or n_inf:
        return problems, None  # cannot safely interpolate through NaN/Inf
    if not np.all(np.diff(data[:, COL_T]) > 0):
        return ["time column not strictly increasing"], None
    max_delta = float(np.max(np.abs(data[:, COL_DELTA])))
    if max_delta > 25:
        problems.append(f"|delta| out of range: max {max_delta:.2f} deg")
    return problems, data


def resample_hold(t_src, arr, t_dst):
    """Linear-interpolate arr(t_src) onto t_dst; for any t_dst outside
    [t_src[0], t_src[-1]] the query is clipped to that boundary BEFORE evaluating,
    which holds the boundary value rather than extrapolating. Identical strategy
    to recon/build_fields_h5.py::resample_axis0 (fill_value="extrapolate" there is
    only a numerical safety valve for the exact-boundary case -- it never actually
    extrapolates because the query is always pre-clipped into range)."""
    f = interp1d(t_src, arr, axis=0, bounds_error=False, fill_value="extrapolate")
    return f(np.clip(t_dst, t_src[0], t_src[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default="final/design/run_matrix.csv")
    ap.add_argument("--root", default="/work/u10677113/NACA2312/dataset_v6")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--n-times", type=int, default=None,
                     help="common sample count for the shared time grid; "
                          "default = largest raw row count among the valid runs")
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
        csv_path = sim_dir / "structural_trajectory.csv"
        if not csv_path.exists():
            skipped.append((sim_id, f"missing structural_trajectory.csv in {sim_dir}"))
            continue

        problems, data = validate_csv(csv_path)
        if data is None:
            skipped.append((sim_id, "; ".join(problems) if problems else "invalid CSV"))
            continue
        if problems:
            print(f"  [{sim_id}] kept with warning(s): {'; '.join(problems)}")

        loaded[dest].append((sim_id, family, data))

    if skipped:
        print(f"\n{len(skipped)} run(s) skipped:")
        for sim_id, reason in skipped:
            print(f"  - {sim_id}: {reason}")
    else:
        print("\nall matrix rows validated cleanly.")

    n_times = args.n_times
    if n_times is None:
        counts = [d.shape[0] for lst in loaded.values() for (_, _, d) in lst]
        if not counts:
            sys.exit("no valid runs loaded -- cannot pick n_times (pass --n-times to force one)")
        n_times = max(counts)
    t_common = np.linspace(0.0, t_max, n_times)
    dt = t_common[1] - t_common[0] if n_times > 1 else float("nan")
    print(f"common grid: n_times={n_times}, dt={dt:.3e} s\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    any_written = False
    for split_name, out_name in (("train", "GLA_train.h5"), ("valid", "GLA_valid.h5"), ("test", "GLA_test.h5")):
        sims = loaded[split_name]
        if not sims:
            print(f"no simulations for {split_name}, {out_name} not created.")
            continue

        N = len(sims)
        input_signals = np.zeros((N, n_times, 6))
        output_signals = np.zeros((N, n_times, 1, 2))
        # Which samples are real. v6 runs have per-run t_end (v5 was a flat 3.0 s),
        # so on a shared grid spanning T_MAX every run shorter than T_MAX is padded
        # by holding its last value. That padding is FABRICATED: the real system is
        # still ringing down there, not sitting flat. Campaign-wide it is 23.6% of
        # the grid, 31% for family A, and 49% for the shortest run. Training on it
        # unmasked teaches the model that the response goes constant.
        # valid_mask[i, k] is True where run i has real data at t_common[k].
        valid_mask = np.zeros((N, n_times), dtype=bool)
        families = []
        for i, (sim_id, family, data) in enumerate(sims):
            t_src = data[:, COL_T]
            valid_mask[i] = t_common <= t_src[-1]
            sig = np.column_stack([
                data[:, COL_H], data[:, COL_HD], data[:, COL_A],
                data[:, COL_AD], data[:, COL_DELTA], data[:, COL_WGUST],
            ])
            input_signals[i] = resample_hold(t_src, sig, t_common)
            fymz = np.column_stack([data[:, COL_FY], data[:, COL_MZ]])
            output_signals[i, :, 0, :] = resample_hold(t_src, fymz, t_common)
            families.append(family.encode())

        input_parameters = np.full((N, 1), U_INF)
        output_fields = np.zeros((N, n_times, 1, 2))  # placeholder, same convention as v5 GLA files

        out_path = out_dir / out_name
        with h5py.File(out_path, "w") as f:
            f.create_dataset("points", data=np.array([[0.0, 0.0]]))
            f.create_dataset("times", data=t_common)
            f.create_dataset("input_parameters", data=input_parameters)
            f.create_dataset("input_signals", data=input_signals)
            f.create_dataset("output_signals", data=output_signals)
            f.create_dataset("output_fields", data=output_fields)
            f.create_dataset("sim_families", data=np.array(families))
            f.create_dataset("valid_mask", data=valid_mask)
        any_written = True
        pad = 100.0 * (1.0 - valid_mask.mean())
        print(f"saved {out_path}  [{N} sims, input_signals {input_signals.shape}, "
              f"output_signals {output_signals.shape}, {pad:.1f}% padded]")
        if pad > 0:
            print(f"    NOTE: {pad:.1f}% of this split's grid is held-last-value padding. "
                  f"Weight the loss by valid_mask, or the model learns a flat tail.")

    if not any_written:
        sys.exit("no GLA_*.h5 file written -- every split was empty after validation")


if __name__ == "__main__":
    main()

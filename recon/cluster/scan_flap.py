#!/usr/bin/env python3
"""Rank dataset_v5 sims by flap-actuation magnitude, to pick a run for mesh viz.

Reads each sim's structural_trajectory.csv (cheap, lives on /work) and scores the
flap channel delta(t): peak-to-peak range, max|delta| and max slew rate |ddelta/dt|.
Family A is gust-only (delta == 0); families B (flap-only) and Cc (gust+flap)
actually move the flap, so they rank at the top. Also reports the gust amplitude W
and whether a *reconstructed* OpenFOAM case (with moving-mesh time dirs) is already
sitting in one of the given case roots.

Remember: /scratch_local is node-local and purged after ~30 days, so the OpenFOAM
case usually has to be (re)generated with field_run_flap.pbs before it can be
exported. This scanner only needs the CSVs, so the login node is enough.

Pure stdlib (csv only) so it runs on a bare login node without numpy.

Examples:
  python3 scan_flap.py --top 20
  python3 scan_flap.py --root /work/u10677113/NACA2312/dataset_v5 \
      --case-root /scratch_local/$USER --case-root /work/u10677113/NACA2312/recon_fields
"""
import argparse
import csv
import os
import re
from pathlib import Path

# structural_trajectory.csv column order (see data/preprocess_GLA.py):
# t, h, hd, alpha, ad, Fy, Mz, W_gust, delta
COL = {"t": 0, "h": 1, "hd": 2, "alpha": 3, "ad": 4,
       "Fy": 5, "Mz": 6, "W_gust": 7, "delta": 8}

TIME_DIR_RE = re.compile(r"^\d+(\.\d+)?([eE][+-]?\d+)?$")  # OpenFOAM numeric time dir


def read_traj(csv_path):
    """Return (t, delta, W) lists from a structural_trajectory.csv, or None on error."""
    t, delta, w = [], [], []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) <= COL["delta"]:
                    continue
                t.append(float(row[COL["t"]]))
                delta.append(float(row[COL["delta"]]))
                w.append(float(row[COL["W_gust"]]))
    except Exception as e:
        return None, str(e)
    if len(t) < 2:
        return None, "fewer than 2 rows"
    return (t, delta, w), None


def score(t, delta, w):
    dmin, dmax = min(delta), max(delta)
    drange = dmax - dmin
    dabs = max(abs(dmin), abs(dmax))
    rate = 0.0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt > 0:
            rate = max(rate, abs(delta[i] - delta[i - 1]) / dt)
    return {
        "delta_range": drange,
        "delta_abs": dabs,
        "delta_rate": rate,
        "w_max": max(w),
        "n": len(t),
        "t_end": t[-1],
    }


def find_case(sim, case_roots):
    """Look for a reconstructed OpenFOAM case for `sim` under the given roots.

    Tries <root>/fldrun_<sim>, <root>/<sim>, <root>/<sim>/case. Returns
    (path, n_time_dirs) for the first dir that has numeric time dirs, else (None, 0).
    """
    for root in case_roots:
        root = Path(os.path.expandvars(root))
        for cand in (root / f"fldrun_{sim}", root / sim, root / sim / "case"):
            if not cand.is_dir():
                continue
            n = sum(1 for p in cand.iterdir()
                    if p.is_dir() and TIME_DIR_RE.match(p.name) and p.name != "0")
            if n > 0:
                return str(cand), n
    return None, 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/work/u10677113/NACA2312/dataset_v5",
                    help="dataset root containing sim_* dirs")
    ap.add_argument("--case-root", action="append", default=[],
                    help="extra root(s) to look for a reconstructed OF case "
                         "(repeatable). Default checks scratch + recon_fields.")
    ap.add_argument("--top", type=int, default=20, help="show top N by delta range")
    ap.add_argument("--min-range", type=float, default=0.0,
                    help="skip sims whose delta range is below this (deg)")
    ap.add_argument("--family", default=None,
                    help="restrict to a family letter (A, B, Cc)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")

    case_roots = args.case_root or [
        f"/scratch_local/{os.environ.get('USER', 'u10677113')}",
        "/work/u10677113/NACA2312/recon_fields",
    ]

    rows = []
    bad = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("sim_"):
            continue
        parts = d.name.split("_")
        fam = parts[1] if len(parts) >= 2 else "?"
        if args.family and fam != args.family:
            continue
        traj, err = read_traj(d / "structural_trajectory.csv")
        if traj is None:
            bad.append((d.name, err))
            continue
        s = score(*traj)
        if s["delta_range"] < args.min_range:
            continue
        case_path, n_case = find_case(d.name, case_roots)
        rows.append((d.name, fam, s, case_path, n_case))

    rows.sort(key=lambda r: r[2]["delta_range"], reverse=True)

    print(f"scanned {root}  ({len(rows)} sims scored, {len(bad)} unreadable)")
    print(f"case roots checked: {case_roots}")
    print()
    hdr = (f"{'sim':<22} {'fam':<3} {'d_range':>8} {'|d|max':>7} "
           f"{'rate':>9} {'W_max':>7} {'nrow':>6} {'case?':>6}  case_path")
    print(hdr)
    print("-" * len(hdr))
    for name, fam, s, case_path, n_case in rows[:args.top]:
        flag = f"{n_case}t" if case_path else "-"
        print(f"{name:<22} {fam:<3} {s['delta_range']:8.3f} {s['delta_abs']:7.3f} "
              f"{s['delta_rate']:9.1f} {s['w_max']:7.3f} {s['n']:6d} {flag:>6}  "
              f"{case_path or ''}")

    if bad:
        print(f"\n{len(bad)} unreadable (first 10):")
        for name, err in bad[:10]:
            print(f"  {name}: {err}")

    print("\nunits: delta in deg, rate in deg/s, W_max = peak gust vertical velocity.")
    print("A=gust-only (delta 0), B=flap-only, Cc=gust+flap. Pick a B/Cc with large")
    print("d_range (and rate, if you want a fast flap). 'case?' = Nt means a")
    print("reconstructed OF case with N time dirs is already on disk (ready to export);")
    print("'-' means regenerate it with:  qsub -v SIM=<sim> field_run_flap.pbs")


if __name__ == "__main__":
    main()

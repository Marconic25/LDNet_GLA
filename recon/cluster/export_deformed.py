#!/usr/bin/env python3
"""Export a reconstructed OpenFOAM case as a ParaView time series with the *moving*
mesh preserved, so the flap is seen deflecting.

Unlike extract_fields.py (which deliberately freezes the grid at the first snapshot
because it only needs the field values on a fixed reference cloud), this reads the
mesh geometry at *every* selected time, so the point positions follow the dynamic
mesh. It slices at the z mid-plane by default, giving a clean 2D deforming grid with
the airfoil+flap hole moving through it, and writes:

  <out>/<name>_XXXX.vtp    one per timestep (slice; or .vtu with --full3d)
  <out>/<name>.pvd         ParaView collection tying the files to their physical times

Open <name>.pvd in ParaView, set representation to "Surface With Edges" to see the
mesh, colour by p or U, step through time — the flap moves.

Runs in the pyvista env used by extract_fields.py (e.g. ~/cosim_env). Example:
  source ~/cosim_env/bin/activate
  python3 export_deformed.py --case /scratch_local/$USER/fldrun_sim_Cc_060_test \
      --out /work/u10677113/NACA2312/recon_fields/sim_Cc_060_test/vtk \
      --name sim_Cc_060_test --auto
"""
import argparse
import csv
from pathlib import Path

Z_MID = 0.125  # mid-span slice plane, matches extract_fields.py


def read_delta(case):
    """Return (t_list, delta_list) from the case's structural_trajectory.csv, or None."""
    p = Path(case) / "structural_trajectory.csv"
    if not p.exists():
        return None
    t, d = [], []
    with open(p, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 9:
                t.append(float(row[0]))
                d.append(float(row[8]))
    return (t, d) if t else None


def nearest(values, target):
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


def pick_times(reader_times, case, mode, stride, explicit):
    """Choose which physical times to export."""
    times = [t for t in reader_times if t > 0.0]
    if explicit:
        return [times[nearest(times, t)] for t in explicit]
    if mode == "auto":
        dd = read_delta(case)
        if dd is None:
            print("[auto] no structural_trajectory.csv; falling back to first/mid/last")
            return [times[0], times[len(times) // 2], times[-1]]
        t, d = dd
        i_min = min(range(len(d)), key=lambda i: d[i])
        i_max = max(range(len(d)), key=lambda i: d[i])
        i_zero = min(range(len(d)), key=lambda i: abs(d[i]))
        targets = sorted({t[i_zero], t[i_min], t[i_max]})
        print(f"[auto] delta extrema: min={d[i_min]:.3f}deg @t={t[i_min]:.4f}, "
              f"max={d[i_max]:.3f}deg @t={t[i_max]:.4f}, "
              f"near-zero @t={t[i_zero]:.4f}")
        return [times[nearest(times, tt)] for tt in targets]
    # stride mode: every Nth available field time
    return times[::max(1, stride)]


def write_pvd(out, name, entries, suffix):
    """entries: list of (index, physical_time)."""
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             '  <Collection>']
    for idx, t in entries:
        lines.append(f'    <DataSet timestep="{t:.6g}" group="" part="0" '
                     f'file="{name}_{idx:04d}.{suffix}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    (Path(out) / f"{name}.pvd").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="reconstructed OpenFOAM case dir")
    ap.add_argument("--out", required=True, help="output dir for the .vtp/.vtu + .pvd")
    ap.add_argument("--name", default="sim", help="basename for the series files")
    ap.add_argument("--z", type=float, default=Z_MID, help="slice plane z")
    ap.add_argument("--full3d", action="store_true",
                    help="export the full internal mesh (.vtu) instead of a z-slice")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--auto", action="store_true",
                     help="export 3 snapshots: delta min, ~0, and max (default)")
    grp.add_argument("--stride", type=int, default=0,
                     help="export every Nth field time (a reduced full series)")
    grp.add_argument("--times", type=float, nargs="+",
                     help="explicit physical times (nearest available is used)")
    args = ap.parse_args()

    import pyvista as pv

    case = Path(args.case)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    foam = case / "export.foam"
    created = not foam.exists()
    if created:
        foam.touch()
    try:
        reader = pv.OpenFOAMReader(str(foam))
        all_times = list(reader.time_values)
        if not all_times:
            raise SystemExit("no time directories found in case")

        mode = "stride" if args.stride else ("times" if args.times else "auto")
        sel = pick_times(all_times, case, mode, args.stride, args.times)
        # de-dup while preserving order
        seen, times = set(), []
        for t in sel:
            if t not in seen:
                seen.add(t)
                times.append(t)
        print(f"case has {len(all_times)} times; exporting {len(times)}: "
              f"[{', '.join(f'{t:.4f}' for t in times)}]", flush=True)

        suffix = "vtu" if args.full3d else "vtp"
        entries = []
        for idx, t in enumerate(times):
            reader.set_active_time_value(t)
            mesh = reader.read()["internalMesh"]  # geometry at THIS time (moving)
            if args.full3d:
                surf = mesh.cell_data_to_point_data()
            else:
                surf = (mesh.slice(normal="z", origin=(0.0, 0.0, args.z))
                            .cell_data_to_point_data()
                            .triangulate())
            fpath = out / f"{args.name}_{idx:04d}.{suffix}"
            surf.save(str(fpath))
            entries.append((idx, t))
            print(f"  wrote {fpath.name}  (t={t:.4f}s, {surf.n_points} pts)", flush=True)

        write_pvd(out, args.name, entries, suffix)
        print(f"\nseries index: {out / (args.name + '.pvd')}")
        print("open the .pvd in ParaView, use 'Surface With Edges' to see the mesh.")
    finally:
        if created:
            foam.unlink()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Incremental, single-timestep field extractor for the v6 co-simulation driver.

recon/extract_fields.py reads a WHOLE reconstructed case in one process and loops
over every time directory. That is impossible for dataset_v6: with purgeWrite 0 and
Nwin=29 (dt=3.16e-5 s), a full 3 s run writes ~3 274 coupling windows, and keeping
every processor time directory until a single end-of-run reconstructPar would need
~419 GB of node-local scratch (see final/README.md / final/ESTIMATE.md ยง5).

The fix is to reconstruct, slice, and delete ONE time directory at a time, from
inside the co-simulation driver's window loop. This script is the "slice" half of
that: the driver (Agent A's cluster/cosim_driver_final.py) calls it once per window,
right after `reconstructPar -time <T>` and right before it purges that time dir.

CLI contract -- the driver depends on this exactly, do not change flag names,
required args, or output paths:

  --case C --time T --out O --name N
      Read exactly the ONE reconstructed time directory T present in case C, slice
      and crop it the same way recon/extract_fields.py does, and write it as
      O/chunks/f_<key>.npy, shape (Npts, 3) float32, channel order (Ux, Uy, p).
      <key> is a canonical, round-trip-safe re-encoding of the parsed --time value
      (see format_time_key below) -- NOT necessarily a literal copy of the --time
      string -- so the driver must not try to predict chunk filenames itself; only
      --finalize's output (fields_<N>.npy, field_times.npy) is a stable contract.
      On the FIRST call for a given O (no mesh_points.npy yet), this also writes
      O/mesh_points.npy [Npts,2] float32 and O/mesh_triangles.npy [Ntri,3] int32,
      computed from that first snapshot exactly as recon/extract_fields.py does.
      An internal O/_crop_mask.npy is also written on the first call and reused on
      every later call so all snapshots select the same physical points (the
      dynamic mesh morphs with constant topology -- see recon/extract_fields.py's
      module docstring -- so a crop computed once stays valid for the whole run,
      the same assumption that script already relies on within a single process).

  --finalize --out O --name N
      Stack every O/chunks/f_*.npy in ascending NUMERIC time order (never
      lexicographic -- OpenFOAM's `general` time format at timePrecision 6 mixes
      fixed and scientific notation, e.g. "3.16e-05" then "0.0001264": as STRINGS
      "0.0001264" < "3.16e-05", which is backwards) into O/fields_<N>.npy
      [T,Npts,3] float32, write O/field_times.npy [T] float32, then remove
      O/chunks/. Tolerant of a few missing or malformed chunks: logs how many and
      continues -- losing one snapshot out of ~860 must not destroy a ~5 h run.

Exit codes: 0 on success (or a tolerated partial finalize), 1 on any real failure,
with a message on stderr explaining what to do about it. Never leaves a half-written
.npy behind (chunks and outputs are written to a temp path and atomically renamed).

Run inside cosim_env (pyvista) on the cluster:
  source ~/cosim_env/bin/activate
  python3 extract_fields_step.py --case "$SCRATCH" --time 0.0001264 \\
      --out /work/u10677113/NACA2312/recon_fields/sim_v6_003 --name sim_v6_003
  ... (repeated once per window; the driver purges $SCRATCH's time dir right after)
  python3 extract_fields_step.py --finalize \\
      --out /work/u10677113/NACA2312/recon_fields/sim_v6_003 --name sim_v6_003

Offline self-check (no pyvista, no OpenFOAM case needed) -- exercises the
time-parsing / numeric-sort / finalize-tolerance logic on synthetic data:
  python3 extract_fields_step.py --selftest
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np

# --- slice plane / crop window -------------------------------------------------
# Copied from recon/extract_fields.py (near+wake crop in chords, mid-span slice) so
# v5 and v6 field data stay directly comparable. Not imported: extract_fields.py
# puts everything (including pyvista import) inside main(), so there is nothing
# clean to import, and this script must not require recon/ on PYTHONPATH on the
# cluster.
CROP_XMIN, CROP_XMAX = -1.0, 4.0
CROP_YMIN, CROP_YMAX = -1.0, 1.0
Z_MID = 0.125  # mid-span slice plane

# Point-ordering guard (see the long comment in write_step()). Anchors are the
# cropped points furthest from the body, where the diffused mesh motion has
# decayed to ~nothing; a reordering would displace them by O(1) chords.
BODY_X, BODY_Y = 0.5, 0.0   # airfoil mid-chord, the centre motion decays away from
ANCHOR_FRAC = 0.05          # fraction of cropped points used as anchors (clamped 200..2000)
ANCHOR_TOL = 0.05           # [chords] max tolerated far-field displacement

CHUNK_RE = re.compile(r"^f_(.+)\.npy$")


def fail(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# --- time parsing / canonicalization -------------------------------------------

def parse_of_time(s: str) -> float:
    """Parse an OpenFOAM time-directory name (`general` format, timePrecision 6).

    Python's float() already parses both the fixed ("0.0001264") and scientific
    ("3.16e-05") forms the `general` writer emits, so this is a thin, deliberate
    wrapper -- the point is to have exactly ONE place in this file that does the
    parsing, so the numeric-sort fix below cannot be bypassed by a stray
    string-sort call somewhere else.
    """
    return float(s.strip())


def format_time_key(t: float) -> str:
    """Canonical, round-trip-safe string for a time value, used in chunk filenames.

    repr() of a Python float is the shortest decimal string that parses back to
    the exact same float (guaranteed since Python 3.1), so
    parse_of_time(format_time_key(t)) == t bit-for-bit -- regardless of whether the
    original --time argument used fixed or scientific notation, or had extra
    whitespace / trailing zeros.
    """
    return repr(float(t))


def chunk_path(out: Path, t: float) -> Path:
    return out / "chunks" / f"f_{format_time_key(t)}.npy"


def sorted_chunks(chunks_dir: Path):
    """Return [(t, path), ...] for every f_*.npy in chunks_dir, sorted by NUMERIC
    time, ascending.

    Never sort chunk filenames as strings. OpenFOAM's `general` time format mixes
    fixed and scientific notation across a run (small early times print as
    "3.16e-05", later times as "0.0001264"), and lexicographic order does not
    track time order for that mix -- the classic version of this bug is simpler
    ("0.1" vs "0.09"-style string sorts) but the real trap here is fixed-vs-
    scientific notation, which is worse because it silently reorders whole chunks
    of the run rather than just two adjacent samples.
    """
    out = []
    for p in chunks_dir.glob("f_*.npy"):
        m = CHUNK_RE.match(p.name)
        if not m:
            continue
        try:
            t = parse_of_time(m.group(1))
        except ValueError:
            continue
        out.append((t, p))
    out.sort(key=lambda pair: pair[0])
    return out


def _atomic_save(path: Path, arr: np.ndarray):
    """np.save + os.replace, so a killed process never leaves a half-written or
    wrongly-suffixed file at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.stem + ".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


# --- pyvista-dependent extraction (single time directory) ----------------------

def slice_surf(reader, t, z):
    """Identical to recon/extract_fields.py::slice_surf."""
    reader.set_active_time_value(t)
    mesh = reader.read()["internalMesh"]
    return (mesh.slice(normal="z", origin=(0.0, 0.0, z))
                .cell_data_to_point_data()
                .triangulate())


def run_step(args):
    if not args.case or args.time is None:
        fail("single-timestep mode requires --case and --time")
    if not args.out or not args.name:
        fail("--out and --name are required")

    try:
        t_requested = parse_of_time(args.time)
    except ValueError:
        fail(f"--time {args.time!r} is not a parseable float")
        return

    out = Path(args.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)

    try:
        import pyvista as pv
    except ImportError as e:
        fail(
            "pyvista is not importable in this Python environment.\n"
            "  On the cluster, activate the cosim env first:\n"
            "    source ~/cosim_env/bin/activate\n"
            f"  (import error: {e})"
        )
        return

    case = Path(args.case)
    if not case.exists():
        fail(f"case dir does not exist: {case}")
        return

    foam = case / "extract.foam"
    created = not foam.exists()
    if created:
        foam.touch()
    try:
        try:
            reader = pv.OpenFOAMReader(str(foam))
            times = np.array(reader.time_values, dtype=float)
        except Exception as e:
            fail(f"could not open OpenFOAM case {case}: {e}")
            return

        if times.size == 0:
            fail(f"no field time directories found in case {case}")
            return

        idx = int(np.argmin(np.abs(times - t_requested)))
        tol = max(1e-9, abs(t_requested) * 1e-6)
        if abs(times[idx] - t_requested) > tol:
            fail(
                f"requested time {t_requested!r} not found in case {case} "
                f"(closest available: {times[idx]!r}; all available: {times.tolist()})"
            )
            return

        try:
            surf = slice_surf(reader, times[idx], Z_MID)
            u = surf.point_data["U"]
            p = surf.point_data["p"]
            pts_full = surf.points[:, :2]
        except Exception as e:
            fail(f"failed to read/slice t={t_requested!r} from {case}: {e}")
            return

        mesh_points_path = out / "mesh_points.npy"
        crop_mask_path = out / "_crop_mask.npy"
        anchor_idx_path = out / "_anchor_idx.npy"
        anchor_xy_path = out / "_anchor_xy.npy"
        first_call = not (mesh_points_path.exists() and crop_mask_path.exists())

        # ── Point-ordering guard ────────────────────────────────────────────────
        # Every snapshot of a run is written by a SEPARATE process (the driver
        # shells out once per field-write window), and each one indexes its values
        # by position into a single shared mesh_points.npy. If the slice ever came
        # back in a different point order, the chunks would be silently scrambled
        # and nothing downstream would notice until training.
        #
        # Two checks that do NOT work, both established empirically on job 31041:
        #   - Comparing all point positions: this is a moving-mesh case (heave,
        #     pitch, flap), so surf.points legitimately differ at every time.
        #   - Hashing the slice connectivity: the slice re-triangulates the cut
        #     polygons as the mesh deforms, so surf.faces changes at EVERY
        #     timestep (54/55 snapshots rejected). Connectivity is not invariant.
        #
        # What the same run showed is that the point COUNT is constant, i.e. the
        # plane keeps cutting the same cells — only the triangulation of the cut
        # polygons moves. So the ordering check has to be positional but restricted
        # to points that barely move: the far field. Mesh motion is diffused from
        # the body outwards and decays, so points furthest from the airfoil shift
        # by O(1e-3) chords at most, whereas any permutation would displace them by
        # O(1) chords. ANCHOR_TOL sits orders of magnitude between the two.
        # The observed max anchor displacement is logged every call, so if the real
        # far-field motion ever approaches the tolerance it shows up in run.log
        # rather than silently weakening the guard.
        if first_call:
            crop = ((pts_full[:, 0] >= CROP_XMIN) & (pts_full[:, 0] <= CROP_XMAX) &
                    (pts_full[:, 1] >= CROP_YMIN) & (pts_full[:, 1] <= CROP_YMAX))
            n_crop = int(crop.sum())
            if n_crop == 0:
                fail(f"crop window selects 0 points at t={t_requested!r} -- check case geometry/units")
                return
            old_to_new = np.full(len(pts_full), -1, dtype=int)
            old_to_new[crop] = np.arange(n_crop)
            faces = surf.faces.reshape(-1, 4)[:, 1:]
            tri_mask = np.all(crop[faces], axis=1)
            faces_crop = old_to_new[faces[tri_mask]]

            _atomic_save(crop_mask_path, crop)

            # Anchors: the ANCHOR_FRAC of cropped points furthest from the body,
            # which is where mesh motion has decayed to almost nothing.
            pts_crop = pts_full[crop]
            d_body = np.hypot(pts_crop[:, 0] - BODY_X, pts_crop[:, 1] - BODY_Y)
            n_anchor = int(np.clip(round(ANCHOR_FRAC * n_crop), 200, 2000))
            n_anchor = min(n_anchor, n_crop)
            anchor_idx = np.argsort(d_body)[-n_anchor:].astype(np.int64)
            _atomic_save(anchor_idx_path, anchor_idx)
            _atomic_save(anchor_xy_path, pts_crop[anchor_idx].astype(np.float64))

            _atomic_save(mesh_points_path, pts_crop.astype(np.float32))
            _atomic_save(out / "mesh_triangles.npy", faces_crop.astype(np.int32))
            print(f"[first call] mesh: {n_crop} pts, {len(faces_crop)} tri, "
                  f"{n_anchor} far-field anchors (>= {d_body[anchor_idx].min():.2f} "
                  f"chords from the body) -> {out}", flush=True)
        else:
            crop = np.load(crop_mask_path)
            if crop.shape[0] != pts_full.shape[0]:
                fail(
                    f"mesh point count changed ({pts_full.shape[0]} now vs "
                    f"{crop.shape[0]} recorded at the first call) -- the "
                    f"constant-topology assumption behind the crop mask broke at "
                    f"t={t_requested!r}"
                )
                return
            if anchor_idx_path.exists() and anchor_xy_path.exists():
                anchor_idx = np.load(anchor_idx_path)
                anchor_ref = np.load(anchor_xy_path)
                anchor_now = pts_full[crop][anchor_idx]
                disp = float(np.abs(anchor_now - anchor_ref).max())
                if disp > ANCHOR_TOL:
                    fail(
                        f"far-field points moved by {disp:.4g} chords at "
                        f"t={t_requested!r}, over the {ANCHOR_TOL} tolerance. The "
                        f"point count still matches, so the crop mask would apply "
                        f"cleanly and this snapshot would be SILENTLY reordered "
                        f"relative to mesh_points.npy. Refusing to write. If the "
                        f"far field genuinely moves this much in your case, raise "
                        f"ANCHOR_TOL -- but check for a reordering first."
                    )
                    return
                # The driver runs this script with capture_output=True and only
                # echoes stdout when the call FAILS, so a printed line here is
                # invisible on the happy path -- which is exactly the path where
                # we need evidence the guard actually ran. Append to a file the
                # job copies out instead, so every run carries a record of the
                # real far-field motion and ANCHOR_TOL's true margin.
                with open(out / "_guard_log.txt", "a") as gl:
                    gl.write(f"{t_requested!r} {disp:.6g}\n")
                print(f"  [guard] max far-field displacement {disp:.3g} chords "
                      f"(tol {ANCHOR_TOL})", flush=True)
            else:
                # Output predates the guard, or the anchor files were lost. Adopt
                # the current geometry rather than disabling the check for good.
                pts_crop = pts_full[crop]
                d_body = np.hypot(pts_crop[:, 0] - BODY_X, pts_crop[:, 1] - BODY_Y)
                n_anchor = int(np.clip(round(ANCHOR_FRAC * crop.sum()), 200, 2000))
                n_anchor = min(n_anchor, int(crop.sum()))
                anchor_idx = np.argsort(d_body)[-n_anchor:].astype(np.int64)
                _atomic_save(anchor_idx_path, anchor_idx)
                _atomic_save(anchor_xy_path, pts_crop[anchor_idx].astype(np.float64))
                print(f"  [guard] adopted {n_anchor} anchors from t={t_requested!r}",
                      flush=True)

        n_crop = int(crop.sum())
        chunk = np.empty((n_crop, 3), dtype=np.float32)
        chunk[:, 0] = u[crop, 0]
        chunk[:, 1] = u[crop, 1]
        chunk[:, 2] = p[crop]

        cpath = chunk_path(out, t_requested)
        _atomic_save(cpath, chunk)
        print(f"t={t_requested!r} -> chunks/{cpath.name}  ({n_crop} pts)", flush=True)
    finally:
        if created:
            try:
                foam.unlink()
            except OSError:
                pass


# --- finalize (stack chunks/ into one fields_<name>.npy) ------------------------

def run_finalize(args):
    if not args.out or not args.name:
        fail("--out and --name are required")
        return

    out = Path(args.out)
    chunks_dir = out / "chunks"
    if not chunks_dir.exists():
        fail(f"no chunks/ directory in {out} -- nothing to finalize")
        return

    pairs = sorted_chunks(chunks_dir)
    if not pairs:
        fail(f"no f_*.npy chunks found in {chunks_dir}")
        return

    times = np.array([t for t, _ in pairs], dtype=float)

    # Best-effort gap detection: there is no external manifest of expected times at
    # this point in the pipeline, so we infer suspected missing snapshots from the
    # spacing between the chunks that DID arrive. This only ever logs -- it never
    # blocks the finalize, per the "one missing snapshot out of ~860 must not
    # destroy a 5 h run" requirement.
    n_missing_est = 0
    if len(times) >= 3:
        dts = np.diff(times)
        med = float(np.median(dts))
        if med > 0:
            for dt in dts:
                ratio = dt / med
                if ratio > 1.5:
                    n_missing_est += int(round(ratio)) - 1
    if n_missing_est:
        print(
            f"WARNING: ~{n_missing_est} snapshot(s) appear missing from the time "
            f"series (gap analysis over {len(times)} chunks that did arrive) -- "
            f"continuing anyway",
            flush=True,
        )

    first = np.load(pairs[0][1])
    npts = first.shape[0]

    fields = np.zeros((len(pairs), npts, 3), dtype=np.float32)
    kept_times = []
    n_bad = 0
    write_i = 0
    for t, p in pairs:
        try:
            arr = np.load(p)
        except Exception as e:
            print(f"WARNING: could not load {p.name}: {e} -- skipping", flush=True)
            n_bad += 1
            continue
        if arr.shape != (npts, 3):
            print(
                f"WARNING: {p.name} has shape {arr.shape}, expected {(npts, 3)} "
                f"-- skipping",
                flush=True,
            )
            n_bad += 1
            continue
        fields[write_i] = arr
        kept_times.append(t)
        write_i += 1

    if write_i == 0:
        fail("every chunk failed to load or had a mismatched shape -- nothing to finalize")
        return

    fields = fields[:write_i]
    field_times = np.array(kept_times, dtype=np.float32)

    _atomic_save(out / f"fields_{args.name}.npy", fields)
    _atomic_save(out / "field_times.npy", field_times)

    shutil.rmtree(chunks_dir, ignore_errors=True)

    print(
        f"finalized {fields.shape} ({fields.nbytes / 1e6:.1f} MB) from "
        f"{len(pairs)} chunk(s) found ({n_bad} unreadable/malformed skipped, "
        f"~{n_missing_est} suspected missing from timing gaps) -> {out}",
        flush=True,
    )


# --- offline self-check ---------------------------------------------------------

def _selftest() -> bool:
    import tempfile

    ok = True

    def check(name, cond):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"[{status}] {name}")

    # 1. Parsing OpenFOAM `general`-format time-directory names (fixed + scientific).
    cases = {
        "0.1": 0.1,
        "0.09": 0.09,
        "3.16e-05": 3.16e-05,
        "1e-05": 1e-05,
        "0.0001264": 0.0001264,
        " 0.5 ": 0.5,
        "2": 2.0,
        "0": 0.0,
    }
    for s, expected in cases.items():
        check(f"parse_of_time({s!r}) == {expected!r}", parse_of_time(s) == expected)

    # 2. format_time_key round-trips exactly (bit-for-bit) through parse_of_time.
    values = [0.1, 0.09, 3.16e-05, 0.0001264, 1.0, 9.0, 10.0, 1e-6, 1234.5678, 0.0]
    for v in values:
        key = format_time_key(v)
        back = parse_of_time(key)
        check(f"round-trip {v!r} -> {key!r} -> {back!r}", back == v)

    # 3. sorted_chunks() returns NUMERIC order on a synthetic set of filenames that
    #    includes two lexicographic traps:
    #      - the literally-cited example: "0.1" vs "0.09"
    #      - the real OpenFOAM trap: `general` format mixes fixed and scientific
    #        notation across a run ("3.16e-05" then, later, "0.0001264"), and as
    #        STRINGS "0.0001264" < "3.16e-05" even though 3.16e-05 (0.0000316) is
    #        numerically the smaller value
    trap_times = [0.1, 0.09, 10.0, 9.0, 3.16e-05, 0.0001264, 0.0, 1.5, 1234.5678]
    with tempfile.TemporaryDirectory() as td:
        chunks = Path(td) / "chunks"
        chunks.mkdir()
        for t in trap_times:
            np.save(chunks / f"f_{format_time_key(t)}.npy", np.zeros((2, 3), dtype=np.float32))

        got = [t for t, _ in sorted_chunks(chunks)]
        expected_order = sorted(trap_times)
        check(f"sorted_chunks() numeric order matches sorted(): {got}", got == expected_order)

        # Confirm plain lexicographic filename sort would have gotten this WRONG,
        # so the check above is exercising the fix and not passing by accident.
        lexi_names = sorted(p.name for p in chunks.glob("f_*.npy"))
        lexi_times = [parse_of_time(CHUNK_RE.match(n).group(1)) for n in lexi_names]
        check(
            "plain lexicographic filename sort IS wrong on this data "
            "(confirms the numeric-sort fix is load-bearing)",
            lexi_times != expected_order,
        )

    # 4. run_finalize() end-to-end on synthetic chunks: tolerates one malformed
    #    chunk and a timing gap without failing, per the "losing one snapshot out
    #    of ~860 must not destroy a 5 h run" requirement.
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sim_selftest"
        chunks = out / "chunks"
        chunks.mkdir(parents=True)
        good_times = [0.0, 0.0001, 0.0002, 0.0004, 0.0005]  # note the missing 0.0003
        for t in good_times:
            np.save(chunks / f"f_{format_time_key(t)}.npy", np.full((4, 3), t, dtype=np.float32))
        # one malformed chunk (wrong shape) mixed in among the good ones
        np.save(chunks / f"f_{format_time_key(0.00035)}.npy", np.ones((4, 2), dtype=np.float32))

        class Args:
            pass

        a = Args()
        a.out = str(out)
        a.name = "selftest"
        try:
            run_finalize(a)
            fields = np.load(out / "fields_selftest.npy")
            ftimes = np.load(out / "field_times.npy")
            check(
                "finalize skips the malformed chunk and keeps the 5 good ones",
                fields.shape == (5, 4, 3) and len(ftimes) == 5,
            )
            check("finalize removes chunks/ afterward", not chunks.exists())
        except SystemExit:
            check("finalize end-to-end ran without a hard failure", False)

    print("SELFTEST: " + ("PASS" if ok else "FAIL"))
    return ok


# --- CLI -------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--case", default=None, help="reconstructed OpenFOAM case dir (single-timestep mode)")
    ap.add_argument("--time", default=None, help="OpenFOAM time-directory name to read, e.g. 0.0001264 or 3.16e-05")
    ap.add_argument("--out", default=None, help="output dir (holds chunks/, mesh_*.npy, fields_<name>.npy)")
    ap.add_argument("--name", default=None, help="sim name (fields_<name>.npy)")
    ap.add_argument("--finalize", action="store_true", help="stack chunks/ into fields_<name>.npy and clean up")
    ap.add_argument("--selftest", action="store_true", help="run offline time-parsing/sorting self-checks and exit")
    return ap


def main():
    args = build_parser().parse_args()

    if args.selftest:
        ok = _selftest()
        sys.exit(0 if ok else 1)

    if args.finalize:
        run_finalize(args)
    else:
        run_step(args)


if __name__ == "__main__":
    main()

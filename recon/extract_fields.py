#!/usr/bin/env python3
"""Standalone OpenFOAM field extractor (cluster-side).

Reads a *reconstructed* OpenFOAM case, slices at the z mid-plane, crops to the
near+wake window, and saves the velocity/pressure fields per snapshot:

  fields_<name>.npy   [T, N_pts, 3] float32   channels = (Ux, Uy, p)
  mesh_points.npy     [N_pts, 2]    float32   reference grid (first snapshot)
  mesh_triangles.npy  [N_tri, 3]    int32     triangulation for plotting
  field_times.npy     [T]           float32   physical times

Decoupled on purpose from the project's extract_data.py timeseries/metadata path:
the global metadata_v5.csv was found inconsistent with the actually-run per-sim
params, and extract_data.main() skips fields if its timeseries step throws. Here we
only touch the CFD case. The LDNet input signals (h,hd,alpha,ad,delta,W_gust) come
separately from each sim's saved structural_trajectory.csv.

Crop window and z-plane match the original extractor (extract_data.extract_fields).
The dynamic mesh morphs (constant topology, tiny displacements), so a crop mask +
reference grid computed once on the first snapshot is valid for all snapshots.

Run inside cosim_env (pyvista). Example:
  python extract_fields.py --case "$SCRATCH" --out /work/.../recon_fields/sim_A_025_test \
      --name sim_A_025_test --stride 1
"""
import argparse
from pathlib import Path
import numpy as np

# Near+wake crop (chords); identical to extract_data.py
CROP_XMIN, CROP_XMAX = -1.0, 4.0
CROP_YMIN, CROP_YMAX = -1.0, 1.0
Z_MID = 0.125  # mid-span slice plane


def slice_surf(reader, t, z):
    reader.set_active_time_value(t)
    mesh = reader.read()["internalMesh"]
    return (mesh.slice(normal="z", origin=(0.0, 0.0, z))
                .cell_data_to_point_data()
                .triangulate())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="reconstructed OpenFOAM case dir")
    ap.add_argument("--out", required=True, help="output dir for npy files")
    ap.add_argument("--name", default="sim", help="sim name (fields_<name>.npy)")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth time dir")
    ap.add_argument("--z", type=float, default=Z_MID, help="slice plane z")
    args = ap.parse_args()

    import pyvista as pv

    case = Path(args.case)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    foam = case / "extract.foam"
    created = not foam.exists()
    if created:
        foam.touch()
    try:
        reader = pv.OpenFOAMReader(str(foam))
        times = np.array(reader.time_values)
        if times.size and times[0] == 0.0:
            times = times[1:]
        if args.stride > 1:
            times = times[::args.stride]
        if times.size == 0:
            raise RuntimeError("no field time directories found in case")
        print(f"{len(times)} field timesteps (stride={args.stride}), "
              f"t=[{times[0]:.4f}, {times[-1]:.4f}]", flush=True)

        # Reference grid + crop mask from the first snapshot
        surf0 = slice_surf(reader, times[0], args.z)
        pts = surf0.points[:, :2]
        crop = ((pts[:, 0] >= CROP_XMIN) & (pts[:, 0] <= CROP_XMAX) &
                (pts[:, 1] >= CROP_YMIN) & (pts[:, 1] <= CROP_YMAX))
        n_crop = int(crop.sum())
        old_to_new = np.full(len(pts), -1, dtype=int)
        old_to_new[crop] = np.arange(n_crop)
        faces = surf0.faces.reshape(-1, 4)[:, 1:]
        tri_mask = np.all(crop[faces], axis=1)
        faces_crop = old_to_new[faces[tri_mask]]
        np.save(out / "mesh_points.npy", pts[crop].astype(np.float32))
        np.save(out / "mesh_triangles.npy", faces_crop.astype(np.int32))
        print(f"mesh: {n_crop} pts, {len(faces_crop)} tri", flush=True)

        fields = np.zeros((len(times), n_crop, 3), dtype=np.float32)
        for i, t in enumerate(times):
            if i % 20 == 0:
                print(f"  snapshot {i}/{len(times)} (t={t:.4f}s)...", flush=True)
            surf = slice_surf(reader, t, args.z)
            fields[i, :, 0] = surf.point_data["U"][crop, 0]
            fields[i, :, 1] = surf.point_data["U"][crop, 1]
            fields[i, :, 2] = surf.point_data["p"][crop]

        np.save(out / f"fields_{args.name}.npy", fields)
        np.save(out / "field_times.npy", times.astype(np.float32))
        print(f"saved fields {fields.shape} ({fields.nbytes/1e6:.0f} MB) -> {out}",
              flush=True)
    finally:
        if created:
            foam.unlink()


if __name__ == "__main__":
    main()

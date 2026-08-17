#!/usr/bin/env python3
"""Build an exaggerated mesh-deformation comparison from an exported .vtp series.

The OpenFOAM pointDisplacement field is only written to some time dirs, so it is
missing from many exported frames and ParaView's Warp By Vector can't use it. Here
we recover a reliable displacement straight from the geometry: the mesh topology is
constant, so disp(frame) = points(frame) - points(reference), with the first frame
(flap ~= 0, body ~= 0) as the undeformed reference.

Outputs a small, turnkey comparison (open the .pvd or the individual files):
  reference.vtp       undeformed mesh (flap 0, body 0)
  deformed_real.vtp   real deformation at the flap-peak frame, carries a "disp"
                      vector field -> Warp By Vector on "disp" to dial any factor
  deformed_xN.vtp     pre-baked exaggeration reference + N*disp (flap ~= 14 deg)

Usage:
  python make_warp_demo.py --dir recon/data/sim_Cc_008_train_vtk \
      --name sim_Cc_008_train --peak 23 --factor 3.03
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pyvista as pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder with the exported .vtp series")
    ap.add_argument("--name", required=True, help="series basename")
    ap.add_argument("--peak", type=int, default=-1,
                    help="frame index to exaggerate (-1 = auto: max displacement)")
    ap.add_argument("--factor", type=float, default=3.03,
                    help="total exaggeration factor (flap_real*factor ~= target)")
    ap.add_argument("--out", default=None, help="output dir (default <dir>/warp_demo)")
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(glob.glob(str(d / f"{args.name}_*.vtp")))
    if not files:
        raise SystemExit(f"no {args.name}_*.vtp in {d}")
    out = Path(args.out) if args.out else d / "warp_demo"
    out.mkdir(parents=True, exist_ok=True)

    ref = pv.read(files[0])
    refpts = ref.points.copy()
    n = refpts.shape[0]
    print(f"{len(files)} frames, reference = {Path(files[0]).name} ({n} pts)")

    # locate peak frame + sanity-check point ordering (far-field disp must be ~0)
    peak_i, peak_mag = args.peak, None
    if args.peak < 0:
        best = -1.0
        for i, f in enumerate(files):
            m = pv.read(f)
            if m.n_points != n:
                continue
            mag = np.linalg.norm(m.points - refpts, axis=1).max()
            if mag > best:
                best, peak_i = mag, i
        peak_mag = best

    peak = pv.read(files[peak_i])
    if peak.n_points != n:
        raise SystemExit(f"frame {peak_i} has {peak.n_points} pts != {n} (topology changed)")
    disp = (peak.points - refpts).astype(np.float32)
    dmag = np.linalg.norm(disp, axis=1)
    print(f"peak frame = {peak_i} ({Path(files[peak_i]).name})")
    print(f"disp magnitude: max={dmag.max():.4f}  median={np.median(dmag):.5f}  "
          f"min={dmag.min():.6f}  (min≈0 confirms consistent point ordering)")

    # reference (undeformed)
    ref.save(str(out / "reference.vtp"))

    # real deformation + a warpable "disp" field
    peak["disp"] = disp
    peak.save(str(out / "deformed_real.vtp"))

    # pre-baked exaggeration: reference + factor*disp
    exag = peak.copy()
    exag.points = refpts + args.factor * disp
    exag["disp"] = disp
    tag = f"{args.factor:.2f}".replace(".", "p")
    exag.save(str(out / f"deformed_x{tag}.vtp"))

    print(f"\nwrote -> {out}")
    print("  reference.vtp        (undeformed)")
    print("  deformed_real.vtp    (real; Warp By Vector on 'disp', factor F -> (1+F)x)")
    print(f"  deformed_x{tag}.vtp  (pre-baked flap ~= {4.62*args.factor:.1f} deg)")
    print("In ParaView open all three, colour them differently / overlay to compare.")


if __name__ == "__main__":
    main()

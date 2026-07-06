#!/bin/bash
RECON=/work/u10677113/NACA2312/recon
TRI=/work/u10677113/NACA2312/recon_fields/sim_A_025_test_T0p05/mesh_triangles.npy

for L in 1 5 10; do
  echo "=== compare l$L ==="
  python3 -u $RECON/viz_fields.py compare \
    --recon $RECON/results/rom_l${L} --name sim_A_025 --tri $TRI \
    --latent $L --out $RECON/results/compare_l${L}.png
done

echo "=== compare-video l10 vy ==="
python3 -u $RECON/viz_fields.py compare-video \
  --recon $RECON/results/rom_l10 --name sim_A_025 --tri $TRI \
  --channel vy --stride 1 --fps 12 --out $RECON/results/compare_video_l10_vy.gif

echo "DONE"

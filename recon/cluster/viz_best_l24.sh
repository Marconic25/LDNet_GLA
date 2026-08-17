#!/bin/bash
# Showcase of the campaign's best model so far (depth study L24_ds10, test 1.313e-2):
# reconstruct the gust+flap test sim, then FOM|ROM|error compare panels at three
# instants, side by side with the old production best (final_div L6 ds1) at the same
# instants. Light login-node use (single forward pass + matplotlib).
set -e
RECON=/work/u10677113/NACA2312/recon
TRI=/work/u10677113/NACA2312/recon_fields/sim_Cc_060_test/mesh_triangles.npy
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c "
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd $RECON
  echo '=== reconstruct L24_ds10 on Cc_060 ==='
  python3 -u reconstruct_fields.py --model models/depth_study/L24_ds10/latent_10 \
      --data data/FIELDS_Cc060.h5 --index 0 --name sim_Cc_060 \
      --out results/depth_rom_cc_L24ds10
  echo '=== compare panels ==='
  for IDX in 40 75 110; do
    python3 -u viz_fields.py compare --recon results/depth_rom_cc_L24ds10 \
        --name sim_Cc_060 --tri $TRI --latent '10 (L24 deep)' --idx \$IDX \
        --out results/showcase/L24ds10_cc060_idx\$IDX.png
    python3 -u viz_fields.py compare --recon results/div_rom_cc_l1 \
        --name sim_Cc_060 --tri $TRI --latent '1 (L6 baseline)' --idx \$IDX \
        --out results/showcase/L6ds1_cc060_idx\$IDX.png
  done
  ls -la results/showcase/
  echo VIZ_DONE"

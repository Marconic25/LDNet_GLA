#!/bin/bash
# Smoke for the local mesh-graph message-passing decoder lever
# (--graph-decoder / GraphRelaxDecoder), the post-closure literature-driven
# candidate (DYNAMIC_CONTRIBUTION_LITERATURE_NOTES.md recommendation B,
# "Read, Write, Relax" arXiv:2608.21677). Follows the exact pattern of
# smoke_stall_localdec.sh.
set -e
RECON=/work/u10677113/NACA2312/recon
TRI=/work/u10677113/NACA2312/recon_fields/sim_Cc_060_test/mesh_triangles.npy
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c "
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd $RECON
  rm -rf models/stall_smoke_graphdec; mkdir -p models/stall_smoke_graphdec

  echo '=== SMOKE 1: --graph-decoder ON (mean-split + coral o10) ==='
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke_graphdec/on --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 1024 --mean-split --decoder coral --siren-omega0 10 \
    --graph-decoder --sampling graph --graph-nodes analysis/graph_nodes.npy --tri $TRI \
    --graph-hidden 8 --graph-relax-steps 2 \
    | grep -E 'graph-decoder|arch:|NRMSE'
  python3 - <<EOF
import json
c=json.load(open('models/stall_smoke_graphdec/on/latent_1/config.json'))
assert c['graph_decoder'] is True, c.get('graph_decoder')
assert c['graph']['n_nodes'] == 200, c.get('graph')
print('OK: config graph_decoder=True, n_nodes=200')
EOF
  python3 -u reconstruct_fields.py --model models/stall_smoke_graphdec/on/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/stall_smoke_graphdec/rom_on
  python3 -c \"import numpy as np; r=np.load('models/stall_smoke_graphdec/rom_on/rom_smoke.npy'); assert np.isfinite(r).all() and r[...,0].max()>30.0; print('OK: graph-decoder recon finite, total-units vx max', round(float(r[...,0].max()),1))\"

  echo '=== SMOKE 2: baseline (--graph-decoder OFF) still byte-behaves (regression guard) ==='
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke_graphdec/off --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 \
    | grep -E 'arch:|NRMSE'

  echo STALL_SMOKE_GRAPHDEC_ALL_DONE
"

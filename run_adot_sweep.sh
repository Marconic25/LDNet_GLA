#!/bin/bash
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash -c 'pip install scipy h5py -q 2>/dev/null; cd /work/u10677113/LDNet_GLA/clean && python3 -u adot_sweep.py'
echo RUN_DONE

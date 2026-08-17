#!/bin/bash
# Wait for the A_015 test extraction to finish, then report the snapshot count.
FT=/work/u10677113/NACA2312/recon_fields/sim_A_015_train/field_times.npy
until [ -f "$FT" ]; do sleep 300; done
sleep 10
N=$(apptainer exec --bind /work/u10677113:/work/u10677113 \
    /work/u10677113/tensorflow_gpu.sif \
    python3 -c "import numpy as np; print(len(np.load('$FT')))" 2>/dev/null)
echo "A015_EXTRACTION_DONE snapshots=$N" > /work/u10677113/NACA2312/recon_fields/A015_TEST_RESULT.txt

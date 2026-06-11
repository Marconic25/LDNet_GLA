#!/bin/bash
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash -c 'pip install scipy h5py -q 2>/dev/null; cd /work/u10677113/LDNet_GLA/clean && for S in sim_A_025_test sim_A_027_test; do echo "##### TRAJ $S #####"; for QV in 1000 10000 100000; do QH=$QV QA=$QV NSTEPS=1000 NGRID=15 RS=0.0001,0.00001 SIM=$S python3 -u optimal_test.py; done; done'
echo OPTIMAL2_DONE

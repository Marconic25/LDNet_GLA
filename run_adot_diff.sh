#!/bin/bash
# Sweep one-step controller with Q_alpha_ddot = Q*(alpha_dot_next - alpha_dot_curr)^2
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash -c '
pip install scipy h5py -q 2>/dev/null
cd /work/u10677113/LDNet_GLA/clean
echo "=== Design A: Q_alpha_ddot sweep (one-step, NH=1) ==="
for S in sim_A_025_test sim_A_027_test; do
  echo ""
  echo "########## SIM=$S ##########"
  for QAD in 10 100 1000 10000 100000; do
    NSTEPS=1000 NGRID=15 NH=1 SIM=$S QAD=$QAD RS=0.0001,0.001,0.01,0.1 python3 -u optimal_test.py
  done
done
'
echo "ADOT_DIFF_SWEEP_DONE"

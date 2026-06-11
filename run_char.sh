#!/bin/bash
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash -c 'pip install scipy matplotlib h5py -q 2>/dev/null; cd /work/u10677113/LDNet_GLA/clean
echo "### gain characterization vs W0 (Tg=1.0, LPF=0.7, TEND=2.0) ###"
for W in 5 10 15 20 30; do
  for GR in "100 0.01" "30 0.01" "10 0.001"; do
    set -- $GR
    W0=$W TG=1.0 TEND=2.0 QAD=$1 RW=$2 LPF=0.7 PLOT=0 python3 -u mpc_gust.py
  done
done
echo "### Tg effect (W0=20, LPF=0.7) ###"
for T in 0.5 1.0 2.0; do W0=20 TG=$T TEND=3.0 QAD=30 RW=0.001 LPF=0.7 PLOT=0 python3 -u mpc_gust.py; done'
echo CHAR_DONE

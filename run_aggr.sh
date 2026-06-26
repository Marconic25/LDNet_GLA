#!/bin/bash
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash -c 'pip install scipy matplotlib h5py -q 2>/dev/null; cd /work/u10677113/LDNet_GLA/clean
for WT in "20:0.5" "30:1.0" "11.46:1.12"; do
  W="${WT%:*}"; T="${WT#*:}"; echo "##### W0=$W Tg=$T #####"
  for DR in "0.95:0.01" "0.85:0.01" "0.7:0.01" "0.85:0.001" "0.6:0.001"; do
    D="${DR%:*}"; R="${DR#*:}"
    W0=$W TG=$T TEND=3.0 QAD=30 RW=$R DLPF=$D NGRID=15 DAMULT=3 PLOT=0 python3 -u mpc_gust.py
  done
done'
echo AGGR_DONE

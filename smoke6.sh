#!/bin/bash
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash << 'INNER'
cd /work/u10677113/LDNet_GLA/clean
echo "=W10T2_sched="
DAMULT=3 SCHED=1 W0=10 TG=2.0 python3 -u mpc_gust.py
echo "=W10T2_R1e4="
DAMULT=3 SCHED=0 QAD=30 RW=0.0001 DLPF=0.85 W0=10 TG=2.0 python3 -u mpc_gust.py
echo "=W10T2_R5e4="
DAMULT=3 SCHED=0 QAD=30 RW=0.0005 DLPF=0.85 W0=10 TG=2.0 python3 -u mpc_gust.py
echo "=W10T2_R1e4_D75="
DAMULT=3 SCHED=0 QAD=30 RW=0.0001 DLPF=0.75 W0=10 TG=2.0 python3 -u mpc_gust.py
echo "=DONE="
INNER

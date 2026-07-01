#!/bin/bash
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash << 'INNER'
cd /work/u10677113/LDNet_GLA/clean
echo =W10T05=
DAMULT=3 SCHED=1 W0=10 TG=0.5 python3 -u mpc_gust.py
echo =W30T1=
DAMULT=3 SCHED=1 W0=30 TG=1.0 python3 -u mpc_gust.py
echo =W20T1=
DAMULT=3 SCHED=1 W0=20 TG=1.0 python3 -u mpc_gust.py
echo =DONE=
INNER

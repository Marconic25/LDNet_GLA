#!/bin/bash
# smoke18: regenerate W30/T2 plot with updated schedule (DLPF=0.85, RQUIET=1e-5, QAD=100)
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash << 'INNER'
cd /work/u10677113/LDNet_GLA/clean
mkdir -p results
echo =PLOT_W30T2_new=
DAMULT=3 SCHED=1 NGRID=15 TEND=3.0 PLOT=1 W0=30 TG=2.0 python3 -u mpc_gust.py
echo =DONE=
INNER

#!/bin/bash
# smoke15: generate plots for all 9 CS-25.341 cells + design point, SCHED=1
cd /work/u10677113/LDNet_GLA
mkdir -p /work/u10677113/LDNet_GLA/clean/results
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash << 'INNER'
cd /work/u10677113/LDNet_GLA/clean
mkdir -p results
for W in 10 20 30; do
  for T in 0.5 1.0 2.0; do
    echo =W${W}T${T}=
    DAMULT=3 SCHED=1 NGRID=15 TEND=3.0 PLOT=1 W0=$W TG=$T python3 -u mpc_gust.py
  done
done
echo =design_pt=
DAMULT=3 SCHED=1 NGRID=15 TEND=3.0 PLOT=1 W0=11.46 TG=1.12 python3 -u mpc_gust.py
echo =DONE=
INNER

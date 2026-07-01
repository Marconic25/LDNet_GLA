#!/bin/bash
# smoke17: W30/T2 RQUIET test + final Tg=2 plots with updated schedule
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash << 'INNER'
cd /work/u10677113/LDNet_GLA/clean
mkdir -p results

# ---- W30/T2: RQUIET=1e-5 test (same mechanism that fixed W20/T2 oscillations) ----
echo =W30T2_RQ1e5=
DAMULT=3 SCHED=0 QAD=30 RW=1e-2 DLPF=0.75 NH=6 DMAX=14 RQUIET=1e-5 W0=30 TG=2.0 python3 -u mpc_gust.py

echo =W30T2_RQ1e5_QAD100=
DAMULT=3 SCHED=0 QAD=100 RW=1e-2 DLPF=0.75 NH=6 DMAX=14 RQUIET=1e-5 W0=30 TG=2.0 python3 -u mpc_gust.py

echo =W30T2_RQ1e5_D85=
DAMULT=3 SCHED=0 QAD=100 RW=1e-2 DLPF=0.85 NH=6 DMAX=14 RQUIET=1e-5 W0=30 TG=2.0 python3 -u mpc_gust.py

# ---- Final plots: W10/T2, W20/T2 with updated schedule (QAD=100 and QAD=200) ----
echo =PLOT_W10T2=
DAMULT=3 SCHED=1 NGRID=15 TEND=3.0 PLOT=1 W0=10 TG=2.0 python3 -u mpc_gust.py

echo =PLOT_W20T2=
DAMULT=3 SCHED=1 NGRID=15 TEND=3.0 PLOT=1 W0=20 TG=2.0 python3 -u mpc_gust.py

# W30/T2: plot with current schedule (RQUIET=0.1, QAD=30) as reference
echo =PLOT_W30T2=
DAMULT=3 SCHED=1 NGRID=15 TEND=3.0 PLOT=1 W0=30 TG=2.0 python3 -u mpc_gust.py

echo =DONE=
INNER

#!/bin/bash
# smoke16: alpha_dot oscillation reduction for W10/T2, W20/T2, W30/T2
# Current: W10 QAD=0, W20 QAD=30, W30 QAD=30 → oscillations visible in plots
# Hypothesis: higher QAD damps alpha_dot with modest CLred loss
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash << 'INNER'
cd /work/u10677113/LDNet_GLA/clean
mkdir -p results

# ---- W10/T2: currently QAD=0 (reversal-safe), test adding alpha_dot cost ----
echo =W10T2_QAD30=
DAMULT=3 SCHED=0 QAD=30 RW=1e-5 DLPF=0.85 NH=6 DMAX=1.7 RQUIET=1e-5 W0=10 TG=2.0 python3 -u mpc_gust.py

echo =W10T2_QAD100=
DAMULT=3 SCHED=0 QAD=100 RW=1e-5 DLPF=0.85 NH=6 DMAX=1.7 RQUIET=1e-5 W0=10 TG=2.0 python3 -u mpc_gust.py

echo =W10T2_QAD300=
DAMULT=3 SCHED=0 QAD=300 RW=1e-5 DLPF=0.85 NH=6 DMAX=1.7 RQUIET=1e-5 W0=10 TG=2.0 python3 -u mpc_gust.py

# ---- W20/T2: currently QAD=30, RQUIET=1e-5 ----
echo =W20T2_QAD100=
DAMULT=3 SCHED=0 QAD=100 RW=1e-3 DLPF=0.85 NH=6 DMAX=14 RQUIET=1e-5 W0=20 TG=2.0 python3 -u mpc_gust.py

echo =W20T2_QAD200=
DAMULT=3 SCHED=0 QAD=200 RW=1e-3 DLPF=0.85 NH=6 DMAX=14 RQUIET=1e-5 W0=20 TG=2.0 python3 -u mpc_gust.py

echo =W20T2_QAD300=
DAMULT=3 SCHED=0 QAD=300 RW=1e-3 DLPF=0.85 NH=6 DMAX=14 RQUIET=1e-5 W0=20 TG=2.0 python3 -u mpc_gust.py

# ---- W30/T2: currently QAD=30, RQUIET=0.1 ----
echo =W30T2_QAD100=
DAMULT=3 SCHED=0 QAD=100 RW=1e-2 DLPF=0.75 NH=6 DMAX=14 RQUIET=0.1 W0=30 TG=2.0 python3 -u mpc_gust.py

echo =W30T2_QAD200=
DAMULT=3 SCHED=0 QAD=200 RW=1e-2 DLPF=0.75 NH=6 DMAX=14 RQUIET=0.1 W0=30 TG=2.0 python3 -u mpc_gust.py

echo =W30T2_QAD300=
DAMULT=3 SCHED=0 QAD=300 RW=1e-2 DLPF=0.75 NH=6 DMAX=14 RQUIET=0.1 W0=30 TG=2.0 python3 -u mpc_gust.py

# ---- Regression check: current SCHED=1 values for all 3 cells ----
echo =W10T2_sched=
DAMULT=3 SCHED=1 W0=10 TG=2.0 python3 -u mpc_gust.py

echo =W20T2_sched=
DAMULT=3 SCHED=1 W0=20 TG=2.0 python3 -u mpc_gust.py

echo =W30T2_sched=
DAMULT=3 SCHED=1 W0=30 TG=2.0 python3 -u mpc_gust.py

echo =DONE=
INNER

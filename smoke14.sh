#!/bin/bash
# smoke14: W20/T0.5 and W30/T0.5 — ring-down fix + prop baseline
# Issue: short gust ends at 0.5s, RQUIET=0.1 releases flap, ring-down t=0.5-1.0s raises CLexc
# Fix hypothesis: RQUIET=1e-5 → R=1e-5 post-gust → active ring-down damping
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash << 'INNER'
cd /work/u10677113/LDNet_GLA/clean

# ---- Prop+D95 baseline (true standard comparison) ----
echo =prop_D95_W20T05=
DAMULT=3 SCHED=0 QAD=30 RW=1e-3 DLPF=0.95 NH=6 DMAX=14 PROP=1 GAIN=-40 W0=20 TG=0.5 python3 -u mpc_gust.py

echo =prop_D95_W30T05=
DAMULT=3 SCHED=0 QAD=30 RW=5e-3 DLPF=0.95 NH=6 DMAX=14 PROP=1 GAIN=-40 W0=30 TG=0.5 python3 -u mpc_gust.py

# ---- RQUIET=1e-5 fix: active ring-down damping after gust ----
echo =W20T05_RQ1e5=
DAMULT=3 SCHED=0 QAD=30 RW=1e-3 DLPF=0.75 NH=6 DMAX=14 RQUIET=1e-5 W0=20 TG=0.5 python3 -u mpc_gust.py

echo =W30T05_RQ1e5=
DAMULT=3 SCHED=0 QAD=30 RW=5e-3 DLPF=0.75 NH=6 DMAX=14 RQUIET=1e-5 W0=30 TG=0.5 python3 -u mpc_gust.py

# ---- DLPF=0.85 fix: smoother flap → less impulsive pitch excitation ----
echo =W20T05_D85=
DAMULT=3 SCHED=0 QAD=30 RW=1e-3 DLPF=0.85 NH=6 DMAX=14 RQUIET=0.1 W0=20 TG=0.5 python3 -u mpc_gust.py

echo =W30T05_D85=
DAMULT=3 SCHED=0 QAD=30 RW=5e-3 DLPF=0.85 NH=6 DMAX=14 RQUIET=0.1 W0=30 TG=0.5 python3 -u mpc_gust.py

# ---- Combined: RQUIET=1e-5 + DLPF=0.85 ----
echo =W20T05_RQ1e5_D85=
DAMULT=3 SCHED=0 QAD=30 RW=1e-3 DLPF=0.85 NH=6 DMAX=14 RQUIET=1e-5 W0=20 TG=0.5 python3 -u mpc_gust.py

echo =W30T05_RQ1e5_D85=
DAMULT=3 SCHED=0 QAD=30 RW=5e-3 DLPF=0.85 NH=6 DMAX=14 RQUIET=1e-5 W0=30 TG=0.5 python3 -u mpc_gust.py

echo =DONE=
INNER

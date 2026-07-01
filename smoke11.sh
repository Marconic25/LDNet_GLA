#!/bin/bash
# smoke11: W10/T2 RQUIET fix + standard prop baseline at DLPF=0.95 for all Tg=2 cells
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash << 'INNER'
cd /work/u10677113/LDNet_GLA/clean

# ---- Standard prop baseline (DLPF=0.95, no schedule) ----
# This is the true reference: prop with standard settings, not schedule-tuned.
# MPC must beat THESE values to satisfy "MPC deve battere tutti i prop gain".

echo =prop_D95_W10T2=
DAMULT=3 SCHED=0 QAD=0 RW=1e-5 DLPF=0.95 NH=6 DMAX=1.7 PROP=1 GAIN=-40 W0=10 TG=2.0 python3 -u mpc_gust.py

echo =prop_D95_W20T2=
DAMULT=3 SCHED=0 QAD=0 RW=1e-5 DLPF=0.95 NH=6 DMAX=14 PROP=1 GAIN=-40 W0=20 TG=2.0 python3 -u mpc_gust.py

echo =prop_D95_W30T2=
DAMULT=3 SCHED=0 QAD=0 RW=1e-5 DLPF=0.95 NH=6 DMAX=14 PROP=1 GAIN=-40 W0=30 TG=2.0 python3 -u mpc_gust.py

# ---- W10/T2: RQUIET sweep — lower RQUIET makes MPC start earlier ----
# RQUIET=0.1 (default): MPC starts at wn~0.91 (t~0.25s)
# RQUIET=0.01:          MPC starts at wn~0.61 (t~0.14s) — 110ms earlier
# RQUIET=1e-5 (=RW):    MPC starts immediately (wn~0)

echo =W10T2_RQ001=
DAMULT=3 SCHED=0 QAD=0 RW=1e-5 DLPF=0.85 NH=6 DMAX=1.7 RQUIET=0.01 W0=10 TG=2.0 python3 -u mpc_gust.py

echo =W10T2_RQ1e5=
DAMULT=3 SCHED=0 QAD=0 RW=1e-5 DLPF=0.85 NH=6 DMAX=1.7 RQUIET=1e-5 W0=10 TG=2.0 python3 -u mpc_gust.py

# ---- W10/T2: DMAX sweep with RQUIET=0.01 ----
echo =W10T2_RQ001_DM15=
DAMULT=3 SCHED=0 QAD=0 RW=1e-5 DLPF=0.85 NH=6 DMAX=1.5 RQUIET=0.01 W0=10 TG=2.0 python3 -u mpc_gust.py

echo =W10T2_RQ001_DM20=
DAMULT=3 SCHED=0 QAD=0 RW=1e-5 DLPF=0.85 NH=6 DMAX=2.0 RQUIET=0.01 W0=10 TG=2.0 python3 -u mpc_gust.py

# ---- W30/T2: try lower R to get more flap from MPC ----
# Current: R=0.01, flap=6 deg, +33%. Prop at DLPF=0.95 may give ~25-30%.
# Try R=5e-3 and R=2e-3 to get more flap without exploding.
echo =W30T2_R5e3=
DAMULT=3 SCHED=0 QAD=30 RW=5e-3 DLPF=0.75 NH=6 DMAX=14 RQUIET=0.1 W0=30 TG=2.0 python3 -u mpc_gust.py

echo =W30T2_R2e3=
DAMULT=3 SCHED=0 QAD=30 RW=2e-3 DLPF=0.75 NH=6 DMAX=14 RQUIET=0.1 W0=30 TG=2.0 python3 -u mpc_gust.py

echo =DONE=
INNER

#!/bin/bash
# smoke12: remaining W10/T2 RQUIET tests + W20/T2 amplitude fix
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash << 'INNER'
cd /work/u10677113/LDNet_GLA/clean

# ---- W10/T2: RQUIET sweep ----
# Lower RQUIET shifts wn breakpoint from ~0.91 to ~0.61 (110ms earlier start)
echo =W10T2_RQ001=
DAMULT=3 SCHED=0 QAD=0 RW=1e-5 DLPF=0.85 NH=6 DMAX=1.7 RQUIET=0.01 W0=10 TG=2.0 python3 -u mpc_gust.py

echo =W10T2_RQ1e5=
DAMULT=3 SCHED=0 QAD=0 RW=1e-5 DLPF=0.85 NH=6 DMAX=1.7 RQUIET=1e-5 W0=10 TG=2.0 python3 -u mpc_gust.py

echo =W10T2_RQ001_DM15=
DAMULT=3 SCHED=0 QAD=0 RW=1e-5 DLPF=0.85 NH=6 DMAX=1.5 RQUIET=0.01 W0=10 TG=2.0 python3 -u mpc_gust.py

echo =W10T2_RQ001_DM20=
DAMULT=3 SCHED=0 QAD=0 RW=1e-5 DLPF=0.85 NH=6 DMAX=2.0 RQUIET=0.01 W0=10 TG=2.0 python3 -u mpc_gust.py

# ---- W20/T2: amplitude fix — R too low causes 4deg overshoot, prop uses 2.5deg ----
# Target: R ~ 7.5e-3 to get ~2.5 deg peak (matching prop amplitude and hence CLred)
# Also test R=5e-3 and R=1e-2 to bracket the sweet spot.
echo =W20T2_R7e3=
DAMULT=3 SCHED=0 QAD=30 RW=7.5e-3 DLPF=0.85 NH=6 DMAX=14 RQUIET=0.1 W0=20 TG=2.0 python3 -u mpc_gust.py

echo =W20T2_R5e3=
DAMULT=3 SCHED=0 QAD=30 RW=5e-3 DLPF=0.85 NH=6 DMAX=14 RQUIET=0.1 W0=20 TG=2.0 python3 -u mpc_gust.py

echo =W20T2_R1e2=
DAMULT=3 SCHED=0 QAD=30 RW=1e-2 DLPF=0.85 NH=6 DMAX=14 RQUIET=0.1 W0=20 TG=2.0 python3 -u mpc_gust.py

# ---- W20/T2 regression check at current SCHED=1 settings ----
echo =W20T2_sched=
DAMULT=3 SCHED=1 W0=20 TG=2.0 python3 -u mpc_gust.py

echo =DONE=
INNER

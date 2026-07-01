#!/bin/bash
# smoke10: W10/T2 — DMAX=1.7 amplitude-limited approach + prop comparison
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash << 'INNER'
cd /work/u10677113/LDNet_GLA/clean

# 1. New schedule (SCHED=1): Q=0, R=1e-5, NH=6, DMAX=1.7 for weak+long — prop-like behavior
echo =sched_new=
DAMULT=3 SCHED=1 PROP=1 GAIN=-40 W0=10 TG=2.0 python3 -u mpc_gust.py

# 2. Same settings but explicit (sanity check)
echo =Qzero_R1e5_DMAX17_NH6=
DAMULT=3 SCHED=0 QAD=0 RW=1e-5 DLPF=0.85 NH=6 DMAX=1.7 W0=10 TG=2.0 python3 -u mpc_gust.py

# 3. DMAX=1.7 but with Q_alpha_dot=30 (see if pitch penalty helps or hurts)
echo =Q30_R1e5_DMAX17_NH6=
DAMULT=3 SCHED=0 QAD=30 RW=1e-5 DLPF=0.85 NH=6 DMAX=1.7 W0=10 TG=2.0 python3 -u mpc_gust.py

# 4. Regression: W20/T2 with new schedule (should be back to ~+70%)
echo =W20T2_reg=
DAMULT=3 SCHED=1 PROP=1 GAIN=-40 W0=20 TG=2.0 python3 -u mpc_gust.py

# 5. Regression: W30/T2 with new schedule (should be back to ~+33%)
echo =W30T2_reg=
DAMULT=3 SCHED=1 PROP=1 GAIN=-40 W0=30 TG=2.0 python3 -u mpc_gust.py

echo =DONE=
INNER

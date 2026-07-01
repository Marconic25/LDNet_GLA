#!/bin/bash
# smoke13: W20/T2 ring-down damping via inverted RQUIET (RQUIET<RW)
# Hypothesis: RQUIET=1e-5 < RW=1e-3 inverts R schedule so post-gust R=1e-5 (aggressive)
# while during-gust R=1e-3 (unchanged). Damps ring-down CL, same as W10/T2 fix.
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash << 'INNER'
cd /work/u10677113/LDNet_GLA/clean

# W20/T2 with inverted RQUIET (=1e-5 < RW=1e-3): ring-down damping, same during-gust amplitude
echo =W20T2_RQ1e5=
DAMULT=3 SCHED=0 QAD=30 RW=1e-3 DLPF=0.85 NH=6 DMAX=14 RQUIET=1e-5 W0=20 TG=2.0 python3 -u mpc_gust.py

# W20/T2 reference: RQUIET=RW=1e-3 (effectively no scheduling effect, constant R=1e-3)
echo =W20T2_RQ1e3=
DAMULT=3 SCHED=0 QAD=30 RW=1e-3 DLPF=0.85 NH=6 DMAX=14 RQUIET=1e-3 W0=20 TG=2.0 python3 -u mpc_gust.py

# W30/T2 regression check with same RQUIET trick (R=1e-2, RQ=1e-5)
echo =W30T2_RQ1e5=
DAMULT=3 SCHED=0 QAD=30 RW=1e-2 DLPF=0.75 NH=6 DMAX=14 RQUIET=1e-5 W0=30 TG=2.0 python3 -u mpc_gust.py

echo =DONE=
INNER

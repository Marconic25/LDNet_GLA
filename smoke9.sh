#!/bin/bash
# smoke9: W10/T2 diagnosis — prop baseline + MPC parameter sweep
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash << 'INNER'
cd /work/u10677113/LDNet_GLA/clean

# 1. Prop baseline with SCHED=1 settings (DLPF=0.85, DAMULT=3)
echo =PROP_baseline=
DAMULT=3 SCHED=1 PROP=1 GAIN=-40 W0=10 TG=2.0 python3 -u mpc_gust.py

# 2. MPC: no alpha_dot penalty, R=1e-3, NH=30 — pure CL minimizer
echo =Qzero_R1e3_NH30=
DAMULT=3 SCHED=0 QAD=0 RW=0.001 DLPF=0.85 NH=30 W0=10 TG=2.0 python3 -u mpc_gust.py

# 3. MPC: no alpha_dot penalty, R=1e-4, NH=30 — aggressive pure CL minimizer
echo =Qzero_R1e4_NH30=
DAMULT=3 SCHED=0 QAD=0 RW=0.0001 DLPF=0.85 NH=30 W0=10 TG=2.0 python3 -u mpc_gust.py

# 4. MPC: no alpha_dot penalty, R=1e-4, NH=6 — aggressive, short horizon
echo =Qzero_R1e4_NH6=
DAMULT=3 SCHED=0 QAD=0 RW=0.0001 DLPF=0.85 NH=6 W0=10 TG=2.0 python3 -u mpc_gust.py

# 5. MPC: intermediate R to find sweet spot between 0 flap and 3.2 deg
echo =R3e3_NH30=
DAMULT=3 SCHED=0 QAD=30 RW=0.003 DLPF=0.85 NH=30 W0=10 TG=2.0 python3 -u mpc_gust.py

echo =DONE=
INNER

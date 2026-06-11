#!/bin/bash
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash -c 'pip install scipy matplotlib h5py -q 2>/dev/null; cd /work/u10677113/LDNet_GLA/clean && mkdir -p results/mpc_plots
echo "### scheduled MPC plots (W0 x Tg) ###"
for W in 10 20 30; do for T in 0.5 1.0 2.0; do
  W0=$W TG=$T TEND=3.0 SCHED=1 PLOT=1 OUT=results/mpc_plots/gust python3 -u mpc_gust.py
done; done
W0=11.46 TG=1.12 TEND=3.0 SCHED=1 PLOT=1 OUT=results/mpc_plots/gust python3 -u mpc_gust.py'
echo PLOTS_DONE

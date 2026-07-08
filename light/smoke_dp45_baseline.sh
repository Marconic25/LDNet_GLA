#!/bin/bash
# Smoke regression: open + optimal (R=3e-4, R=1e-4) + combo oracle clean
# at W30/Tg0.4 DAMULT=3, dp45 tree. Run from cluster light/ dir.
cd /work/u10677113/LDNet_GLA/light
APP="apptainer exec --writable-tmpfs --env PYTHONNOUSERSITE=1 --env DAMULT=3 \
  --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash -c "pip install -q scipy h5py matplotlib; python3 -s -u -c \"
import run as R
OL  = R.simulate('open', 30, 0.4)
OPT3 = R.simulate('optimal', 30, 0.4, R=3e-4)
OPT1 = R.simulate('optimal', 30, 0.4, R=1e-4)
COM  = R.simulate('combo',   30, 0.4, R=3e-4, NH=8)
mo3 = R.metrics(OPT3, OL, 0.4); mo1 = R.metrics(OPT1, OL, 0.4)
mc  = R.metrics(COM,  OL, 0.4)
import numpy as np
cex0 = float(np.max(np.abs(OL['CL'] - R.CLTRIM)))
print(f'open cex0     = {cex0:.4f}   (rk4 ref: 0.4600)')
print(f'optimal R=3e-4: {mo3[\"clred\"]:+.2f}%  (rk4 ref: +76.58%)')
print(f'optimal R=1e-4: {mo1[\"clred\"]:+.2f}%  (rk4 ref: +80.67%)')
print(f'combo   R=3e-4: {mc[\"clred\"]:+.2f}%   (rk4 ref: +80.5%)')
print('# DONE')
\""

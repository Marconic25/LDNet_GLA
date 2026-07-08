#!/usr/bin/env python3
"""dp45 baseline smoke: open + optimal (R=3e-4, R=1e-4) + combo oracle clean
at W30/Tg0.4 DAMULT=3. Print 4 anchor numbers then # DONE."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R
import numpy as np

OL   = R.simulate('open',    30, 0.4)
OPT3 = R.simulate('optimal', 30, 0.4, R=3e-4)
OPT1 = R.simulate('optimal', 30, 0.4, R=1e-4)
COM  = R.simulate('combo',   30, 0.4, R=3e-4, NH=8)

mo3 = R.metrics(OPT3, OL, 0.4)
mo1 = R.metrics(OPT1, OL, 0.4)
mc  = R.metrics(COM,  OL, 0.4)

cex0 = float(np.max(np.abs(OL['CL'] - R.CLTRIM)))
print(f'open cex0     = {cex0:.4f}   (rk4 ref: 0.4600)', flush=True)
print(f'optimal R=3e-4: {mo3["clred"]:+.2f}%  (rk4 ref: +76.58%)', flush=True)
print(f'optimal R=1e-4: {mo1["clred"]:+.2f}%  (rk4 ref: +80.67%)', flush=True)
print(f'combo   R=3e-4: {mc["clred"]:+.2f}%   (rk4 ref: +80.5%)', flush=True)
print('# DONE', flush=True)

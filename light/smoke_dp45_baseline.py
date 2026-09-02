#!/usr/bin/env python3
"""dp45 baseline smoke: open + combo oracle clean (MPC N=8, R=3e-4)
at W30/Tg0.4 DAMULT=3. Print 2 anchor numbers then # DONE."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R
import numpy as np

OL  = R.simulate('open',  30, 0.4)
COM = R.simulate('combo', 30, 0.4, R=3e-4, NH=8)

mc = R.metrics(COM, OL, 0.4)

cex0 = float(np.max(np.abs(OL['CL'] - R.CLTRIM)))
print(f'open cex0     = {cex0:.4f}   (rk4 ref: 0.4600)', flush=True)
print(f'combo   R=3e-4: {mc["clred"]:+.2f}%   (rk4 ref: +80.5%)', flush=True)
print('# DONE', flush=True)

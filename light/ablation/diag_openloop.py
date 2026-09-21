"""
Open-loop COUPLED check: run the full aeroelastic simulation (no control)
with each aerodynamic model as the plant, and compare.

The fixed-airfoil probes were misleading: holding the structure at trim
suppresses the LDNet's response (its latent dynamics are driven largely by
the structural motion) while leaving the linear model's direct gust term
untouched. The meaningful question is what each model does when it is
actually coupled to the structure, which is how both are used in the loop.

This reproduces run.py's 'open' arm with a swappable aero model and reports
the peak C_L excursion and the structural response for each.
"""
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'light'))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import structure
structure.D_ALPHA *= float(os.environ.get('DAMULT', '3'))

from linear_aero import LinearUnsteadyAero
from ldnet_aero import LDNetAero

U, RHO, C, DT, S = 80.0, 1.225, 1.0, 0.002, 0.05
q = 0.5 * RHO * U ** 2 * S
X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])


def gust(t, W0, Tg):
    return (W0 / 2.0) * (1 - np.cos(2 * np.pi * t / Tg)) if (0 <= t <= Tg) else 0.0


def simulate(aero, W0, Tg, TEND=3.0):
    n = int(round(TEND / DT)) + 1
    ts = np.arange(n) * DT
    Wt = np.array([gust(t, W0, Tg) for t in ts])
    aero.reset(dt=DT)
    x = X0.copy()
    CL = np.zeros(n); AL = np.zeros(n); AD = np.zeros(n); H = np.zeros(n)
    for i in range(n):
        cl, cm = aero.predict(x, 0.0, float(Wt[i]), U)
        Fy = q * cl; Mz = q * cm * C
        aero.advance(x, 0.0, float(Wt[i]), U, DT)
        x = structure.step_dp45(x, Fy, Mz, DT)
        CL[i] = cl; AL[i] = x[2]; AD[i] = x[3]; H[i] = x[0]
    return ts, Wt, CL, AL, AD, H


def main():
    lin = LinearUnsteadyAero(coeff_path=os.path.join(HERE, 'linear_coeffs.json'))
    ld = LDNetAero(os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10'))

    cases = [(10, 0.4), (20, 0.7), (30, 0.4), (30, 0.3), (30, 1.2)]
    print(f'{"case":>12s}{"model":>8s}{"CLtrim":>9s}{"CLexc":>9s}'
          f'{"|a|max":>10s}{"|ad|max":>10s}{"|h|max":>10s}')
    for (W0, Tg) in cases:
        for name, aero in [('lin', lin), ('LDNet', ld)]:
            ts, Wt, CL, AL, AD, H = simulate(aero, W0, Tg)
            mw = ts <= (Tg + 0.5)
            trim = CL[0]
            exc = np.max(np.abs(CL[mw] - trim))
            print(f'{f"W{W0}/T{Tg}":>12s}{name:>8s}{trim:>9.4f}{exc:>9.4f}'
                  f'{np.max(np.abs(AL[mw])):>10.5f}{np.max(np.abs(AD[mw])):>10.4f}'
                  f'{np.max(np.abs(H[mw])):>10.5f}')
        print()


if __name__ == '__main__':
    main()

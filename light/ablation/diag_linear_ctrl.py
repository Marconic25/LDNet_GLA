"""
Scrutinise the linear-controller failure before accepting it.

The linear arm returned -3.7% CLred with the flap pinned at 14 deg and
pitch_ratio 2.94. A result that extreme must be checked for an implementation
fault -- above all a sign error -- rather than reported as a physics finding.
If the linear model has the wrong sign for dC_L/ddelta, its MPC would push the
flap the wrong way and saturate, which is exactly the observed signature.

Checks
------
1. Sign and magnitude of dC_L/ddelta for BOTH models along a real gust
   trajectory (not at trim, where the earlier static probe was misleading):
   if they disagree in sign, the failure is a bug.
2. The flap command time history from both controllers on the same cell:
   does the linear one push the correct direction and merely overshoot, or
   does it invert?
3. What the linear controller BELIEVES it is achieving (its own predicted
   C_L over the horizon) versus what the plant actually delivers -- the
   model-mismatch signature.
"""
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'light'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import structure
structure.D_ALPHA *= 3.0

from ldnet_aero import LDNetAero
from linear_aero import LinearUnsteadyAero
from optimal import MPCPreviewController

U, RHO, C, DT, S = 80.0, 1.225, 1.0, 0.002, 0.05
q = 0.5 * RHO * U ** 2 * S
X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
MD = os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10')


def gust(t, W0, Tg):
    return (W0 / 2.0) * (1 - np.cos(2 * np.pi * t / Tg)) if (0 <= t <= Tg) else 0.0


def flap_sensitivity_along_trajectory(W0=30, Tg=0.4):
    """dC_L/ddelta for both models at states visited during the real gust."""
    ld = LDNetAero(MD); ld.reset(dt=DT)
    lin = LinearUnsteadyAero(coeff_path=os.path.join(HERE, 'linear_coeffs.json'))
    lin.reset(dt=DT)

    n = int(round(3.0 / DT)) + 1
    ts = np.arange(n) * DT
    Wt = np.array([gust(t, W0, Tg) for t in ts])
    x = X0.copy()
    print('  dC_L/ddelta [1/deg] at states along the OPEN-LOOP gust:')
    print(f'  {"t":>7s}{"W":>7s}{"LDNet":>11s}{"linear":>11s}{"ratio":>9s}')
    for i in range(n):
        Wi = float(Wt[i])
        if i % 40 == 0 and i <= int(1.0 / DT):
            a1 = ld.predict(x, -1.0, Wi, U)[0]; b1 = ld.predict(x, 1.0, Wi, U)[0]
            a2 = lin.predict(x, -1.0, Wi, U)[0]; b2 = lin.predict(x, 1.0, Wi, U)[0]
            s_n = (b1 - a1) / 2.0
            s_l = (b2 - a2) / 2.0
            print(f'  {ts[i]:>7.3f}{Wi:>7.2f}{s_n:>11.5f}{s_l:>11.5f}'
                  f'{(s_n/s_l if s_l else float("nan")):>9.2f}')
        cl, cm = ld.predict(x, 0.0, Wi, U)
        ld.advance(x, 0.0, Wi, U, DT)
        lin.advance(x, 0.0, Wi, U, DT)
        x = structure.step_dp45(x, q * cl, q * cm * C, DT)


def flap_histories(W0=30, Tg=0.4, R=3e-4, TEND=1.2):
    """Flap command from each controller, plant = LDNet in both."""
    out = {}
    for kind in ['ldnet', 'linear']:
        p = LDNetAero(MD); p.reset(dt=DT)
        CLTRIM = float(p.predict(X0, 0., 0., U)[0])
        if kind == 'ldnet':
            internal = LDNetAero(MD); internal.reset(dt=DT)
        else:
            internal = LinearUnsteadyAero(
                coeff_path=os.path.join(HERE, 'linear_coeffs.json'))
            internal.reset(dt=DT)
        ctrl = MPCPreviewController(internal, U=U, dt=DT, rho=RHO, S=S, C=C,
                                    C_L_trim=CLTRIM, N=8, R=R, R_du=0.0,
                                    G=161, delta_max=14., delta_dot_max=300.)
        ctrl.reset()
        n = int(round(TEND / DT)) + 1
        ts = np.arange(n) * DT
        Wt = np.array([gust(t, W0, Tg) for t in ts])
        x = X0.copy()
        de_h = np.zeros(n); cl_h = np.zeros(n)
        for i in range(n):
            Wi = float(Wt[i])
            lo = i + 1; hi = min(i + 9, n)
            w = np.zeros(8); w[:hi - lo] = Wt[lo:hi]
            de = ctrl.compute(x, w, Wi)
            ctrl._delta_prev = de
            cl, cm = p.predict(x, de, Wi, U)
            p.advance(x, de, Wi, U, DT)
            x = structure.step_dp45(x, q * cl, q * cm * C, DT)
            de_h[i] = de; cl_h[i] = cl
        out[kind] = (ts, Wt, de_h, cl_h, CLTRIM)

    ts, Wt, de_n, cl_n, trim = out['ldnet']
    _, _, de_l, cl_l, _ = out['linear']
    print(f'\n  flap command and resulting C_L (trim={trim:.4f}):')
    print(f'  {"t":>7s}{"W":>7s}{"de_LDNet":>10s}{"de_lin":>9s}'
          f'{"CL_LDNet":>10s}{"CL_lin":>9s}')
    for i in range(0, len(ts), max(1, len(ts) // 20)):
        print(f'  {ts[i]:>7.3f}{Wt[i]:>7.2f}{de_n[i]:>10.2f}{de_l[i]:>9.2f}'
              f'{cl_n[i]:>10.4f}{cl_l[i]:>9.4f}')


if __name__ == '__main__':
    print('=== 1. flap sensitivity along trajectory ===')
    flap_sensitivity_along_trajectory()
    print('\n=== 2. flap command histories (W30/Tg0.4) ===')
    flap_histories()

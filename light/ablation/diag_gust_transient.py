"""
Check the linear model's GUST response in a realistic transient.

The static gust sweep produced C_L = -4.2 at W = 40 m/s, which is absurd, but
that operating point is not one the model is meant to cover: with the airfoil
held fixed the Kussner lag states stay at zero, so the raw gust downwash gets
the full CL_g = -10.05 with no wake attenuation. The gain was identified on
trajectories where those lags are active.

This script replays the actual 1-cosine gust used in chapter 3 through the
lag dynamics (airfoil held at trim, flap zero) and reports C_L(t) for the
linear model against the LDNet. If the linear model is sane here, the static
sweep was a meaningless probe; if it is still wild, the gust path is wrong
and must be fixed before any closed-loop run.
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

from linear_aero import LinearUnsteadyAero
from ldnet_aero import LDNetAero

U = 80.0
DT = 0.002
X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])


def gust(t, W0, Tg):
    return (W0 / 2.0) * (1 - np.cos(2 * np.pi * t / Tg)) if (0 <= t <= Tg) else 0.0


def run(W0, Tg, lin, ld):
    T = 1.5 * Tg
    n = int(T / DT)
    lin.reset(dt=DT); ld.reset(dt=DT)
    rows = []
    for i in range(n):
        t = i * DT
        W = gust(t, W0, Tg)
        cl_l = lin.predict(X0, 0.0, W, U)[0]
        cl_n = ld.predict(X0, 0.0, W, U)[0]
        rows.append((t, W, cl_l, cl_n))
        lin.advance(X0, 0.0, W, U, DT)
        ld.advance(X0, 0.0, W, U, DT)
    return np.array(rows)


def main():
    lin = LinearUnsteadyAero(coeff_path=os.path.join(HERE, 'linear_coeffs.json'))
    ld = LDNetAero(os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10'))

    for (W0, Tg) in [(10, 0.4), (20, 0.7), (30, 0.4), (30, 1.2)]:
        r = run(W0, Tg, lin, ld)
        cl_l, cl_n = r[:, 2], r[:, 3]
        print(f'=== W0={W0} Tg={Tg} ===')
        print(f'  linear C_L range [{cl_l.min():+.4f}, {cl_l.max():+.4f}]'
              f'  peak excursion {np.max(np.abs(cl_l - cl_l[0])):.4f}')
        print(f'  LDNet  C_L range [{cl_n.min():+.4f}, {cl_n.max():+.4f}]'
              f'  peak excursion {np.max(np.abs(cl_n - cl_n[0])):.4f}')
        k = max(1, len(r) // 8)
        print(f'  {"t":>7s}{"W":>8s}{"CL_lin":>10s}{"CL_LDNet":>10s}')
        for i in range(0, len(r), k):
            print(f'  {r[i,0]:>7.3f}{r[i,1]:>8.2f}{r[i,2]:>10.4f}{r[i,3]:>10.4f}')
        print()


if __name__ == '__main__':
    main()

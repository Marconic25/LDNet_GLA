"""
Open-loop comparison of the fitted linear unsteady model against the LDNet,
BEFORE either is put inside the control loop.

Purpose: establish that the linear baseline is a competent model in the
regime it claims to cover, so that any closed-loop gap is attributable to the
physics it omits and not to a botched implementation. Also quantifies the
flap nonlinearity directly -- the LDNet's dC_L/ddelta is not constant, which
is exactly what the linear structure cannot represent.

Outputs
-------
1. Static flap sweep at trim: C_L(delta) for both models, plus the local
   slope, showing where the linear secant diverges from the LDNet tangent.
2. Static gust sweep at trim.
3. Trajectory replay: both models driven with the SAME recorded inputs from
   held-out test trajectories, reporting NRMSE against the CFD targets.
"""
import json
import os
import sys
import numpy as np
import h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'light'))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from linear_aero import LinearUnsteadyAero
from ldnet_aero import LDNetAero

RHO, S, C = 1.225, 0.05, 1.0
U = 80.0
X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])


def static_sweeps(lin, ld):
    print('=== static flap sweep at trim (W=0) ===')
    print(f'{"delta":>8s}{"C_L lin":>12s}{"C_L LDNet":>12s}{"diff":>10s}')
    prev = None
    for d in [-14, -10, -7, -5, -2, 0, 2, 5, 7, 10, 14]:
        cl_l, _ = lin.predict(X0, d, 0.0, U)
        cl_n, _ = ld.predict(X0, d, 0.0, U)
        print(f'{d:>8.1f}{cl_l:>12.5f}{cl_n:>12.5f}{cl_l-cl_n:>10.5f}')

    print('\n  local LDNet slope dC_L/ddelta [1/deg] (central diff):')
    for d in [-12, -8, -4, 0, 4, 8, 12]:
        a, _ = ld.predict(X0, d - 1.0, 0.0, U)
        b, _ = ld.predict(X0, d + 1.0, 0.0, U)
        sl_n = (b - a) / 2.0
        a2, _ = lin.predict(X0, d - 1.0, 0.0, U)
        b2, _ = lin.predict(X0, d + 1.0, 0.0, U)
        sl_l = (b2 - a2) / 2.0
        print(f'    delta={d:+5.1f}  LDNet {sl_n:+.5f}   linear {sl_l:+.5f}'
              f'   ratio {sl_n/sl_l if sl_l else float("nan"):+.3f}')

    print('\n=== static gust sweep at trim (delta=0) ===')
    print(f'{"W":>8s}{"C_L lin":>12s}{"C_L LDNet":>12s}{"diff":>10s}')
    for W in [0, 5, 10, 20, 30, 40]:
        cl_l, _ = lin.predict(X0, 0.0, W, U)
        cl_n, _ = ld.predict(X0, 0.0, W, U)
        print(f'{W:>8.1f}{cl_l:>12.5f}{cl_n:>12.5f}{cl_l-cl_n:>10.5f}')


def replay(lin, ld, n_traj=6, stride=1):
    """Drive both models with identical recorded inputs; compare to CFD."""
    with h5py.File(os.path.join(ROOT, 'data', 'GLA_test.h5'), 'r') as f:
        sig = np.array(f['input_signals'])
        out = np.array(f['output_signals'])
        par = np.array(f['input_parameters'])
        times = np.array(f['times'])
    dt = float(times[1] - times[0])

    print(f'\n=== trajectory replay on held-out test split '
          f'(first {n_traj} trajectories, dt={dt}) ===')
    print(f'{"traj":>6s}{"NRMSE lin":>12s}{"NRMSE LDNet":>14s}{"ratio":>9s}')

    tot_l, tot_n = [], []
    for n in range(min(n_traj, sig.shape[0])):
        h, hd, a, ad, delta, W = [sig[n, :, j] for j in range(6)]
        Uf = max(float(par[n, 0]), 1.0)
        q = 0.5 * RHO * Uf ** 2 * S
        CL_true = out[n, :, 0, 0] / q

        lin.reset(dt=dt)
        ld.reset(dt=dt)
        T = len(h)
        cl_l = np.zeros(T); cl_n = np.zeros(T)
        for i in range(0, T, stride):
            st = (h[i], hd[i], a[i], ad[i])
            cl_l[i] = lin.predict(st, delta[i], W[i], Uf)[0]
            cl_n[i] = ld.predict(st, delta[i], W[i], Uf)[0]
            lin.advance(st, delta[i], W[i], Uf, dt * stride)
            ld.advance(st, delta[i], W[i], Uf, dt * stride)
        idx = np.arange(0, T, stride)
        rng = CL_true[idx].max() - CL_true[idx].min()
        nl = np.sqrt(np.mean((cl_l[idx] - CL_true[idx]) ** 2)) / rng
        nn = np.sqrt(np.mean((cl_n[idx] - CL_true[idx]) ** 2)) / rng
        tot_l.append(nl); tot_n.append(nn)
        print(f'{n:>6d}{nl:>12.4f}{nn:>14.4f}{nl/nn if nn else float("nan"):>9.2f}')

    print(f'{"mean":>6s}{np.mean(tot_l):>12.4f}{np.mean(tot_n):>14.4f}'
          f'{np.mean(tot_l)/np.mean(tot_n):>9.2f}')


def main():
    which = os.environ.get('COEFFS', 'linear_coeffs.json')
    lin = LinearUnsteadyAero(coeff_path=os.path.join(HERE, which))
    ld = LDNetAero(os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10'))
    lin.reset(dt=0.002); ld.reset(dt=0.002)
    print(f'[coeffs] {which}\n')
    static_sweeps(lin, ld)
    replay(lin, ld, n_traj=int(os.environ.get('NTRAJ', '6')),
           stride=int(os.environ.get('RSTRIDE', '7')))


if __name__ == '__main__':
    main()

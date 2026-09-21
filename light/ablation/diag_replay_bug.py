"""
Find why the CFD forecast probe reported an LDNet RMS error of 0.63 -- larger
than the whole C_L range, and impossible to reconcile with the verified
+80.5% closed-loop result from the same checkpoint.

Hypotheses
----------
H1  The test split's dt (2.8e-4 s) is ~7x finer than the LDNet's training
    dt_ref (2e-3 s). advance() sub-steps with n_sub = round(dt/dt_sub), which
    for dt < dt_sub rounds to 1 and then scales dz by dt/dt_ref ~ 0.14 -- the
    latent is driven at the wrong rate, so z never reaches the right state.
    The existing _warmup_from_csv() explicitly subsamples by exactly this
    stride, which is strong evidence the model must be driven at dt_ref.
H2  A 0-step (not 8-step) comparison would already be wrong, which would
    point at normalisation rather than latent integration.

This script measures the 0-step reconstruction error at the true states for
both models, then repeats the LDNet at the correct stride, to separate the
two.
"""
import os
import sys
import numpy as np
import h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'light'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

from linear_aero import LinearUnsteadyAero
from ldnet_aero import LDNetAero

RHO, S = 1.225, 0.05
MD = os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10')


def main():
    with h5py.File(os.path.join(ROOT, 'data', 'GLA_test.h5'), 'r') as f:
        sig = np.array(f['input_signals'][0])
        out = np.array(f['output_signals'][0])
        par = float(np.array(f['input_parameters'])[0, 0])
        times = np.array(f['times'])
    dt_raw = float(times[1] - times[0])
    U = max(par, 1.0)
    q = 0.5 * RHO * U ** 2 * S
    CL_true = out[:, 0, 0] / q

    ld = LDNetAero(MD)
    print(f'raw dt = {dt_raw:.6g} s,  LDNet dt_ref = {ld._dt_ref:.6g} s,'
          f'  ratio = {ld._dt_ref/dt_raw:.2f}')
    print(f'CL_true range [{CL_true.min():.4f}, {CL_true.max():.4f}]\n')

    h, hd, a, ad, delta, W = [sig[:, j] for j in range(6)]

    # --- 0-step reconstruction while advancing at the RAW dt (the bug) -------
    ld.reset(dt=dt_raw)
    errs = []
    for i in range(0, 4000):
        st = (h[i], hd[i], a[i], ad[i])
        cl = ld.predict(st, delta[i], W[i], U)[0]
        errs.append(cl - CL_true[i])
        ld.advance(st, delta[i], W[i], U, dt_raw)
    print(f'LDNet 0-step RMS, advancing at raw dt : '
          f'{np.sqrt(np.mean(np.square(errs))):.5f}')

    # --- 0-step reconstruction at the TRAINING stride -----------------------
    stride = max(1, round(ld._dt_ref / dt_raw))
    ld.reset(dt=ld._dt_ref)
    errs2 = []
    idx = list(range(0, 4000, stride))
    for i in idx:
        st = (h[i], hd[i], a[i], ad[i])
        cl = ld.predict(st, delta[i], W[i], U)[0]
        errs2.append(cl - CL_true[i])
        ld.advance(st, delta[i], W[i], U, ld._dt_ref)
    print(f'LDNet 0-step RMS, advancing at dt_ref (stride={stride}): '
          f'{np.sqrt(np.mean(np.square(errs2))):.5f}')

    # --- same for the linear model, which has no such constraint ------------
    lin = LinearUnsteadyAero(coeff_path=os.path.join(HERE, 'linear_coeffs.json'))
    lin.reset(dt=dt_raw)
    errs3 = []
    for i in range(0, 4000):
        st = (h[i], hd[i], a[i], ad[i])
        errs3.append(lin.predict(st, delta[i], W[i], U)[0] - CL_true[i])
        lin.advance(st, delta[i], W[i], U, dt_raw)
    print(f'linear 0-step RMS, raw dt             : '
          f'{np.sqrt(np.mean(np.square(errs3))):.5f}')

    lin.reset(dt=ld._dt_ref)
    errs4 = []
    for i in idx:
        st = (h[i], hd[i], a[i], ad[i])
        errs4.append(lin.predict(st, delta[i], W[i], U)[0] - CL_true[i])
        lin.advance(st, delta[i], W[i], U, ld._dt_ref)
    print(f'linear 0-step RMS, dt_ref stride      : '
          f'{np.sqrt(np.mean(np.square(errs4))):.5f}')

    print(f'\ndelta range on this trajectory: '
          f'[{delta.min():.3f}, {delta.max():.3f}] deg')
    print(f'frac |delta|>2 deg: {np.mean(np.abs(delta)>2):.3f}')


if __name__ == '__main__':
    main()

"""
Give the linear baseline its best possible shot: refit the flap gain in the
regime that actually matters for control, and verify closed-loop.

Motivation
----------
The MPC-internals probe showed the linear controller's failure is a
systematic OVER-COMMAND: it asks for 2-5x more flap than the plant needs,
because its constant dC_L/ddelta (+0.0253 /deg) underestimates the LDNet's
state-dependent value (0.018-0.047, typically ~0.030-0.038 near trim). Since
that is a GAIN error, it is only fair to ask whether correcting the gain
rescues the linear model. If it does, the earlier collapse was a tuning
artefact and must not be reported as evidence of nonlinearity.

Three fairer variants are produced:

  A 'ctrlgain' : CL_d rescaled to match the plant's flap sensitivity measured
                 near the operating point (delta in [-5,5] deg, at trim).
                 This is the single most favourable constant a linear model
                 could carry for this loop.
  B 'lsq_flap' : CL_d refitted by least squares on the training data using
                 ONLY samples with |delta| > 2 deg, so the flap gain is
                 estimated where the flap is actually active rather than
                 dominated by the delta ~ 0 bulk.
  C 'secant'   : CL_d set to the secant slope of the plant's C_L(delta) over
                 the full +-14 deg range -- the best constant approximation
                 in a large-deflection sense.

Each is written as its own coefficient file so the sweep can be re-run
against whichever is strongest. Reporting the ablation against a baseline
that has NOT had this treatment would overstate the LDNet's advantage.
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'light'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import numpy as np
from ldnet_aero import LDNetAero

MD = os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10')
X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
U = 80.0


def plant_flap_sensitivity(rng_deg=5.0, n=21):
    """Mean dC_L/ddelta of the plant near trim, over +-rng_deg."""
    ld = LDNetAero(MD); ld.reset(dt=0.002)
    ds = np.linspace(-rng_deg, rng_deg, n)
    cl = np.array([ld.predict(X0, float(d), 0.0, U)[0] for d in ds])
    # least-squares slope over the window, per degree
    A = np.column_stack([ds, np.ones_like(ds)])
    b, *_ = np.linalg.lstsq(A, cl, rcond=None)
    return float(b[0]), ds, cl


def secant_slope(rng_deg=14.0):
    ld = LDNetAero(MD); ld.reset(dt=0.002)
    a = ld.predict(X0, -rng_deg, 0.0, U)[0]
    b = ld.predict(X0, +rng_deg, 0.0, U)[0]
    return float((b - a) / (2 * rng_deg))


def lsq_flap_gain(min_deg=2.0, stride=10):
    """Refit CL_d on training samples where the flap is actually moving."""
    from fit_linear import build
    X, yL, _ = build('train', stride=stride)
    m = np.abs(np.rad2deg(X[:, 2])) > min_deg
    print(f'  [lsq_flap] {m.sum()} / {len(m)} samples with |delta| > {min_deg} deg')
    base = json.load(open(os.path.join(HERE, 'linear_coeffs.json')))
    # hold the other gains fixed, refit only CL_d and the offset on this subset
    off = (base['CL_a'] * X[m, 0] + base['CL_g'] * X[m, 1]
           + base['AM_L'] * X[m, 3])
    A = np.column_stack([X[m, 2], np.ones(int(m.sum()))])
    b, *_ = np.linalg.lstsq(A, yL[m] - off, rcond=None)
    return float(b[0]), float(b[1])


def main():
    base = json.load(open(os.path.join(HERE, 'linear_coeffs.json')))
    cur_per_deg = base['CL_d'] * np.pi / 180.0
    print(f'current linear CL_d = {base["CL_d"]:.5f} /rad '
          f'= {cur_per_deg:.5f} /deg')

    s_near, ds, cl = plant_flap_sensitivity(5.0)
    s_sec = secant_slope(14.0)
    print(f'plant dC_L/ddelta near trim (+-5 deg) = {s_near:.5f} /deg')
    print(f'plant secant slope over +-14 deg      = {s_sec:.5f} /deg')

    variants = {}

    a = dict(base)
    a['CL_d'] = float(s_near * 180.0 / np.pi)
    a['_note'] = 'CL_d matched to plant sensitivity near trim (+-5 deg)'
    variants['linear_coeffs_ctrlgain.json'] = a

    c = dict(base)
    c['CL_d'] = float(s_sec * 180.0 / np.pi)
    c['_note'] = 'CL_d = plant secant slope over +-14 deg'
    variants['linear_coeffs_secant.json'] = c

    g, off = lsq_flap_gain()
    b = dict(base)
    b['CL_d'] = g
    b['CL_0'] = off
    b['_note'] = 'CL_d refit on |delta|>2deg training samples'
    variants['linear_coeffs_lsqflap.json'] = b

    for fn, v in variants.items():
        with open(os.path.join(HERE, fn), 'w') as f:
            json.dump(v, f, indent=2)
        print(f'  wrote {fn}: CL_d = {v["CL_d"]:.5f} /rad '
              f'= {v["CL_d"]*np.pi/180:.5f} /deg')


if __name__ == '__main__':
    main()

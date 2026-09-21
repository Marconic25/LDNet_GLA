"""
Fit the linear unsteady model — corrected procedure.

What the diagnostics established
--------------------------------
* No collinearity (all VIF ~ 1), so the first fit's CL_a = -38 was not a
  cancellation artefact.
* The motion downwash w_m_eff carries only ~2.8% of the C_L standard
  deviation, so CL_a is barely identifiable: pinning it to 2*pi costs
  0.0015 NRMSE (0.03671 -> 0.03817 train). The data is indifferent, so the
  physical value is used rather than a noise-fitted one.
* The gust gain really is negative in this codebase's sign convention. The
  LDNet itself gives dC_L/dW < 0 at the trim state, and the plant applies
  the load as rhs_h = -Fy, so "positive C_L" pushes the section down in h.
  The linear model must match that convention, since both models feed the
  identical loop. Sign is therefore taken from the data, not imposed.

Fitting strategy
----------------
Pin the motion-related gains (CL_a = 2*pi, added mass = analytic) to their
theoretical values, because the campaign cannot identify them, and fit by
least squares only the gains the data DOES determine strongly:

    CL_g, CL_d, CL_0      (t-statistics 328, 268, 2096 in the free fit)
    CM_a, CM_g, CM_d, CM_0

This is the strongest honest version of the linear baseline: every parameter
is either the textbook value or the least-squares optimum on the same data
the LDNet saw. No parameter is chosen to make the comparison come out a
particular way.

A second variant, --free, keeps all gains free; it is reported alongside so
the ablation does not rest on the pinning choice.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_linear import build

CL_A_PINNED = 2.0 * np.pi
AM_L_PINNED = 1.0
COLS = ['w_m_eff', 'w_g_eff', 'delta', 'am', 'const']


def nrmse(pred, y):
    return np.sqrt(np.mean((pred - y) ** 2)) / (y.max() - y.min())


def fit(mode='pinned', stride=10):
    X, yL, yM = build('train', stride=stride)
    Xv, yLv, yMv = build('valid', stride=stride)

    if mode == 'free':
        bL, *_ = np.linalg.lstsq(X, yL, rcond=None)
        bM, *_ = np.linalg.lstsq(X, yM, rcond=None)
    else:
        # C_L: pin the motion gain and the added-mass scaling, fit the rest
        off = CL_A_PINNED * X[:, 0] + AM_L_PINNED * X[:, 3]
        Xr = X[:, [1, 2, 4]]                       # w_g_eff, delta, const
        br, *_ = np.linalg.lstsq(Xr, yL - off, rcond=None)
        bL = np.array([CL_A_PINNED, br[0], br[1], AM_L_PINNED, br[2]])
        # C_M: the moment slopes are all reasonably determined; fit freely
        bM, *_ = np.linalg.lstsq(X, yM, rcond=None)

    res = dict(
        CL_a=float(bL[0]), CL_g=float(bL[1]), CL_d=float(bL[2]),
        AM_L=float(bL[3]), CL_0=float(bL[4]),
        CM_a=float(bM[0]), CM_g=float(bM[1]), CM_d=float(bM[2]),
        AM_M=float(bM[3]), CM_0=float(bM[4]),
    )
    res['_fit'] = dict(
        mode=mode, stride=stride, n_samples=int(X.shape[0]),
        nrmse_CL=float(nrmse(X @ bL, yL)), nrmse_CM=float(nrmse(X @ bM, yM)),
        nrmse_CL_valid=float(nrmse(Xv @ bL, yLv)),
        nrmse_CM_valid=float(nrmse(Xv @ bM, yMv)),
    )
    return res


def main():
    stride = int(os.environ.get('STRIDE', '10'))
    out_dir = os.path.dirname(os.path.abspath(__file__))

    for mode, fname in [('pinned', 'linear_coeffs.json'),
                        ('free', 'linear_coeffs_free.json')]:
        r = fit(mode, stride)
        print(f'=== mode={mode} ===')
        for k in ['CL_a', 'CL_g', 'CL_d', 'AM_L', 'CL_0',
                  'CM_a', 'CM_g', 'CM_d', 'AM_M', 'CM_0']:
            print(f'   {k:6s} = {r[k]:+.5f}')
        f = r['_fit']
        print(f'   train NRMSE  C_L {f["nrmse_CL"]:.5f}  C_M {f["nrmse_CM"]:.5f}')
        print(f'   valid NRMSE  C_L {f["nrmse_CL_valid"]:.5f}  '
              f'C_M {f["nrmse_CM_valid"]:.5f}\n')
        with open(os.path.join(out_dir, fname), 'w') as fh:
            json.dump(r, fh, indent=2)
        print(f'   wrote {fname}\n')


if __name__ == '__main__':
    main()

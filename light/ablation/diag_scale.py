"""
Check how much each regressor actually contributes to C_L, and how well the
motion gain is identified.

The collinearity diagnostic came back clean (all VIF ~ 1), so the -38
lift-curve slope is not a cancellation artefact. The suspicion is now scale:
w_m_eff has RMS 9.4e-4 against delta's 5.7e-2, so the motion column carries
~60x less signal and its gain is poorly determined -- a large wrong CL_a
costs almost nothing in training NRMSE but is physically meaningless and
would misbehave once the MPC drives the structural states.

This script measures:
  1. the contribution of each fitted term to C_L (RMS of gain*column);
  2. the standard error on each gain from the least-squares covariance;
  3. how much training NRMSE degrades if CL_a is PINNED to 2*pi and the rest
     refitted -- i.e. whether the data actually prefers -38 or is indifferent.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_linear import build

COLS = ['w_m_eff', 'w_g_eff', 'delta', 'am', 'const']


def nrmse(pred, y):
    return np.sqrt(np.mean((pred - y) ** 2)) / (y.max() - y.min())


def main():
    stride = int(os.environ.get('STRIDE', '10'))
    X, yL, yM = build('train', stride=stride)
    Xv, yLv, yMv = build('valid', stride=stride)

    b, *_ = np.linalg.lstsq(X, yL, rcond=None)
    resid = yL - X @ b
    n, p = X.shape
    sigma2 = np.sum(resid ** 2) / (n - p)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    print('term contributions to C_L and gain uncertainty:\n')
    print(f'{"col":10s}{"gain":>12s}{"std err":>12s}{"t":>10s}{"RMS(gain*col)":>16s}')
    for i, c in enumerate(COLS):
        contrib = np.sqrt(np.mean((b[i] * X[:, i]) ** 2))
        t = b[i] / se[i] if se[i] > 0 else np.inf
        print(f'{c:10s}{b[i]:>12.4f}{se[i]:>12.4f}{t:>10.1f}{contrib:>16.5f}')

    print(f'\nunconstrained  train NRMSE {nrmse(X @ b, yL):.5f}'
          f'   valid {nrmse(Xv @ b, yLv):.5f}')

    # --- pin CL_a to the thin-airfoil value and refit the rest --------------
    for pin in [2 * np.pi, 5.0, 6.0, 7.0]:
        y_adj = yL - pin * X[:, 0]
        Xr = X[:, 1:]
        br, *_ = np.linalg.lstsq(Xr, y_adj, rcond=None)
        pred = pin * X[:, 0] + Xr @ br
        predv = pin * Xv[:, 0] + Xv[:, 1:] @ br
        print(f'CL_a pinned to {pin:6.4f}: train NRMSE {nrmse(pred, yL):.5f}'
              f'   valid {nrmse(predv, yLv):.5f}'
              f'   | CL_g={br[0]:+.3f} CL_d={br[1]:+.4f}')

    # --- how much of C_L variance does the motion term explain at all? -------
    print(f'\nstd(C_L) = {yL.std():.5f}')
    print(f'std(2pi * w_m_eff) = {(2*np.pi*X[:,0]).std():.5f}'
          f'   ({(2*np.pi*X[:,0]).std()/yL.std()*100:.2f}% of C_L std)')
    print(f'std(CL_g_fit * w_g_eff) = {(b[1]*X[:,1]).std():.5f}')


if __name__ == '__main__':
    main()

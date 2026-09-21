"""
Diagnose the conditioning of the linear-unsteady regression.

The first unconstrained fit returned CL_a = -38, CL_g = -9.2 with a 3.7%
training NRMSE. A negative lift-curve slope is unphysical, so the design
matrix is almost certainly ill-conditioned: in the training campaign the
structural states are themselves the RESPONSE to the gust, so alpha, hd/U and
W/U are strongly correlated and many gain combinations fit equally well.

A model fitted in that regime is fine for interpolating the training
trajectories but wrong inside a control loop, where the MPC drives delta far
off the training correlation structure. This script quantifies the problem
before choosing the remedy.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_linear import build

COLS = ['w_m_eff', 'w_g_eff', 'delta', 'am', 'const']


def main():
    stride = int(os.environ.get('STRIDE', '10'))
    X, yL, yM = build('train', stride=stride)

    print(f'design matrix {X.shape}\n')

    # column scales
    print('column RMS:')
    for i, c in enumerate(COLS):
        print(f'  {c:9s} {np.sqrt(np.mean(X[:, i] ** 2)):.5f}')

    # correlation matrix of the non-constant columns
    Z = X[:, :4]
    Zc = Z - Z.mean(0)
    Cm = np.corrcoef(Zc.T)
    print('\ncorrelation matrix:')
    print('           ' + ''.join(f'{c:>10s}' for c in COLS[:4]))
    for i, c in enumerate(COLS[:4]):
        print(f'  {c:9s}' + ''.join(f'{Cm[i, j]:>10.4f}' for j in range(4)))

    # conditioning
    s = np.linalg.svd(X, compute_uv=False)
    print(f'\nsingular values: {np.array2string(s, precision=4)}')
    print(f'condition number: {s[0] / s[-1]:.3e}')

    # variance inflation factors
    print('\nVIF (>10 => severe collinearity):')
    for i in range(4):
        others = [j for j in range(4) if j != i]
        A = np.column_stack([Z[:, others], np.ones(len(Z))])
        b, *_ = np.linalg.lstsq(A, Z[:, i], rcond=None)
        r2 = 1 - np.sum((Z[:, i] - A @ b) ** 2) / np.sum((Z[:, i] - Z[:, i].mean()) ** 2)
        vif = 1.0 / max(1e-12, 1 - r2)
        print(f'  {COLS[i]:9s} R2={r2:.6f}  VIF={vif:.1f}')

    # How much does delta actually vary in the training data, and is it
    # correlated with the gust? If the campaign moved the flap only while the
    # gust was acting, the fit cannot separate their effects.
    print(f'\ndelta range in fit data: [{X[:, 2].min():.4f}, {X[:, 2].max():.4f}] rad')
    print(f'frac |delta|>0.01 rad : {np.mean(np.abs(X[:, 2]) > 0.01):.3f}')


if __name__ == '__main__':
    main()

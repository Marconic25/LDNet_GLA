"""
Fit the linear unsteady model's gains to the SAME CFD campaign the LDNet
was trained on (data/GLA_train.h5).

Why fit at all: the ablation asks how much of the alleviation is due to the
data-driven surrogate. If the linear baseline carried textbook gains it could
lose simply because 2*pi is the wrong lift-curve slope for this viscous
airfoil at this Reynolds number -- a gain error, not a missing nonlinearity.
Fitting every gain by least squares on the same data gives the linear model
the best parameters available within its own (linear, attached-flow)
structure. Whatever gap remains is then attributable to what the structure
cannot represent.

Procedure
---------
1. Load the 100 training trajectories: input_signals [h,hd,a,ad,delta,W_gust],
   output_signals [F_y, M_z]; convert forces to coefficients with the same
   q_dyn convention used in ldnet_aero.py (q = 0.5*rho*U^2*S, C_M = M_z/(q*c)).
2. Integrate the four Wagner/Kussner lag states along each trajectory with the
   true inputs (the lag dynamics are input-driven and parameter-free, so this
   can be done once, before fitting).
3. Build the regressor matrix
       [w_m_eff, w_g_eff, delta_rad, am, 1]
   and solve the two least-squares problems for C_L and C_M.

The regression is linear in all unknown gains, so this is a single exact
lstsq -- no iteration, no local minima, no tuning knobs the author could
lean on to steer the outcome.
"""
import json
import os
import sys
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linear_aero import A1, B1, A2, B2, G1, GB1, G2, GB2

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
RHO, S, C = 1.225, 0.05, 1.0
A_EA, A_34 = 0.40, 0.75
DT = 0.002


def integrate_lags(w_m, w_g, U, dt, c=C):
    """
    Integrate the four lag states along one trajectory (RK4, vectorised in
    time by stepping -- the system is linear and input-driven).

    w_m, w_g : (T,) downwash histories
    Returns  : (T,4) lag states, aligned so z[i] is the state BEFORE the
               load at step i is evaluated (matches predict()/batch_step()).
    """
    T = len(w_m)
    z = np.zeros((T, 4))
    bvec = np.array([B1, B2, GB1, GB2])
    k = 2.0 * max(float(U), 1.0) / c
    zc = np.zeros(4)
    for i in range(T):
        z[i] = zc
        wv = np.array([w_m[i], w_m[i], w_g[i], w_g[i]])

        def f(zz):
            return k * (-bvec * zz + wv)

        k1 = f(zc); k2 = f(zc + 0.5 * dt * k1)
        k3 = f(zc + 0.5 * dt * k2); k4 = f(zc + dt * k3)
        zc = zc + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def build(tag='train', stride=5, max_traj=None):
    """Assemble the regressor matrix and targets over a dataset split."""
    path = os.path.join(ROOT, 'data', f'GLA_{tag}.h5')
    with h5py.File(path, 'r') as f:
        sig = np.array(f['input_signals'])          # (N,T,6)
        out = np.array(f['output_signals'])         # (N,T,1,2)
        par = np.array(f['input_parameters'])       # (N,1)
        times = np.array(f['times'])

    N = sig.shape[0] if max_traj is None else min(max_traj, sig.shape[0])
    dt = float(times[1] - times[0])

    Xs, yL, yM = [], [], []
    for n in range(N):
        h, hd, a, ad, delta, W = [sig[n, :, j] for j in range(6)]
        Fy = out[n, :, 0, 0]
        Mz = out[n, :, 0, 1]
        U = float(par[n, 0])
        Uf = max(U, 1.0)
        q = 0.5 * RHO * Uf ** 2 * S

        CL = Fy / q
        CM = Mz / (q * C)

        w_m = a + hd / Uf + (C / Uf) * (A_34 - A_EA) * ad
        w_g = W / Uf
        z = integrate_lags(w_m, w_g, Uf, dt)

        w_m_eff = w_m - A1 * B1 * z[:, 0] - A2 * B2 * z[:, 1]
        w_g_eff = w_g - G1 * GB1 * z[:, 2] - G2 * GB2 * z[:, 3]
        am = (np.pi * C / (2.0 * Uf)) * ad
        d = np.deg2rad(delta)

        X = np.stack([w_m_eff, w_g_eff, d, am, np.ones_like(d)], axis=1)
        Xs.append(X[::stride])
        yL.append(CL[::stride])
        yM.append(CM[::stride])

    return (np.concatenate(Xs), np.concatenate(yL), np.concatenate(yM))


def main():
    stride = int(os.environ.get('STRIDE', '5'))
    print(f'[fit] loading train split (stride={stride}) ...', flush=True)
    X, yL, yM = build('train', stride=stride)
    print(f'[fit] design matrix {X.shape}, targets {yL.shape}', flush=True)

    bL, resL, rankL, _ = np.linalg.lstsq(X, yL, rcond=None)
    bM, resM, rankM, _ = np.linalg.lstsq(X, yM, rcond=None)

    predL = X @ bL
    predM = X @ bM
    nrmseL = np.sqrt(np.mean((predL - yL) ** 2)) / (yL.max() - yL.min())
    nrmseM = np.sqrt(np.mean((predM - yM) ** 2)) / (yM.max() - yM.min())

    coeff = dict(
        CL_a=float(bL[0]), CL_g=float(bL[1]), CL_d=float(bL[2]),
        AM_L=float(bL[3]), CL_0=float(bL[4]),
        CM_a=float(bM[0]), CM_g=float(bM[1]), CM_d=float(bM[2]),
        AM_M=float(bM[3]), CM_0=float(bM[4]),
        _fit=dict(nrmse_CL=float(nrmseL), nrmse_CM=float(nrmseM),
                  n_samples=int(X.shape[0]), stride=stride,
                  rank_L=int(rankL), rank_M=int(rankM)),
    )

    print('\n[fit] fitted gains:')
    for k in ['CL_a', 'CL_g', 'CL_d', 'AM_L', 'CL_0',
              'CM_a', 'CM_g', 'CM_d', 'AM_M', 'CM_0']:
        print(f'   {k:6s} = {coeff[k]:+.5f}')
    print(f'\n[fit] train NRMSE  C_L {nrmseL:.4f}   C_M {nrmseM:.4f}')
    print(f'[fit] (reference: 2*pi = {2*np.pi:.4f})')

    # held-out check
    Xv, yLv, yMv = build('valid', stride=stride)
    nL = np.sqrt(np.mean((Xv @ bL - yLv) ** 2)) / (yLv.max() - yLv.min())
    nM = np.sqrt(np.mean((Xv @ bM - yMv) ** 2)) / (yMv.max() - yMv.min())
    coeff['_fit']['nrmse_CL_valid'] = float(nL)
    coeff['_fit']['nrmse_CM_valid'] = float(nM)
    print(f'[fit] valid NRMSE  C_L {nL:.4f}   C_M {nM:.4f}')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'linear_coeffs.json')
    with open(out, 'w') as f:
        json.dump(coeff, f, indent=2)
    print(f'\n[fit] wrote {out}')


if __name__ == '__main__':
    main()

"""
Resolve the sign of the gust-lift gain.

The fit returns CL_g ~ -9.2 with t = -328: strongly determined, and
physically backwards (an upward gust should INCREASE lift). Before pinning
anything, establish the sign conventions actually used by the plant, because
run.py applies the loads as

    Fy = q*cl ;  structure.rhs does  rhs_h = -Fy - D_H*hd - K_H*h

i.e. Fy enters the heave equation with a MINUS sign, so a positive C_L in
this codebase pushes the section DOWN in the h coordinate. The dataset's F_y
may therefore carry the opposite sign to the usual "lift positive up".

Checks
------
1. Sign of C_L vs W in the raw training data (no model at all): correlate
   the gust signal against the load signal directly.
2. Same for delta: does a positive flap deflection raise or lower F_y?
3. What the LDNet itself predicts: sweep W and delta through the trained
   model at the trim state and report the slopes. This is the ground truth
   the linear model must match, since both feed the SAME loop.
"""
import os
import sys
import numpy as np
import h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
RHO, S, C = 1.225, 0.05, 1.0


def raw_data_signs():
    with h5py.File(os.path.join(ROOT, 'data', 'GLA_train.h5'), 'r') as f:
        sig = np.array(f['input_signals'])
        out = np.array(f['output_signals'])
        par = np.array(f['input_parameters'])

    W = sig[:, :, 5].ravel()
    d = sig[:, :, 4].ravel()
    Fy = out[:, :, 0, 0].ravel()

    print('--- raw dataset correlations ---')
    print(f'  corr(W_gust, F_y)  = {np.corrcoef(W, Fy)[0,1]:+.4f}')
    print(f'  corr(delta,  F_y)  = {np.corrcoef(d, Fy)[0,1]:+.4f}')
    print(f'  F_y range [{Fy.min():.2f}, {Fy.max():.2f}]  mean {Fy.mean():.2f}')
    print(f'  W   range [{W.min():.2f}, {W.max():.2f}]')
    print(f'  delta range [{d.min():.2f}, {d.max():.2f}]')

    # slope of F_y on W, controlling for nothing (crude but sign-revealing)
    A = np.column_stack([W, d, np.ones_like(W)])
    b, *_ = np.linalg.lstsq(A, Fy, rcond=None)
    print(f'  crude slopes: dFy/dW = {b[0]:+.4f} N/(m/s),'
          f'  dFy/ddelta = {b[1]:+.4f} N/deg')


def ldnet_signs():
    """Probe the trained LDNet the loop actually uses."""
    sys.path.insert(0, os.path.join(ROOT, 'light'))
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from ldnet_aero import LDNetAero

    md = os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10')
    aero = LDNetAero(md)
    aero.reset(dt=0.002)
    X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
    U = 80.0

    print('\n--- LDNet probe at trim state (frozen z=0) ---')
    cl0, cm0 = aero.predict(X0, 0.0, 0.0, U)
    print(f'  C_L(W=0, d=0) = {cl0:+.5f}   C_M = {cm0:+.5f}')

    print('  gust sweep (delta=0):')
    for W in [0.0, 5.0, 10.0, 20.0, 30.0]:
        cl, cm = aero.predict(X0, 0.0, W, U)
        print(f'    W={W:5.1f}  C_L={cl:+.5f}  dC_L={cl-cl0:+.5f}')

    print('  flap sweep (W=0):')
    for d in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        cl, cm = aero.predict(X0, d, 0.0, U)
        print(f'    delta={d:+6.1f}  C_L={cl:+.5f}  dC_L={cl-cl0:+.5f}')


if __name__ == '__main__':
    raw_data_signs()
    ldnet_signs()

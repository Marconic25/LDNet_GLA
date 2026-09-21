"""
Why does a model with a 0.125 forecast error control so much worse than one
with 0.013?

The linear model's 8-step C_L forecast error against CFD is ~10x the LDNet's
-- bad, but not obviously fatal. Yet closed-loop it goes to -45% on a mild
cell where the LDNet reaches +89%. Before treating that as the finding, work
out the mechanism, because a mechanism that turns out to be an implementation
detail (rather than physics) would invalidate the comparison.

What the MPC actually needs from its internal model is not absolute accuracy
but the correct SENSITIVITY of C_L to the flap over the horizon: it picks
argmin over a grid of delta. A model with a large constant bias but the right
gradient still picks the right flap. A model with a small bias but the wrong
gradient does not.

This script dumps, at a few instants of a real gust, the cost curve J(delta)
that each controller minimises, together with the C_L the PLANT would
actually produce for the same delta. That directly exposes whether the
linear model's argmin is displaced, and by how much.
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
from optimal import MPCPreviewController, dp45_batch
import run_ablation as RA

U, RHO, C, DT, S = 80.0, 1.225, 1.0, 0.002, 0.05
q = 0.5 * RHO * U ** 2 * S
X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
MD = os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10')


def horizon_cost(model, z0, x0, dg, w_seq, trim, R, N=8, lam=0.0):
    """Reproduce MPCPreviewController.compute's cost curve for a given model."""
    G = len(dg)
    z_b = np.tile(np.asarray(z0, float).reshape(1, -1), (G, 1))
    x_b = np.tile(np.asarray(x0, float).reshape(1, -1), (G, 1))
    J = R * dg ** 2
    CL0 = None
    for k in range(N):
        Wk = float(w_seq[k])
        CL, CM, z_new = model.batch_step(z_b, x_b, dg, Wk, U, DT)
        if k == 0:
            CL0 = CL.copy()
        z_b = z_new - lam * z_b
        x_b = dp45_batch(x_b, q * CL, q * CM * C, DT)
        J = J + (CL - trim) ** 2
    return J, CL0


def main():
    W0 = float(os.environ.get('W0', '10'))
    Tg = float(os.environ.get('TG', '0.4'))
    R = float(os.environ.get('RW', '3e-4'))

    plant = LDNetAero(MD); plant.reset(dt=DT)
    trim = float(plant.predict(X0, 0., 0., U)[0])

    lin = LinearUnsteadyAero(coeff_path=os.path.join(HERE, 'linear_coeffs.json'))
    lin.reset(dt=DT)
    ldc = LDNetAero(MD); ldc.reset(dt=DT)

    n = int(round(1.0 / DT)) + 1
    ts = np.arange(n) * DT
    Wt = np.array([RA.gust(t, W0, Tg) for t in ts])
    dg = np.linspace(-14, 14, 161)

    # drive the plant OPEN-LOOP and probe both cost curves along the way
    x = X0.copy()
    z_lin = np.zeros(lin._num_z)
    z_ld = np.zeros(ldc._num_z)

    probes = [int(0.05 / DT), int(0.10 / DT), int(0.15 / DT), int(0.20 / DT),
              int(0.30 / DT)]
    print(f'W0={W0:g} Tg={Tg:g} R={R:g} trim={trim:.4f}\n')
    for i in range(n):
        Wi = float(Wt[i])
        if i in probes:
            lo = i + 1; hi = min(i + 9, n)
            w = np.zeros(8); w[:hi - lo] = Wt[lo:hi]

            J_l, CL_l = horizon_cost(lin, z_lin, x, dg, w, trim, R, lam=0.0)
            J_n, CL_n = horizon_cost(ldc, z_ld, x, dg, w, trim, R,
                                     lam=ldc._z_leak)
            d_l = dg[int(np.argmin(J_l))]
            d_n = dg[int(np.argmin(J_n))]

            # what the PLANT actually gives for those two choices
            cl_plant_l = plant.predict(x, d_l, Wi, U)[0]
            cl_plant_n = plant.predict(x, d_n, Wi, U)[0]

            print(f't={ts[i]:.3f} W={Wi:6.2f}')
            print(f'   linear picks d={d_l:+7.2f}  -> plant C_L='
                  f'{cl_plant_l:.4f}  (err vs trim {cl_plant_l-trim:+.4f})')
            print(f'   LDNet  picks d={d_n:+7.2f}  -> plant C_L='
                  f'{cl_plant_n:.4f}  (err vs trim {cl_plant_n-trim:+.4f})')
            # what the linear model BELIEVED it would get at its own choice
            jl = int(np.argmin(J_l))
            print(f'   linear believed first-step C_L={CL_l[jl]:.4f}, '
                  f'plant gave {cl_plant_l:.4f}  '
                  f'(belief error {CL_l[jl]-cl_plant_l:+.4f})')
            # the flap that would ACTUALLY have been best on the plant now
            best_d, best_e = None, 1e9
            for d in dg[::8]:
                e = abs(plant.predict(x, d, Wi, U)[0] - trim)
                if e < best_e:
                    best_e, best_d = e, d
            print(f'   plant-optimal instantaneous d={best_d:+7.2f} '
                  f'(|C_L-trim|={best_e:.4f})\n')

        cl, cm = plant.predict(x, 0.0, Wi, U)
        z_lin = lin.advance_z(z_lin, x, 0.0, Wi, U, DT)
        z_ld = ldc.advance_z(z_ld, x, 0.0, Wi, U, DT)
        plant.advance(x, 0.0, Wi, U, DT)
        x = structure.step_dp45(x, q * cl, q * cm * C, DT)


if __name__ == '__main__':
    main()

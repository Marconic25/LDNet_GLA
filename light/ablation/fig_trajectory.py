"""
Example-trajectory forecast: true CFD C_L(t) against the LDNet's and the
linear model's rolling 8-step horizon prediction, on the held-out trajectory
with the largest peak deflection -- the severe case where run_cfd_horizon.py
shows the two models diverge most.

Reuses the exact horizon-rolling logic of run_cfd_horizon.py (same model
reset, same resynchronisation between probes, same dt_model/sub handling),
but keeps the per-probe N-step-ahead C_L forecast instead of collapsing it to
an RMS error, so it can be plotted against the true trace.
"""
import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'light'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

from linear_aero import LinearUnsteadyAero
from ldnet_aero import LDNetAero

# Same rcParams as light/plots_ldnet_accuracy.py, "shared with the chapter-3
# figures": serif + Computer Modern math to match the LaTeX body text, not
# matplotlib's default DejaVu Sans.
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'savefig.dpi': 300,
})

RHO, S = 1.225, 0.05
MD = os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10')
BLUE = '#1c3c78'
RED = '#a83232'
GREY = '#444444'


def horizon_forecast_trace(model, sig, U, dt, sub, N, stride, is_ldnet):
    """Same rolling scheme as run_cfd_horizon.py, but keeps the N-step-ahead
    forecast value (not just its RMS) at each probe, indexed by the time of
    the forecast point (probe + N steps), so it lines up with CL_true."""
    T = sig.shape[0]
    h, hd, a, ad, delta, W = [sig[:, j] for j in range(6)]
    model.reset(dt=dt)
    grid = list(range(0, T - (N + 1) * sub, sub))
    idx, pred = [], []
    for gi, i in enumerate(grid):
        if gi % stride == 0:
            z = np.array(model._z, dtype=float, copy=True)
            cl = None
            for k in range(N):
                j = i + k * sub
                st = (h[j], hd[j], a[j], ad[j])
                if is_ldnet:
                    cl = model._reconstruct(
                        z, model._normalize_signals(h[j], hd[j], a[j], ad[j],
                                                    delta[j], W[j]), U)[0]
                else:
                    cl = model._loads(z, st, delta[j], W[j], U)[0]
                z = model.advance_z(z, st, delta[j], W[j], U, dt)
            idx.append(i + (N - 1) * sub)
            pred.append(float(cl))
        st = (h[i], hd[i], a[i], ad[i])
        model.advance(st, delta[i], W[i], U, dt)
    return np.array(idx), np.array(pred)


def main():
    split = os.environ.get('SPLIT', 'test')
    N = int(os.environ.get('NH', '8'))
    stride = int(os.environ.get('PSTRIDE', '6'))

    with h5py.File(os.path.join(ROOT, 'data', f'GLA_{split}.h5'), 'r') as f:
        sig_all = np.array(f['input_signals'])
        out_all = np.array(f['output_signals'])
        par_all = np.array(f['input_parameters'])
        times = np.array(f['times'])
    dt_raw = float(times[1] - times[0])

    lin = LinearUnsteadyAero(coeff_path=os.path.join(HERE, 'linear_coeffs.json'))
    ld = LDNetAero(MD)
    dt_model = ld._dt_ref
    sub = max(1, round(dt_model / dt_raw))

    dmax = np.max(np.abs(sig_all[:, :, 4]), axis=1)
    n = int(np.argmax(dmax))
    U = max(float(par_all[n, 0]), 1.0)
    q = 0.5 * RHO * U ** 2 * S
    CL_true = out_all[n, :, 0, 0] / q
    t = np.arange(len(CL_true)) * dt_raw
    sig = sig_all[n]

    print(f'[traj] trajectory {n}, |delta|max={dmax[n]:.1f} deg, U={U:.1f} m/s')

    il, pl = horizon_forecast_trace(lin, sig, U, dt_model, sub, N, stride, False)
    ind, pn = horizon_forecast_trace(ld, sig, U, dt_model, sub, N, stride, True)

    tl = il * dt_raw
    tn = ind * dt_raw
    CL_at_l = CL_true[il]
    CL_at_n = CL_true[ind]

    rms_l = float(np.sqrt(np.mean((pl - CL_at_l) ** 2)))
    rms_n = float(np.sqrt(np.mean((pn - CL_at_n) ** 2)))
    print(f'[traj] RMS on this trajectory: linear {rms_l:.5f}  LDNet {rms_n:.5f}'
          f'  ratio {rms_l/rms_n:.2f}x')

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(t, CL_true, color=GREY, lw=1.4, label='CFD (true)')
    ax.plot(tl, pl, color=RED, lw=1.1, ls='--', label='linear unsteady, 8-step forecast')
    ax.plot(tn, pn, color=BLUE, lw=1.1, ls='--', label='LDNet, 8-step forecast')
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$C_L$')
    ax.legend(loc='best', framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(HERE, 'fig_ablation_trajectory.png')
    fig.savefig(out)
    print(f'[traj] wrote {out}')


if __name__ == '__main__':
    main()

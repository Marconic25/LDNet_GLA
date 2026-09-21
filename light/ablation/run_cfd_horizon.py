"""
Plant-neutral comparison: score each controller's MPC horizon against CFD.

The caveat
----------
The closed-loop ablation uses the LDNet as the plant, so its controller has a
perfect internal model. A live CFD plant would settle the question but costs
~370x per run. The previous attempt (run_cfd_plant.py) tried to recover flap
authority from the campaign by partial regression; that failed, and the
failure is worth recording: in the campaign the flap is commanded IN RESPONSE
to the gust, so delta is correlated with alpha, W and ad, and a partial slope
cannot be separated from them. Trajectory data with a closed-loop flap
history cannot yield a clean dC_L/ddelta by regression.

What this script does instead
-----------------------------
It removes the plant privilege without needing a delta sweep, by testing the
one thing the MPC actually consumes: the predicted C_L trajectory over the
horizon, at the TRUE flap history the CFD ran.

For probe points along held-out CFD trajectories, each model is given the
true state and the true delta history, and rolls its own horizon forward.
The forecast is compared to the CFD C_L over the same window. Neither model
is the plant; CFD is. The horizon is exactly the MPC's (N=8, dt=2e-3).

This measures forecast quality on the real system under the real flap
commands -- the quantity that determines whether the MPC's argmin is placed
correctly. It does not simulate a closed loop against CFD (that needs the
sweep), and is reported as what it is.

Reported per severity band so the envelope region is located, and with a
paired bootstrap so the difference carries an interval rather than a bare
ratio.
"""
import os
import sys
import json
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

RHO, S, C = 1.225, 0.05, 1.0
MD = os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10')


def horizon_errors(model, sig, CL_true, U, dt, sub, N=8, stride=20,
                   is_ldnet=False):
    """
    Per-probe RMS error of the model's N-step C_L forecast against CFD.

    Returns an array of (rms_over_horizon, |delta|max_in_window, W_max_in_window).
    The model is resynchronised with the truth between probes, so this is
    forecast quality, not accumulated drift -- matching how the MPC restarts
    its horizon from the current estimate every step.
    """
    T = len(CL_true)
    h, hd, a, ad, delta, W = [sig[:, j] for j in range(6)]
    model.reset(dt=dt)
    grid = list(range(0, T - (N + 1) * sub, sub))
    out = []
    for gi, i in enumerate(grid):
        if gi % stride == 0:
            z = np.array(model._z, dtype=float, copy=True)
            errs = []
            for k in range(N):
                j = i + k * sub
                st = (h[j], hd[j], a[j], ad[j])
                if is_ldnet:
                    cl = model._reconstruct(
                        z, model._normalize_signals(h[j], hd[j], a[j], ad[j],
                                                    delta[j], W[j]), U)[0]
                else:
                    cl = model._loads(z, st, delta[j], W[j], U)[0]
                errs.append(cl - CL_true[j])
                z = model.advance_z(z, st, delta[j], W[j], U, dt)
            w = slice(i, i + N * sub)
            out.append((float(np.sqrt(np.mean(np.square(errs)))),
                        float(np.max(np.abs(delta[w]))),
                        float(np.max(W[w]))))
        st = (h[i], hd[i], a[i], ad[i])
        model.advance(st, delta[i], W[i], U, dt)
    return np.array(out)


def boot_ratio(el, en, n=4000, seed=0):
    """Paired bootstrap CI for RMS(linear)/RMS(ldnet)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(el))
    r = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        a = np.sqrt(np.mean(el[s] ** 2))
        b = np.sqrt(np.mean(en[s] ** 2))
        if b > 0:
            r.append(a / b)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    split = os.environ.get('SPLIT', 'test')
    n_traj = int(os.environ.get('NTRAJ', '10'))
    N = int(os.environ.get('NH', '8'))
    stride = int(os.environ.get('PSTRIDE', '12'))

    with h5py.File(os.path.join(ROOT, 'data', f'GLA_{split}.h5'), 'r') as f:
        sig_all = np.array(f['input_signals'])
        out_all = np.array(f['output_signals'])
        par_all = np.array(f['input_parameters'])
        times = np.array(f['times'])
    dt_raw = float(times[1] - times[0])

    lin = LinearUnsteadyAero(
        coeff_path=os.path.join(HERE,
                                os.environ.get('COEFFS', 'linear_coeffs.json')))
    ld = LDNetAero(MD)
    dt_model = ld._dt_ref
    sub = max(1, round(dt_model / dt_raw))

    dmax = np.max(np.abs(sig_all[:, :, 4]), axis=1)
    chosen = [int(i) for i in np.argsort(-dmax)[:n_traj]]
    print(f'[cfd-horizon] split={split} N={N} sub={sub} '
          f'trajectories={chosen}')
    print(f'[cfd-horizon] plant = CFD (neither model); both see the true '
          f'state and the true flap history\n')

    EL, EN = [], []
    for n in chosen:
        U = max(float(par_all[n, 0]), 1.0)
        q = 0.5 * RHO * U ** 2 * S
        CL_true = out_all[n, :, 0, 0] / q
        el = horizon_errors(lin, sig_all[n], CL_true, U, dt_model, sub, N,
                            stride, False)
        en = horizon_errors(ld, sig_all[n], CL_true, U, dt_model, sub, N,
                            stride, True)
        EL.append(el); EN.append(en)
        print(f'  traj {n:2d} (|d|max={dmax[n]:5.1f} deg): '
              f'linear {np.sqrt(np.mean(el[:,0]**2)):.5f}   '
              f'LDNet {np.sqrt(np.mean(en[:,0]**2)):.5f}', flush=True)

    EL = np.concatenate(EL); EN = np.concatenate(EN)
    rl = float(np.sqrt(np.mean(EL[:, 0] ** 2)))
    rn = float(np.sqrt(np.mean(EN[:, 0] ** 2)))
    lo, hi = boot_ratio(EL[:, 0], EN[:, 0])
    print(f'\n=== {N}-step C_L forecast vs CFD ({len(EL)} probes) ===')
    print(f'  linear RMS {rl:.5f}   LDNet RMS {rn:.5f}   '
          f'ratio {rl/rn:.2f}x  [95% CI {lo:.2f}, {hi:.2f}]')
    wins = int(np.sum(EN[:, 0] < EL[:, 0]))
    print(f'  LDNet better at {wins}/{len(EL)} probes '
          f'({wins/len(EL)*100:.1f}%)')

    for name, col, edges in [('|delta|max [deg]', 1, [0, 2, 5, 8, 11, 100]),
                             ('gust Wmax [m/s]', 2, [0, 5, 15, 25, 35, 100])]:
        print(f'\n  by {name}:')
        print(f'  {"bin":>12s}{"n":>7s}{"linear":>10s}{"LDNet":>10s}{"ratio":>8s}')
        for a, b in zip(edges[:-1], edges[1:]):
            m = (EL[:, col] >= a) & (EL[:, col] < b)
            if m.sum() < 10:
                continue
            x = float(np.sqrt(np.mean(EL[m, 0] ** 2)))
            y = float(np.sqrt(np.mean(EN[m, 0] ** 2)))
            print(f'  {f"{a}-{b}":>12s}{int(m.sum()):>7d}{x:>10.5f}'
                  f'{y:>10.5f}{x/y:>8.2f}')

    with open(os.path.join(HERE, 'cfd_horizon.json'), 'w') as f:
        json.dump(dict(rms_linear=rl, rms_ldnet=rn, ratio=rl / rn,
                       ci=[lo, hi], n_probes=int(len(EL)),
                       win_frac=wins / len(EL)), f, indent=2)


if __name__ == '__main__':
    main()

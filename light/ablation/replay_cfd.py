"""
Third-plant check: how well does each internal model PREDICT the held-out CFD
data it would be steering against?

Why this is needed
------------------
The closed-loop ablation uses the LDNet as the plant in both arms. That
flatters the LDNet, whose controller then has a perfect internal model, and
it is the first thing a reader should challenge. Running the loop against CFD
instead is the ideal answer but costs ~370x more per evaluation and is out of
reach here.

What IS affordable is to remove the plant asymmetry from the part of the
problem that drives the controller: prediction accuracy. The MPC's only use
of its internal model is to forecast C_L over an 8-step horizon for each
candidate flap setting. This script measures exactly that forecast, for both
models, against held-out CFD trajectories -- data neither model can have
memorised in the closed-loop sense, and which is the true reference for both.

If the LDNet's horizon forecast is substantially better on CFD ground truth,
then its closed-loop advantage is not an artefact of it being the plant: it
is predicting the real system better, which is what the MPC needs.

Metrics
-------
  * N-step-ahead C_L forecast error (the quantity the MPC actually minimises)
  * error stratified by |delta| and by gust severity, to locate WHERE the
    linear model degrades -- the 'region of the envelope' the thesis asks to
    identify.
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


def horizon_forecast_error(model, sig, CL_true, U, dt_model, N=8, stride=25,
                           is_ldnet=False, sub=1):
    """
    Walk the trajectory; at each probe point, roll the model forward N steps
    with the RECORDED inputs and compare its C_L forecast to CFD.

    The model's state is kept synchronised with the true trajectory between
    probes (the MPC likewise starts each horizon from its current estimate),
    so this isolates forecast quality, not long-run drift.

    IMPORTANT -- time base. The CFD files are stored at dt_raw ~ 2.8e-4 s,
    about 7x finer than the LDNet's training dt_ref = 2e-3 s. LDNetAero
    integrates its latent with n_sub = round(dt/dt_sub) forward-Euler
    sub-steps, so feeding it the raw dt rounds n_sub to 1 and scales dz by
    dt/dt_ref ~ 0.14: the latent is then driven at the wrong rate and never
    reaches the correct state. Driving it at the raw dt gives an RMS 0-step
    error of 0.569 against 0.0094 at the proper stride -- a 60x artefact that
    has nothing to do with model quality. (The same reasoning is already
    encoded in LDNetAero._warmup_from_csv, which subsamples by exactly this
    stride.) Both models are therefore stepped on the dt_ref grid, `sub`
    being the subsampling stride into the raw arrays.
    """
    T = len(CL_true)
    h, hd, a, ad, delta, W = [sig[:, j] for j in range(6)]
    model.reset(dt=dt_model)

    errs = []       # (err, |delta| at probe, W at probe)
    grid = list(range(0, T - (N + 1) * sub, sub))
    for gi, i in enumerate(grid):
        if gi % stride == 0:
            # roll a COPY of the latent forward N model-steps
            z = np.array(model._z, dtype=float, copy=True)
            for k in range(N):
                j = i + k * sub
                st = (h[j], hd[j], a[j], ad[j])
                cl = (model._reconstruct(z, model._normalize_signals(
                          h[j], hd[j], a[j], ad[j], delta[j], W[j]), U)[0]
                      if is_ldnet else
                      model._loads(z, st, delta[j], W[j], U)[0])
                if k == N - 1:
                    errs.append((cl - CL_true[j], abs(delta[i]), W[i]))
                z = model.advance_z(z, st, delta[j], W[j], U, dt_model)
        # keep the model synchronised with the truth
        st = (h[i], hd[i], a[i], ad[i])
        model.advance(st, delta[i], W[i], U, dt_model)
    return np.array(errs)


def main():
    n_traj = int(os.environ.get('NTRAJ', '8'))
    N = int(os.environ.get('NH', '8'))
    stride = int(os.environ.get('PSTRIDE', '25'))
    split = os.environ.get('SPLIT', 'test')

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
    print(f'[replay] split={split} dt_raw={dt_raw:.6g} dt_model={dt_model:.6g} '
          f'sub={sub}')

    # Prefer trajectories that actually MOVE the flap: the flap nonlinearity is
    # the mechanism under test, and a delta==0 trajectory cannot probe it.
    dmax = np.max(np.abs(sig_all[:, :, 4]), axis=1)
    order = np.argsort(-dmax)
    chosen = [int(i) for i in order[:n_traj]]
    print(f'[replay] trajectories by |delta|max: '
          f'{[(int(i), round(float(dmax[i]),1)) for i in chosen]}')

    agg = {'linear': [], 'ldnet': []}
    for n in chosen:
        U = max(float(par_all[n, 0]), 1.0)
        q = 0.5 * RHO * U ** 2 * S
        CL_true = out_all[n, :, 0, 0] / q
        sig = sig_all[n]
        e_l = horizon_forecast_error(lin, sig, CL_true, U, dt_model, N, stride,
                                     False, sub)
        e_n = horizon_forecast_error(ld, sig, CL_true, U, dt_model, N, stride,
                                     True, sub)
        agg['linear'].append(e_l)
        agg['ldnet'].append(e_n)
        print(f'  traj {n} (|d|max={dmax[n]:.1f}): RMS {N}-step err  '
              f'linear {np.sqrt(np.mean(e_l[:,0]**2)):.5f}'
              f'   LDNet {np.sqrt(np.mean(e_n[:,0]**2)):.5f}', flush=True)

    EL = np.concatenate(agg['linear'])
    EN = np.concatenate(agg['ldnet'])
    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))

    print(f'\n=== {N}-step C_L forecast error vs held-out CFD ===')
    print(f'  overall RMS   linear {rms(EL[:,0]):.5f}   LDNet {rms(EN[:,0]):.5f}'
          f'   ratio {rms(EL[:,0])/rms(EN[:,0]):.2f}x')

    print(f'\n  stratified by |delta| [deg]:')
    print(f'  {"bin":>14s}{"n":>7s}{"linear":>10s}{"LDNet":>10s}{"ratio":>8s}')
    edges = [0, 2, 5, 8, 11, 100]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (EL[:, 1] >= lo) & (EL[:, 1] < hi)
        if m.sum() < 10:
            continue
        rl, rn = rms(EL[m, 0]), rms(EN[m, 0])
        print(f'  {f"{lo}-{hi}":>14s}{m.sum():>7d}{rl:>10.5f}{rn:>10.5f}'
              f'{rl/rn:>8.2f}')

    print(f'\n  stratified by gust W [m/s]:')
    print(f'  {"bin":>14s}{"n":>7s}{"linear":>10s}{"LDNet":>10s}{"ratio":>8s}')
    edges = [0, 5, 15, 25, 35, 100]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (EL[:, 2] >= lo) & (EL[:, 2] < hi)
        if m.sum() < 10:
            continue
        rl, rn = rms(EL[m, 0]), rms(EN[m, 0])
        print(f'  {f"{lo}-{hi}":>14s}{m.sum():>7d}{rl:>10.5f}{rn:>10.5f}'
              f'{rl/rn:>8.2f}')

    res = dict(N=N, n_traj=n_traj,
               rms_linear=rms(EL[:, 0]), rms_ldnet=rms(EN[:, 0]))
    with open(os.path.join(HERE, 'cfd_forecast.json'), 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()

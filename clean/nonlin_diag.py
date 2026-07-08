"""
Nonlinearity diagnostic: can a fixed-gain (clipped-linear) classical controller
reproduce the model-based controller's optimal flap command delta*(x, W)?

For each gust cell we run the deployed receding-horizon MPC closed-loop (fixed
simple cost, no per-cell schedule), and at every step record the model's FREE
optimum delta*_free = argmin_delta J_rollout(x, z, W) over a fine delta grid
(the unconstrained best the model wants -- the rate limit is a shared actuator
constraint, excluded here).

Then we ask the decisive question with least-squares fits of delta*_free on the
signals a classical controller could sense:

  H1: e_CL                       (1-channel prop on lift  -- the current baseline)
  H2: e_CL, ad                   (2-channel PD: lift + pitch-rate -- the objection)
  H3: h, hd, al, ad              (full linear state feedback, NO gust sensor)
  H4: h, hd, al, ad, W           (state + gust sensor)
  H5: h, hd, al, ad, W, e_CL     (everything observable)

A clip on delta only matters where delta* saturates; we therefore also report the
INTERIOR fraction (|delta*| < 0.95*delta_max) and run the fits on interior-only
samples. If even H5 leaves a large residual on interior samples, then delta* is a
genuinely nonlinear / latent-state-dependent function that NO fixed-gain clipped
law can reproduce -> the model advantage is real. If H2 already fits near-perfectly,
a 2-channel clipped PD replicates the model -> the thesis is weak.
"""
import os, numpy as np
import mpc_gust as M
from controller import Controller

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM

GRID = {
    'W10/T0.5': (10., 0.5), 'W10/T1.0': (10., 1.0), 'W10/T2.0': (10., 2.0),
    'W20/T0.5': (20., 0.5), 'W20/T1.0': (20., 1.0), 'W20/T2.0': (20., 2.0),
    'W30/T0.5': (30., 0.5), 'W30/T1.0': (30., 1.0), 'W30/T2.0': (30., 2.0),
    'design'  : (11.46, 1.12),
}
DECISIVE = ['W10/T2.0', 'W30/T1.0', 'W20/T1.0', 'design']


def run_cell(W0, Tg, TEND, NH, NGRID, QAD, RW, DLPF, DMAX, FINE):
    """Closed-loop deployed MPC (fixed cost). Records, per step, the realized state
    and the model's FREE optimum delta*_free over a fine grid."""
    a = M.a
    N = int(round(TEND / DT)) + 1
    tg = np.arange(N) * DT
    Wt = np.array([M.gust(t, W0, Tg) for t in tg])
    a.reset(dt=DT)
    ctrl = Controller(aero_predict=a.predict, U=U, dt=DT, Q_h=0., Q_alpha=0.,
                      Q_alpha_dot=QAD, Q_CL=1.0, R=RW, R_du=0.0, n_grid=NGRID,
                      global_search=True, causal_basin=False, mpc_horizon=NH, aero=a,
                      target_lpf=0.0, C_L_trim=CLTRIM, Fy_trim=0., Mz_trim=0.,
                      delta_max=DMAX, delta_dot_max=300.)
    ctrl.reset()
    fine = np.linspace(-DMAX, DMAX, FINE)
    x = M.X0.copy(); de_f = 0.0; de_f2 = 0.0
    rec = {k: [] for k in ['t', 'W', 'h', 'hd', 'al', 'ad', 'CL', 'dfree', 'de']}
    for i in range(N):
        # FREE optimum the model wants at the current (x, z, W) -- no rate limit
        J = ctrl._rollout_cost_batch(fine, x, float(Wt[i]))
        dfree = float(fine[int(np.argmin(J))])
        # deployed (rate-limited + smoothed) command, to advance the realistic trajectory
        de_raw = ctrl.compute(x, W_hat=float(Wt[i]))
        de_f = DLPF * de_f + (1.0 - DLPF) * de_raw
        de_f2 = DLPF * de_f2 + (1.0 - DLPF) * de_f
        de = de_f2
        ctrl._delta_prev = de
        cl, cm = a.predict(x, de, Wt[i], U)
        rec['t'].append(tg[i]); rec['W'].append(Wt[i])
        rec['h'].append(x[0]); rec['hd'].append(x[1]); rec['al'].append(x[2]); rec['ad'].append(x[3])
        rec['CL'].append(float(cl)); rec['dfree'].append(dfree); rec['de'].append(de)
        a.advance(x, de, Wt[i], U, DT); x = M.structure.step_dp45(x, M.q * cl, M.q * cm * M.C, DT)
    out = {k: np.array(v) for k, v in rec.items()}
    out['_win'] = out['t'] <= (Tg + 0.5)
    return out


def fit(y, X):
    """Least squares y ~ [1, X]. Returns (R2, resid_rms)."""
    A = np.column_stack([np.ones(len(y)), X]) if X.ndim == 2 else np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    ss_res = float(np.sum(r ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
    return r2, float(np.sqrt(np.mean(r ** 2)))


if __name__ == '__main__':
    sel = os.environ.get('CELLS', 'decisive')
    cells = DECISIVE if sel == 'decisive' else list(GRID.keys())
    TEND = float(os.environ.get('TEND', '2.5'))
    NH = int(os.environ.get('NH', '6')); NGRID = int(os.environ.get('NGRID', '15'))
    FINE = int(os.environ.get('FINE', '81'))
    QAD = float(os.environ.get('QAD', '100')); RW = float(os.environ.get('RW', '0.01'))
    DLPF = float(os.environ.get('DLPF', '0.95')); DMAX = float(os.environ.get('DMAX', '14.'))
    print(f'# nonlin_diag  cells={sel}  TEND={TEND} NH={NH} NGRID={NGRID} FINE={FINE} '
          f'QAD={QAD} RW={RW} DLPF={DLPF} DMAX={DMAX} DAMULT={os.environ.get("DAMULT","1")}', flush=True)
    print(f'# CLTRIM={CLTRIM:.5f}', flush=True)

    pool = {k: [] for k in ['dfree', 'eCL', 'ad', 'h', 'hd', 'al', 'W', 'interior']}
    print('\n## per-cell: peak |delta*_free|, interior fraction (|d*|<0.95 dmax)', flush=True)
    print(f'{"cell":10s} {"peak|d*|":>9s} {"mean|d*|":>9s} {"interior%":>9s}', flush=True)
    percell = {}
    for name in cells:
        W0, Tg = GRID[name]
        r = run_cell(W0, Tg, TEND, NH, NGRID, QAD, RW, DLPF, DMAX, FINE)
        w = r['_win']
        df = r['dfree'][w]
        interior = np.abs(df) < 0.95 * DMAX
        peak = float(np.max(np.abs(df))); mean = float(np.mean(np.abs(df)))
        intf = float(np.mean(interior)) * 100.0
        percell[name] = peak
        print(f'{name:10s} {peak:9.2f} {mean:9.2f} {intf:9.1f}', flush=True)
        pool['dfree'].append(df)
        pool['eCL'].append(r['CL'][w] - CLTRIM)
        pool['ad'].append(r['ad'][w]); pool['h'].append(r['h'][w])
        pool['hd'].append(r['hd'][w]); pool['al'].append(r['al'][w]); pool['W'].append(r['W'][w])
        pool['interior'].append(interior)

    P = {k: np.concatenate(v) for k, v in pool.items()}
    y = P['dfree']; intr = P['interior']
    spread = np.array(list(percell.values()))
    print(f'\n# cross-cell delta*_free peak spread: min={spread.min():.2f} max={spread.max():.2f} '
          f'ratio={spread.max()/max(spread.min(),1e-9):.1f}x', flush=True)
    print(f'# pooled interior fraction = {100.0*np.mean(intr):.1f}%   '
          f'(samples: {len(y)} total, {int(np.sum(intr))} interior)', flush=True)

    feats = {
        'H1 e_CL              ': ['eCL'],
        'H2 e_CL,ad           ': ['eCL', 'ad'],
        'H3 state(h,hd,al,ad) ': ['h', 'hd', 'al', 'ad'],
        'H4 state+W           ': ['h', 'hd', 'al', 'ad', 'W'],
        'H5 state+W+e_CL      ': ['h', 'hd', 'al', 'ad', 'W', 'eCL'],
    }
    for tag, mask in [('ALL gust-window', np.ones(len(y), bool)), ('INTERIOR only', intr)]:
        yy = y[mask]
        sd = float(np.std(yy))
        print(f'\n## linear-fit of delta*_free  [{tag}]   N={len(yy)}  std(d*)={sd:.3f} deg', flush=True)
        print(f'{"hypothesis":22s} {"R2":>7s} {"resid_rms[deg]":>14s} {"resid/std":>10s}', flush=True)
        for tag2, fl in feats.items():
            X = np.column_stack([P[f][mask] for f in fl])
            r2, rms = fit(yy, X)
            print(f'{tag2:22s} {r2:7.3f} {rms:14.3f} {rms/max(sd,1e-9):10.1%}', flush=True)
    print('\n# DONE', flush=True)

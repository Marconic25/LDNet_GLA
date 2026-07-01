"""
Single-step optimal controller vs proportional.

Physics: a one-step-optimal controller on a locally-linear plant IS a proportional
law. With J(delta)=Q_CL*(C_L(delta)-trim)^2 + R*delta^2 and C_L~=C_L0+s*delta
(s=dC_L/ddelta), the optimum is delta* = -[Q_CL*s/(Q_CL*s^2+R)]*(C_L0-trim) = -K*(C_L0-trim).
So in the quasi-steady regime the best fixed-gain proportional already realizes the
optimum. The single-step optimal can only BEAT it where s varies (nonlinearity): its
gain K self-adapts, and at the pitch-reversal (s->0 / sign flip) K->0 so it stops,
where a fixed gain pushes the wrong way.

This script runs the single-step optimal (controller.Controller with mpc_horizon=1,
causal_basin=True, NO heavy smoothing) against the best fixed-gain ProportionalController
across linear (expect tie) and nonlinear (expect win) cells, and records the effective
gain K(t)=delta/(C_L0-trim) for the money plot.
"""
import os, numpy as np
import mpc_gust as M
from controller import Controller, ProportionalController

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
TEND = float(os.environ.get('TEND', '2.5'))
NGRID = int(os.environ.get('NGRID', '15'))
R_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]   # single-step control-penalty sweep (its tuning knob)

GRID = {
    'W10/T1.0': (10., 1.0), 'W20/T1.0': (20., 1.0),         # ~linear  -> expect tie
    'W10/T2.0': (10., 2.0),                                 # pitch reversal -> expect win
    'W30/T1.0': (30., 1.0), 'W30/T0.5': (30., 0.5),         # high amplitude -> expect win
}
LINEAR = {'W10/T1.0', 'W20/T1.0'}


def win_stats(al, CL, de, t, Tg):
    mw = t <= (Tg + 0.5)
    cex = float(np.max(np.abs(CL[mw] - CLTRIM)))
    pk = float(np.max(np.abs(al[mw])))
    dmx = float(np.max(np.abs(de[mw])))
    return cex, pk, dmx


def run_open(W0, Tg):
    a = M.a; N = int(round(TEND / DT)) + 1
    t = np.arange(N) * DT; Wt = np.array([M.gust(tt, W0, Tg) for tt in t])
    a.reset(dt=DT); x = M.X0.copy()
    al = []; CL = []; de = []
    for i in range(N):
        cl, cm = a.predict(x, 0.0, Wt[i], U)
        a.advance(x, 0.0, Wt[i], U, DT); x = M.structure.step_rk4(x, q * cl, q * cm * C, DT)
        al.append(x[2]); CL.append(float(cl)); de.append(0.0)
    return t, Wt, np.array(al), np.array(CL), np.array(de), None


def run_onestep(W0, Tg, R, QAD=0.0, DMAX=14.):
    a = M.a; N = int(round(TEND / DT)) + 1
    t = np.arange(N) * DT; Wt = np.array([M.gust(tt, W0, Tg) for tt in t])
    a.reset(dt=DT)
    ctrl = Controller(aero_predict=a.predict, U=U, dt=DT, Q_h=0., Q_alpha=0., Q_alpha_dot=QAD,
                      Q_CL=1.0, R=R, n_grid=NGRID, global_search=True, causal_basin=True,
                      mpc_horizon=1, aero=a, target_lpf=0.0, C_L_trim=CLTRIM,
                      Fy_trim=0., Mz_trim=0., delta_max=DMAX, delta_dot_max=300.)
    ctrl.reset()
    x = M.X0.copy()
    al = []; CL = []; de = []; K = []
    for i in range(N):
        cl0 = float(a.predict(x, 0.0, Wt[i], U)[0])           # lift the controller sees (zero flap)
        d = ctrl.compute(x, W_hat=float(Wt[i]))                # rate-limited single-step optimum (DLPF=0)
        cl, cm = a.predict(x, d, Wt[i], U)
        a.advance(x, d, Wt[i], U, DT); x = M.structure.step_rk4(x, q * cl, q * cm * C, DT)
        e = cl0 - CLTRIM
        al.append(x[2]); CL.append(float(cl)); de.append(d)
        K.append(d / e if abs(e) > 1e-3 else np.nan)
    return t, Wt, np.array(al), np.array(CL), np.array(de), np.array(K)


def run_prop(W0, Tg, gain, DMAX=14.):
    a = M.a; N = int(round(TEND / DT)) + 1
    t = np.arange(N) * DT; Wt = np.array([M.gust(tt, W0, Tg) for tt in t])
    a.reset(dt=DT)
    ctrl = ProportionalController(C_L_trim=CLTRIM, gain=gain, dt=DT, delta_max=DMAX, delta_dot_max=300.)
    ctrl.reset()
    x = M.X0.copy(); de_prev = 0.0
    al = []; CL = []; de = []; K = []
    for i in range(N):
        cl0 = float(a.predict(x, 0.0, Wt[i], U)[0])
        clm = float(a.predict(x, de_prev, Wt[i], U)[0])       # measured lift with current flap
        d = ctrl.compute(clm)                                  # clip + rate limit, DLPF=0
        cl, cm = a.predict(x, d, Wt[i], U)
        a.advance(x, d, Wt[i], U, DT); x = M.structure.step_rk4(x, q * cl, q * cm * C, DT)
        de_prev = d; e = cl0 - CLTRIM
        al.append(x[2]); CL.append(float(cl)); de.append(d)
        K.append(d / e if abs(e) > 1e-3 else np.nan)
    return t, Wt, np.array(al), np.array(CL), np.array(de), np.array(K)


def main():
    print(f'# onestep(R-swept) vs proportional  TEND={TEND} NGRID={NGRID} DAMULT={os.environ.get("DAMULT","1")}', flush=True)
    print(f'# CLTRIM={CLTRIM:.5f}  R_GRID={R_GRID}', flush=True)
    gains = [-10., -20., -40., -60., -80., -120., -160.]
    print(f'\n{"cell":10s} {"kind":7s} | {"open":>6s} | {"prop: CLred pitch (g*)":>24s} | {"1step: CLred pitch (R*)":>26s}', flush=True)
    traces = {}
    for name, (W0, Tg) in GRID.items():
        kind = 'linear' if name in LINEAR else 'nonlin'
        t, Wt, alO, CLo, deO, _ = run_open(W0, Tg)
        cex0, pk0, _ = win_stats(alO, CLo, deO, t, Tg)
        # best proportional gain (max CLred)
        bp = (None, -1e9, None)
        for g in gains:
            r = run_prop(W0, Tg, g)
            ce, pk, _ = win_stats(r[2], r[3], r[4], t, Tg)
            clred = (cex0 - ce) / cex0 * 100
            if clred > bp[1]: bp = (g, clred, (pk, r))
        gstar, pclred, (ppk, rprop) = bp
        # best single-step R (max CLred) -- its tuning knob, matched to prop's gain sweep
        bo = (None, -1e9, None)
        for R in R_GRID:
            r1 = run_onestep(W0, Tg, R)
            ce1, pk1, _ = win_stats(r1[2], r1[3], r1[4], t, Tg)
            clred = (cex0 - ce1) / cex0 * 100
            if clred > bo[1]: bo = (R, clred, (pk1, r1))
        Rstar, oclred, (opk, rone) = bo
        print(f'{name:10s} {kind:7s} | {cex0:6.3f} | g*={gstar:6.0f} {pclred:7.1f}% {ppk/pk0:5.2f} | '
              f'R*={Rstar:8.0e} {oclred:7.1f}% {opk/pk0:5.2f}', flush=True)
        traces[name] = dict(t=t, W=Wt, prop=rprop, one=rone, gstar=gstar, Rstar=Rstar, pk0=pk0, cex0=cex0)

    # save traces for the money plot (W30/T0.5 dramatic + W30/T1)
    out = {}
    for name in ['W30/T0.5', 'W30/T1.0']:
        if name in traces:
            tr = traces[name]; tag = name.replace('/', '_').replace('.', '')
            out[f'{tag}_t'] = tr['t']; out[f'{tag}_W'] = tr['W']
            out[f'{tag}_prop_de'] = tr['prop'][4]; out[f'{tag}_prop_CL'] = tr['prop'][3]; out[f'{tag}_prop_K'] = tr['prop'][5]
            out[f'{tag}_one_de'] = tr['one'][4]; out[f'{tag}_one_CL'] = tr['one'][3]; out[f'{tag}_one_K'] = tr['one'][5]
            out[f'{tag}_one_al'] = tr['one'][2]; out[f'{tag}_prop_al'] = tr['prop'][2]
            out[f'{tag}_gstar'] = tr['gstar']; out[f'{tag}_Rstar'] = tr['Rstar']
    import os as _o; _o.makedirs('results', exist_ok=True)
    np.savez('results/onestep_traces.npz', **out)
    print('\n# saved results/onestep_traces.npz (for money plot)', flush=True)
    print('# DONE', flush=True)


if __name__ == '__main__':
    main()

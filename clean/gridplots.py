"""Full-grid time histories: open vs best proportional vs single-step optimal.
Runs every gust cell, records all quantities (h, alpha, alpha_dot, C_L, C_M, delta),
saves to results/gridplots.npz for local plotting. No extra flap smoothing (DLPF=0,
rate-limited only); single-step optimal = Controller(mpc_horizon=1, causal_basin=True)."""
import os, numpy as np
import mpc_gust as M
from controller import Controller

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
TEND = float(os.environ.get('TEND', '3.0'))
ROPT = float(os.environ.get('ROPT', '1e-3'))
GAINS = [-10., -20., -40., -60., -80., -120., -160.]
GRID = {
    'W10/T0.5': (10., 0.5), 'W10/T1.0': (10., 1.0), 'W10/T2.0': (10., 2.0),
    'W20/T0.5': (20., 0.5), 'W20/T1.0': (20., 1.0), 'W20/T2.0': (20., 2.0),
    'W30/T0.5': (30., 0.5), 'W30/T1.0': (30., 1.0), 'W30/T2.0': (30., 2.0),
    'design'  : (11.46, 1.12),
}
QTY = ['h', 'al', 'ad', 'CL', 'CM', 'de']


def clexc(r, Tg):
    mw = r['_t'] <= (Tg + 0.5)
    return float(np.max(np.abs(r['CL'][mw] - CLTRIM)))


def sim_opt(W0, Tg):
    a = M.a; N = int(round(TEND / DT)) + 1
    t = np.arange(N) * DT; Wt = np.array([M.gust(tt, W0, Tg) for tt in t])
    a.reset(dt=DT)
    ctrl = Controller(aero_predict=a.predict, U=U, dt=DT, Q_h=0., Q_alpha=0., Q_alpha_dot=0.,
                      Q_CL=1.0, R=ROPT, n_grid=15, global_search=True, causal_basin=True,
                      mpc_horizon=1, aero=a, target_lpf=0.0, C_L_trim=CLTRIM,
                      Fy_trim=0., Mz_trim=0., delta_max=14., delta_dot_max=300.)
    ctrl.reset(); x = M.X0.copy()
    R = {k: [] for k in QTY}
    for i in range(N):
        d = ctrl.compute(x, W_hat=float(Wt[i]))
        cl, cm = a.predict(x, d, Wt[i], U)
        a.advance(x, d, Wt[i], U, DT); x = M.structure.step_rk4(x, q * cl, q * cm * C, DT)
        R['h'].append(x[0]); R['al'].append(x[2]); R['ad'].append(x[3])
        R['CL'].append(float(cl)); R['CM'].append(float(cm)); R['de'].append(d)
    o = {k: np.array(v) for k, v in R.items()}; o['_t'] = t; o['_Wt'] = Wt
    return o


def main():
    print(f'# gridplots  TEND={TEND} ROPT={ROPT} DAMULT={os.environ.get("DAMULT","1")}', flush=True)
    out = {}
    for name, (W0, Tg) in GRID.items():
        tag = name.replace('/', '_').replace('.', '')
        OL = M.simulate('open', W0, Tg, TEND=TEND, DLPF=0.0)
        cex0 = clexc(OL, Tg)
        bp = (None, -1e9, None)
        for g in GAINS:
            r = M.simulate('prop', W0, Tg, gain=g, TEND=TEND, DLPF=0.0)
            cr = (cex0 - clexc(r, Tg)) / cex0 * 100
            if cr > bp[1]: bp = (g, cr, r)
        gstar, pcr, PR = bp
        OP = sim_opt(W0, Tg); ocr = (cex0 - clexc(OP, Tg)) / cex0 * 100
        for arm, r in [('open', OL), ('prop', PR), ('opt', OP)]:
            for k in QTY:
                out[f'{tag}_{arm}_{k}'] = r[k]
        out[f'{tag}_t'] = OL['_t']; out[f'{tag}_W'] = OL['_Wt']
        out[f'{tag}_gstar'] = float(gstar); out[f'{tag}_Tg'] = float(Tg)
        out[f'{tag}_pcr'] = float(pcr); out[f'{tag}_ocr'] = float(ocr)
        print(f'{name:10s}: prop g*={gstar:6.0f} CLred={pcr:5.1f}%  |  opt(R={ROPT:g}) CLred={ocr:5.1f}%', flush=True)
    import os as _o; _o.makedirs('results', exist_ok=True)
    np.savez('results/gridplots.npz', **out)
    print('# saved results/gridplots.npz', flush=True)
    print('# DONE', flush=True)


if __name__ == '__main__':
    main()

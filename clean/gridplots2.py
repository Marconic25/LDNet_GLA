"""Full-grid time histories: open vs best proportional vs single-step optimal,
with the single-step control weight R TUNED per cell (best CL reduction).

The R sweep is vectorized: all R candidates run as parallel closed-loop single-step
sims sharing one gust, with the per-step argmin over a fine delta-grid done in batched
TF calls (like the PD batched sweep). This reproduces Controller(mpc_horizon=1,
causal_basin=True) with Q_h=Q_alpha=Q_alpha_dot=0 (one-step cost = (CL-trim)^2 + R*delta^2),
leak-corrected, and is ~1 sim per cell instead of len(R_GRID).

Saves results/gridplots.npz (open/prop/opt full traces per cell) for local plotting.
"""
import os, numpy as np
import mpc_gust as M
from controller import Controller

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
LAM = float(M.a._z_leak)
_dummy = Controller(aero_predict=M.a.predict, U=U, dt=DT)
_rk4b = _dummy._rk4_batch
TEND = float(os.environ.get('TEND', '3.0'))
GAINS = [-10., -20., -40., -60., -80., -120., -160.]
R_GRID = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
DMAX = 14.0; G = 161
GRID = {
    'W10/T0.5': (10., 0.5), 'W10/T1.0': (10., 1.0), 'W10/T2.0': (10., 2.0),
    'W20/T0.5': (20., 0.5), 'W20/T1.0': (20., 1.0), 'W20/T2.0': (20., 2.0),
    'W30/T0.5': (30., 0.5), 'W30/T1.0': (30., 1.0), 'W30/T2.0': (30., 2.0),
    'design'  : (11.46, 1.12),
}
QTY = ['h', 'al', 'ad', 'CL', 'CM', 'de']


def clexc(CL, t, Tg):
    mw = t <= (Tg + 0.5)
    return float(np.max(np.abs(CL[mw] - CLTRIM)))


def batch_onestep(W0, Tg):
    """Parallel single-step optimal for all R in R_GRID. Returns t, Wt, and per-R full
    traces dict[qty] of shape (N, B)."""
    a = M.a; B = len(R_GRID)
    N = int(round(TEND / DT)) + 1
    t = np.arange(N) * DT; Wt = np.array([M.gust(tt, W0, Tg) for tt in t])
    a.reset(dt=DT)
    z_b = np.tile(np.asarray(a._z, float).reshape(1, -1), (B, 1))
    x_b = np.tile(np.asarray(M.X0, float).reshape(1, -1), (B, 1))
    prev = np.zeros(B); rate = 300.0 * DT
    dg = np.linspace(-DMAX, DMAX, G)                      # (G,)
    rec = {k: np.zeros((N, B)) for k in QTY}
    for i in range(N):
        Wi = float(Wt[i])
        cl0, _, _ = a.batch_step(z_b, x_b, np.zeros(B), Wi, U, DT)          # (B,) CL at delta=0
        xb_big = np.repeat(x_b, G, axis=0); zb_big = np.repeat(z_b, G, axis=0)
        d_big = np.tile(dg, B)
        CLg, _, _ = a.batch_step(zb_big, xb_big, d_big, Wi, U, DT)          # (B*G,)
        CLg = CLg.reshape(B, G)
        cost = (CLg - CLTRIM) ** 2 + R_GRID[:, None] * dg[None, :] ** 2     # (B,G)
        neg = dg[None, :] <= 0.0
        causal = np.where(cl0[:, None] >= CLTRIM, neg, ~neg)               # lift-reducing half
        ratem = np.abs(dg[None, :] - prev[:, None]) <= rate + 1e-9
        cost = np.where(causal & ratem, cost, np.inf)
        d = dg[np.argmin(cost, axis=1)]                                     # (B,)
        cl, cm, z_new = a.batch_step(z_b, x_b, d, Wi, U, DT)
        z_b = z_new - LAM * z_b
        x_b = _rk4b(x_b, q * cl, q * cm * C, DT)
        prev = d
        rec['h'][i] = x_b[:, 0]; rec['al'][i] = x_b[:, 2]; rec['ad'][i] = x_b[:, 3]
        rec['CL'][i] = cl; rec['CM'][i] = cm; rec['de'][i] = d
    return t, Wt, rec


def main():
    print(f'# gridplots2 (R-tuned, batched)  TEND={TEND} R_GRID={list(R_GRID)} G={G} DAMULT={os.environ.get("DAMULT","1")}', flush=True)
    out = {}
    for name, (W0, Tg) in GRID.items():
        tag = name.replace('/', '_').replace('.', '')
        OL = M.simulate('open', W0, Tg, TEND=TEND, DLPF=0.0)
        t = OL['_t']; cex0 = clexc(OL['CL'], t, Tg)
        # best proportional
        bp = (None, -1e9, None)
        for g in GAINS:
            r = M.simulate('prop', W0, Tg, gain=g, TEND=TEND, DLPF=0.0)
            cr = (cex0 - clexc(r['CL'], t, Tg)) / cex0 * 100
            if cr > bp[1]: bp = (g, cr, r)
        gstar, pcr, PR = bp
        # single-step optimal, R tuned (batched). Objective is minimal: (CL-trim)^2 + R*delta^2.
        # R is picked to MAXIMIZE CL reduction while NOT exciting pitch beyond open-loop (<=TOL).
        _, W, rec = batch_onestep(W0, Tg)
        mw = t <= (Tg + 0.5)
        pitch_open = float(np.max(np.abs(OL['al'][mw])))
        crs = np.array([(cex0 - clexc(rec['CL'][:, j], t, Tg)) / cex0 * 100 for j in range(len(R_GRID))])
        prs = np.array([float(np.max(np.abs(rec['al'][mw, j]))) / max(pitch_open, 1e-12) for j in range(len(R_GRID))])
        TOL = 1.10
        ok = np.where(prs <= TOL)[0]
        jb = int(ok[np.argmax(crs[ok])]) if len(ok) else int(np.argmin(prs))
        Rstar = float(R_GRID[jb]); ocr = float(crs[jb]); opr = float(prs[jb])
        for arm, r in [('open', OL), ('prop', PR)]:
            for k in QTY:
                out[f'{tag}_{arm}_{k}'] = r[k]
        for k in QTY:
            out[f'{tag}_opt_{k}'] = rec[k][:, jb]
        out[f'{tag}_t'] = t; out[f'{tag}_W'] = W
        out[f'{tag}_gstar'] = float(gstar); out[f'{tag}_Rstar'] = Rstar; out[f'{tag}_Tg'] = float(Tg)
        out[f'{tag}_pcr'] = float(pcr); out[f'{tag}_ocr'] = ocr
        print(f'{name:10s}: prop g*={gstar:6.0f} CLred={pcr:5.1f}%  |  opt R*={Rstar:8.0e} CLred={ocr:5.1f}%', flush=True)
    import os as _o; _o.makedirs('results', exist_ok=True)
    np.savez('results/gridplots.npz', **out)
    print('# saved results/gridplots.npz', flush=True)
    print('# DONE', flush=True)


if __name__ == '__main__':
    main()

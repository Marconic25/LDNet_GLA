"""Time-history traces (open / prop-W / opt-W) for large-k (Tg=0.30) and small-k
(Tg=1.20) at W10/W20/W30. Both controllers know the gust W. Saves results/showcase.npz."""
import os, numpy as np
import mpc_gust as M
from controller import Controller

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
LAM = float(M.a._z_leak)
_rk4b = Controller(aero_predict=M.a.predict, U=U, dt=DT)._rk4_batch
TEND = float(os.environ.get('TEND', '3.0'))
DMAX = 14.0; G = 161
R_GRID = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
CELLS = {f'W{w}/T{tg:.2f}': (float(w), tg) for tg in [0.30, 1.20] for w in [10, 20, 30]}
QTY = ['h', 'al', 'ad', 'CL', 'CM', 'de']


def clexc(CL, t, Tg): return float(np.max(np.abs(CL[t <= (Tg + 0.5)] - CLTRIM)))


def batch_prop_tr(W0, Tg, G1, GW):
    a = M.a; B = len(G1); N = int(round(TEND / DT)) + 1
    t = np.arange(N) * DT; Wt = np.array([M.gust(tt, W0, Tg) for tt in t])
    a.reset(dt=DT)
    z_b = np.tile(np.asarray(a._z, float).reshape(1, -1), (B, 1))
    x_b = np.tile(np.asarray(M.X0, float).reshape(1, -1), (B, 1))
    prev = np.zeros(B); reach = 300.0 * DT
    G1 = np.asarray(G1, float); GW = np.asarray(GW, float)
    rec = {k: np.zeros((N, B)) for k in QTY}
    for i in range(N):
        Wi = float(Wt[i])
        clm, _, _ = a.batch_step(z_b, x_b, prev, Wi, U, DT)
        d = G1 * (clm - CLTRIM) + GW * Wi
        d = np.clip(d, -DMAX, DMAX); d = np.clip(d, prev - reach, prev + reach)
        cl, cm, z_new = a.batch_step(z_b, x_b, d, Wi, U, DT)
        z_b = z_new - LAM * z_b; x_b = _rk4b(x_b, q * cl, q * cm * C, DT); prev = d
        rec['h'][i] = x_b[:, 0]; rec['al'][i] = x_b[:, 2]; rec['ad'][i] = x_b[:, 3]
        rec['CL'][i] = cl; rec['CM'][i] = cm; rec['de'][i] = d
    return t, Wt, rec


def batch_optW_tr(W0, Tg):
    a = M.a; B = len(R_GRID); N = int(round(TEND / DT)) + 1
    t = np.arange(N) * DT; Wt = np.array([M.gust(tt, W0, Tg) for tt in t])
    a.reset(dt=DT)
    z_b = np.tile(np.asarray(a._z, float).reshape(1, -1), (B, 1))
    x_b = np.tile(np.asarray(M.X0, float).reshape(1, -1), (B, 1))
    prev = np.zeros(B); reach = 300.0 * DT; dg = np.linspace(-DMAX, DMAX, G)
    rec = {k: np.zeros((N, B)) for k in QTY}
    for i in range(N):
        Wi = float(Wt[i])
        cl0, _, _ = a.batch_step(z_b, x_b, np.zeros(B), Wi, U, DT)
        CLg, _, _ = a.batch_step(np.repeat(z_b, G, 0), np.repeat(x_b, G, 0), np.tile(dg, B), Wi, U, DT)
        CLg = CLg.reshape(B, G)
        cost = (CLg - CLTRIM) ** 2 + R_GRID[:, None] * dg[None, :] ** 2
        neg = dg[None, :] <= 0.0
        causal = np.where(cl0[:, None] >= CLTRIM, neg, ~neg)
        ratem = np.abs(dg[None, :] - prev[:, None]) <= reach + 1e-9
        cost = np.where(causal & ratem, cost, np.inf)
        d = dg[np.argmin(cost, axis=1)]
        cl, cm, z_new = a.batch_step(z_b, x_b, d, Wi, U, DT)
        z_b = z_new - LAM * z_b; x_b = _rk4b(x_b, q * cl, q * cm * C, DT); prev = d
        rec['h'][i] = x_b[:, 0]; rec['al'][i] = x_b[:, 2]; rec['ad'][i] = x_b[:, 3]
        rec['CL'][i] = cl; rec['CM'][i] = cm; rec['de'][i] = d
    return t, Wt, rec


def main():
    print(f'# showcase traces  TEND={TEND} DAMULT={os.environ.get("DAMULT","1")}', flush=True)
    g1s = [-10., -20., -40., -60., -80., -120., -160.]
    gws = [-1.0, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0.0]
    G1 = np.array([a for a in g1s for _ in gws]); GW = np.array([b for _ in g1s for b in gws])
    out = {}
    for name, (W0, Tg) in CELLS.items():
        tag = name.replace('/', '_').replace('.', '')
        OL = M.simulate('open', W0, Tg, TEND=TEND, DLPF=0.0)
        t = OL['_t']; cex0 = clexc(OL['CL'], t, Tg)
        _, W, rp = batch_prop_tr(W0, Tg, G1, GW)
        crp = np.array([(cex0 - clexc(rp['CL'][:, j], t, Tg)) / cex0 * 100 for j in range(len(G1))])
        jp = int(np.argmax(crp))
        _, _, ro = batch_optW_tr(W0, Tg)
        cro = np.array([(cex0 - clexc(ro['CL'][:, j], t, Tg)) / cex0 * 100 for j in range(len(R_GRID))])
        jo = int(np.argmax(cro))
        for k in QTY:
            out[f'{tag}_open_{k}'] = OL[k]; out[f'{tag}_prop_{k}'] = rp[k][:, jp]; out[f'{tag}_opt_{k}'] = ro[k][:, jo]
        out[f'{tag}_t'] = t; out[f'{tag}_W'] = W; out[f'{tag}_Tg'] = float(Tg)
        out[f'{tag}_pcr'] = float(crp[jp]); out[f'{tag}_ocr'] = float(cro[jo])
        out[f'{tag}_g1'] = float(G1[jp]); out[f'{tag}_gw'] = float(GW[jp]); out[f'{tag}_Rstar'] = float(R_GRID[jo])
        print(f'{name:10s}: prop-W CLred={crp[jp]:5.1f}%  opt-W CLred={cro[jo]:5.1f}%', flush=True)
    import os as _o; _o.makedirs('results', exist_ok=True)
    np.savez('results/showcase.npz', **out)
    print('# saved results/showcase.npz', flush=True); print('# DONE', flush=True)


if __name__ == '__main__':
    main()

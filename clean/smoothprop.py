"""Smoothed proportional (2nd-order low-pass on the command, DLPF swept) vs raw prop
vs opt, on 3 cases. Trade-off: does smoothing kill the chatter (TV of delta) without
losing C_L reduction? If opt stays above the prop's CLred-vs-chatter frontier, the
optimal's smoothness is a genuine advantage. Both know W."""
import os, numpy as np
import mpc_gust as M
from controller import Controller

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
LAM = float(M.a._z_leak)
_rk4b = Controller(aero_predict=M.a.predict, U=U, dt=DT)._rk4_batch
TEND = float(os.environ.get('TEND', '3.0'))
DMAX = 14.0; G = 161
R_GRID = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
CASES = {'W48/T0.30': (48., 0.30), 'W48/T0.40': (48., 0.40), 'W48/T0.50': (48., 0.50)}
DLPF_GRID = [0.0, 0.7, 0.9]


def metrics(cl_b, de_hist, mw):
    """cl_b: (N,B) CL; de_hist: (N,B) delta. Returns clexc_b, TV_b over window."""
    cex = np.max(np.abs(cl_b[mw] - CLTRIM), axis=0)
    dd = np.abs(np.diff(de_hist[mw], axis=0))
    tv = np.sum(dd, axis=0)
    return cex, tv


def batch_prop_dlpf(W0, Tg, G1, GW, DLPF):
    a = M.a; B = len(G1); N = int(round(TEND / DT)) + 1
    t = np.arange(N) * DT; Wt = np.array([M.gust(tt, W0, Tg) for tt in t]); mw = t <= (Tg + 0.5)
    a.reset(dt=DT)
    z_b = np.tile(np.asarray(a._z, float).reshape(1, -1), (B, 1))
    x_b = np.tile(np.asarray(M.X0, float).reshape(1, -1), (B, 1))
    prev = np.zeros(B); df = np.zeros(B); df2 = np.zeros(B); reach = 300.0 * DT
    G1 = np.asarray(G1, float); GW = np.asarray(GW, float)
    CLh = np.zeros((N, B)); deh = np.zeros((N, B))
    for i in range(N):
        Wi = float(Wt[i])
        clm, _, _ = a.batch_step(z_b, x_b, prev, Wi, U, DT)
        d = G1 * (clm - CLTRIM) + GW * Wi
        d = np.clip(d, -DMAX, DMAX)
        if DLPF > 0:
            df = DLPF * df + (1 - DLPF) * d; df2 = DLPF * df2 + (1 - DLPF) * df; d = df2
        d = np.clip(d, prev - reach, prev + reach); prev = d
        cl, cm, z_new = a.batch_step(z_b, x_b, d, Wi, U, DT)
        z_b = z_new - LAM * z_b; x_b = _rk4b(x_b, q * cl, q * cm * C, DT)
        CLh[i] = cl; deh[i] = d
    cex, tv = metrics(CLh, deh, mw)
    return cex, tv


def batch_opt(W0, Tg):
    a = M.a; B = len(R_GRID); N = int(round(TEND / DT)) + 1
    t = np.arange(N) * DT; Wt = np.array([M.gust(tt, W0, Tg) for tt in t]); mw = t <= (Tg + 0.5)
    a.reset(dt=DT)
    z_b = np.tile(np.asarray(a._z, float).reshape(1, -1), (B, 1))
    x_b = np.tile(np.asarray(M.X0, float).reshape(1, -1), (B, 1))
    prev = np.zeros(B); reach = 300.0 * DT; dg = np.linspace(-DMAX, DMAX, G)
    CLh = np.zeros((N, B)); deh = np.zeros((N, B))
    for i in range(N):
        Wi = float(Wt[i])
        cl0, _, _ = a.batch_step(z_b, x_b, np.zeros(B), Wi, U, DT)
        CLg, _, _ = a.batch_step(np.repeat(z_b, G, 0), np.repeat(x_b, G, 0), np.tile(dg, B), Wi, U, DT)
        CLg = CLg.reshape(B, G)
        cost = (CLg - CLTRIM) ** 2 + R_GRID[:, None] * dg[None, :] ** 2
        neg = dg[None, :] <= 0.0
        causal = np.where(cl0[:, None] >= CLTRIM, neg, ~neg)
        ratem = np.abs(dg[None, :] - prev[:, None]) <= reach + 1e-9
        d = dg[np.argmin(np.where(causal & ratem, cost, np.inf), axis=1)]
        cl, cm, z_new = a.batch_step(z_b, x_b, d, Wi, U, DT)
        z_b = z_new - LAM * z_b; x_b = _rk4b(x_b, q * cl, q * cm * C, DT); prev = d
        CLh[i] = cl; deh[i] = d
    cex, tv = metrics(CLh, deh, mw)
    return cex, tv


def main():
    print(f'# smoothprop  TEND={TEND} DAMULT={os.environ.get("DAMULT","1")}', flush=True)
    g1s = [-10., -20., -40., -60., -80., -120., -160.]
    gws = [-1.0, -0.6, -0.4, -0.2, -0.1, 0.0]
    G1 = np.array([a for a in g1s for _ in gws]); GW = np.array([b for _ in g1s for b in gws])
    for name, (W0, Tg) in CASES.items():
        OL = M.simulate('open', W0, Tg, TEND=TEND, DLPF=0.0)
        t = OL['_t']; mw = t <= (Tg + 0.5)
        cex0 = float(np.max(np.abs(OL['CL'][mw] - CLTRIM)))
        print(f'\n{name}  (k={np.pi/(80*Tg):.3f})   open clexc={cex0:.3f}', flush=True)
        print(f'  {"arm":18s} {"CLred%":>7s} {"TV(delta)":>10s}', flush=True)
        for DLPF in DLPF_GRID:
            cex, tv = batch_prop_dlpf(W0, Tg, G1, GW, DLPF)
            cr = (cex0 - cex) / cex0 * 100
            j = int(np.argmax(cr))
            tag = 'prop raw' if DLPF == 0 else f'prop DLPF={DLPF}'
            print(f'  {tag:18s} {cr[j]:7.1f} {tv[j]:10.1f}', flush=True)
        ce, tvo = batch_opt(W0, Tg); cro = (cex0 - ce) / cex0 * 100; jo = int(np.argmax(cro))
        print(f'  {"opt":18s} {cro[jo]:7.1f} {tvo[jo]:10.1f}', flush=True)
    print('\n# DONE', flush=True)


if __name__ == '__main__':
    main()

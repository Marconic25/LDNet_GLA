"""Equal-information comparison (BOTH know the gust W):
  prop-CL : delta = g_CL*(C_L_meas - trim)                 [reactive, no W]
  prop-W  : delta = g_CL*(C_L_meas - trim) + g_W*W(t)      [+ gust feedforward, knows W]
  opt-W   : single-step optimal with W oracle              [model + optimization + W]
If prop-W reaches opt-W, the model's advantage was the INFORMATION (W), realizable by a
simple feedforward; if opt-W still wins, the model/optimization adds value at equal info.
Batched sweeps; pick best CLred with pitch <= open-loop (~10%). Limit gust cases."""
import os, numpy as np
import mpc_gust as M
from controller import Controller

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
LAM = float(M.a._z_leak)
_rk4b = Controller(aero_predict=M.a.predict, U=U, dt=DT)._rk4_batch
TEND = float(os.environ.get('TEND', '3.0'))
DMAX = 14.0; G = 161
# full in-envelope grid (Tg in [0.30,1.20], includes high k = pi/(80 Tg) up to 0.131)
TG_LIST = [0.30, 0.40, 0.50, 0.70, 1.00, 1.20]
GRID = {f'W{int(w)}/T{tg:.2f}': (float(w), float(tg)) for w in [10, 20, 30] for tg in TG_LIST}
R_GRID = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
def kred(Tg): return np.pi / (80.0 * Tg)


def clexc(CL, t, Tg): return float(np.max(np.abs(CL[t <= (Tg + 0.5)] - CLTRIM)))
def pitchpk(al, t, Tg): return float(np.max(np.abs(al[t <= (Tg + 0.5)])))


def batch_prop(W0, Tg, G1, GW):
    """delta = G1*(C_L_meas-trim) + GW*W. Returns (clexc_b, pitch_b, CL_traj0? ) -> just metrics."""
    a = M.a; B = len(G1)
    N = int(round(TEND / DT)) + 1
    t = np.arange(N) * DT; Wt = np.array([M.gust(tt, W0, Tg) for tt in t])
    a.reset(dt=DT)
    z_b = np.tile(np.asarray(a._z, float).reshape(1, -1), (B, 1))
    x_b = np.tile(np.asarray(M.X0, float).reshape(1, -1), (B, 1))
    prev = np.zeros(B); reach = 300.0 * DT
    G1 = np.asarray(G1, float); GW = np.asarray(GW, float)
    cex = np.zeros(B); pk = np.zeros(B)
    for i in range(N):
        Wi = float(Wt[i])
        clm, _, _ = a.batch_step(z_b, x_b, prev, Wi, U, DT)          # measured C_L
        d = G1 * (clm - CLTRIM) + GW * Wi
        d = np.clip(d, -DMAX, DMAX); d = np.clip(d, prev - reach, prev + reach)
        cl, cm, z_new = a.batch_step(z_b, x_b, d, Wi, U, DT)
        z_b = z_new - LAM * z_b
        x_b = _rk4b(x_b, q * cl, q * cm * C, DT)
        prev = d
        if t[i] <= (Tg + 0.5):
            cex = np.maximum(cex, np.abs(cl - CLTRIM)); pk = np.maximum(pk, np.abs(x_b[:, 2]))
    return cex, pk


def batch_optW(W0, Tg):
    """Single-step optimal WITH gust oracle (uses true W), R swept. Returns (clexc_b, pitch_b) per R."""
    a = M.a; B = len(R_GRID)
    N = int(round(TEND / DT)) + 1
    t = np.arange(N) * DT; Wt = np.array([M.gust(tt, W0, Tg) for tt in t])
    a.reset(dt=DT)
    z_b = np.tile(np.asarray(a._z, float).reshape(1, -1), (B, 1))
    x_b = np.tile(np.asarray(M.X0, float).reshape(1, -1), (B, 1))
    prev = np.zeros(B); reach = 300.0 * DT
    dg = np.linspace(-DMAX, DMAX, G)
    cex = np.zeros(B); pk = np.zeros(B)
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
        z_b = z_new - LAM * z_b
        x_b = _rk4b(x_b, q * cl, q * cm * C, DT)
        prev = d
        if t[i] <= (Tg + 0.5):
            cex = np.maximum(cex, np.abs(cl - CLTRIM)); pk = np.maximum(pk, np.abs(x_b[:, 2]))
    return cex, pk


def pick(cex_b, pk_b, cex0, pk0, tol=1.10):
    """Unconstrained max CL reduction (robust, no pitch-fallback artifact); pitch reported
    alongside so we can see if the winner pays in torsion."""
    cr = (cex0 - cex_b) / cex0 * 100; pr = pk_b / max(pk0, 1e-12)
    j = int(np.argmax(cr))
    return float(cr[j]), float(pr[j]), j


def main():
    print(f'# propw (equal info: both know W) full in-envelope grid  TEND={TEND} DAMULT={os.environ.get("DAMULT","1")}', flush=True)
    g1s = [-10., -20., -40., -60., -80., -120., -160.]
    gws = [-1.0, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0.0]
    G1w = np.array([a for a in g1s for _ in gws]); GWw = np.array([b for _ in g1s for b in gws])
    print(f'\n{"cell":10s} {"k":>6s} | {"prop-W: CLred pitch (g1,gw)":>28s} | {"opt-W: CLred pitch":>18s} | {"margin":>7s}', flush=True)
    print('-' * 80, flush=True)
    for name, (W0, Tg) in GRID.items():
        OL = M.simulate('open', W0, Tg, TEND=TEND, DLPF=0.0)
        t = OL['_t']; cex0 = clexc(OL['CL'], t, Tg); pk0 = pitchpk(OL['al'], t, Tg)
        cw, pw = batch_prop(W0, Tg, G1w, GWw); crw, prw, jw = pick(cw, pw, cex0, pk0)
        co, po = batch_optW(W0, Tg); cro, pro, _ = pick(co, po, cex0, pk0)
        print(f'{name:10s} {kred(Tg):6.3f} | {crw:6.1f}% {prw:5.2f} ({G1w[jw]:4.0f},{GWw[jw]:5.2f}) '
              f'| {cro:6.1f}% {pro:5.2f} | {cro-crw:6.1f}', flush=True)
    print('\n# margin = opt-W - prop-W. ~0 -> advantage was INFORMATION (feedforward), not the model.', flush=True)
    print('# DONE', flush=True)


if __name__ == '__main__':
    main()

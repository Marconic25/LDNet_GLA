"""
Phase-0 reduced-frequency sweep: does the MPC-PD margin grow with k?

In-distribution (training T_g in [0.30, 1.20]), no retraining. Fixed W0, sweep T_g
from 1.20 down to 0.30 -> k = pi/(80*T_g) from 0.033 to 0.131. For each T_g we trace
a small model frontier (QAD x R) and a PD2 frontier (batched, LEAK-CORRECTED), on the
neutral axes CLred% and pitch_ratio, and report the max CLred each side can reach at
matched pitch -> the physical prediction is that the MPC margin grows with k.

Leak fix: LDNetAero.batch_step omits the latent damping (1-lambda); fine over the MPC's
6-step horizon but wrong over a full closed-loop rollout. Here we correct each advance:
z_b = z_new - lambda*z_old  (= (1-lambda)*z + dz), matching scalar advance() (n_sub=1).
"""
import os, numpy as np
import mpc_gust as M
import structure as _st
from controller import Controller

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
LAM = float(M.a._z_leak)                       # latent leak (0.003)
_dummy = Controller(aero_predict=M.a.predict, U=U, dt=DT)
_rk4b = _dummy._rk4_batch

W0 = float(os.environ.get('W0', '20'))
TEND = float(os.environ.get('TEND', '2.5'))
TG_GRID = [1.20, 1.00, 0.70, 0.50, 0.40, 0.35, 0.30]
MODEL_FIXED = dict(NH=6, NGRID=15, DLPF=0.95, SCHED=False)


def kred(Tg): return np.pi / (80.0 * Tg)


def win_stats(r, Tg):
    mw = M._win(r['_t'], Tg)
    return (float(np.max(np.abs(r['CL'][mw] - CLTRIM))), float(np.max(np.abs(r['al'][mw]))))


def batch_pd_traj(Tg, G1, G2, DLPF=0.95, DMAX=14.):
    """Leak-corrected batched 2-ch-PD rollout. Returns (clexc_b, pitchpeak_b)."""
    a = M.a; B = len(G1)
    N = int(round(TEND / DT)) + 1
    tg = np.arange(N) * DT
    Wt = np.array([M.gust(t, W0, Tg) for t in tg])
    a.reset(dt=DT)
    z_b = np.tile(np.asarray(a._z, float).reshape(1, -1), (B, 1))
    x_b = np.tile(np.asarray(M.X0, float).reshape(1, -1), (B, 1))
    G1 = np.asarray(G1, float); G2 = np.asarray(G2, float)
    de_f = np.zeros(B); de_f2 = np.zeros(B); prev = np.zeros(B); rate = 300.0 * DT
    clexc = np.zeros(B); pitchpk = np.zeros(B)
    for i in range(N):
        Wi = float(Wt[i])
        clm, _, _ = a.batch_step(z_b, x_b, de_f2, Wi, U, DT)       # read-only measure
        d = G1 * (clm - CLTRIM) + G2 * x_b[:, 3]
        d = np.clip(d, -DMAX, DMAX); d = np.clip(d, prev - rate, prev + rate); prev = d
        de_f = DLPF * de_f + (1 - DLPF) * d; de_f2 = DLPF * de_f2 + (1 - DLPF) * de_f
        cl, cm, z_new = a.batch_step(z_b, x_b, de_f2, Wi, U, DT)   # force + advance
        z_b = z_new - LAM * z_b                                    # <-- leak correction
        x_b = _rk4b(x_b, q * cl, q * cm * C, DT)
        if tg[i] <= (Tg + 0.5):
            clexc = np.maximum(clexc, np.abs(cl - CLTRIM))
            pitchpk = np.maximum(pitchpk, np.abs(x_b[:, 2]))
    return clexc, pitchpk


def best_clred_at(pts, cex, pop, thr):
    """pts: list of (clexc, pitch). Max CLred% among points with pitch_ratio <= thr."""
    cand = [(cex - ce) / cex * 100 for ce, pk in pts if pk / pop <= thr + 1e-9]
    return max(cand) if cand else float('nan')


def main():
    print(f'# ksweep  W0={W0}  TEND={TEND}  DAMULT={os.environ.get("DAMULT","1")}', flush=True)
    print(f'# CLTRIM={CLTRIM:.5f}  leak={LAM}', flush=True)
    # PD2 gain grid (shared across Tg)
    g1s = [-10., -20., -40., -60., -80., -120.]
    g2s = [-16., -12., -8., -4., -2., 0., 2., 4.]
    G1 = np.array([a for a in g1s for _ in g2s]); G2 = np.array([b for _ in g1s for b in g2s])
    # model knob grid
    MQAD = [0., 100.]; MR = [1e-3, 1e-2]

    print(f'\n{"Tg":>5s} {"k":>6s} | {"mdl_max":>7s} {"pd2_max":>7s} {"margin":>7s} | '
          f'{"mdl@1.10":>8s} {"pd2@1.10":>8s} {"marg110":>8s}', flush=True)
    print('-' * 78, flush=True)
    for Tg in TG_GRID:
        OL = M.simulate('open', W0, Tg, TEND=TEND, QAD=100., RW=0.01, **MODEL_FIXED)
        cex, pop = win_stats(OL, Tg)
        # model frontier points
        mpts = []
        for QAD in MQAD:
            for Rm in MR:
                r = M.simulate('mpc', W0, Tg, TEND=TEND, RW=Rm, QAD=QAD, **MODEL_FIXED)
                mpts.append(win_stats(r, Tg))
        # PD2 frontier points (leak-correct batched)
        ce_b, pk_b = batch_pd_traj(Tg, G1, G2)
        ppts = list(zip(ce_b.tolist(), pk_b.tolist()))
        # unconstrained max CLred and iso-pitch (<=1.10) max CLred
        m_max = best_clred_at(mpts, cex, pop, 1e9)
        p_max = best_clred_at(ppts, cex, pop, 1e9)
        m_110 = best_clred_at(mpts, cex, pop, 1.10)
        p_110 = best_clred_at(ppts, cex, pop, 1.10)
        print(f'{Tg:5.2f} {kred(Tg):6.3f} | {m_max:7.1f} {p_max:7.1f} {m_max-p_max:7.1f} | '
              f'{m_110:8.1f} {p_110:8.1f} {(m_110-p_110):8.1f}', flush=True)
    print('\n# margin>0 => model wins; trend should rise with k (smaller Tg)', flush=True)
    print('# DONE', flush=True)


if __name__ == '__main__':
    main()

"""
Phase-0 (corrected): reduced-frequency sweep with the MPC FORMULATION FIXED.

The previous ksweep was invalid: the MPC was crippled by the 2nd-order flap
smoothing (DLPF=0.95), which delays/attenuates the command and is fatal on fast
gusts. The diagnostic showed that stripping it (DLPF=0, no gust-gating) recovers
the MPC to 73% at Tg=0.50. Here we re-run the k-sweep with BOTH controllers on
the same, physically-limited post-processing: DLPF=0 + 300 deg/s rate limit (an
actuator limit, not an extra filter), and trace each side's CL-vs-pitch frontier.

Verdict per k: max CLred at iso-pitch (pitch_ratio <= thr), model vs PD. Does the
model now BEAT the PD (margin>0), and how does the margin vary with k?
"""
import os, numpy as np
import mpc_gust as M
import structure as _st
from controller import Controller

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
LAM = float(M.a._z_leak)
_dummy = Controller(aero_predict=M.a.predict, U=U, dt=DT)
_rk4b = _dummy._rk4_batch

W0 = float(os.environ.get('W0', '20'))
TEND = float(os.environ.get('TEND', '2.5'))
TG_GRID = [1.20, 1.00, 0.70, 0.50, 0.40, 0.35, 0.30]
# fixed MPC formulation: no extra smoothing, no gust-gating (RQUIET=R), rate-limited only
RMPC = 1e-3
MODEL_FIXED = dict(NH=6, NGRID=15, DLPF=0.0, RQUIET=RMPC, RW=RMPC, SCHED=False)
PD_DLPF = 0.0


def kred(Tg): return np.pi / (80.0 * Tg)


def win_stats(r, Tg):
    mw = M._win(r['_t'], Tg)
    return (float(np.max(np.abs(r['CL'][mw] - CLTRIM))), float(np.max(np.abs(r['al'][mw]))))


def batch_pd_traj(Tg, G1, G2, DLPF=PD_DLPF, DMAX=14.):
    a = M.a; B = len(G1)
    N = int(round(TEND / DT)) + 1
    tg = np.arange(N) * DT; Wt = np.array([M.gust(t, W0, Tg) for t in tg])
    a.reset(dt=DT)
    z_b = np.tile(np.asarray(a._z, float).reshape(1, -1), (B, 1))
    x_b = np.tile(np.asarray(M.X0, float).reshape(1, -1), (B, 1))
    G1 = np.asarray(G1, float); G2 = np.asarray(G2, float)
    de_f = np.zeros(B); de_f2 = np.zeros(B); prev = np.zeros(B); rate = 300.0 * DT
    clexc = np.zeros(B); pitchpk = np.zeros(B)
    for i in range(N):
        Wi = float(Wt[i])
        clm, _, _ = a.batch_step(z_b, x_b, de_f2, Wi, U, DT)
        d = G1 * (clm - CLTRIM) + G2 * x_b[:, 3]
        d = np.clip(d, -DMAX, DMAX); d = np.clip(d, prev - rate, prev + rate); prev = d
        if DLPF > 0:
            de_f = DLPF * de_f + (1 - DLPF) * d; de_f2 = DLPF * de_f2 + (1 - DLPF) * de_f
        else:
            de_f2 = d
        cl, cm, z_new = a.batch_step(z_b, x_b, de_f2, Wi, U, DT)
        z_b = z_new - LAM * z_b
        x_b = _rk4b(x_b, q * cl, q * cm * C, DT)
        if tg[i] <= (Tg + 0.5):
            clexc = np.maximum(clexc, np.abs(cl - CLTRIM))
            pitchpk = np.maximum(pitchpk, np.abs(x_b[:, 2]))
    return clexc, pitchpk


def best_at(pts, cex, pop, thr):
    cand = [(cex - ce) / cex * 100 for ce, pk in pts if pk / pop <= thr + 1e-9]
    return max(cand) if cand else float('nan')


def main():
    print(f'# ksweep2 (MPC formulation fixed: DLPF=0, no gating)  W0={W0} TEND={TEND} DAMULT={os.environ.get("DAMULT","1")}', flush=True)
    g1s = [-10., -20., -40., -60., -80., -120.]; g2s = [-16., -12., -8., -4., -2., 0., 2., 4.]
    G1 = np.array([a for a in g1s for _ in g2s]); G2 = np.array([b for _ in g1s for b in g2s])
    MQAD = [0., 30., 100., 300.]
    print(f'\n{"Tg":>5s} {"k":>6s} | {"mdl_max":>7s} {"pd_max":>6s} {"marg":>6s} | '
          f'{"mdl@115":>7s} {"pd@115":>6s} {"marg":>6s} | {"mdl@105":>7s} {"pd@105":>6s} {"marg":>6s}', flush=True)
    print('-' * 92, flush=True)
    for Tg in TG_GRID:
        OL = M.simulate('open', W0, Tg, TEND=TEND, QAD=100., **MODEL_FIXED)
        cex, pop = win_stats(OL, Tg)
        mpts = []
        for QAD in MQAD:
            r = M.simulate('mpc', W0, Tg, TEND=TEND, QAD=QAD, **MODEL_FIXED)
            mpts.append(win_stats(r, Tg))
        ce_b, pk_b = batch_pd_traj(Tg, G1, G2)
        ppts = list(zip(ce_b.tolist(), pk_b.tolist()))
        row = []
        for thr in [1e9, 1.15, 1.05]:
            m = best_at(mpts, cex, pop, thr); p = best_at(ppts, cex, pop, thr)
            row += [m, p, m - p]
        print(f'{Tg:5.2f} {kred(Tg):6.3f} | {row[0]:7.1f} {row[1]:6.1f} {row[2]:6.1f} | '
              f'{row[3]:7.1f} {row[4]:6.1f} {row[5]:6.1f} | {row[6]:7.1f} {row[7]:6.1f} {row[8]:6.1f}', flush=True)
    print('\n# marg = model - PD (>0 model wins). mdl@115 = max CLred with pitch_ratio<=1.15', flush=True)
    print('# DONE', flush=True)


if __name__ == '__main__':
    main()

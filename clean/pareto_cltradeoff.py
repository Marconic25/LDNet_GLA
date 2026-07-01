"""
Definitive test: the C_L-vs-pitch Pareto frontier on a pitch-active cell.

The thesis claim ("reduce C_L without exciting torsion") is decided by whether the
model achieves MORE C_L reduction at the SAME torsion excitation than the best
fixed-gain PD. We trace each controller's trade-off frontier by sweeping its
trade-off knob:
  * model: pitch weight QAD (with R fixed, full control authority)
  * PD2  : the gain pair (g1, g2)
and plot CLred% vs pitch_ratio = peak|alpha|_closed / peak|alpha|_open.

If the model frontier dominates (lies above-left of the PD frontier), the model is
genuinely better -- it is not merely more timid, it buys more lift reduction per
unit of torsion. If the frontiers overlap, a tuned PD is as good.
"""
import os, numpy as np
import mpc_gust as M
import structure as _st
from controller import Controller

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
_dummy = Controller(aero_predict=M.a.predict, U=U, dt=DT)
_rk4b = _dummy._rk4_batch

CELL = os.environ.get('CELL', 'W30/T1.0')
GRID = {'W30/T0.5': (30., 0.5), 'W30/T1.0': (30., 1.0), 'W20/T0.5': (20., 0.5),
        'W20/T1.0': (20., 1.0), 'W10/T1.0': (10., 1.0)}
W0, Tg = GRID[CELL]
TEND = float(os.environ.get('TEND', '2.5'))
MODEL_FIXED = dict(NH=6, NGRID=15, DLPF=0.95, SCHED=False)


def win_stats(r):
    mw = M._win(r['_t'], Tg)
    return (float(np.max(np.abs(r['CL'][mw] - CLTRIM))), float(np.max(np.abs(r['al'][mw]))))


def batch_pd_traj(G1, G2, DLPF=0.95, DMAX=14.):
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
        clm, _, _ = a.batch_step(z_b, x_b, de_f2, Wi, U, DT)
        d = G1 * (clm - CLTRIM) + G2 * x_b[:, 3]
        d = np.clip(d, -DMAX, DMAX); d = np.clip(d, prev - rate, prev + rate); prev = d
        de_f = DLPF * de_f + (1 - DLPF) * d; de_f2 = DLPF * de_f2 + (1 - DLPF) * de_f
        cl, cm, z_b = a.batch_step(z_b, x_b, de_f2, Wi, U, DT)
        x_b = _rk4b(x_b, q * cl, q * cm * C, DT)
        if tg[i] <= (Tg + 0.5):
            clexc = np.maximum(clexc, np.abs(cl - CLTRIM))
            pitchpk = np.maximum(pitchpk, np.abs(x_b[:, 2]))
    return clexc, pitchpk


def pareto(pts):
    """pts: list of (pitch_ratio, clred). Keep non-dominated (low pitch, high clred)."""
    pts = sorted(pts, key=lambda p: (p[0], -p[1]))
    out = []; best = -1e9
    for pr, cr in pts:
        if cr > best + 1e-9:
            out.append((pr, cr)); best = cr
    return out


def main():
    print(f'# pareto  cell={CELL}  W0={W0} Tg={Tg}  TEND={TEND}  DAMULT={os.environ.get("DAMULT","1")}', flush=True)
    OL = M.simulate('open', W0, Tg, TEND=TEND, QAD=100., RW=0.01, **MODEL_FIXED)
    cex, pop = win_stats(OL)
    print(f'# open-loop: clexc={cex:.4f}  pitchpeak={pop:.5f}', flush=True)

    # ---- model frontier: sweep its FULL knob space (QAD x R), symmetric to the PD's (g1,g2) ----
    print('\n## MODEL points (sweep QAD x R)', flush=True)
    print(f'{"QAD":>6s} {"R":>9s} {"CLred%":>7s} {"pitch":>7s}', flush=True)
    mpts = []
    for QAD in [0., 3., 10., 30., 100., 300., 1000.]:
        for Rm in [3e-4, 1e-3, 3e-3, 1e-2]:
            r = M.simulate('mpc', W0, Tg, TEND=TEND, RW=Rm, QAD=QAD, **MODEL_FIXED)
            ce, pk = win_stats(r); cr = (cex - ce) / cex * 100; pr = pk / pop
            mpts.append((pr, cr))
            print(f'{QAD:6.0f} {Rm:9.1e} {cr:7.1f} {pr:7.3f}', flush=True)

    # ---- PD2 frontier: dense (g1,g2) sweep, batched ----
    g1s = [-10., -20., -40., -60., -80., -120., -160., -220.]
    g2s = [-24., -16., -12., -8., -4., -2., 0., 2., 4.]
    G1 = np.array([a for a in g1s for _ in g2s]); G2 = np.array([b for _ in g1s for b in g2s])
    ce_b, pk_b = batch_pd_traj(G1, G2)
    pdpts = [(pk_b[i] / pop, (cex - ce_b[i]) / cex * 100) for i in range(len(G1))]
    pdfront = pareto(pdpts)
    print('\n## PD2 frontier (Pareto over g1,g2 sweep)', flush=True)
    print(f'{"pitch":>7s} {"CLred%":>7s}', flush=True)
    for pr, cr in pdfront:
        print(f'{pr:7.3f} {cr:7.1f}', flush=True)

    # ---- model Pareto frontier (envelope of the QAD x R sweep) ----
    mfront = pareto(mpts)
    print('\n## MODEL frontier (Pareto over QAD x R sweep)', flush=True)
    print(f'{"pitch":>7s} {"CLred%":>7s}', flush=True)
    for pr, cr in mfront:
        print(f'{pr:7.3f} {cr:7.1f}', flush=True)

    # ---- verdict: at matched pitch levels, who gives more CLred ----
    print('\n## VERDICT: max CLred achievable at pitch_ratio <= threshold', flush=True)
    print(f'{"pitch<=":>8s} {"model":>8s} {"PD2":>8s} {"winner":>8s}', flush=True)
    for thr in [1.00, 1.05, 1.10, 1.20, 1.40]:
        mc = max([cr for pr, cr in mpts if pr <= thr + 1e-9], default=float('nan'))
        pc = max([cr for pr, cr in pdpts if pr <= thr + 1e-9], default=float('nan'))
        win = 'model' if (mc == mc and (pc != pc or mc > pc)) else ('PD2' if pc == pc else '-')
        print(f'{thr:8.2f} {mc:8.1f} {pc:8.1f} {win:>8s}', flush=True)
    print('\n# DONE', flush=True)


if __name__ == '__main__':
    main()

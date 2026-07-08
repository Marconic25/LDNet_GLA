"""
Pass-2 CORRECTED: fair single-tuning comparison, model vs best fixed-gain PD,
on the thesis objective -- reduce peak C_L WITHOUT exciting torsion.

Fixes vs perf_pd_vs_model.py:
  * Objective is the thesis objective, not the pitch-energy J:
        score(cell) = clexc_closed/clexc_open  +  LAMBDA * max(0, pitch_ratio - 1)
    where clexc = peak |C_L - C_L_trim| and pitch_ratio = peak|alpha|_closed / peak|alpha|_open.
    Lower = better: less residual lift AND no torsion excited beyond open-loop.
  * The MODEL is tuned too (single R across the envelope, like the PD's single gain),
    so neither side is handicapped. Both get one config for the whole envelope.

Controllers: model (MPC, tuned R) ; PD1 = clip(g1*eCL) ; PD2 = clip(g1*eCL + g2*adot).
Reports per cell: CLred% (higher=better) and pitch_ratio (>1 = torsion excited).
"""
import os, numpy as np
import mpc_gust as M
import structure as _st
from controller import Controller

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
LAMBDA = float(os.environ.get('LAMBDA', '2.0'))
_dummy = Controller(aero_predict=M.a.predict, U=U, dt=DT)
_rk4b = _dummy._rk4_batch

GRID = {
    'W10/T0.5': (10., 0.5), 'W10/T1.0': (10., 1.0), 'W10/T2.0': (10., 2.0),
    'W20/T0.5': (20., 0.5), 'W20/T1.0': (20., 1.0), 'W20/T2.0': (20., 2.0),
    'W30/T0.5': (30., 0.5), 'W30/T1.0': (30., 1.0), 'W30/T2.0': (30., 2.0),
    'design'  : (11.46, 1.12),
}
MODEL_FIXED = dict(NH=6, NGRID=15, QAD=100., DLPF=0.95, SCHED=False)


def win_stats(r, Tg):
    """Return (clexc, pitchpeak) over the gust+ring-down window."""
    mw = M._win(r['_t'], Tg)
    clexc = float(np.max(np.abs(r['CL'][mw] - CLTRIM)))
    pitchpk = float(np.max(np.abs(r['al'][mw])))
    return clexc, pitchpk


def batch_pd_traj(W0, Tg, G1, G2, TEND, DLPF=0.95, DMAX=14.):
    """Vectorized B parallel 2-ch-PD sims. Returns (clexc_b, pitchpeak_b) per candidate."""
    a = M.a; B = len(G1)
    N = int(round(TEND / DT)) + 1
    tg = np.arange(N) * DT
    Wt = np.array([M.gust(t, W0, Tg) for t in tg])
    a.reset(dt=DT)
    z_b = np.tile(np.asarray(a._z, dtype=float).reshape(1, -1), (B, 1))
    x_b = np.tile(np.asarray(M.X0, dtype=float).reshape(1, -1), (B, 1))
    G1 = np.asarray(G1, float); G2 = np.asarray(G2, float)
    de_f = np.zeros(B); de_f2 = np.zeros(B); prev = np.zeros(B)
    rate = 300.0 * DT
    clexc = np.zeros(B); pitchpk = np.zeros(B)
    for i in range(N):
        Wi = float(Wt[i])
        clm, _, _ = a.batch_step(z_b, x_b, de_f2, Wi, U, DT)
        d = G1 * (clm - CLTRIM) + G2 * x_b[:, 3]
        d = np.clip(d, -DMAX, DMAX); d = np.clip(d, prev - rate, prev + rate); prev = d
        de_f = DLPF * de_f + (1.0 - DLPF) * d
        de_f2 = DLPF * de_f2 + (1.0 - DLPF) * de_f; de = de_f2
        cl, cm, z_b = a.batch_step(z_b, x_b, de, Wi, U, DT)
        x_b = _rk4b(x_b, q * cl, q * cm * C, DT)
        if tg[i] <= (Tg + 0.5):
            clexc = np.maximum(clexc, np.abs(cl - CLTRIM))
            pitchpk = np.maximum(pitchpk, np.abs(x_b[:, 2]))
    return clexc, pitchpk


def sim_pd2_full(W0, Tg, g1, g2, TEND, DLPF=0.95, DMAX=14.):
    """Faithful scalar 2-ch PD (for final reporting), returns metrics dict + (clexc,pitch)."""
    a = M.a
    N = int(round(TEND / DT)) + 1
    tg = np.arange(N) * DT
    Wt = np.array([M.gust(t, W0, Tg) for t in tg])
    a.reset(dt=DT)
    x = M.X0.copy(); de_f = 0.0; de_f2 = 0.0; prev = 0.0; rate = 300.0 * DT
    R = {k: [] for k in ['al', 'ad', 'de', 'CL']}
    for i in range(N):
        clm, _ = a.predict(x, de_f2, float(Wt[i]), U)
        d = g1 * (float(clm) - CLTRIM) + g2 * float(x[3])
        d = float(np.clip(d, -DMAX, DMAX)); d = float(np.clip(d, prev - rate, prev + rate)); prev = d
        de_f = DLPF * de_f + (1.0 - DLPF) * d
        de_f2 = DLPF * de_f2 + (1.0 - DLPF) * de_f; de = de_f2
        cl, cm = a.predict(x, de, Wt[i], U)
        a.advance(x, de, Wt[i], U, DT); x = M.structure.step_dp45(x, q * cl, q * cm * C, DT)
        R['al'].append(x[2]); R['ad'].append(x[3]); R['de'].append(de); R['CL'].append(float(cl))
    out = {k: np.array(v) for k, v in R.items()}; out['_t'] = tg
    return out


DECISIVE = ['W30/T0.5', 'W30/T1.0', 'W10/T2.0']


def main():
    TEND = float(os.environ.get('TEND', '2.5'))
    sel = os.environ.get('CELLS', 'decisive')
    names = DECISIVE if sel == 'decisive' else list(GRID.keys())
    cells = [(n, GRID[n]) for n in names]
    print(f'# perf2  TEND={TEND}  LAMBDA={LAMBDA}  model={MODEL_FIXED}  DAMULT={os.environ.get("DAMULT","1")}', flush=True)

    # ---- open-loop reference ----
    OPEN = {}
    for name, (W0, Tg) in cells:
        r = M.simulate('open', W0, Tg, TEND=TEND, **MODEL_FIXED)
        OPEN[name] = win_stats(r, Tg)
    print('# open-loop refs (clexc, pitchpeak) computed', flush=True)

    def score(clexc, pitch, ref):
        cex, pop = ref
        return (clexc / max(cex, 1e-12)) + LAMBDA * max(0.0, pitch / max(pop, 1e-12) - 1.0)

    # ---- tune model: single R across envelope ----
    R_grid = [5e-4, 1e-3, 3e-3, 1e-2]
    modscore = {R: 0.0 for R in R_grid}; modcell = {R: {} for R in R_grid}
    for R in R_grid:
        for name, (W0, Tg) in cells:
            r = M.simulate('mpc', W0, Tg, TEND=TEND, RW=R, **MODEL_FIXED)
            ce, pk = win_stats(r, Tg)
            modcell[R][name] = (ce, pk)
            modscore[R] += score(ce, pk, OPEN[name])
        modscore[R] /= len(cells)
        print(f'  [tune model] R={R:8.1e}  mean score={modscore[R]:.4f}', flush=True)
    Rb = min(R_grid, key=lambda R: modscore[R])
    print(f'# best model R={Rb:.1e}  mean score={modscore[Rb]:.4f}', flush=True)

    # ---- tune PD1 (batched) ----
    g1_grid = np.array([-10., -20., -40., -60., -80., -120., -160.])
    s1 = np.zeros(len(g1_grid))
    for name, (W0, Tg) in cells:
        ce, pk = batch_pd_traj(W0, Tg, g1_grid, np.zeros_like(g1_grid), TEND)
        ref = OPEN[name]
        s1 += (ce / max(ref[0], 1e-12)) + LAMBDA * np.maximum(0.0, pk / max(ref[1], 1e-12) - 1.0)
    s1 /= len(cells)
    g1b = float(g1_grid[int(np.argmin(s1))])
    print(f'# best PD1 g1={g1b}  mean score={s1.min():.4f}', flush=True)

    # ---- tune PD2 (batched) ----
    g1_2 = [-20., -40., -80., -120., -160.]
    g2_2 = [-16., -12., -8., -4., -2., 0., 2., 4., 8.]
    G1 = np.array([a for a in g1_2 for _ in g2_2]); G2 = np.array([b for _ in g1_2 for b in g2_2])
    s2 = np.zeros(len(G1))
    for name, (W0, Tg) in cells:
        ce, pk = batch_pd_traj(W0, Tg, G1, G2, TEND)
        ref = OPEN[name]
        s2 += (ce / max(ref[0], 1e-12)) + LAMBDA * np.maximum(0.0, pk / max(ref[1], 1e-12) - 1.0)
    s2 /= len(cells)
    bb = int(np.argmin(s2)); g1c = float(G1[bb]); g2c = float(G2[bb])
    print(f'# best PD2 g1={g1c} g2={g2c}  mean score={s2.min():.4f}', flush=True)

    # ---- final per-cell table ----
    print(f'\n## per-cell  CLred% (higher=better) | pitch_ratio (>1.00 = torsion excited)', flush=True)
    print(f'{"cell":10s} | {"model(R=%.0e)"%Rb:>16s} | {"PD1":>14s} | {"PD2":>14s}', flush=True)
    print(f'{"":10s} | {"CLred":>7s} {"pitch":>7s} | {"CLred":>6s} {"pitch":>6s} | {"CLred":>6s} {"pitch":>6s}', flush=True)
    agg = {'model': [], 'pd1': [], 'pd2': []}
    for name, (W0, Tg) in cells:
        cex, pop = OPEN[name]
        ce_m, pk_m = modcell[Rb][name]
        r1 = sim_pd2_full(W0, Tg, g1b, 0.0, TEND); ce1, pk1 = win_stats(r1, Tg)
        r2 = sim_pd2_full(W0, Tg, g1c, g2c, TEND); ce2, pk2 = win_stats(r2, Tg)
        cr_m = (cex - ce_m) / cex * 100; cr_1 = (cex - ce1) / cex * 100; cr_2 = (cex - ce2) / cex * 100
        pr_m = pk_m / pop; pr_1 = pk1 / pop; pr_2 = pk2 / pop
        agg['model'].append((cr_m, pr_m)); agg['pd1'].append((cr_1, pr_1)); agg['pd2'].append((cr_2, pr_2))
        print(f'{name:10s} | {cr_m:7.0f} {pr_m:7.2f} | {cr_1:6.0f} {pr_1:6.2f} | {cr_2:6.0f} {pr_2:6.2f}', flush=True)
    mm = {k: (np.mean([a for a, _ in v]), np.mean([b for _, b in v])) for k, v in agg.items()}
    print(f'{"MEAN":10s} | {mm["model"][0]:7.0f} {mm["model"][1]:7.2f} | '
          f'{mm["pd1"][0]:6.0f} {mm["pd1"][1]:6.2f} | {mm["pd2"][0]:6.0f} {mm["pd2"][1]:6.2f}', flush=True)
    print('\n# DONE', flush=True)


if __name__ == '__main__':
    main()

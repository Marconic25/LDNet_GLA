"""
Pass-2 performance test: best fixed-gain classical PD (1- and 2-channel) vs the
model-based controller, closed-loop, across the full gust envelope.

Fair comparison: every arm is scored on the SAME objective
    J = Q_CL*sum(dCL^2) + Q_ad*sum(adot^2) + R*sum(delta^2)   (mpc_gust.SCORE_W)
summed over the gust window. The model directly minimizes this J; the PDs are
TUNED (gain sweep) to minimize the envelope-aggregate of the same J -- i.e. the
classical controllers get their single best gain across the whole envelope, the
strongest fair shot. We then report, per cell, J/J_open (fraction of the
open-loop objective that remains), the peak-C_L reduction %, and the pitch
excitation (alpha_dot RMS).

Controllers:
  open : delta = 0
  model: fixed-cost receding-horizon MPC (ONE config, all cells)
  pd1  : delta = clip(g1 * (C_L - C_L_trim))                 [1-channel, lift]
  pd2  : delta = clip(g1 * (C_L - C_L_trim) + g2 * alpha_dot)[2-channel, +pitch-rate]
Both PDs share the model's flap saturation, rate limit, and 2nd-order smoothing.
"""
import os, numpy as np
import mpc_gust as M
import structure as _st
from controller import Controller

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
_dummy = Controller(aero_predict=M.a.predict, U=U, dt=DT)
_rk4b = _dummy._rk4_batch   # batched structural RK4 (same plant as mpc_gust)


def batch_pd_J(W0, Tg, G1, G2, TEND, DLPF=0.95, DMAX=14.):
    """Vectorized: run B=len(G1) parallel 2-channel-PD closed-loop sims (one per gain
    candidate) sharing one gust, using batched TF calls. dt_ref==DT so batch_step is
    exact (n_sub=1) -> identical plant to the scalar sim. Returns J (window) per candidate."""
    a = M.a; B = len(G1)
    N = int(round(TEND / DT)) + 1
    tg = np.arange(N) * DT
    Wt = np.array([M.gust(t, W0, Tg) for t in tg])
    a.reset(dt=DT)
    z_b = np.tile(np.asarray(a._z, dtype=float).reshape(1, -1), (B, 1))
    x_b = np.tile(np.asarray(M.X0, dtype=float).reshape(1, -1), (B, 1))
    G1 = np.asarray(G1, float); G2 = np.asarray(G2, float)
    de_f = np.zeros(B); de_f2 = np.zeros(B); prev = np.zeros(B)
    rate = 300.0 * DT; w = M.SCORE_W; J = np.zeros(B)
    for i in range(N):
        Wi = float(Wt[i])
        clm, _, _ = a.batch_step(z_b, x_b, de_f2, Wi, U, DT)     # read-only measure at applied flap
        d = G1 * (clm - CLTRIM) + G2 * x_b[:, 3]
        d = np.clip(d, -DMAX, DMAX); d = np.clip(d, prev - rate, prev + rate); prev = d
        de_f = DLPF * de_f + (1.0 - DLPF) * d
        de_f2 = DLPF * de_f2 + (1.0 - DLPF) * de_f; de = de_f2
        cl, cm, z_b = a.batch_step(z_b, x_b, de, Wi, U, DT)      # force + advance z
        x_b = _rk4b(x_b, q * cl, q * cm * C, DT)
        if tg[i] <= (Tg + 0.5):
            J += w['Q_CL'] * (cl - CLTRIM) ** 2 + w['Q_ad'] * x_b[:, 3] ** 2 + w['R'] * de ** 2
    return J

GRID = {
    'W10/T0.5': (10., 0.5), 'W10/T1.0': (10., 1.0), 'W10/T2.0': (10., 2.0),
    'W20/T0.5': (20., 0.5), 'W20/T1.0': (20., 1.0), 'W20/T2.0': (20., 2.0),
    'W30/T0.5': (30., 0.5), 'W30/T1.0': (30., 1.0), 'W30/T2.0': (30., 2.0),
    'design'  : (11.46, 1.12),
}

MODEL_KW = dict(NH=6, NGRID=15, QAD=100., RW=0.01, DLPF=0.95, SCHED=False)


def sim_pd2(W0, Tg, g1, g2, TEND, DLPF=0.95, DMAX=14.):
    """2-channel PD closed-loop, mirroring mpc_gust.simulate's prop arm + smoothing.
    g2=0 reproduces the 1-channel prop law exactly (cross-check)."""
    a = M.a
    N = int(round(TEND / DT)) + 1
    tg = np.arange(N) * DT
    Wt = np.array([M.gust(t, W0, Tg) for t in tg])
    a.reset(dt=DT)
    x = M.X0.copy(); de_f = 0.0; de_f2 = 0.0; prev = 0.0
    rate = 300.0 * DT
    R = {k: [] for k in ['h', 'hd', 'al', 'ad', 'hdd', 'add', 'de', 'CL', 'CM', 'Fy']}
    for i in range(N):
        clm, _ = a.predict(x, de_f2, float(Wt[i]), U)          # measured C_L with applied flap
        d = g1 * (float(clm) - CLTRIM) + g2 * float(x[3])       # lift + pitch-rate feedback
        d = float(np.clip(d, -DMAX, DMAX))
        d = float(np.clip(d, prev - rate, prev + rate)); prev = d
        de_f = DLPF * de_f + (1.0 - DLPF) * d                   # same 2nd-order smoothing chain
        de_f2 = DLPF * de_f2 + (1.0 - DLPF) * de_f
        de = de_f2
        cl, cm = a.predict(x, de, Wt[i], U); Fy = q * cl; Mz = q * cm * C
        der = M.structure.rhs(x, Fy, Mz)
        a.advance(x, de, Wt[i], U, DT); x = M.structure.step_dp45(x, Fy, Mz, DT)
        for k, v in zip(['h', 'hd', 'al', 'ad', 'hdd', 'add', 'de', 'CL', 'CM', 'Fy'],
                        [x[0], x[1], x[2], x[3], der[1], der[3], de, float(cl), float(cm), Fy]):
            R[k].append(v)
    out = {k: np.array(v) for k, v in R.items()}
    out['_t'] = tg; out['_Wt'] = Wt; out['_comp_ms'] = 0.0
    out['_cfg'] = dict(QAD=0, RW=0, DLPF=DLPF, gain=g1)
    return out


def main():
    TEND = float(os.environ.get('TEND', '2.5'))
    cells = list(GRID.keys())
    print(f'# perf_pd_vs_model  TEND={TEND}  model={MODEL_KW}  DAMULT={os.environ.get("DAMULT","1")}', flush=True)

    # ---- reference: open-loop + model, per cell ----
    OL = {}; MP = {}; Jopen = {}; Jmodel = {}
    for name, (W0, Tg) in GRID.items():
        OL[name] = M.simulate('open', W0, Tg, TEND=TEND, **MODEL_KW)
        MP[name] = M.simulate('mpc',  W0, Tg, TEND=TEND, **MODEL_KW)
        mw = M._win(OL[name]['_t'], Tg)
        Jopen[name] = M.achieved_J(OL[name], mw)
        Jmodel[name] = M.achieved_J(MP[name], mw)

    def aggregate(Jdict):
        return float(np.mean([Jdict[n] / max(Jopen[n], 1e-12) for n in cells]))

    cell_list = list(GRID.items())

    # ---- tune PD-1ch: batched sweep over g1, minimize envelope-mean J/Jopen ----
    g1_grid = np.array([-10., -20., -40., -60., -80., -120., -160.])
    agg1 = np.zeros(len(g1_grid))
    for name, (W0, Tg) in cell_list:
        agg1 += batch_pd_J(W0, Tg, g1_grid, np.zeros_like(g1_grid), TEND) / max(Jopen[name], 1e-12)
    agg1 /= len(cells)
    b1 = int(np.argmin(agg1)); g1b = float(g1_grid[b1])
    for g, a in zip(g1_grid, agg1):
        print(f'  [tune pd1] g1={g:7.1f}  mean J/Jopen={a:.4f}{"  <-- best" if g == g1b else ""}', flush=True)
    print(f'# best PD1: g1={g1b}  mean J/Jopen={agg1[b1]:.4f}', flush=True)

    # ---- tune PD-2ch: batched sweep over (g1,g2) ----
    g1_2 = [-20., -40., -80., -120.]
    g2_2 = [-12., -8., -4., -2., 0., 2., 4., 8., 12.]
    G1 = np.array([a for a in g1_2 for _ in g2_2])
    G2 = np.array([b for _ in g1_2 for b in g2_2])
    agg2 = np.zeros(len(G1))
    for name, (W0, Tg) in cell_list:
        agg2 += batch_pd_J(W0, Tg, G1, G2, TEND) / max(Jopen[name], 1e-12)
    agg2 /= len(cells)
    b2 = int(np.argmin(agg2)); g1c = float(G1[b2]); g2c = float(G2[b2])
    print(f'# best PD2: g1={g1c} g2={g2c}  mean J/Jopen={agg2[b2]:.4f}', flush=True)

    # ---- final per-cell table: rerun best PDs to get full metrics ----
    print('\n## per-cell  (J/Jopen: lower=better;  CLred%: higher=better;  adot_RMS deg/s)', flush=True)
    hdr = f'{"cell":10s} | {"model":>21s} | {"PD1 (lift)":>21s} | {"PD2 (lift+pitch)":>21s}'
    print(hdr, flush=True)
    print(f'{"":10s} | {"J/Jo":>6s} {"CLred":>6s} {"ad":>6s} | {"J/Jo":>6s} {"CLred":>6s} {"ad":>6s} | {"J/Jo":>6s} {"CLred":>6s} {"ad":>6s}', flush=True)
    agg = {'model': [], 'pd1': [], 'pd2': []}
    for name, (W0, Tg) in GRID.items():
        mw = M._win(OL[name]['_t'], Tg)
        mm = M.metrics(MP[name], OL[name], Tg)
        r1 = sim_pd2(W0, Tg, g1b, 0.0, TEND); m1 = M.metrics(r1, OL[name], Tg)
        r2 = sim_pd2(W0, Tg, g1c, g2c, TEND); m2 = M.metrics(r2, OL[name], Tg)
        jm = Jmodel[name] / Jopen[name]
        j1 = M.achieved_J(r1, mw) / Jopen[name]; j2 = M.achieved_J(r2, mw) / Jopen[name]
        agg['model'].append(jm); agg['pd1'].append(j1); agg['pd2'].append(j2)
        print(f'{name:10s} | {jm:6.3f} {mm["clred"]:6.0f} {mm["adrms"]:6.2f} '
              f'| {j1:6.3f} {m1["clred"]:6.0f} {m1["adrms"]:6.2f} '
              f'| {j2:6.3f} {m2["clred"]:6.0f} {m2["adrms"]:6.2f}', flush=True)
    print(f'{"MEAN":10s} | {np.mean(agg["model"]):6.3f} {"":6s} {"":6s} '
          f'| {np.mean(agg["pd1"]):6.3f} {"":6s} {"":6s} '
          f'| {np.mean(agg["pd2"]):6.3f}', flush=True)
    print('\n# DONE', flush=True)


if __name__ == '__main__':
    main()

"""
Diagnose the MPC collapse at high k (Tg=0.50, W0=20): is it the FORMULATION
(constant-delta + gust-gated R + 2nd-order smoothing) or fundamental?

We run the MPC with the sluggishness sources stripped one at a time:
  baseline   : DLPF=0.95, gating (RQUIET=0.1)   <- the collapsed config
  no-smooth  : DLPF=0.0
  no-gate    : RQUIET=RW (aggressive throughout)
  both       : DLPF=0.0 AND RQUIET=RW
and compare to the best 2-channel PD. If the MPC RECOVERS when stripped ->
the collapse is formulation (fixable). If it stays ~0% -> fundamental.

Outputs a summary table + a W/CL/delta time-history plot.
"""
import os, numpy as np
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
import mpc_gust as M

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
W0 = float(os.environ.get('W0', '20')); Tg = float(os.environ.get('TG', '0.50'))
TEND = float(os.environ.get('TEND', '2.5'))
RW = 1e-3; QAD = 0.0
MK = dict(NH=6, NGRID=15, SCHED=False, QAD=QAD, RW=RW)


def stats(r):
    mw = M._win(r['_t'], Tg)
    cex = float(np.max(np.abs(r['CL'][mw] - CLTRIM)))
    pk = float(np.max(np.abs(r['al'][mw])))
    de = r['de']; t = r['_t']
    dmax = float(np.max(np.abs(de[mw]))); tpk = float(t[mw][int(np.argmax(np.abs(de[mw])))])
    return cex, pk, dmax, tpk


def sim_pd2(g1, g2, DLPF=0.95, DMAX=14.):
    a = M.a; N = int(round(TEND / DT)) + 1
    tg = np.arange(N) * DT; Wt = np.array([M.gust(t, W0, Tg) for t in tg])
    a.reset(dt=DT); x = M.X0.copy(); de_f = 0.; de_f2 = 0.; prev = 0.; rate = 300. * DT
    R = {k: [] for k in ['al', 'ad', 'de', 'CL']}
    for i in range(N):
        clm, _ = a.predict(x, de_f2, float(Wt[i]), U)
        d = g1 * (float(clm) - CLTRIM) + g2 * float(x[3])
        d = float(np.clip(d, -DMAX, DMAX)); d = float(np.clip(d, prev - rate, prev + rate)); prev = d
        de_f = DLPF * de_f + (1 - DLPF) * d; de_f2 = DLPF * de_f2 + (1 - DLPF) * de_f
        cl, cm = a.predict(x, de_f2, Wt[i], U)
        a.advance(x, de_f2, Wt[i], U, DT); x = M.structure.step_rk4(x, q * cl, q * cm * C, DT)
        R['al'].append(x[2]); R['ad'].append(x[3]); R['de'].append(de_f2); R['CL'].append(float(cl))
    out = {k: np.array(v) for k, v in R.items()}; out['_t'] = tg; out['_Wt'] = Wt
    return out


def main():
    print(f'# diag_collapse  W0={W0} Tg={Tg} k={np.pi/(80*Tg):.3f}  R={RW} QAD={QAD}  DAMULT={os.environ.get("DAMULT","1")}', flush=True)
    OL = M.simulate('open', W0, Tg, TEND=TEND, DLPF=0.95, RQUIET=0.1, **MK)
    cex0, pk0, _, _ = stats(OL)
    print(f'# open: clexc={cex0:.4f} pitchpeak={pk0:.5f}  (gust peak at t={Tg/2:.3f}s)', flush=True)

    variants = {
        'MPC baseline (smooth+gate)': dict(DLPF=0.95, RQUIET=0.1),
        'MPC no-smooth':              dict(DLPF=0.0,  RQUIET=0.1),
        'MPC no-gate':                dict(DLPF=0.95, RQUIET=RW),
        'MPC both stripped':          dict(DLPF=0.0,  RQUIET=RW),
    }
    runs = {}
    print(f'\n{"variant":28s} {"CLred%":>7s} {"pitch":>6s} {"|d|max":>7s} {"t(dmax)":>8s}', flush=True)
    for name, kw in variants.items():
        r = M.simulate('mpc', W0, Tg, TEND=TEND, **{**MK, **kw}); runs[name] = r
        ce, pk, dmax, tpk = stats(r)
        print(f'{name:28s} {(cex0-ce)/cex0*100:7.1f} {pk/pk0:6.2f} {dmax:7.2f} {tpk:8.3f}', flush=True)

    # best PD2 (small scalar sweep)
    best = (None, -1e9, None)
    for g1 in [-20., -40., -60., -80., -120.]:
        for g2 in [-12., -8., -4., 0.]:
            r = sim_pd2(g1, g2); ce, pk, _, _ = stats(r)
            if pk / pk0 <= 1.15 and (cex0 - ce) / cex0 * 100 > best[1]:
                best = ((g1, g2), (cex0 - ce) / cex0 * 100, r)
    rpd = best[2]; ce, pk, dmax, tpk = stats(rpd)
    print(f'{"PD2 best "+str(best[0]):28s} {(cex0-ce)/cex0*100:7.1f} {pk/pk0:6.2f} {dmax:7.2f} {tpk:8.3f}', flush=True)
    runs['PD2 best'] = rpd

    # ---- plot (optional) ----
    if not HAS_PLT:
        print('\n# matplotlib unavailable -> skipping plot (numbers above are the verdict)', flush=True)
        print('# DONE', flush=True); return
    sel = ['MPC baseline (smooth+gate)', 'MPC both stripped', 'PD2 best']
    cols = {'MPC baseline (smooth+gate)': 'crimson', 'MPC both stripped': 'darkorange', 'PD2 best': 'seagreen'}
    t = OL['_t']; mw = t <= TEND
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax[0].plot(t[mw], OL['_Wt'][mw], 'k', lw=1.3); ax[0].set_ylabel('W gust [m/s]')
    ax[1].plot(t[mw], (OL['CL'][mw] - CLTRIM), 'k--', lw=1.0, label='open')
    for s in sel:
        ax[1].plot(t[mw], runs[s]['CL'][mw] - CLTRIM, cols[s], lw=1.3, label=s)
        ax[2].plot(t[mw], runs[s]['de'][mw], cols[s], lw=1.3, label=s)
    ax[1].set_ylabel('C_L - C_L_trim'); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    ax[2].set_ylabel('delta [deg]'); ax[2].set_xlabel('t [s]'); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    for a in ax: a.axvline(Tg/2, color='gray', ls=':', lw=.8); a.axvspan(0, Tg, color='lightblue', alpha=.15)
    ax[0].set_title(f'MPC collapse diagnostic  W0={W0} Tg={Tg} (k={np.pi/(80*Tg):.3f})')
    fig.tight_layout(); fn = f'results/diag_collapse_W{int(W0)}_Tg{int(Tg*100)}.png'
    import os as _o; _o.makedirs('results', exist_ok=True); fig.savefig(fn, dpi=120)
    print(f'\n# saved {fn}', flush=True)
    print('# DONE', flush=True)


if __name__ == '__main__':
    main()

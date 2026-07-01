"""Targeted trace dump for the W30/T0.5 money plot: best proportional vs best
single-step optimal (small R sweep), saving full time histories."""
import os, numpy as np
import onestep as O

W0, Tg = 30.0, 0.5
t, W, alO, CLo, deO, _ = O.run_open(W0, Tg)
cex0 = float(np.max(np.abs(CLo[t <= Tg + 0.5] - O.CLTRIM)))
pk0 = float(np.max(np.abs(alO[t <= Tg + 0.5])))
print(f'# W30/T0.5  open clexc={cex0:.4f} pitchpeak={pk0:.5f}', flush=True)


def stats(r):
    return O.win_stats(r[2], r[3], r[4], t, Tg)

# best proportional
bp = (None, -1e9, None)
for g in [-10., -20., -40., -60., -80., -120., -160.]:
    r = O.run_prop(W0, Tg, g); ce, pk, _ = stats(r); cr = (cex0 - ce) / cex0 * 100
    print(f'  prop g={g:6.0f}: CLred={cr:6.1f}% pitch={pk/pk0:.2f}', flush=True)
    if cr > bp[1]: bp = (g, cr, r)
gstar, pcr, rp = bp

# best single-step (small R sweep)
bo = (None, -1e9, None)
for R in [3e-3, 1e-3, 3e-4]:
    r = O.run_onestep(W0, Tg, R); ce, pk, _ = stats(r); cr = (cex0 - ce) / cex0 * 100
    print(f'  1step R={R:8.0e}: CLred={cr:6.1f}% pitch={pk/pk0:.2f}', flush=True)
    if cr > bo[1]: bo = (R, cr, r)
Rstar, ocr, ro = bo

print(f'# BEST prop g*={gstar} {pcr:.1f}%   |   1step R*={Rstar:.0e} {ocr:.1f}%', flush=True)
import os as _o; _o.makedirs('results', exist_ok=True)
np.savez('results/w30t05_traces.npz', t=t, W=W, cex0=cex0, pk0=pk0,
         prop_de=rp[4], prop_CL=rp[3], prop_K=rp[5], prop_al=rp[2],
         one_de=ro[4], one_CL=ro[3], one_K=ro[5], one_al=ro[2],
         gstar=float(gstar), Rstar=float(Rstar), pcr=pcr, ocr=ocr)
print('# saved results/w30t05_traces.npz', flush=True)
print('# DONE', flush=True)

"""
Fix the two cs25_wnext defects WITHOUT changing the objective:
  (1) staircase delta  -> finer flap grid + parabolic sub-cell refine (continuous
      argmin; cost J unchanged).
  (2) low-CLred cells  -> hypothesis: staircase flap excites the pitch mode, so the
      pitch<=1.10*open pick rule forces a gentle R and CLred collapses; smoothing
      the flap should cut pitch excitation and let the aggressive R stay in budget.

Run ONLY the low-CLred cells + one good control cell, comparing baseline vs a
smoothed config, full R-sweep, with a flap-roughness metric. Save delta-comparison
plots so we can eyeball the staircase.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import harness as H
from controllers import OptGrid

CELLS = [(30.0, 0.40),   # good control
         (20.0, 0.50), (30.0, 0.30), (30.0, 0.50),
         (30.0, 1.00), (30.0, 1.20)]
R_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
PITCH_TOL = 1.10
CONFIGS = [('base   G161      ', dict(G=161, refine=False)),
           ('refine G161      ', dict(G=161, refine=True)),
           ('smooth G481+ref  ', dict(G=481, refine=True))]
OUTD = 'results_cs25_fix'; os.makedirs(OUTD, exist_ok=True)


def chatter(de, mw):
    d = de[mw]
    tv = float(np.sum(np.abs(np.diff(d))))            # total variation [deg]
    rough = float(np.sum(np.abs(np.diff(d, 2))))      # staircase roughness (2nd diff) [deg]
    return tv, rough


print(f"# cs25_fix  cells={[f'W{int(w)}/T{t:.2f}' for w,t in CELLS]}  DAMULT={os.environ.get('DAMULT','1')}", flush=True)
for (W0, Tg) in CELLS:
    cell = f'W{int(W0)}/T{Tg:.2f}'
    OL = H.scalar_rollout(None, W0, Tg)
    t = OL['_t']; mw = H.window(t, Tg)
    pk0 = float(np.max(np.abs(OL['al'][mw]))); cex0 = H.cex_of(OL['CL'], t, Tg)
    print(f"\n=== {cell}  cex0={cex0:.4f} ===", flush=True)
    best_by_cfg = {}
    for cname, cfg in CONFIGS:
        rows = []
        for R in R_GRID:
            r = H.scalar_rollout(OptGrid(R=R, gate='hard', use_wnext=True, **cfg), W0, Tg)
            m = H.metrics(r, OL, Tg); pr = m['pitchpk']/max(pk0, 1e-12)
            tv, rough = chatter(r['de'], mw)
            rows.append((R, m['clred'], pr, m['flap_max'], rough, m['flag'], r))
        crs = np.array([x[1] for x in rows]); prs = np.array([x[2] for x in rows])
        ok = np.where(prs <= PITCH_TOL)[0]
        jb = int(ok[np.argmax(crs[ok])]) if len(ok) else int(np.argmin(prs))
        R, cl, pr, fm, rough, flag, r = rows[jb]
        best_by_cfg[cname] = (r, R, cl, pr, rough)
        print(f"  {cname}: per-R CLred/pitch -> " +
              "  ".join(f"{x[1]:+.0f}%/{x[2]:.2f}" for x in rows), flush=True)
        print(f"  {cname}: R*={R:g} CLred={cl:+.1f}% pitch={pr:.2f} flap={fm:.1f} "
              f"rough={rough:.2f} {flag}", flush=True)
    # delta-comparison plot: base vs smooth
    fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    rb = best_by_cfg['base   G161      '][0]; rs = best_by_cfg['smooth G481+ref  '][0]
    ax[0].axhline(H.CLTRIM, color='gray', lw=0.7, ls=':')
    ax[0].plot(t, OL['CL'], 'k--', lw=1.2, label='open')
    ax[0].plot(t, rb['CL'], color='0.6', lw=1.2, label=f"base ({best_by_cfg['base   G161      '][2]:+.0f}%)")
    ax[0].plot(t, rs['CL'], 'r-', lw=1.3, label=f"smooth ({best_by_cfg['smooth G481+ref  '][2]:+.0f}%)")
    ax[0].set_ylabel('C_L'); ax[0].legend(fontsize=7, frameon=False); ax[0].axvspan(0, Tg, alpha=0.12, color='#aad4f5')
    ax[1].plot(t, rb['de'], color='0.6', lw=1.2, label=f"base rough={best_by_cfg['base   G161      '][4]:.1f}")
    ax[1].plot(t, rs['de'], 'r-', lw=1.3, label=f"smooth rough={best_by_cfg['smooth G481+ref  '][4]:.1f}")
    ax[1].set_ylabel('delta [deg]'); ax[1].set_xlabel('time [s]'); ax[1].legend(fontsize=7, frameon=False)
    ax[1].axvspan(0, Tg, alpha=0.12, color='#aad4f5'); ax[1].set_xlim(0, min(2.0, Tg+1.0))
    fig.suptitle(f'{cell}  base vs smooth (G481+refine)', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTD, f"fix_{cell.replace('/','_')}.png"), dpi=140, bbox_inches='tight')
    plt.close(fig)
print("\n# DONE", flush=True)

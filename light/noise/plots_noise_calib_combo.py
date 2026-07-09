"""
Figure: CLred vs calibration error (bias, gain) for the E2-combo.
Reads results/A2_calib.npz. Run locally after scp.
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
# our own npz (harness save_records object-array schema) -- trusted local file
recs = list(np.load(os.path.join(DIR, 'A2_calib.npz'), allow_pickle=True)['records'])
pts  = [r for r in recs if r.get('kind') == 'point']

ANCHOR = 80.51

def series(arm, frac):
    rows = sorted([r for r in pts if r['arm'] == arm
                   and abs(float(r['frac']) - frac) < 1e-9],
                  key=lambda r: float(r['value']))
    if arm == 'bias':
        x = [float(r['value']) * 100 for r in rows]          # %*W0
    else:
        x = [float(r['value']) for r in rows]
    return (x, [float(r['mean']) for r in rows], [float(r['lo']) for r in rows],
            [float(r['hi']) for r in rows], [int(r['nflag']) for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, arm, xl in [(axes[0], 'bias', 'bias [% of W0]'),
                    (axes[1], 'gain', 'gain [-]')]:
    x, m, lo, hi, nf = series(arm, 0.0)
    # anchor (bias=0) belongs to both panels
    if arm == 'gain' and 1.0 not in x:
        x = x + [1.0]; m = m + [ANCHOR]; lo = lo + [ANCHOR]
        hi = hi + [ANCHOR]; nf = nf + [0]
        order = np.argsort(x)
        x  = list(np.array(x)[order]);  m  = list(np.array(m)[order])
        lo = list(np.array(lo)[order]); hi = list(np.array(hi)[order])
        nf = list(np.array(nf)[order])
    ax.plot(x, m, 'o-', color='tab:blue', label='clean (deterministic)')
    for xi, mi, f in zip(x, m, nf):
        if f:
            ax.plot(xi, mi, 'o', mfc='none', mec='red', ms=12, mew=1.5)
    xn, mn, lon, hin, nfn = series(arm, 0.02)
    if xn:
        yerr = [np.array(mn) - np.array(lon), np.array(hin) - np.array(mn)]
        ax.errorbar(xn, mn, yerr=yerr, fmt='s', color='tab:orange',
                    capsize=3, label='sigma=2% x 6 seeds')
        for xi, mi, f in zip(xn, mn, nfn):
            if f:
                ax.plot(xi, mi, 's', mfc='none', mec='red', ms=12, mew=1.5)
    ax.axhline(ANCHOR, color='gray', lw=0.7, ls=':', label=f'clean anchor +{ANCHOR}%')
    ax.axhline(0, color='gray', lw=0.7)
    ax.set_xlabel(xl); ax.set_ylabel('CLred [%]')
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
axes[0].set_title('A2 -- sensor bias (red ring = flagged)')
axes[1].set_title('A2 -- sensor gain')
fig.suptitle('E2-combo calibration robustness (W30/Tg0.4, DAMULT=3)', y=1.02)
plt.tight_layout()
fn = os.path.join(DIR, 'fig_noise_calib_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}')

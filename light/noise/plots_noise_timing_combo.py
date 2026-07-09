"""
Figure: CLred vs timing error (preview shift, estimator refit period) for the
E2-combo. Reads results/B2_timing.npz. Run locally after scp.
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
# our own npz (harness save_records object-array schema) -- trusted local file
recs = list(np.load(os.path.join(DIR, 'B2_timing.npz'), allow_pickle=True)['records'])
pts  = [r for r in recs if r.get('kind') == 'point']

ANCHOR = 80.51
DT_MS  = 2.0

def series(arm, frac):
    rows = sorted([r for r in pts if r['arm'] == arm
                   and abs(float(r['frac']) - frac) < 1e-9],
                  key=lambda r: float(r['value']))
    x = [float(r['value']) * DT_MS for r in rows]            # steps -> ms
    return (x, [float(r['mean']) for r in rows], [float(r['lo']) for r in rows],
            [float(r['hi']) for r in rows], [int(r['nflag']) for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# left: shift
ax = axes[0]
x, m, lo, hi, nf = series('shift', 0.0)
ax.plot(x, m, 'o-', color='tab:blue', label='clean (deterministic)')
for xi, mi, f in zip(x, m, nf):
    if f:
        ax.plot(xi, mi, 'o', mfc='none', mec='red', ms=12, mew=1.5)
xn, mn, lon, hin, nfn = series('shift', 0.02)
if xn:
    yerr = [np.array(mn) - np.array(lon), np.array(hin) - np.array(mn)]
    ax.errorbar(xn, mn, yerr=yerr, fmt='s', color='tab:orange',
                capsize=3, label='sigma=2% x 6 seeds')
    for xi, mi, f in zip(xn, mn, nfn):
        if f:
            ax.plot(xi, mi, 's', mfc='none', mec='red', ms=12, mew=1.5)
ax.axhline(ANCHOR, color='gray', lw=0.7, ls=':', label=f'clean anchor +{ANCHOR}%')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel('preview shift [ms]  (>0 = early, <0 = late)')
ax.set_ylabel('CLred [%]')
ax.set_title('B2 -- systematic preview shift (red ring = flagged)')
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)

# right: refit period
ax = axes[1]
x, m, lo, hi, nf = series('refit', 0.02)
yerr = [np.array(m) - np.array(lo), np.array(hi) - np.array(m)]
ax.errorbar(x, m, yerr=yerr, fmt='o-', color='tab:blue', capsize=3,
            label='sigma=2% x 6 seeds')
for xi, mi, f in zip(x, m, nf):
    if f:
        ax.plot(xi, mi, 'o', mfc='none', mec='red', ms=12, mew=1.5)
ax.axvline(100.0, color='tab:green', lw=1.0, ls='--', label='DLR refit 10 Hz (100 ms)')
ax.axhline(ANCHOR, color='gray', lw=0.7, ls=':', label=f'clean anchor +{ANCHOR}%')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel('estimator refit period [ms]')
ax.set_ylabel('CLred [%]')
ax.set_title('B2 -- estimator refit rate')
ax.set_xlim(0, 110)
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)

fig.suptitle('E2-combo timing robustness (W30/Tg0.4, DAMULT=3)', y=1.02)
plt.tight_layout()
fn = os.path.join(DIR, 'fig_noise_timing_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}')

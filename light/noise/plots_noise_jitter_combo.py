"""
Figure: CLred vs per-shot spatial jitter for the E2-combo.
Reads results/C2_jitter.npz. Run locally after scp.
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
# our own npz (harness save_records object-array schema) -- trusted local file
recs = list(np.load(os.path.join(DIR, 'C2_jitter.npz'), allow_pickle=True)['records'])
pts  = [r for r in recs if r.get('kind') == 'point']

ANCHOR = 80.51
DX_M   = 0.16   # node spacing U*dt [m]

def series(frac):
    rows = sorted([r for r in pts if abs(float(r['frac']) - frac) < 1e-9],
                  key=lambda r: float(r['value']))
    return ([float(r['value']) for r in rows],
            [float(r['mean']) for r in rows], [float(r['lo']) for r in rows],
            [float(r['hi']) for r in rows], [int(r['nflag']) for r in rows],
            [int(r['n']) if 'n' in r else len(r['clred']) for r in rows])

fig, ax = plt.subplots(figsize=(7, 4.2))
x, m, lo, hi, nf, nn = series(0.0)
yerr = [np.array(m) - np.array(lo), np.array(hi) - np.array(m)]
ax.errorbar(x, m, yerr=yerr, fmt='o-', color='tab:blue', capsize=3,
            label='jitter only (6 seeds; k=0 deterministic)')
for xi, mi, f in zip(x, m, nf):
    if f:
        ax.plot(xi, mi, 'o', mfc='none', mec='red', ms=12, mew=1.5)
xn, mn, lon, hin, nfn, _ = series(0.02)
if xn:
    yerr = [np.array(mn) - np.array(lon), np.array(hin) - np.array(mn)]
    ax.errorbar(xn, mn, yerr=yerr, fmt='s', color='tab:orange', capsize=3,
                label='jitter + sigma=2% x 6 seeds')
    for xi, mi, f in zip(xn, mn, nfn):
        if f:
            ax.plot(xi, mi, 's', mfc='none', mec='red', ms=12, mew=1.5)
ax.axhline(ANCHOR, color='gray', lw=0.7, ls=':', label=f'clean anchor +{ANCHOR}%')
ax.axhline(0, color='gray', lw=0.7)
ticks = sorted(set(x) | set(xn))
ax.set_xticks(ticks)
ax.set_xticklabels([f'{int(t)}\n({t*DX_M:.2f} m)' for t in ticks])
ax.set_xlabel('per-shot node jitter k  (range error)')
ax.set_ylabel('CLred [%]')
ax.set_title('C2 -- spatial per-shot jitter (red ring = flagged)\n'
             'E2-combo, W30/Tg0.4, DAMULT=3')
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
fn = os.path.join(DIR, 'fig_noise_jitter_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}')

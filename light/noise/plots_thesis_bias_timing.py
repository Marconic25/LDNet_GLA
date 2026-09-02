"""
Thesis figure (chapter 3): systematic-error robustness of the fused-preview
MPC on the two critical axes — (a) calibration bias of the wind measurement,
(b) systematic preview time shift. Home cell W30/Tg0.4, DAMULT=3.

Reads results/A2_calib.npz (arm='bias') and results/B2_timing.npz
(arm='shift'). Generates fig_ch3_bias_timing.png in results/ and
light/latex/Images/.

Style follows light/latex/AGENTS.md ("Linee guida per i plot").
Run locally after scp:  python3 -s -u plots_thesis_bias_timing.py
"""
import os
import shutil
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS   = os.path.dirname(os.path.abspath(__file__))
DIR     = os.path.join(_THIS, 'results')
IMG_DIR = os.path.join(_THIS, '..', 'latex', 'Images')
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'savefig.dpi': 300,
})
C_CLEAN, C_NOISY, C_FLAG = '#4477AA', '#EE7733', '#CC3311'
DT_MS = 2.0


def load_pts(fname):
    recs = list(np.load(os.path.join(DIR, fname), allow_pickle=True)['records'])
    return [r for r in recs if r.get('kind') == 'point']


def series(pts, arm, frac, xscale):
    rows = sorted((r for r in pts if r['arm'] == arm
                   and abs(float(r['frac']) - frac) < 1e-9),
                  key=lambda r: float(r['value']))
    return (np.array([float(r['value']) * xscale for r in rows]),
            np.array([float(r['mean']) for r in rows]),
            np.array([float(r['lo'])   for r in rows]),
            np.array([float(r['hi'])   for r in rows]),
            np.array([int(r['nflag'])  for r in rows]))


def panel(ax, pts, arm, xscale, anchor):
    x, m, _, _, nf = series(pts, arm, 0.0, xscale)
    ax.plot(x, m, 'o-', ms=4, color=C_CLEAN, label='noise-free (deterministic)')
    ax.plot(x[nf > 0], m[nf > 0], 'o', ms=9, mfc='none', mec=C_FLAG, mew=1.2,
            label='stability flag raised')
    xn, mn, lon, hin, nfn = series(pts, arm, 0.02, xscale)
    if len(xn):
        ax.errorbar(xn, mn, yerr=[mn - lon, hin - mn], fmt='s', ms=4,
                    color=C_NOISY, capsize=2.5, lw=1,
                    label=r'$\sigma = 2\%\,W_0$, 6 seeds')
        ax.plot(xn[nfn > 0], mn[nfn > 0], 's', ms=9, mfc='none', mec=C_FLAG,
                mew=1.2)
    ax.axhline(anchor, color='k', ls=':', lw=0.8)
    ax.axhline(0, color='0.5', lw=0.7)
    ax.set_ylabel('CLred [%]')
    ax.set_ylim(-15, 100)


pts_a = load_pts('A2_calib.npz')
pts_b = load_pts('B2_timing.npz')
anchor = float([r for r in pts_a if r['arm'] == 'bias'
                and abs(float(r['value'])) < 1e-12][0]['mean'])

fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True)
panel(ax[0], pts_a, 'bias', 100.0, anchor)          # value*100 = % of W0
ax[0].set_xlabel(r'measurement bias [% of $W_0$]')
ax[0].legend(frameon=False, loc='lower left')
ax[0].text(-10, anchor + 2, 'noise-free value', ha='left', fontsize=7)

panel(ax[1], pts_b, 'shift', DT_MS, anchor)          # steps -> ms
ax[1].set_xlabel('preview shift [ms]  (> 0 = early)')
ax[1].set_ylabel(None)

fig.tight_layout(w_pad=1.0)
fn = os.path.join(DIR, 'fig_ch3_bias_timing.png')
fig.savefig(fn, bbox_inches='tight'); plt.close(fig)
shutil.copy(fn, os.path.join(IMG_DIR, 'fig_ch3_bias_timing.png'))
print(f'saved {fn} (+ latex/Images)')

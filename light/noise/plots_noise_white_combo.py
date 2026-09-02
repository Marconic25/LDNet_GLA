"""
Thesis figure (chapter 3): CLred vs raw white-noise sigma for the MPC with
inverse-variance sensor fusion (Jmax=50, N=8), home cell W30/Tg0.4, 6 seeds.

Reads results/W_combo.npz (arm='combo' records only). Generates
fig_ch3_noise_white.png in results/ and light/latex/Images/.

Style follows light/latex/AGENTS.md ("Linee guida per i plot").
Run locally after scp:  python3 -s -u plots_noise_white_combo.py
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
    'legend.fontsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'savefig.dpi': 300,
})
C_CLOSED = '#4477AA'

recs  = list(np.load(os.path.join(DIR, 'W_combo.npz'), allow_pickle=True)['records'])
combo = sorted((r for r in recs if r.get('kind') == 'point' and r.get('arm') == 'combo'),
               key=lambda r: float(r['frac']))

pcts  = [100 * float(r['frac'])  for r in combo]
mean  = [float(r['mean'])        for r in combo]
lo    = [float(r['lo'])          for r in combo]
hi    = [float(r['hi'])          for r in combo]
nflag = [int(r['nflag'])         for r in combo]
nseed = [len(r['seeds'])         for r in combo]
sdel  = [float(r['sigma_del'])   for r in combo]
clean = mean[0]

x = np.arange(len(pcts))          # category spacing: sweep levels, not linear
fig, ax = plt.subplots(figsize=(4.4, 2.9))
ax.fill_between(x, lo, hi, alpha=0.25, color=C_CLOSED, lw=0,
                label='min--max over seeds')
ax.plot(x, mean, 'o-', ms=4, color=C_CLOSED, label='mean over seeds')
flg = [i for i in range(len(pcts)) if nflag[i] > 0]
ax.plot(x[flg], [mean[i] for i in flg], 'o', ms=9,
        mfc='none', mec='#CC3311', mew=1.2, label='stability flag raised')
for i in flg:
    ax.annotate(f'{nflag[i]}/{nseed[i]}', (x[i], hi[i]),
                textcoords='offset points', xytext=(0, 6),
                ha='center', fontsize=7, color='#CC3311')
ax.axhline(clean, color='k', ls=':', lw=0.8)
ax.text(0.05, clean - 2.5, 'noise-free value', ha='left', va='top', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([f'{p:g}\n{s:.2f}' for p, s in zip(pcts, sdel)])
ax.set_xlabel(r'raw noise $\sigma / W_0$ [%] (upper) '
              r'/ delivered $\sigma_{\mathrm{del}}$ [m/s] (lower)')
ax.set_ylabel('CLred [%]')
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc='lower left')
fig.tight_layout()

fn = os.path.join(DIR, 'fig_ch3_noise_white.png')
fig.savefig(fn, bbox_inches='tight'); plt.close(fig)
shutil.copy(fn, os.path.join(IMG_DIR, 'fig_ch3_noise_white.png'))
print(f'saved {fn} (+ latex/Images)')

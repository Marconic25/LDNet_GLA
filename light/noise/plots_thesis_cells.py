"""
Thesis figure (chapter 3): generality of the fused-preview MPC (untuned,
Jmax=50, N=8, R=3e-4) across three gust cells, under flat white noise
(sigma = 2% W0) and under the range-proportional lidar noise model
(raw LOS sigma 1-3 m/s). 6 seeds per condition; dashed line = noise-free value.

Reads results/E2_combo_flat.npz, E2_combo_dlr.npz, E2_combo_cells_W10T07.npz,
E2_combo_cells_W30T07.npz. Generates fig_ch3_cases.png in results/ and
light/latex/Images/.

Style follows light/latex/AGENTS.md ("Linee guida per i plot").
Run locally after scp:  python3 -s -u plots_thesis_cells.py
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
    'axes.axisbelow': True,
    'savefig.dpi': 300,
})
C_FLAT, C_DLR, C_FLAG = '#4477AA', '#228833', '#CC3311'


def load(name):
    return list(np.load(os.path.join(DIR, name), allow_pickle=True)['records'])


def pt(recs, **sel):
    out = [r for r in recs if r.get('kind') == 'point'
           and all(r.get(k) == v for k, v in sel.items())]
    assert len(out) == 1, f"{sel} -> {len(out)} matches"
    return out[0]


CF  = load('E2_combo_flat.npz')
CD  = load('E2_combo_dlr.npz')
W10 = load('E2_combo_cells_W10T07.npz')
W30 = load('E2_combo_cells_W30T07.npz')

cells = [
    ('$W_0=30$ m/s, $T_g=0.4$ s',
     {'anchor':     pt(CF, arm='anchor', R_du=0.0),
      'combo-flat': pt(CF, arm='combo', detail='flat Rdu=0'),
      'combo-dlr':  pt(CD, arm='combo', detail='dlr lam=1 Rdu=0')}),
    ('$W_0=10$ m/s, $T_g=0.7$ s',
     {a: pt(W10, arm=a) for a in ('anchor', 'combo-flat', 'combo-dlr')}),
    ('$W_0=30$ m/s, $T_g=0.7$ s',
     {a: pt(W30, arm=a) for a in ('anchor', 'combo-flat', 'combo-dlr')}),
]
arms = [('combo-flat', r'white noise, $\sigma = 2\%\,W_0$', C_FLAT),
        ('combo-dlr',  'lidar model, raw LOS 1–3 m/s',     C_DLR)]

fig, ax = plt.subplots(figsize=(4.6, 2.9))
BW = 0.28
for k, (aname, alab, col) in enumerate(arms):
    for i, (_, d) in enumerate(cells):
        r = d[aname]
        x = i + (k - 0.5) * BW
        ax.bar(x, r['mean'], width=BW * 0.9, color=col, alpha=0.85,
               label=alab if i == 0 else None)
        if r['nflag']:
            ax.text(x, 2, f"{int(r['nflag'])}/{len(r['seeds'])}",
                    ha='center', fontsize=7, color=C_FLAG)

for i, (_, d) in enumerate(cells):
    m = float(d['anchor']['mean'])
    ax.plot([i - 0.8 * BW - BW / 2, i + 0.8 * BW + BW / 2], [m, m],
            'k--', lw=0.9,
            label='noise-free value' if i == 0 else None)
    ax.text(i, m + 1.5, f'{m:+.1f}', ha='center', fontsize=7)

ax.set_xticks(range(len(cells)))
ax.set_xticklabels([c[0] for c in cells])
ax.set_ylabel('CLred [%]')
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc='lower left')
fig.tight_layout()

fn = os.path.join(DIR, 'fig_ch3_cells.png')
fig.savefig(fn, bbox_inches='tight'); plt.close(fig)
shutil.copy(fn, os.path.join(IMG_DIR, 'fig_ch3_cells.png'))
print(f'saved {fn} (+ latex/Images)')

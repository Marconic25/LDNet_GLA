"""
CS-25 plots for the E2-combo controller.

Reads:
  results_cs25_combo/traces_W{10,20,30}.npz (combo)

Generates in results_cs25_combo/:
  heatmap_clred_combo.png   – CLred heatmap (combo)
  summary_lines_combo.png   – CLred vs Tg, one line per W0
  summary.md                – per-cell table

Run after the study is complete:
  python3 -s -u cs25_combo_plots.py
"""
import csv, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

U = 80.0
W_LIST  = [10, 20, 30]
TG_LIST = [0.30, 0.40, 0.50, 0.70, 1.00, 1.20]
_THIS   = os.path.dirname(os.path.abspath(__file__))
DIR_COMBO = os.path.join(_THIS, '..', 'results_cs25_combo')
os.makedirs(DIR_COMBO, exist_ok=True)


def H_ft(Tg): return U * Tg / 2.0 / 0.3048
def kred(Tg): return np.pi / (U * Tg)


def load_row(d, w):
    path = os.path.join(d, f'traces_W{w}.npz')
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=False)


rows = []
for w in W_LIST:
    d_combo = load_row(DIR_COMBO, w)
    for Tg in TG_LIST:
        tag = f'Tg{Tg:.2f}'
        row = dict(W0=w, Tg=Tg, H_ft=round(H_ft(Tg)), k=round(kred(Tg), 3))
        if d_combo is None:
            row['combo_clred'] = float('nan')
            row['combo_Rstar'] = float('nan')
            row['combo_fmax']  = float('nan')
            row['combo_pitch'] = float('nan')
            row['combo_flag']  = '?'
        else:
            jb   = int(d_combo[f'{tag}_jb'])
            row['combo_clred'] = round(float(d_combo[f'{tag}_clred']), 1)
            row['combo_Rstar'] = float(d_combo[f'{tag}_Rstar'])
            row['combo_fmax']  = round(float(d_combo[f'{tag}_fmax'][jb]), 1)
            row['combo_pitch'] = round(float(d_combo[f'{tag}_pitch']), 2)
            row['combo_flag']  = str(d_combo[f'{tag}_flags'][jb])
        rows.append(row)

# --- heatmap ---
fig, ax = plt.subplots(figsize=(8, 3.8))
label, title = 'combo', 'E2-combo (FusedSensor + MPC N=8)'
M  = np.array([[r[f'{label}_clred'] for r in rows if r['W0'] == w] for w in W_LIST])
RS = np.array([[r[f'{label}_Rstar'] for r in rows if r['W0'] == w] for w in W_LIST])
FL = np.array([[r[f'{label}_flag']  for r in rows if r['W0'] == w] for w in W_LIST])
vmin = min(0., float(np.nanmin(M))); vmax = max(10., float(np.nanmax(M)))
pc = ax.pcolormesh(np.arange(len(TG_LIST)+1), np.arange(len(W_LIST)+1), M,
                   cmap='RdYlGn', vmin=vmin, vmax=vmax,
                   edgecolors='white', linewidth=2)
for i, ww in enumerate(W_LIST):
    for j, Tg in enumerate(TG_LIST):
        warn = ' !' if FL[i, j] else ''
        ax.text(j+0.5, i+0.5, f'{M[i,j]:+.0f}%{warn}\nR={RS[i,j]:g}',
                ha='center', va='center', fontsize=8,
                color=('darkred' if FL[i, j] else 'black'))
ax.set_xticks(np.arange(len(TG_LIST))+0.5)
ax.set_xticklabels([f'Tg={Tg:g}s\nk={kred(Tg):.3f}\nH={H_ft(Tg):.0f}ft'
                    for Tg in TG_LIST], fontsize=8)
ax.set_yticks(np.arange(len(W_LIST))+0.5)
ax.set_yticklabels([f'W0={ww}' for ww in W_LIST], fontsize=9)
ax.set_title(title, fontsize=9)
fig.colorbar(pc, ax=ax).set_label('CLred [%]', fontsize=9)

fig.suptitle('CS-25.341 – CLred (DAMULT=3, R* per cell, no-explosion pick)', fontsize=9)
plt.tight_layout()
fn = os.path.join(DIR_COMBO, 'heatmap_clred_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}', flush=True)

# --- line plot ---
fig, ax = plt.subplots(figsize=(7, 4))
colors = ['tab:blue', 'tab:orange', 'tab:red']
for w, col in zip(W_LIST, colors):
    cr_combo = [r['combo_clred'] for r in rows if r['W0'] == w]
    ax.plot(TG_LIST, cr_combo, 's-', color=col, label=f'combo W0={w}')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel('Tg [s]'); ax.set_ylabel('CLred [%]')
ax.set_title('CS-25 – E2-combo')
ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2, frameon=False)
plt.tight_layout()
fn = os.path.join(DIR_COMBO, 'summary_lines_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}', flush=True)

# --- summary.md ---
md = [
    '# CS-25.341 – E2-combo (light/)\n\n',
    'Controller: DAMULT=3, TEND=3 s, dmax=14 deg, rate 300 deg/s.\n'
    'R* per cell = MAX CLred with no explosion flag; fallback: min pitch.\n'
    'Combo: FusedSensor Jmax=50, MPC N=8, R=R*, R_du=0, oracle preview\n'
    'w_seq=Wt[i+1:i+N+1], dp45 horizon.\n\n',
    '| W0 | Tg | H [ft] | k | '
    'combo CLred | combo R* | combo flap | combo pitch | combo flag |\n',
    '|---|---|---|---|---|---|---|---|---|\n',
]
for r in rows:
    md.append(
        f"| {r['W0']} | {r['Tg']:.2f} | {r['H_ft']} | {r['k']} "
        f"| {r['combo_clred']:+.1f} | {r['combo_Rstar']:g} | {r['combo_fmax']} "
        f"| {r['combo_pitch']} | {r['combo_flag']} |\n"
    )
with open(os.path.join(DIR_COMBO, 'summary.md'), 'w') as f:
    f.writelines(md)
print(f'saved {DIR_COMBO}/summary.md', flush=True)
print('# PLOTS DONE', flush=True)

"""
CS-25 study plots: per-cell 4-panel time histories (style of
tests/test_optimal.py), annotated CLred heatmap, summary.csv + summary.md.

Reads results_cs25/traces_W{10,20,30}.npz written by cs25_study.py.
Run AFTER all three row jobs are done:
    python3 -s -u cs25_plots.py
"""
import csv, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

U = 80.0
W_LIST = [10, 20, 30]
TG_LIST = [0.30, 0.40, 0.50, 0.70, 1.00, 1.20]
# Resolve relative to this file (light/tests/), not the cwd
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_cs25')


def H_m(Tg): return U * Tg / 2.0
def H_ft(Tg): return H_m(Tg) / 0.3048
def kred(Tg): return np.pi / (U * Tg)


data = {w: np.load(f'{DIR}/traces_W{w}.npz', allow_pickle=False) for w in W_LIST}
CLTRIM = float(data[W_LIST[0]]['CLTRIM'])

OL_KW = dict(color='black', linestyle='--', linewidth=1.5, label='open')
OPT_KW = dict(color='red', linestyle='-', linewidth=1.5, label='optimal (wnext)')
SH_KW = dict(alpha=0.15, color='#aad4f5', zorder=0)

rows = []
for w in W_LIST:
    d = data[w]
    for Tg in TG_LIST:
        tag = f'Tg{Tg:.2f}'
        t = d[f'{tag}_t']; Wt = d[f'{tag}_Wt']
        Rstar = float(d[f'{tag}_Rstar']); clred = float(d[f'{tag}_clred'])
        pitch = float(d[f'{tag}_pitch']); cex0 = float(d[f'{tag}_cex0'])
        jb = int(d[f'{tag}_jb']); fm = float(d[f'{tag}_fmax'][jb])
        flag = str(d[f'{tag}_flags'][jb])
        cex = cex0 * (1.0 - clred / 100.0)
        rows.append(dict(W0=w, Tg=Tg, H_m=round(H_m(Tg), 1), H_ft=round(H_ft(Tg)),
                         k=round(kred(Tg), 3), cex0=round(cex0, 4), cex=round(cex, 4),
                         clred=round(clred, 1), Rstar=Rstar, flap_max=round(fm, 1),
                         pitch_ratio=round(pitch, 2), flag=flag))

        fig, axes = plt.subplots(4, 1, figsize=(5, 7), sharex=True)
        panels = [
            (d[f'{tag}_open_CL'], d[f'{tag}_opt_CL'], r'$C_L$'),
            (d[f'{tag}_open_de'], d[f'{tag}_opt_de'], r'$\delta$ [deg]'),
            (None, None, r'$W$ [m/s]'),
            (d[f'{tag}_open_ad'] * 180 / np.pi, d[f'{tag}_opt_ad'] * 180 / np.pi,
             r'$\dot\alpha$ [deg/s]'),
        ]
        for ax, (yo, yc, lab) in zip(axes, panels):
            if yo is None:
                ax.plot(t, Wt, color='green', linewidth=1.5, label='W_gust')
            else:
                ax.plot(t, yo, **OL_KW)
                ax.plot(t, yc, **OPT_KW)
            if lab == r'$C_L$':
                ax.axhline(CLTRIM, color='gray', lw=0.7, ls=':')
            ax.axvspan(0.0, Tg, **SH_KW)
            ax.set_ylabel(lab, fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=8)
        axes[0].legend(fontsize=8, frameon=False)
        axes[2].legend(fontsize=8, frameon=False)
        axes[-1].set_xlabel('time [s]', fontsize=9)
        fig.suptitle(f'W0={w} m/s  Tg={Tg:g} s  (H={H_ft(Tg):.0f} ft, k={kred(Tg):.3f})\n'
                     f'R*={Rstar:g}  DAMULT=3  CLred={clred:+.1f}%'
                     + (f'  [{flag}]' if flag else ''), fontsize=9)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fn = f'{DIR}/traj_W{w}_Tg' + f'{Tg:.2f}'.replace('.', '') + '.png'
        fig.savefig(fn, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'saved {fn}', flush=True)

# ---------------------------------------------------------------------------
# Annotated heatmap of CLred (heatmap, not contour: only 3 W rows —
# interpolation would be misleading)
# ---------------------------------------------------------------------------
M = np.array([[r['clred'] for r in rows if r['W0'] == w] for w in W_LIST])
RS = np.array([[r['Rstar'] for r in rows if r['W0'] == w] for w in W_LIST])
FL = np.array([[r['flag'] for r in rows if r['W0'] == w] for w in W_LIST])
PR = np.array([[r['pitch_ratio'] for r in rows if r['W0'] == w] for w in W_LIST])

fig, ax = plt.subplots(figsize=(10, 3.8))
vmin = min(0.0, float(M.min())); vmax = max(10.0, float(M.max()))
pc = ax.pcolormesh(np.arange(len(TG_LIST) + 1), np.arange(len(W_LIST) + 1), M,
                   cmap='RdYlGn', vmin=vmin, vmax=vmax,
                   edgecolors='white', linewidth=2)
for i, w in enumerate(W_LIST):
    for j, Tg in enumerate(TG_LIST):
        warn = ' !' if FL[i, j] else ''
        ax.text(j + 0.5, i + 0.5, f'{M[i, j]:+.0f}%{warn}\nR={RS[i, j]:g}',
                ha='center', va='center', fontsize=8,
                color=('darkred' if FL[i, j] else 'black'))
ax.set_xticks(np.arange(len(TG_LIST)) + 0.5)
ax.set_xticklabels([f'Tg={Tg:g}s\nk={kred(Tg):.3f}\nH={H_ft(Tg):.0f}ft'
                    for Tg in TG_LIST], fontsize=8)
ax.set_yticks(np.arange(len(W_LIST)) + 0.5)
ax.set_yticklabels([f'W0={w}' for w in W_LIST], fontsize=9)
ax.set_title('One-step optimal (wnext + refine) — C_L excursion reduction, '
             'CS-25.341 discrete gust grid  (DAMULT=3, R* per cell, no-explosion pick)',
             fontsize=9)
cb = fig.colorbar(pc, ax=ax); cb.set_label('CLred [%]', fontsize=9)
plt.tight_layout()
fig.savefig(f'{DIR}/heatmap_clred.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'saved {DIR}/heatmap_clred.png', flush=True)

# ---------------------------------------------------------------------------
# Summary line plot: CLred vs Tg, one line per W0
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
for w, col in zip(W_LIST, ['tab:blue', 'tab:orange', 'tab:red']):
    cr = [r['clred'] for r in rows if r['W0'] == w]
    ax.plot(TG_LIST, cr, 'o-', color=col, label=f'W0={w} m/s')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel('Tg [s]'); ax.set_ylabel('CLred [%]')
ax.set_title('CS-25 gust study — one-step optimal wnext (R* per cell)')
ax.grid(alpha=0.3); ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(f'{DIR}/summary_lines.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'saved {DIR}/summary_lines.png', flush=True)

# ---------------------------------------------------------------------------
# summary.csv + summary.md
# ---------------------------------------------------------------------------
with open(f'{DIR}/summary.csv', 'w', newline='') as f:
    wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wcsv.writeheader(); wcsv.writerows(rows)

md = [
    '# CS-25.341 discrete gust study — final one-step optimal, wnext + refine (light/)\n\n',
    'Scalar rollout of light/optimal.py (candidates evaluated at W(t+dt), parabolic argmin\n'
    'refine), DAMULT=3, TEND=3 s, DLPF=0, dmax=14 deg, rate 300 deg/s.\n'
    'R* per cell = MAX CLred subject to NO explosion flag (alpha_dot/alpha_ddot/h_ddot\n'
    '< 3x open loop); pitch ratio reported for transparency. H = U*Tg/2, k = pi/(U*Tg),\n'
    'U=80 m/s.\n\n',
    'The W(t+dt) evaluation makes the good closed-loop branch a robust attractor\n'
    '(ensemble spread 0.4 CLred pts at W30/Tg0.4) — see 76/NOTES.md and the\n'
    'light/optimal.py docstring for the full investigation and the FP-artifact history.\n\n',
    '| W0 [m/s] | Tg [s] | H [m] | H [ft] | k | cex0 | cex | CLred [%] | R* | flap [deg] | pitch | flag |\n',
    '|---|---|---|---|---|---|---|---|---|---|---|---|\n',
]
for r in rows:
    md.append(f"| {r['W0']} | {r['Tg']:.2f} | {r['H_m']} | {r['H_ft']} | {r['k']} "
              f"| {r['cex0']:.4f} | {r['cex']:.4f} | {r['clred']:+.1f} | {r['Rstar']:g} "
              f"| {r['flap_max']} | {r['pitch_ratio']} | {r['flag']} |\n")
with open(f'{DIR}/summary.md', 'w') as f:
    f.writelines(md)
print(f'saved {DIR}/summary.csv and summary.md', flush=True)
print('# PLOTS DONE', flush=True)

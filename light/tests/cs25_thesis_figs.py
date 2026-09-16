"""
Thesis figures (chapter 3) for the noise-free CS-25 parametric grid.

Reads results_cs25_combo/traces_W{10,20,30}.npz and generates, in the same
folder and in light/latex/Images/:
  fig_ch3_trace_tests.png   – nominal time histories, Test 1 (W20/Tg0.7) and
                              Test 2 (W10/Tg0.4) side by side in one figure
  fig_ch3_envelope.png      – CLred and |delta|max vs gust gradient H

Style follows light/latex/AGENTS.md ("Linee guida per i plot"):
serif + cm mathtext, ~9 pt at print width, no in-figure titles, 300 dpi.
Colour palette shared with the Study-1 figures (plots_ldnet_accuracy.py):
open-loop red (#CC3311), closed-loop blue (#4477AA), flap orange (#EE7733).

Cells listed in FOM_CSV also carry their full-order (OpenFOAM FSI) counterpart,
overlaid dashed in the same colours: colour encodes the loop, linestyle the
plant. In the full-order closed loop the controller still predicts with the
LDNet surrogate but is fed the true CFD structural state every window, so its
flap history differs from the reduced-order one.

Run:  python3 -s -u cs25_thesis_figs.py
"""
import os
import shutil
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

U = 80.0
RHO = 1.225
S   = 0.05
Q   = 0.5 * RHO * U**2 * S   # dynamic pressure x reference area [N], matches light/run.py
W_LIST  = [10, 20, 30]
TG_LIST = [0.30, 0.40, 0.50, 0.70, 1.00, 1.20]
DMAX    = 14.0
_THIS   = os.path.dirname(os.path.abspath(__file__))
DIR     = os.path.join(_THIS, '..', 'results_cs25_combo')
IMG_DIR = os.path.join(_THIS, '..', 'latex', 'Images')
os.makedirs(IMG_DIR, exist_ok=True)

# Full-order (OpenFOAM FSI) counterparts of the two ROM traces, from real
# co-simulation runs (recon/cluster/mpc_fom_verify_rtag.pbs). In the closed-loop
# run the MPC still PREDICTS with the LDNet surrogate, but it is fed the true
# FOM structural state every window and re-plans against it, so its delta(t)
# genuinely differs from the ROM's.
#
# Only the full-order CLOSED loop is drawn; the open-loop FOM run is still
# loaded, because the CLred printed below must divide two excursions measured
# on the SAME plant. Do not read a reduction off the figure by measuring the
# dashed closed loop against the solid red open loop: those are different
# plants, and the ROM underestimates the open-loop peak by 28% (W20/Tg0.70)
# to 45% (W10/Tg0.40) -- see light/dagger_fom/NOTES.md. The caption must quote
# the per-plant numbers this script prints, not a visually inferred one.
#
# Paths are pinned to the *_OLDMODEL_backup copies: the un-suffixed
# Rsweep_*_win29/ directories are volatile and today hold a mix of retraining
# iterations, while the backup set is the complete, stable production run.
FOM_CSV = {
    (10, 0.40): {
        'closed': '/work/u10677113/NACA2312/mpc_fom_dagger/Rsweep_W10_Tg0.40_R0.0003_win29_OLDMODEL_backup/structural_trajectory.csv',
        'open':   '/work/u10677113/NACA2312/mpc_fom_dagger/OpenLoop_W10_Tg0.40/structural_trajectory.csv',
    },
    (30, 0.40): {
        'closed': '/work/u10677113/NACA2312/mpc_fom_dagger/Rsweep_W30_Tg0.40_R0.0003_win29_OLDMODEL_backup/structural_trajectory.csv',
        'open':   '/work/u10677113/NACA2312/mpc_fom_dagger/OpenLoop_W30_Tg0.40/structural_trajectory.csv',
    },
    (20, 0.70): {
        'closed': '/work/u10677113/NACA2312/mpc_fom_dagger/Rsweep_W20_Tg0.70_R0.0001_win29_OLDMODEL_backup/structural_trajectory.csv',
        'open':   '/work/u10677113/NACA2312/mpc_fom_dagger/OpenLoop_W20_Tg0.70/structural_trajectory.csv',
    },
}


def _read_fom(path, t_show):
    if not os.path.exists(path):
        return None
    d = np.genfromtxt(path, delimiter=',', names=True)
    m = d['t'] <= t_show
    return d['t'][m], d['Fy'][m] / Q, d['delta'][m]


def fom_data(W0, Tg, t_show):
    """Return {'open': (t, CL, delta), 'closed': (...)} for cells with FOM runs."""
    spec = FOM_CSV.get((W0, round(Tg, 2)))
    if spec is None:
        return None
    out = {k: _read_fom(p, t_show) for k, p in spec.items()}
    return None if out.get('closed') is None else out


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

# Colour encodes the quantity (red open-loop C_L / light blue closed-loop C_L /
# orange flap), linestyle the plant: solid ROM, dashed FOM.
C_OPEN, C_CLOSED, C_FLAP = '#CC3311', '#66CCEE', '#EE7733'
C_W = {10: '#4477AA', 20: '#EE7733', 30: '#CC3311'}   # severity: blue->red


def H_ft(Tg): return U * Tg / 2.0 / 0.3048


def save(fig, name, tight=True):
    fn = os.path.join(DIR, name)
    fig.savefig(fn, bbox_inches='tight' if tight else None); plt.close(fig)
    shutil.copy(fn, os.path.join(IMG_DIR, name))
    print(f'saved {fn} (+ latex/Images)', flush=True)


# --- Fig A: nominal time histories, Test 1 and Test 2 in one merged figure -
TRACE_CELLS = [(20, 0.70, 'Test 1'), (10, 0.40, 'Test 2')]
T_SHOW = max(Tg for _, Tg, _ in TRACE_CELLS) + 0.5


def cell_data(W0, Tg):
    d   = np.load(os.path.join(DIR, f'traces_W{W0}.npz'))
    tag = f'Tg{Tg:.2f}'
    t   = d[f'{tag}_t']
    m   = t <= T_SHOW
    fom = fom_data(W0, Tg, T_SHOW)
    return (t[m], d[f'{tag}_Wt'][m], d[f'{tag}_open_CL'][m],
            d[f'{tag}_opt_CL'][m], d[f'{tag}_opt_de'][m], float(d['CLTRIM']), fom)


def pad(lo, hi, f=0.08):
    s = f * (hi - lo)
    return lo - s, hi + s


DATA = {label: cell_data(W0, Tg) for W0, Tg, label in TRACE_CELLS}


def _spread(v, idx):
    """Min/max over the ROM trace at column idx plus the FOM traces DRAWN.

    Only the full-order closed loop is plotted, so the open-loop FOM run must
    not stretch the axes: including it would leave the range sized for a curve
    the reader never sees, squashing the near-trim detail that matters.
    """
    vals = [v[idx]]
    if v[6]:
        col = 1 if idx in (2, 3) else 2
        vals.append(v[6]['closed'][col])
    return min(a.min() for a in vals), max(a.max() for a in vals)


LIM_W  = pad(0, max(v[1].max() for v in DATA.values()))
LIM_CL = pad(min(min(_spread(v, 2)[0], _spread(v, 3)[0]) for v in DATA.values()),
             max(max(_spread(v, 2)[1], _spread(v, 3)[1]) for v in DATA.values()))
LIM_DE = pad(min(_spread(v, 4)[0] for v in DATA.values()),
             max(_spread(v, 4)[1] for v in DATA.values()))

fig, axs = plt.subplots(3, 2, figsize=(6.3, 4.35), sharex=True, sharey='row')
for col, (W0, Tg, label) in enumerate(TRACE_CELLS):
    t, W, cl_o, cl_c, de, trim, fom = DATA[label]
    ax = axs[:, col]

    ax[0].plot(t, W, color='0.2', lw=1.0)
    ax[0].set_ylim(*LIM_W)

    ax[1].plot(t, cl_o, color=C_OPEN, lw=1.0, label='open loop, ROM')
    ax[1].plot(t, cl_c, color=C_CLOSED, lw=1.3, label='closed loop, ROM')
    if fom is not None:
        ax[1].plot(fom['closed'][0], fom['closed'][1], color=C_CLOSED,
                   lw=1.6, ls='--', label='closed loop, FOM')
    ax[1].axhline(trim, color='k', ls=':', lw=0.8, label=r'$C_{L,\mathrm{trim}}$')
    ax[1].set_ylim(*LIM_CL)

    ax[2].plot(t, de, color=C_FLAP, lw=1.2, label=r'$\delta$, ROM')
    if fom is not None:
        ax[2].plot(fom['closed'][0], fom['closed'][2], color=C_FLAP, lw=1.2,
                   ls='--', label=r'$\delta$, FOM')
    ax[2].set_ylim(*LIM_DE)
    ax[2].set_xlabel(r'$t$ [s]')
    ax[2].set_xlim(0, T_SHOW)

axs[0, 0].set_ylabel(r'$W$ [m/s]')
axs[1, 0].set_ylabel(r'$C_L$ [-]')
axs[2, 0].set_ylabel(r'$\delta$ [deg]')

fig.align_ylabels(axs[:, 0])
# One legend for the whole figure, covering both the C_L and the flap panel;
# six entries fit in no panel without covering a curve, so it goes underneath
# and the layout reserves the strip (this figure is saved without a tight bbox).
# matplotlib fills a multi-column legend column-first, so this handle order
# gives: [C_L ROM pair] [C_L FOM + trim] [flap pair].
fig.tight_layout(h_pad=0.4, w_pad=1.2, rect=[0, 0.105, 1, 1])
h_cl, l_cl = axs[1, 0].get_legend_handles_labels()
h_de, l_de = axs[2, 0].get_legend_handles_labels()
fig.legend(h_cl + h_de, l_cl + l_de, frameon=False, fontsize=7.5, ncol=3,
           loc='lower center', bbox_to_anchor=(0.5, -0.005),
           handlelength=1.9, columnspacing=2.2, labelspacing=0.5)
save(fig, 'fig_ch3_trace_tests.png', tight=False)

# Peak-excursion reduction on each plant, measured the same way on both, so the
# caption can quote numbers that match what the figure shows.
print('\n--- CLred per plant (max|CL - CLtrim| over the plotted window) ---')
for W0, Tg, label in TRACE_CELLS:
    t, W, cl_o, cl_c, de, trim, fom = DATA[label]
    exo_r, exc_r = np.abs(cl_o - trim).max(), np.abs(cl_c - trim).max()
    print(f'{label}  W0={W0} Tg={Tg}')
    print(f'    reduced    exo={exo_r:.4f} exc={exc_r:.4f} '
          f'CLred={100*(exo_r-exc_r)/exo_r:+.1f}%  |d|max={np.abs(de).max():.2f} deg')
    if fom is not None and fom['open'] is not None:
        exo_f = np.abs(fom['open'][1] - trim).max()
        exc_f = np.abs(fom['closed'][1] - trim).max()
        print(f'    full-order exo={exo_f:.4f} exc={exc_f:.4f} '
              f'CLred={100*(exo_f-exc_f)/exo_f:+.1f}%  '
              f'|d|max={np.abs(fom["closed"][2]).max():.2f} deg')

# --- Fig B: envelope lines vs gust gradient H -------------------------------
H = [H_ft(Tg) for Tg in TG_LIST]
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
for W0 in W_LIST:
    d  = np.load(os.path.join(DIR, f'traces_W{W0}.npz'))
    cr, fm = [], []
    for Tg in TG_LIST:
        tag = f'Tg{Tg:.2f}'
        jb  = int(d[f'{tag}_jb'])
        cr.append(float(d[f'{tag}_clred']))
        fm.append(float(d[f'{tag}_fmax'][jb]))
    ax[0].plot(H, cr, 'o-', ms=4, color=C_W[W0],
               label=fr'$W_0 = {W0}$ m/s')
    ax[1].plot(H, fm, 'o-', ms=4, color=C_W[W0])

ax[0].set_xlabel(r'$H$ [ft]'); ax[0].set_ylabel('CLred [%]')
ax[0].set_ylim(0, 100)
ax[0].legend(frameon=False, loc='lower center')
ax[1].axhline(DMAX, color='k', ls='--', lw=0.8)
ax[1].text(H[0], DMAX - 0.6, r'$\delta_{\max}$', va='top', fontsize=8)
ax[1].set_xlabel(r'$H$ [ft]'); ax[1].set_ylabel(r'$|\delta|_{\max}$ [deg]')
ax[1].set_ylim(0, 15)
for a in ax:
    a.set_xticks(H)
    a.set_xticklabels([f'{h:.0f}' for h in H])

fig.tight_layout(w_pad=1.5)
save(fig, 'fig_ch3_envelope.png')
print('# THESIS GRID FIGS DONE', flush=True)

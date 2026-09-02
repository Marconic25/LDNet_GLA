"""
Thesis figures (chapter 3) for the noise-free CS-25 parametric grid.

Reads results_cs25_combo/traces_W{10,20,30}.npz and generates, in the same
folder and in light/latex/Images/:
  fig_ch3_trace_W30Tg04.png  – nominal time histories, severe cell (W30/Tg0.4)
  fig_ch3_trace_W10Tg07.png  – nominal time histories, gentle cell (W10/Tg0.7)
  fig_ch3_envelope.png       – CLred and |delta|max vs gust gradient H

Style follows light/latex/AGENTS.md ("Linee guida per i plot"):
serif + cm mathtext, ~9 pt at print width, no in-figure titles, 300 dpi.

Cells listed in FOM_CSV get an additional "full-order (FSI)" trace overlaid
on the CL/delta panels, read from a real OpenFOAM + structural co-simulation
run (recon/cluster/mpc_fom_verify.pbs), independent of the ROM used
everywhere else in this script.

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

# FOM (full-order FSI co-simulation) verification traces, when available for a
# cell -- overlaid on the ROM-nominal trace as an independent check against
# real OpenFOAM + structural physics (see recon/cluster/mpc_fom_verify.pbs).
# Path is None for cells without a FOM run.
FOM_CSV = {
    (30, 0.40): '/work/u10677113/NACA2312/mpc_fom_dagger/Rsweep_W30_Tg0.40_R0.0003_win29_OLDMODEL_backup/structural_trajectory.csv',
    (30, 0.70): '/work/u10677113/NACA2312/mpc_fom_dagger/Rsweep_W30_Tg0.70_R0.0001_win29_OLDMODEL_backup/structural_trajectory.csv',
    (10, 0.70): '/work/u10677113/NACA2312/mpc_fom_dagger/Rsweep_W10_Tg0.70_R0.0003_win29/structural_trajectory.csv',
    (20, 0.70): '/work/u10677113/NACA2312/mpc_fom_dagger/Rsweep_W20_Tg0.70_R0.0001_win29_OLDMODEL_backup/structural_trajectory.csv',
}


def fom_data(W0, Tg, t_show):
    path = FOM_CSV.get((W0, round(Tg, 2)))
    if path is None or not os.path.exists(path):
        return None
    d = np.genfromtxt(path, delimiter=',', names=True)
    m = d['t'] <= t_show
    return d['t'][m], d['Fy'][m] / Q, d['delta'][m]

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

C_OPEN, C_CLOSED, C_FLAP, C_FOM = '0.45', '#4477AA', '#228833', '#AA3377'
C_W = {10: '#4477AA', 20: '#EE7733', 30: '#CC3311'}   # severity: blue->red


def H_ft(Tg): return U * Tg / 2.0 / 0.3048


def save(fig, name, tight=True):
    fn = os.path.join(DIR, name)
    fig.savefig(fn, bbox_inches='tight' if tight else None); plt.close(fig)
    shutil.copy(fn, os.path.join(IMG_DIR, name))
    print(f'saved {fn} (+ latex/Images)', flush=True)


# --- Fig A: nominal time histories, one cell per file (subfloat pair) -------
# Paired subfloats: identical figsize, identical axis limits (AGENTS.md),
# exported WITHOUT tight crop so both PNGs have the same pixel size.
TRACE_CELLS = [(30, 0.40, 'fig_ch3_trace_W30Tg04.png'),
               (30, 0.70, 'fig_ch3_trace_W30Tg07.png'),
               (10, 0.70, 'fig_ch3_trace_W10Tg07.png'),
               (20, 0.70, 'fig_ch3_trace_W20Tg07.png')]
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


DATA = {name: cell_data(W0, Tg) for W0, Tg, name in TRACE_CELLS}
LIM_W  = pad(0, max(v[1].max() for v in DATA.values()))
LIM_CL = pad(min(min(v[2].min(), v[3].min(),
                      v[6][1].min() if v[6] else v[3].min()) for v in DATA.values()),
             max(max(v[2].max(), v[3].max(),
                      v[6][1].max() if v[6] else v[3].max()) for v in DATA.values()))
LIM_DE = pad(min(min(v[4].min(), v[6][2].min() if v[6] else v[4].min()) for v in DATA.values()),
             max(max(v[4].max(), v[6][2].max() if v[6] else v[4].max()) for v in DATA.values()))


def trace_fig(name, legend=False):
    t, W, cl_o, cl_c, de, trim, fom = DATA[name]
    fig, ax = plt.subplots(3, 1, figsize=(3.0, 3.9), sharex=True)
    ax[0].plot(t, W, color='0.2', lw=1.0)
    ax[0].set_ylabel(r'$W$ [m/s]'); ax[0].set_ylim(*LIM_W)

    ax[1].plot(t, cl_o, color=C_OPEN, lw=1.0, label='open loop')
    ax[1].plot(t, cl_c, color=C_CLOSED, lw=1.2, label='closed loop')
    if fom is not None:
        ax[1].plot(fom[0], fom[1], color=C_FOM, lw=1.2, ls='--',
                   label='full-order (FSI)')
    ax[1].axhline(trim, color='k', ls=':', lw=0.8,
                  label=r'$C_{L,\mathrm{trim}}$')
    ax[1].set_ylabel(r'$C_L$ [-]'); ax[1].set_ylim(*LIM_CL)
    if legend:
        ax[1].legend(frameon=False, loc='upper left', fontsize=7,
                     borderaxespad=0.2, handlelength=1.6)

    ax[2].plot(t, de, color=C_FLAP, lw=1.2)
    if fom is not None:
        ax[2].plot(fom[0], fom[2], color=C_FOM, lw=1.2, ls='--')
    ax[2].set_ylabel(r'$\delta$ [deg]'); ax[2].set_ylim(*LIM_DE)
    ax[2].set_xlabel(r'$t$ [s]')
    ax[2].set_xlim(0, T_SHOW)

    fig.align_ylabels(ax)
    fig.tight_layout(h_pad=0.4)
    save(fig, name, tight=False)


trace_fig('fig_ch3_trace_W30Tg04.png')
trace_fig('fig_ch3_trace_W30Tg07.png', legend=True)
trace_fig('fig_ch3_trace_W10Tg07.png')
trace_fig('fig_ch3_trace_W20Tg07.png')

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

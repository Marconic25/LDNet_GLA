"""
Thesis figures (chapter 2): open-loop load-reconstruction accuracy of the LDNet
(F_y, M_z) — number-of-inputs study, latent-state sensitivity, definitive-model
test traces and training loss.

Consumes the artifacts written by src/sensitivity_latent.py under
    results/sensitivity/in{2,4,6}/{latent_N/{metrics.json,traces.npz,loss_history.npz},
                                   summary/all_metrics.json}
and writes, to light/latex/Images/:
    fig_ch2_input_study.png        NRMSE and 1-rho vs d_s, 2 vs 4 vs 6 inputs (combined)
    fig_ch2_latent_sensitivity.png NRMSE and 1-rho vs d_s (6 inputs; combined, F_y, M_z)
    fig_ch2_loads_traces.png       F_y/M_z FOM vs LDNet, families A/B/Cc (definitive model)
    fig_ch2_loss.png               train/valid loss vs epochs (definitive model)

Matplotlib only (no TensorFlow). Robust to missing inputs: any figure whose data is
not on disk is skipped with a message, so it can be run incrementally as the cluster
sweeps complete. Style follows light/latex/AGENTS.md ("Linee guida per i plot").

Run locally after pulling results/:  python3 -s -u light/plots_ldnet_accuracy.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS   = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.environ.get('SENS_ROOT', os.path.join(_THIS, '..', 'results', 'sensitivity'))
IMG_DIR = os.path.join(_THIS, 'latex', 'Images')
os.makedirs(IMG_DIR, exist_ok=True)

DEF_DS = int(os.environ.get('DEF_DS', '10'))   # definitive latent dimension (6-input model)
# Definitive/deployed model directory for the loads traces: the closed-loop
# rollout control model, evaluated open-loop (leak-aware). Falls back to the
# teacher-forced sweep dir in6/latent_{DEF_DS} if this path has no traces.
DEF_MODEL_DIR = os.environ.get(
    'DEF_MODEL_DIR', os.path.join(ROOT, 'rollout_eval', 'in6', f'latent_{DEF_DS}'))

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

# Sober, consistent palette (shared with the chapter-3 figures)
C_COMB = 'k'
C_FY   = '#4477AA'
C_MZ   = '#CC3311'
C_SET  = {'2': '#CC3311', '4': '#EE7733', '6': '#4477AA'}
LBL_SET = {
    '2': r'2 inputs ($W,\delta$)',
    '4': r'4 inputs ($+\,h,\alpha$)',
    '6': r'6 inputs ($+\,\dot h,\dot\alpha$)',
}
MK_SET = {'2': 'o', '4': 's', '6': '^'}


def _save(fig, name):
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def load_metrics(set_id):
    """Return the sorted metrics list for an input set, or None if absent.

    Falls back to the flat legacy layout (results/sensitivity/summary) for the
    6-input sweep, which pre-dates the per-input-set directories.
    """
    candidates = [os.path.join(ROOT, f'in{set_id}', 'summary', 'all_metrics.json')]
    if set_id == '6':
        candidates.append(os.path.join(ROOT, 'summary', 'all_metrics.json'))
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                m = json.load(f)
            m.sort(key=lambda d: d['num_latent_states'])
            return m
    return None


def _latent_dir(set_id, ds):
    """Resolve latent_{ds} dir for an input set (flat-layout fallback for in6)."""
    candidates = [os.path.join(ROOT, f'in{set_id}', f'latent_{ds}')]
    if set_id == '6':
        candidates.append(os.path.join(ROOT, f'latent_{ds}'))
    for d in candidates:
        if os.path.isdir(d):
            return d
    return None


# ---------------------------------------------------------------------------
def fig_input_study():
    """NRMSE and 1-rho vs d_s, comparing the 2/4/6-input configurations (combined)."""
    data = {s: load_metrics(s) for s in ('2', '4', '6')}
    have = [s for s in ('2', '4', '6') if data[s]]
    if not have:
        print('  SKIP fig_ch2_input_study: no all_metrics.json for any input set')
        return
    if len(have) < 3:
        print(f'  NOTE fig_ch2_input_study: only input sets {have} available '
              f'(missing {sorted(set("246") - set(have))}); plotting what exists')

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for s in have:
        m = data[s]
        ds = [d['num_latent_states'] for d in m]
        axs[0].semilogy(ds, [d['NRMSE'] for d in m], MK_SET[s] + '-',
                        color=C_SET[s], ms=4, label=LBL_SET[s])
        axs[1].semilogy(ds, [1 - d['rho'] for d in m], MK_SET[s] + '-',
                        color=C_SET[s], ms=4, label=LBL_SET[s])
    axs[0].set_ylabel('NRMSE')
    axs[0].legend()
    axs[1].set_ylabel(r'$1-\rho$')
    all_ds = sorted({d['num_latent_states'] for s in have for d in data[s]})
    for ax in axs:
        ax.set_xlabel(r'latent dimension $d_s$')
        ax.set_xticks(all_ds)
        ax.grid(True, which='both', ls=':')
    fig.tight_layout()
    _save(fig, 'fig_ch2_input_study.png')


def fig_latent_sensitivity():
    """NRMSE and 1-rho vs d_s for the 6-input config (combined, F_y, M_z)."""
    m = load_metrics('6')
    if m is None:
        print('  SKIP fig_ch2_latent_sensitivity: no 6-input all_metrics.json')
        return
    ds = [d['num_latent_states'] for d in m]

    fig, axs = plt.subplots(2, 1, figsize=(4.8, 4.4), sharex=True)
    axs[0].semilogy(ds, [d['NRMSE']     for d in m], 'o-',  color=C_COMB, ms=4, label='combined')
    axs[0].semilogy(ds, [d['NRMSE_F_y'] for d in m], '^--', color=C_FY,   ms=4, label=r'$F_y$')
    axs[0].semilogy(ds, [d['NRMSE_M_z'] for d in m], 's--', color=C_MZ,   ms=4, label=r'$M_z$')
    axs[0].set_ylabel('NRMSE')
    axs[0].legend()

    axs[1].semilogy(ds, [1 - d['rho']     for d in m], 'o-',  color=C_COMB, ms=4, label='combined')
    axs[1].semilogy(ds, [1 - d['rho_F_y'] for d in m], '^--', color=C_FY,   ms=4, label=r'$F_y$')
    axs[1].semilogy(ds, [1 - d['rho_M_z'] for d in m], 's--', color=C_MZ,   ms=4, label=r'$M_z$')
    axs[1].set_ylabel(r'$1-\rho$')
    axs[1].set_xlabel(r'latent dimension $d_s$')
    axs[1].set_xticks(ds)
    axs[0].grid(True, which='both', ls=':')
    axs[1].grid(True, which='both', ls=':')
    fig.tight_layout()
    _save(fig, 'fig_ch2_latent_sensitivity.png')


def fig_loads_traces():
    """F_y and M_z, FOM vs LDNet, for families A/B/Cc (definitive 6-input, d_s=DEF_DS)."""
    # Prefer the deployed rollout control model (open-loop leak-aware traces);
    # fall back to the teacher-forced sweep model if unavailable.
    tpath = os.path.join(DEF_MODEL_DIR, 'traces.npz')
    if not os.path.exists(tpath):
        ldir = _latent_dir('6', DEF_DS)
        tpath = os.path.join(ldir, 'traces.npz') if ldir else None
    if not tpath or not os.path.exists(tpath):
        print(f'  SKIP fig_ch2_loads_traces: no traces.npz for the definitive model '
              f'(looked in {DEF_MODEL_DIR} and in6/latent_{DEF_DS})')
        return
    d = np.load(tpath)
    t = d['time']
    fams = [f for f in ('A', 'B', 'Cc') if f'{f}_fom_Fy' in d.files]

    # Relative error is expressed in % of each load's global (all-family) range,
    # the same normalization as the reported NRMSE, so the error panels share one
    # comparable scale. The F_y and M_z rows use common y-limits across families
    # (taken from the widest family) so the panels are comparable at a glance.
    fy_all = np.concatenate([d[f'{f}_fom_Fy'] for f in fams])
    mz_all = np.concatenate([d[f'{f}_fom_Mz'] for f in fams])
    fy_rng = fy_all.max() - fy_all.min()
    mz_rng = mz_all.max() - mz_all.min()

    def _lims(arr_list, pad=0.05):
        lo = min(a.min() for a in arr_list); hi = max(a.max() for a in arr_list)
        m = pad * (hi - lo)
        return lo - m, hi + m
    fy_lim = _lims([d[f'{f}_fom_Fy'] for f in fams] + [d[f'{f}_rom_Fy'] for f in fams])
    mz_lim = _lims([d[f'{f}_fom_Mz'] for f in fams] + [d[f'{f}_rom_Mz'] for f in fams])

    err_max = 0.0
    for f in fams:
        eF = 100 * np.abs(d[f'{f}_fom_Fy'] - d[f'{f}_rom_Fy']) / fy_rng
        eM = 100 * np.abs(d[f'{f}_fom_Mz'] - d[f'{f}_rom_Mz']) / mz_rng
        err_max = max(err_max, eF.max(), eM.max())

    fig, axs = plt.subplots(3, len(fams), figsize=(6.3, 5.2), sharex=True, squeeze=False)
    for col, fam in enumerate(fams):
        axs[0, col].plot(t, d[f'{fam}_fom_Fy'], '-',  color=C_COMB, lw=1.1, label='FOM')
        axs[0, col].plot(t, d[f'{fam}_rom_Fy'], '--', color=C_MZ,   lw=1.1, label='LDNet')
        axs[0, col].set_title(f'Family {fam}')
        axs[0, col].set_ylim(fy_lim)
        axs[1, col].plot(t, d[f'{fam}_fom_Mz'], '-',  color=C_COMB, lw=1.1)
        axs[1, col].plot(t, d[f'{fam}_rom_Mz'], '--', color=C_MZ,   lw=1.1)
        axs[1, col].set_ylim(mz_lim)
        axs[2, col].plot(t, 100 * np.abs(d[f'{fam}_fom_Fy'] - d[f'{fam}_rom_Fy']) / fy_rng,
                         '-', color=C_FY, lw=1.0, label=r'$|\Delta F_y|$')
        axs[2, col].plot(t, 100 * np.abs(d[f'{fam}_fom_Mz'] - d[f'{fam}_rom_Mz']) / mz_rng,
                         '--', color=C_MZ, lw=1.0, label=r'$|\Delta M_z|$')
        axs[2, col].set_ylim(0, 1.05 * err_max)
        axs[2, col].set_xlabel('$t$ [s]')
    axs[0, 0].set_ylabel(r'$F_y$ [N]')
    axs[1, 0].set_ylabel(r'$M_z$ [N$\cdot$m]')
    axs[2, 0].set_ylabel('rel. error [%]')
    axs[0, 0].legend()
    axs[2, 0].legend()
    fig.tight_layout()
    _save(fig, 'fig_ch2_loads_traces.png')


def fig_loss():
    """Training/validation loss vs epochs for the definitive model (log-log)."""
    ldir = _latent_dir('6', DEF_DS)
    lpath = os.path.join(ldir, 'loss_history.npz') if ldir else None
    if not lpath or not os.path.exists(lpath):
        print(f'  SKIP fig_ch2_loss: no loss_history.npz for 6-input d_s={DEF_DS} '
              f'(re-run sensitivity_latent.py to produce it)')
        return
    d = np.load(lpath)
    it, tr, va = d['iterations'], d['train'], d['valid']

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.loglog(it, tr, 'o-', color=C_FY, ms=3, lw=1.0, label='training')
    ax.loglog(it, va, 'o-', color=C_MZ, ms=3, lw=1.0, label='validation')
    ax.set_ylim(top=0.5)
    if 'adam_epochs' in d.files:
        ax.axvline(int(d['adam_epochs']), color='k', ls='--', lw=0.8)
        ax.text(int(d['adam_epochs']), ax.get_ylim()[1],
                r'Adam$\to$L-BFGS', fontsize=7, rotation=90, va='top', ha='right')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.legend()
    ax.grid(True, which='both', ls=':')
    fig.tight_layout()
    _save(fig, 'fig_ch2_loss.png')


if __name__ == '__main__':
    print(f'Reading sensitivity results from: {ROOT}')
    print(f'Writing figures to: {IMG_DIR}')
    fig_input_study()
    fig_latent_sensitivity()
    fig_loads_traces()
    fig_loss()
    print('Done.')

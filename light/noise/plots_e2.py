"""
Chapter figures for the E2 study (LITERATURE.md techniques -> NOTES.md E2/E2C/
E2CC sections). Reads results/E2_*.npz (record schema: harness_noise.py),
needs numpy+matplotlib only (no TF/harness import — runs anywhere).

  fig_E2_money.png   time traces, home cell W30/Tg0.4: worst one-step seed on
                     the raw sigma=2% preview (explodes post-gust) vs a
                     combo-dlr seed (fusion Jmax=50 + MPC N=8, raw LOS noise
                     1-3 m/s) that tracks the clean optimum.
  fig_E2_ladder.png  all E2 arms at the home cell, sigma=2% (dlr: raw 1-3 m/s),
                     mean + [min,max] whiskers, colored by flag count, with
                     the one-step-clean and combo-clean reference lines.
  fig_E2_cells.png   generality: the same untuned combo at the three study
                     cells vs the one-step clean and noisy baselines.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')


def load(name):
    return list(np.load(os.path.join(RES, name), allow_pickle=True)['records'])


def pt(recs, **sel):
    """The unique point record matching all key=value selectors."""
    out = [r for r in recs if r.get('kind') == 'point'
           and all(r.get(k) == v for k, v in sel.items())]
    assert len(out) == 1, f"{sel} -> {len(out)} matches"
    return out[0]


def trajs(recs, prefix):
    return [r for r in recs if r.get('kind') == 'traj'
            and str(r.get('label', '')).startswith(prefix)]


CF = load('E2_combo_flat.npz')
CD = load('E2_combo_dlr.npz')
GA = load('E2_gate.npz')
KA = load('E2_kalman.npz')
MA = load('E2_mpc.npz')
SA = load('E2_sensor.npz')
W10 = load('E2_combo_cells_W10T07.npz')
W30 = load('E2_combo_cells_W30T07.npz')

OPEN = [r for r in CF if r['kind'] == 'open'][0]
TRIM = float(OPEN['CL'][0])          # open loop starts from the trimmed state
CLEAN_1STEP = 76.58                  # study regression anchor (light/run.py)
CLEAN_COMBO = float(pt(CF, arm='anchor', R_du=0.0)['mean'])


# ---- fig 1: money plot -----------------------------------------------------------
def fig_money():
    tn = min(trajs(CF, 'E2C_none_'), key=lambda r: r['clred'])
    td = min(trajs(CD, 'E2C_dlr_best_'), key=lambda r: r['clred'])
    fig, ax = plt.subplots(3, 2, figsize=(10.5, 7.5), sharex=True, sharey='row')
    for j, (tr, ttl) in enumerate([
            (tn, f"one-step, raw preview σ=2%·W0 — worst seed "
                 f"(CLred {tr_clred(tn)}, flag '{tn['flag']}')"),
            (td, f"combo: fusion + MPC N=8, raw LOS 1–3 m/s "
                 f"(CLred {tr_clred(td)}, no flags)")]):
        t = tr['t']
        ax[0, j].plot(t, tr['W'], 'k-', lw=1.6, label='W true')
        ax[0, j].plot(t, tr['Wc'], color='tab:orange', lw=0.7, alpha=0.85,
                      label='W seen (t+dt ch.)')
        ax[1, j].plot(OPEN['t'], OPEN['CL'], color='0.65', lw=1.0, label='open loop')
        ax[1, j].plot(t, tr['CL'], 'b-', lw=1.2, label='closed loop')
        ax[1, j].axhline(TRIM, color='k', ls=':', lw=0.8)
        ax[2, j].plot(t, tr['de'], 'g-', lw=1.2)
        ax[0, j].set_title(ttl, fontsize=9.5)
    ax[0, 0].set_ylabel('W [m/s]'); ax[1, 0].set_ylabel('$C_L$')
    ax[2, 0].set_ylabel(r'$\delta$ [deg]')
    for a in ax[2]: a.set_xlabel('t [s]'); a.set_xlim(0, 1.2)
    ax[0, 0].legend(fontsize=8, loc='upper right')
    ax[1, 0].legend(fontsize=8, loc='upper right')
    fig.suptitle('W30/Tg0.4, DAMULT=3 — raw single-sample preview vs the composed pipeline',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_E2_money.png'), dpi=180)
    print('fig_E2_money.png')


def tr_clred(tr):
    return f"{tr['clred']:+.1f}%"


# ---- fig 2: mitigation ladder at the home cell ------------------------------------
def fig_ladder():
    entries = [   # (label, point record)  — all at sigma=2% unless noted
        ('one-step, raw preview',            pt(CF, arm='none')),
        ('+ Kalman CV (q=1e4)',              pt(KA, arm='kf', detail='q=10000', frac=0.02)),
        ('+ command washout τ=200ms',        pt(GA, arm='wash', detail='tau=200ms', frac=0.02)),
        ('MPC N4, current gust only',        pt(MA, arm='mpc-cur', frac=0.02)),
        ('+ fusion J=50, λ=10 (one-step)',   pt(SA, arm='fuseT', detail='J=50 lam=10', frac=0.02)),
        ('+ fusion J=25 (one-step)',         pt(SA, arm='fuse', detail='J=25', frac=0.02)),
        ('MPC N8, raw preview',              pt(MA, arm='mpcp', detail='N=8', frac=0.02)),
        ('COMBO: fusion + MPC N8',           pt(CF, arm='combo', detail='flat Rdu=0')),
        ('COMBO, raw LOS 1–3 m/s (!)',       pt(CD, arm='combo', detail='dlr lam=1 Rdu=0')),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(entries))[::-1]
    for yi, (lab, r) in zip(y, entries):
        n = len(r['seeds'])
        c = ('tab:green' if r['nflag'] == 0 else
             'tab:orange' if r['nflag'] <= 2 else 'tab:red')
        ax.barh(yi, r['mean'], color=c, alpha=0.75, height=0.62)
        ax.errorbar(r['mean'], yi, xerr=[[r['mean'] - r['lo']], [r['hi'] - r['mean']]],
                    fmt='none', ecolor='k', elinewidth=1.1, capsize=3)
        ax.text(max(r['hi'], r['mean']) + 1.5, yi,
                f"{r['mean']:+.1f}%  ({r['nflag']}/{n} flags)",
                va='center', fontsize=8.5)
    ax.axvline(0, color='k', lw=0.8)
    ax.axvline(CLEAN_1STEP, color='0.4', ls='--', lw=1.0)
    ax.text(CLEAN_1STEP, len(entries) - 0.25, ' one-step clean', fontsize=8, color='0.3')
    ax.axvline(CLEAN_COMBO, color='tab:blue', ls='--', lw=1.0)
    ax.text(CLEAN_COMBO, -0.65, ' combo clean', fontsize=8, color='tab:blue')
    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in entries], fontsize=9)
    ax.set_xlabel('CLred [%] — mean and [min,max] over 6 seeds')
    ax.set_title('W30/Tg0.4, preview noise σ=2%·W0 (0.6 m/s); (!) = raw 1–3 m/s LOS noise',
                 fontsize=10)
    ax.set_xlim(-35, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_E2_ladder.png'), dpi=180)
    print('fig_E2_ladder.png')


# ---- fig 3: generality across cells ------------------------------------------------
def fig_cells():
    home = {'1step-clean': dict(mean=CLEAN_1STEP, lo=CLEAN_1STEP, hi=CLEAN_1STEP, nflag=0, seeds=[0]),
            'anchor':      pt(CF, arm='anchor', R_du=0.0),
            'none':        pt(CF, arm='none'),
            'combo-flat':  pt(CF, arm='combo', detail='flat Rdu=0'),
            'combo-dlr':   pt(CD, arm='combo', detail='dlr lam=1 Rdu=0')}
    cells = [('W30/Tg0.4\n(home, sharp)', home),
             ('W10/Tg0.7\n(gentle; dlr raw = 10–30%·W0)',
              {a: pt(W10, arm=a) for a in
               ('1step-clean', 'anchor', 'none', 'combo-flat', 'combo-dlr')}),
             ('W30/Tg0.7\n(strong, slower)',
              {a: pt(W30, arm=a) for a in
               ('1step-clean', 'anchor', 'none', 'combo-flat', 'combo-dlr')})]
    arms = [('1step-clean', 'one-step clean (oracle)', '0.6'),
            ('none',        'one-step, σ=2% raw',      'tab:red'),
            ('combo-flat',  'combo, σ=2%',             'tab:blue'),
            ('combo-dlr',   'combo, raw LOS 1–3 m/s',  'tab:green')]
    fig, ax = plt.subplots(figsize=(9, 5))
    W = 0.19
    for k, (aname, alab, col) in enumerate(arms):
        xs, ms = [], []
        for i, (cl, d) in enumerate(cells):
            r = d[aname]
            x = i + (k - 1.5) * W
            xs.append(x); ms.append(r['mean'])
            ax.bar(x, r['mean'], width=W * 0.92, color=col, alpha=0.8,
                   label=alab if i == 0 else None)
            if len(r['seeds']) > 1:
                ax.errorbar(x, r['mean'],
                            yerr=[[r['mean'] - r['lo']], [r['hi'] - r['mean']]],
                            fmt='none', ecolor='k', elinewidth=1.0, capsize=2.5)
            if r['nflag']:
                ax.text(x, min(r['lo'], r['mean']) - 4, f"{r['nflag']}/6!",
                        ha='center', fontsize=7.5, color='tab:red')
    for i, (cl, d) in enumerate(cells):   # combo clean anchor marker
        m = d['anchor']['mean']
        ax.plot([i - 2 * W, i + 2 * W], [m, m], 'k--', lw=1.0)
        ax.text(i + 2 * W, m + 0.6, f'combo clean {m:+.1f}', fontsize=7.5,
                ha='right')
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c[0] for c in cells], fontsize=9)
    ax.set_ylabel('CLred [%]')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylim(-15, 103)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.set_title('Generality of the untuned composition (fusion Jmax=50 → MPC N=8, R=3e-4)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_E2_cells.png'), dpi=180)
    print('fig_E2_cells.png')


fig_money()
fig_ladder()
fig_cells()
print('# DONE')

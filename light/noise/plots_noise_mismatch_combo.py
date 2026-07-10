"""
Figure: CLred vs structural / controller-parameter mismatch for the E2-combo.
Reads results/D2_mismatch.npz. Run locally after scp.
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
# our own npz (harness save_records object-array schema) -- trusted local file
recs = list(np.load(os.path.join(DIR, 'D2_mismatch.npz'), allow_pickle=True)['records'])
pts  = [r for r in recs if r.get('kind') == 'point']

ANCHOR = 80.51
U_NOM  = 80.0

def series(arm):
    rows = sorted([r for r in pts if r['arm'] == arm],
                  key=lambda r: float(r['value']))
    return ([float(r['value']) for r in rows],
            [float(r['mean']) for r in rows],
            [int(r['nflag']) for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# left: structural multipliers (plant vs nominal internal model)
ax = axes[0]
for arm, color, label in [('dalpha', 'tab:blue', 'plant D_ALPHA x (ctrl assumes x1)'),
                          ('kalpha', 'tab:green', 'plant K_ALPHA x (ctrl assumes x1)')]:
    x, m, nf = series(arm)
    x = x + [1.0]; m = m + [ANCHOR]; nf = nf + [0]
    order = np.argsort(x)
    x = list(np.array(x)[order]); m = list(np.array(m)[order])
    nf = list(np.array(nf)[order])
    ax.plot(x, m, 'o-', color=color, label=label)
    for xi, mi, f in zip(x, m, nf):
        if f:
            ax.plot(xi, mi, 'o', mfc='none', mec='red', ms=12, mew=1.5)
ax.axhline(ANCHOR, color='gray', lw=0.7, ls=':', label=f'clean anchor +{ANCHOR}%')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel('plant parameter multiplier vs internal model')
ax.set_ylabel('CLred [%]')
ax.set_title('D2 -- structural mismatch (red ring = flagged)')
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)

# right: controller-side estimate errors (U, CLtrim), x = % error
ax = axes[1]
xu, mu, nfu = series('uinf')
xu_pct = [(u / U_NOM - 1.0) * 100 for u in xu]
xc, mc, nfc = series('cltrim')
xc_pct = [(c - 1.0) * 100 for c in xc]
for xs, ms_, nfs, color, mk, label in [
        (xu_pct, mu, nfu, 'tab:purple', 'o', 'controller U error'),
        (xc_pct, mc, nfc, 'tab:brown', 's', 'controller C_L_trim error')]:
    xs = xs + [0.0]; ms_ = ms_ + [ANCHOR]; nfs = nfs + [0]
    order = np.argsort(xs)
    xs = list(np.array(xs)[order]); ms_ = list(np.array(ms_)[order])
    nfs = list(np.array(nfs)[order])
    ax.plot(xs, ms_, mk + '-', color=color, label=label)
    for xi, mi, f in zip(xs, ms_, nfs):
        if f:
            ax.plot(xi, mi, mk, mfc='none', mec='red', ms=12, mew=1.5)
ax.axhline(ANCHOR, color='gray', lw=0.7, ls=':', label=f'clean anchor +{ANCHOR}%')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel('controller estimate error [%]')
ax.set_ylabel('CLred [%]')
ax.set_title('D2 -- controller-side parameter errors')
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)

fig.suptitle('E2-combo model-mismatch robustness (W30/Tg0.4, DAMULT=3)', y=1.02)
plt.tight_layout()
fn = os.path.join(DIR, 'fig_noise_mismatch_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}')

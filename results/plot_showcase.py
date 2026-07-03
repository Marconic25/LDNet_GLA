"""6 figures (open / prop-W / opt-W) for large-k (Tg=0.30) and small-k (Tg=1.20)
at W10/W20/W30. Reads results/showcase.npz."""
import numpy as np, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

D = np.load('/home/marco/LDNet_OF/results/showcase.npz')
CLTRIM = 0.86834
outdir = '/home/marco/LDNet_OF/results/showcase'; os.makedirs(outdir, exist_ok=True)
tags = sorted({k.rsplit('_', 2)[0] for k in D.files if k.endswith('_opt_CL')})
COL = {'open': ('0.55', 'open loop'), 'prop': ('darkorange', 'prop-W'), 'opt': ('seagreen', 'opt-W')}
ROWS = [('CL', 'C_L - C_L_trim', 1.0, CLTRIM), ('CM', 'C_M', 1.0, 0.0),
        ('al', 'alpha [deg]', 180 / np.pi, 0.0), ('ad', 'alpha_dot [deg/s]', 180 / np.pi, 0.0),
        ('de', 'delta [deg]', 1.0, 0.0)]

for tag in tags:
    t = D[f'{tag}_t']; W = D[f'{tag}_W']; Tg = float(D[f'{tag}_Tg'])
    k = np.pi / (80.0 * Tg); pcr = float(D[f'{tag}_pcr']); ocr = float(D[f'{tag}_ocr'])
    mw = t <= min(t[-1], Tg + 1.0)
    fig, ax = plt.subplots(6, 1, figsize=(7.5, 12), sharex=True)
    name = tag.replace('_', '/', 1)
    fig.suptitle(f'{name}   k={k:.3f}   |   prop-W CLred={pcr:.0f}%   opt-W CLred={ocr:.0f}%   '
                 f'(margin {ocr-pcr:+.0f})', fontsize=11)
    for j, (key, lab, sc, off) in enumerate(ROWS):
        for arm, (c, ll) in COL.items():
            if key == 'de' and arm == 'open':
                continue
            y = (D[f'{tag}_{arm}_{key}'] - off) * sc
            ax[j].plot(t[mw], y[mw], color=c, lw=1.4, label=ll)
        ax[j].set_ylabel(lab, fontsize=9); ax[j].grid(alpha=.3); ax[j].axvspan(0, Tg, color='lightblue', alpha=.12)
        if j == 0:
            ax[j].axhline(0, color='gray', lw=.6); ax[j].legend(fontsize=8, loc='upper right')
    ax[5].plot(t[mw], W[mw], 'k', lw=1.3); ax[5].set_ylabel('W gust [m/s]', fontsize=9)
    ax[5].grid(alpha=.3); ax[5].axvspan(0, Tg, color='lightblue', alpha=.12); ax[5].set_xlabel('t [s]')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fn = f'{outdir}/showcase_{tag}.png'; fig.savefig(fn, dpi=110); plt.close(fig); print('saved', fn)
print('done')

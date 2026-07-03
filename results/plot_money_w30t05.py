import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

d = np.load('/home/marco/LDNet_OF/results/w30t05_traces.npz')
CLTRIM = 0.86834
t = d['t']; W = d['W']; g = float(d['gstar']); R = float(d['Rstar'])
pcr = float(d['pcr']); ocr = float(d['ocr'])
mw = t <= 1.5

fig, ax = plt.subplots(4, 1, figsize=(8.2, 10.5), sharex=True)
ax[0].plot(t[mw], W[mw], 'k', lw=1.4); ax[0].set_ylabel('W gust [m/s]')
ax[0].set_title(f'Single-step optimal vs proportional — W30 / Tg=0.5  (strong+sharp gust)\n'
                f'proportional best CLred={pcr:.0f}%  |  single-step CLred={ocr:.0f}%')

ax[1].plot(t[mw], d['prop_CL'][mw] - CLTRIM, color='darkorange', lw=1.4, label=f'proportional (g*={g:.0f})')
ax[1].plot(t[mw], d['one_CL'][mw] - CLTRIM, color='seagreen', lw=1.4, label=f'single-step optimal (R*={R:.0e})')
ax[1].axhline(0, color='gray', lw=.6); ax[1].set_ylabel('C_L - C_L_trim'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

ax[2].plot(t[mw], d['prop_al'][mw] * 180 / np.pi, color='darkorange', lw=1.4, label='proportional')
ax[2].plot(t[mw], d['one_al'][mw] * 180 / np.pi, color='seagreen', lw=1.4, label='single-step optimal')
ax[2].set_ylabel('alpha [deg] (torsion)'); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

ax[3].plot(t[mw], d['prop_de'][mw], color='darkorange', lw=1.4, label='proportional delta')
ax[3].plot(t[mw], d['one_de'][mw], color='seagreen', lw=1.4, label='single-step delta')
ax[3].set_ylabel('delta [deg]'); ax[3].set_xlabel('t [s]'); ax[3].legend(fontsize=8); ax[3].grid(alpha=.3)

for a in ax:
    a.axvspan(0, 0.5, color='lightblue', alpha=.12); a.axvline(0.25, color='gray', ls=':', lw=.7)
fig.tight_layout()
fig.savefig('/home/marco/LDNet_OF/results/money_plot_W30T05.png', dpi=120)
print('saved money_plot_W30T05.png')

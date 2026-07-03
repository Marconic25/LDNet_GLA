import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

d = np.load('/home/marco/LDNet_OF/results/onestep_traces.npz')
CLTRIM = 0.86834
tag = 'W30_T10'; title = 'W30 / Tg=1.0  (k=0.039, strong gust)'
t = d[f'{tag}_t']; W = d[f'{tag}_W']; g = float(d[f'{tag}_gstar'])
p_de = d[f'{tag}_prop_de']; o_de = d[f'{tag}_one_de']
p_CL = d[f'{tag}_prop_CL']; o_CL = d[f'{tag}_one_CL']
o_K = d[f'{tag}_one_K']
mw = t <= 2.0

fig, ax = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
ax[0].plot(t[mw], W[mw], 'k', lw=1.3); ax[0].set_ylabel('W gust [m/s]')
ax[0].set_title('Single-step optimal vs proportional — ' + title)

ax[1].plot(t[mw], p_CL[mw] - CLTRIM, color='darkorange', lw=1.3, label='proportional')
ax[1].plot(t[mw], o_CL[mw] - CLTRIM, color='seagreen', lw=1.3, label='single-step optimal')
ax[1].axhline(0, color='gray', lw=.6); ax[1].set_ylabel('C_L - C_L_trim'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

ax[2].plot(t[mw], p_de[mw], color='darkorange', lw=1.3, label='proportional')
ax[2].plot(t[mw], o_de[mw], color='seagreen', lw=1.3, label='single-step optimal')
ax[2].set_ylabel('delta [deg]'); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

# effective gain |K(t)| = |delta/(C_L0 - trim)|
ax[3].axhline(abs(g), color='darkorange', lw=1.5, ls='--', label=f'proportional fixed |g*|={abs(g):.0f}')
ax[3].plot(t[mw], np.abs(o_K[mw]), color='seagreen', lw=1.3, label='single-step effective |K(t)| (model-adaptive)')
ax[3].set_ylabel('|effective gain|'); ax[3].set_xlabel('t [s]'); ax[3].set_ylim(0, max(60, abs(g)*1.5))
ax[3].legend(fontsize=8); ax[3].grid(alpha=.3)

fig.tight_layout()
fig.savefig('/home/marco/LDNet_OF/results/money_plot_W30T1.png', dpi=120)
print('saved money_plot_W30T1.png')

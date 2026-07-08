"""
Plot for the honest equal-information comparison (honest_home.py output).

4-panel figure (C_L, delta, W, alpha_dot) with three curves: open (black
dashed), prop-W(t+dt) (blue), optimal wnext (red). TF-free: reads
results_honest/honest_home.npz only.

    python3 -s -u honest_plots.py
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_honest')
d = np.load(os.path.join(DIR, 'honest_home.npz'), allow_pickle=False)

t = d['ts']; Wt = d['Wt']; CLTRIM = float(d['CLTRIM'])
W0 = float(d['W0']); Tg = float(d['Tg'])

OL_KW = dict(color='black', linestyle='--', linewidth=1.5, label='open')
PW_KW = dict(color='tab:blue', linestyle='-', linewidth=1.5,
             label=f'prop-W(t+dt)  ({float(d["pw_clred"]):+.1f}%)')
OP_KW = dict(color='red', linestyle='-', linewidth=1.5,
             label=f'optimal wnext ({float(d["opt_clred"]):+.1f}%)')
SH_KW = dict(alpha=0.15, color='#aad4f5', zorder=0)

fig, axes = plt.subplots(4, 1, figsize=(5.5, 7.5), sharex=True)
panels = [('CL', r'$C_L$'), ('de', r'$\delta$ [deg]'),
          (None, r'$W$ [m/s]'), ('ad', r'$\dot\alpha$ [deg/s]')]
for ax, (key, lab) in zip(axes, panels):
    if key is None:
        ax.plot(t, Wt, color='green', linewidth=1.5, label='W_gust')
    else:
        sc = 180/np.pi if key == 'ad' else 1.0
        ax.plot(t, d[f'open_{key}']*sc, **OL_KW)
        ax.plot(t, d[f'pw_{key}']*sc, **PW_KW)
        ax.plot(t, d[f'opt_{key}']*sc, **OP_KW)
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
fig.suptitle(
    f'Honest equal-information comparison — W0={W0:g} m/s, Tg={Tg:g} s, DAMULT=3\n'
    f'both see W(t+dt); prop: gCL={float(d["pw_gCL"]):g}, gW={float(d["pw_gW"]):g}; '
    f'optimal: R*={float(d["opt_R"]):g}  (no-flag pick)', fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fn = os.path.join(DIR, 'honest_W30_Tg04.png')
fig.savefig(fn, dpi=150, bbox_inches='tight')
print(f'saved {fn}', flush=True)
print('# PLOTS DONE', flush=True)

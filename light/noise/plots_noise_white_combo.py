"""
Figure: CLred vs sigma/W0 for E2-combo vs one-step none.
Reads results/W_combo.npz. Run locally after scp.
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
recs = list(np.load(os.path.join(DIR, 'W_combo.npz'), allow_pickle=True)['records'])

combo = [r for r in recs if r.get('kind') == 'point' and r.get('arm') == 'combo']
none  = [r for r in recs if r.get('kind') == 'point' and r.get('arm') == 'none']

combo_f = sorted({float(r['frac']) for r in combo})
none_f  = sorted({float(r['frac']) for r in none})

def get_stats(rows, frac):
    pts = [r for r in rows if abs(float(r['frac']) - frac) < 1e-9]
    if not pts: return None
    r = pts[0]
    return float(r['mean']), float(r['lo']), float(r['hi'])

fig, ax = plt.subplots(figsize=(7, 4))
c_fracs = combo_f
n_fracs = none_f

c_means, c_lo, c_hi = [], [], []
for f in c_fracs:
    s = get_stats(combo, f)
    c_means.append(s[0]); c_lo.append(s[1]); c_hi.append(s[2])

n_means, n_lo, n_hi = [], [], []
for f in n_fracs:
    s = get_stats(none, f)
    if s: n_means.append(s[0]); n_lo.append(s[1]); n_hi.append(s[2])
    else: n_means.append(float('nan')); n_lo.append(float('nan')); n_hi.append(float('nan'))

pcts = [f * 100 for f in c_fracs]
ax.fill_between(pcts, c_lo, c_hi, alpha=0.2, color='tab:blue')
ax.plot(pcts, c_means, 'o-', color='tab:blue', label='E2-combo (Jmax=50, N=8)')

if n_fracs:
    npcts = [f * 100 for f in n_fracs]
    ax.fill_between(npcts, n_lo, n_hi, alpha=0.2, color='tab:red')
    ax.plot(npcts, n_means, 's--', color='tab:red', label='one-step optimal (no fusion)')

ax.axhline(0, color='gray', lw=0.7)
ax.axhline(32.0, color='gray', lw=0.7, ls=':', label='prop-W clean (+32%)')
ax.set_xlabel('Raw measurement noise sigma / W0 [%]')
ax.set_ylabel('CLred [%]')
ax.set_title('White-noise robustness -- E2-combo vs one-step (W30/Tg0.4, DAMULT=3)')
ax.legend(frameon=False); ax.grid(alpha=0.3)
plt.tight_layout()
fn = os.path.join(DIR, 'fig_noise_white_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}')

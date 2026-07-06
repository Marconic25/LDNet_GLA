"""
test_optimal.py — Visual sanity check for the final OptimalController
(wnext + refine: flap candidates evaluated at the next-step gust W(t+dt)).

Runs open-loop and optimal GLA for W0=30 m/s, Tg=0.4 s and produces a
4-panel figure:
  1. C_L vs time
  2. delta [deg] vs time
  3. W_gust [m/s] vs time
  4. alpha_dot [deg/s] vs time

Saves: tests/optimal_W30_Tg04.png  (relative to this file's directory)
"""

import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Allow imports from light/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# DAMULT must be set BEFORE importing run (applied to structure.D_ALPHA at import).
# DAMULT=3 matches the reference study plant. R=3e-4 is the universal choice for
# the final wnext controller (see 76/NOTES.md and the note in optimal.py).
os.environ.setdefault('DAMULT', '3')

from run import simulate, metrics, CLTRIM

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
W0 = 30.0
Tg = 0.4
R  = float(os.environ.get('R', '3e-4'))

# ------------------------------------------------------------------
# Simulations
# ------------------------------------------------------------------
print(f"Running open-loop  (W0={W0}, Tg={Tg}, DAMULT={os.environ['DAMULT']}) ...", flush=True)
r_open = simulate('open',    W0, Tg, R=R, DLPF=0.0, DMAX=14.)

print(f"Running optimal    (W0={W0}, Tg={Tg}, R={R:g}) ...", flush=True)
r_opt  = simulate('optimal', W0, Tg, R=R, DLPF=0.0, DMAX=14.)

# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
m = metrics(r_opt, r_open, Tg)
print(f"\n--- Metrics (optimal vs open) ---", flush=True)
print(f"  CLred%   = {m['clred']:+.1f} %", flush=True)
print(f"  flap_max = {m['flap_max']:.2f} deg", flush=True)
print(f"  adrms    = {m['adrms']:.4f} deg/s", flush=True)
print(f"  comp_ms  = {m['comp_ms']:.2f} ms/step", flush=True)
if m['flag']:
    print(f"  WARNING: {m['flag']}", flush=True)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
t      = r_open['_t']
Wt     = r_open['_Wt']
CL_o   = r_open['CL']
CL_c   = r_opt['CL']
de_o   = r_open['de']           # always 0 for open, kept for reference
de_c   = r_opt['de']
ad_o   = r_open['ad'] * 180.0 / np.pi   # rad/s -> deg/s
ad_c   = r_opt['ad']  * 180.0 / np.pi

OL_KW  = dict(color='black', linestyle='--', linewidth=1.5, label='open')
CL_KW  = dict(color='red',   linestyle='-',  linewidth=1.5, label='optimal')
SH_KW  = dict(alpha=0.15, color='#aad4f5', zorder=0)   # gust-window shade

fig, axes = plt.subplots(4, 1, figsize=(5, 7), sharex=True)

def shade(ax):
    ax.axvspan(0.0, Tg, **SH_KW)

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

# Panel 1 — C_L
ax = axes[0]
ax.plot(t, CL_o, **OL_KW)
ax.plot(t, CL_c, **CL_KW)
ax.set_ylabel(r'$C_L$', fontsize=9)
ax.legend(fontsize=8, frameon=False)
shade(ax); despine(ax)

# Panel 2 — delta
ax = axes[1]
ax.plot(t, de_o, **OL_KW)
ax.plot(t, de_c, **CL_KW)
ax.set_ylabel(r'$\delta$ [deg]', fontsize=9)
shade(ax); despine(ax)

# Panel 3 — W_gust
ax = axes[2]
ax.plot(t, Wt, color='green', linestyle='-', linewidth=1.5, label='W_gust')
ax.set_ylabel(r'$W$ [m/s]', fontsize=9)
ax.legend(fontsize=8, frameon=False)
shade(ax); despine(ax)

# Panel 4 — alpha_dot
ax = axes[3]
ax.plot(t, ad_o, **OL_KW)
ax.plot(t, ad_c, **CL_KW)
ax.set_ylabel(r'$\dot\alpha$ [deg/s]', fontsize=9)
ax.set_xlabel('time [s]', fontsize=9)
shade(ax); despine(ax)

fig.suptitle(f"W0={W0:g} Tg={Tg:g}  R={R:g} DAMULT={os.environ['DAMULT']}  "
             f"CLred={m['clred']:+.1f}%", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.97])

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimal_W30_Tg04.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}", flush=True)

"""
Final money plot at W30/Tg0.4, DAMULT=3: open loop vs best prop-W vs the winning
single-step controller (SS wnext R3e-4). 4 panels: C_L, delta, W_gust, alpha_dot.
Style mirrors light/tests/test_optimal.py. Self-contained (runs the 3 arms fresh).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import harness as H
from controllers import OptGrid, PropW

W0, Tg = 30.0, 0.4
OL = H.scalar_rollout(None, W0, Tg)
cex0 = H.cex_of(OL['CL'], OL['_t'], Tg)

# best prop-W over the full clean/propw.py reference gain grid (fair tuned baseline)
bpw = None
for gCL in (-10., -20., -40., -60., -80., -120., -160.):
    for gW in (-1.0, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0.0):
        r = H.scalar_rollout(PropW(gain_CL=gCL, gain_W=gW), W0, Tg)
        m = H.metrics(r, OL, Tg)
        if bpw is None or m['clred'] > bpw[1]['clred']:
            bpw = (r, m, gCL, gW)
rPW, mPW, gCL, gW = bpw

rWN = H.scalar_rollout(OptGrid(R=3e-4, G=161, gate='hard', use_wnext=True), W0, Tg)
mWN = H.metrics(rWN, OL, Tg)
mOL = H.metrics(OL, OL, Tg)

print(f"open cex0={cex0:.4f}", flush=True)
print(f"prop-W best g=({gCL:g},{gW:g}) CLred={mPW['clred']:+.1f}% flap={mPW['flap_max']:.1f} "
      f"pitch={mPW['pitchpk']*180/np.pi:.3f}", flush=True)
print(f"wnext R3e-4          CLred={mWN['clred']:+.1f}% flap={mWN['flap_max']:.1f} "
      f"pitch={mWN['pitchpk']*180/np.pi:.3f}", flush=True)

t = OL['_t']; Wt = OL['_Wt']
OL_KW = dict(color='black', linestyle='--', linewidth=1.5)
PW_KW = dict(color='tab:orange', linestyle='-', linewidth=1.3)
WN_KW = dict(color='crimson', linestyle='-', linewidth=1.6)
SH_KW = dict(alpha=0.15, color='#aad4f5', zorder=0)

fig, axes = plt.subplots(4, 1, figsize=(5.2, 7.4), sharex=True)

def shade(ax): ax.axvspan(0.0, Tg, **SH_KW)
def despine(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.25)

ax = axes[0]
ax.axhline(H.CLTRIM, color='gray', lw=0.7, ls=':')
ax.plot(t, OL['CL'], label=f'open ({mOL["clexc"]:.3f})', **OL_KW)
ax.plot(t, rPW['CL'], label=f'prop-W ({mPW["clred"]:+.0f}%)', **PW_KW)
ax.plot(t, rWN['CL'], label=f'wnext ({mWN["clred"]:+.0f}%)', **WN_KW)
ax.set_ylabel(r'$C_L$', fontsize=9); ax.legend(fontsize=7.5, frameon=False, loc='lower right')
shade(ax); despine(ax)

ax = axes[1]
ax.plot(t, OL['de'], **OL_KW); ax.plot(t, rPW['de'], **PW_KW); ax.plot(t, rWN['de'], **WN_KW)
ax.set_ylabel(r'$\delta$ [deg]', fontsize=9); shade(ax); despine(ax)

ax = axes[2]
ax.plot(t, Wt, color='green', lw=1.5)
ax.set_ylabel(r'$W$ [m/s]', fontsize=9); shade(ax); despine(ax)

ax = axes[3]
ax.plot(t, OL['ad']*180/np.pi, **OL_KW)
ax.plot(t, rPW['ad']*180/np.pi, **PW_KW)
ax.plot(t, rWN['ad']*180/np.pi, **WN_KW)
ax.set_ylabel(r'$\dot\alpha$ [deg/s]', fontsize=9); ax.set_xlabel('time [s]', fontsize=9)
shade(ax); despine(ax)

fig.suptitle(f"W0={W0:g} Tg={Tg:g} DAMULT=3  |  open cex={mOL['clexc']:.3f}  "
             f"prop-W {mPW['clred']:+.0f}%  wnext {mWN['clred']:+.0f}%", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'money_W30_Tg04.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
np.savez_compressed('results_money.npz', t=t, Wt=Wt, CLTRIM=H.CLTRIM, cex0=cex0,
    OL_CL=OL['CL'], OL_de=OL['de'], OL_ad=OL['ad'],
    PW_CL=rPW['CL'], PW_de=rPW['de'], PW_ad=rPW['ad'], PW_clred=mPW['clred'], PW_g=(gCL,gW),
    WN_CL=rWN['CL'], WN_de=rWN['de'], WN_ad=rWN['ad'], WN_clred=mWN['clred'])
print(f"Saved {out}", flush=True)
print("# DONE", flush=True)

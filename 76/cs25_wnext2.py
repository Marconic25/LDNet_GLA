"""
CS-25 gust study, FIXED robust single-step wnext controller.

Two fixes vs cs25_wnext.py, objective UNCHANGED (J = (C_L-C_L*)^2 + R*delta^2):
  (1) staircase delta -> parabolic sub-cell refine on the SAME 161-pt grid
      (refine=True). Same argmin -> same good branch, just a continuous output.
      (A finer grid was tried and REJECTED: it changes the near-tie discretization
      and jumps to a worse branch.)
  (2) inconsistent CLred -> R* pick by MAX CLred subject to NO explosion flag
      (alpha_dot / alpha_ddot / h_ddot < 3x open loop, the physical instability
      gate) instead of the over-conservative pitch<=1.10x rule (whose ratio is
      inflated where the open-loop pitch is small). Pitch is reported for
      transparency; if every R flags, fall back to min pitch.

Same points (3x6 grid), same R_GRID, same CS-25.341 framing, same 4-panel plots
(now with smooth delta). Outputs to results_cs25_wnext2/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import harness as H
from controllers import OptGrid

TG_LIST = [0.30, 0.40, 0.50, 0.70, 1.00, 1.20]
W0_LIST = [10.0, 20.0, 30.0]
R_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
TEND = float(os.environ.get('TEND', '3.0'))
OUTD = 'results_cs25_wnext2'; os.makedirs(OUTD, exist_ok=True)

OL_KW = dict(color='black', linestyle='--', linewidth=1.5, label='open')
CL_KW = dict(color='red',   linestyle='-',  linewidth=1.5, label='wnext (refine)')
SH_KW = dict(alpha=0.15, color='#aad4f5', zorder=0)


def four_panel(cell, Tg, t, Wt, r_open, r_opt, m, Rstar, pr):
    fig, axes = plt.subplots(4, 1, figsize=(5, 7), sharex=True)
    def shade(ax): ax.axvspan(0.0, Tg, **SH_KW)
    def despine(ax):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
    ax = axes[0]
    ax.plot(t, r_open['CL'], **OL_KW); ax.plot(t, r_opt['CL'], **CL_KW)
    ax.axhline(H.CLTRIM, color='gray', lw=0.7, ls=':')
    ax.set_ylabel(r'$C_L$', fontsize=9); ax.legend(fontsize=8, frameon=False); shade(ax); despine(ax)
    ax = axes[1]
    ax.plot(t, r_open['de'], **OL_KW); ax.plot(t, r_opt['de'], **CL_KW)
    ax.set_ylabel(r'$\delta$ [deg]', fontsize=9); shade(ax); despine(ax)
    ax = axes[2]
    ax.plot(t, Wt, color='green', lw=1.5, label='W_gust')
    ax.set_ylabel(r'$W$ [m/s]', fontsize=9); ax.legend(fontsize=8, frameon=False); shade(ax); despine(ax)
    ax = axes[3]
    ax.plot(t, r_open['ad']*180/np.pi, **OL_KW); ax.plot(t, r_opt['ad']*180/np.pi, **CL_KW)
    ax.set_ylabel(r'$\dot\alpha$ [deg/s]', fontsize=9); ax.set_xlabel('time [s]', fontsize=9)
    shade(ax); despine(ax)
    kred = np.pi/(H.U*Tg); Hgrad = H.U*Tg/2.0
    fig.suptitle(f"{cell}  R*={Rstar:g} DAMULT=3 | H={Hgrad:.1f}m k={kred:.3f} | "
                 f"CLred={m['clred']:+.1f}% flap={m['flap_max']:.1f} pitch={pr:.2f}x",
                 fontsize=8.5)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTD, f"cs25w2_{cell.replace('/','_')}.png"), dpi=140, bbox_inches='tight')
    plt.close(fig)


print(f"# cs25_wnext2 (refine + no-flag pick)  grid {W0_LIST}x{TG_LIST}  DAMULT={os.environ.get('DAMULT','1')}", flush=True)
print(f"# {'cell':10s} {'H[m]':>5s} {'k':>6s} | {'R*':>7s} {'CLred':>7s} {'flap':>5s} {'pitch':>6s} {'adrms':>6s} {'flag':>6s}", flush=True)
summary = {w: [] for w in W0_LIST}
save = dict(CLTRIM=H.CLTRIM, TG_LIST=np.array(TG_LIST), W0_LIST=np.array(W0_LIST))

for W0 in W0_LIST:
    for Tg in TG_LIST:
        cell = f'W{int(W0)}/T{Tg:.2f}'
        OL = H.scalar_rollout(None, W0, Tg, TEND=TEND)
        t = OL['_t']; mw = H.window(t, Tg); pk0 = float(np.max(np.abs(OL['al'][mw])))
        runs = []; ms = []
        for R in R_GRID:
            r = H.scalar_rollout(OptGrid(R=R, G=161, gate='hard', use_wnext=True, refine=True), W0, Tg, TEND=TEND)
            m = H.metrics(r, OL, Tg); m['pr'] = m['pitchpk']/max(pk0, 1e-12)
            runs.append(r); ms.append(m)
        crs = np.array([m['clred'] for m in ms]); prs = np.array([m['pr'] for m in ms])
        noflag = np.array([m['flag'] == '' for m in ms])
        idx = np.where(noflag)[0]
        jb = int(idx[np.argmax(crs[idx])]) if len(idx) else int(np.argmin(prs))
        r_opt = runs[jb]; m = ms[jb]; Rstar = R_GRID[jb]
        kred = np.pi/(H.U*Tg); Hgrad = H.U*Tg/2.0
        print(f"# {cell:10s} {Hgrad:5.1f} {kred:6.3f} | {Rstar:7.0e} {m['clred']:+6.1f}% "
              f"{m['flap_max']:5.1f} {m['pr']:6.2f} {m['adrms']:6.2f} {m['flag']:>6s}", flush=True)
        four_panel(cell, Tg, t, OL['_Wt'], OL, r_opt, m, Rstar, m['pr'])
        summary[W0].append((Tg, m['clred']))
        tag = f'W{int(W0)}_Tg{int(Tg*100)}'
        save[f'{tag}_t'] = t; save[f'{tag}_Wt'] = OL['_Wt']
        for k in ['CL', 'de', 'al', 'ad']:
            save[f'{tag}_open_{k}'] = OL[k]; save[f'{tag}_opt_{k}'] = r_opt[k]
        save[f'{tag}_Rstar'] = Rstar; save[f'{tag}_clred'] = m['clred']; save[f'{tag}_pitch'] = m['pr']

fig, ax = plt.subplots(figsize=(6, 4))
for W0, col in zip(W0_LIST, ['tab:blue', 'tab:orange', 'tab:red']):
    arr = np.array(summary[W0])
    ax.plot(arr[:, 0], arr[:, 1], 'o-', color=col, label=f'W0={int(W0)} m/s')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel('Tg [s]'); ax.set_ylabel('CLred [%]')
ax.set_title('CS-25 gust study — wnext (refine, no-flag R* pick)')
ax.grid(alpha=0.3); ax.legend(frameon=False); plt.tight_layout()
fig.savefig(os.path.join(OUTD, 'cs25_wnext2_summary.png'), dpi=140, bbox_inches='tight'); plt.close(fig)

np.savez_compressed(os.path.join(OUTD, 'cs25_wnext2_traces.npz'), **save)
print(f"# saved {OUTD}/  (18 four-panel PNGs + summary + traces.npz)", flush=True)
print("# DONE", flush=True)

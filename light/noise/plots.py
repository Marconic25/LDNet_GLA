"""
All figures for the noise study, from results/*.npz (record schema in
harness_noise.py). Each axis figure is skipped gracefully if its npz is
missing, so partial results still plot. Also prints the break-even sigma
(optimal mean CLred crossing the prop-W same-noise curve and the flat +32%
noise-free prop line) — the number NOTES.md must quote.

Run on the cluster after the axis jobs:  python3 -s -u plots.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PROP_CLEAN = 32.0    # honest_home no-flag best, prop-W(t+dt), W30/Tg0.4


def load(axis_file):
    p = os.path.join(RD, axis_file)
    if not os.path.exists(p):
        print(f"# skip {axis_file} (missing)", flush=True)
        return None
    # allow_pickle: npz written by our own axis scripts (trusted, local)
    return list(np.load(p, allow_pickle=True)['records'])


def pts(recs, **match):
    out = []
    for r in recs:
        if r.get('kind') != 'point':
            continue
        if all(r.get(k) == v for k, v in match.items()):
            out.append(r)
    return out


def trajs(recs, prefix):
    return [r for r in recs if r.get('kind') == 'traj'
            and str(r.get('label', '')).startswith(prefix)]


def savefig(fig, name):
    p = os.path.join(RD, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"# wrote {p}", flush=True)


# ---------------------------------------------------------------------------
# A) CLred vs sigma + break-even
# ---------------------------------------------------------------------------
A = load('A_white.npz')
if A:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for R, col in ((3e-4, 'tab:blue'), (1e-4, 'tab:orange')):
        p = sorted(pts(A, arm='optimal', R=R), key=lambda r: r['frac'])
        f = np.array([r['frac'] for r in p])
        ax.plot(f * 100, [r['mean'] for r in p], 'o-', color=col,
                label=f'optimal wnext R={R:g}')
        ax.fill_between(f * 100, [r['lo'] for r in p], [r['hi'] for r in p],
                        color=col, alpha=0.15)
    pp = sorted(pts(A, arm='prop'), key=lambda r: r['frac'])
    if pp:
        fp = np.array([r['frac'] for r in pp])
        ax.plot(fp * 100, [r['mean'] for r in pp], 's--', color='tab:green',
                label='prop-W (same noisy preview)')
        ax.fill_between(fp * 100, [r['lo'] for r in pp], [r['hi'] for r in pp],
                        color='tab:green', alpha=0.15)
    ax.axhline(PROP_CLEAN, color='tab:green', lw=0.8, ls=':',
               label=f'prop-W noise-free (+{PROP_CLEAN:.0f}%)')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('preview noise  $\\sigma / W_0$  [%]')
    ax.set_ylabel('CLred [%] (mean, min-max band over seeds)')
    ax.set_title('A) white preview noise — W30/Tg0.4, DAMULT=3')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    savefig(fig, 'fig_A_clred_vs_sigma.png')

    # break-even: first crossing of the optimal mean below the prop references
    for R in (3e-4, 1e-4):
        p = sorted(pts(A, arm='optimal', R=R), key=lambda r: r['frac'])
        f = np.array([r['frac'] for r in p]); m = np.array([r['mean'] for r in p])
        for name, ref in [('prop noise-free +32.0', np.full_like(f, PROP_CLEAN))] + \
                ([('prop same-noise', np.interp(f, [r['frac'] for r in pp],
                                                [r['mean'] for r in pp]))] if pp else []):
            d = m - ref
            if (d <= 0).any():
                k = int(np.argmax(d <= 0))
                if k == 0:
                    fx = f[0]
                else:
                    fx = f[k-1] + (f[k]-f[k-1]) * d[k-1] / (d[k-1] - d[k])
                print(f"# break-even R={R:g} vs {name}: sigma/W0 = {fx*100:.2f}%",
                      flush=True)
            else:
                print(f"# break-even R={R:g} vs {name}: none up to "
                      f"{f[-1]*100:.0f}%", flush=True)

# ---------------------------------------------------------------------------
# B) CLred vs tau
# ---------------------------------------------------------------------------
B = load('B_bandlim.npz')
if B:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for R, col in ((3e-4, 'tab:blue'), (1e-4, 'tab:orange')):
        p = sorted(pts(B, arm='optimal', R=R), key=lambda r: r['tau_ms'])
        tau = np.array([r['tau_ms'] for r in p])
        ax.plot(tau, [r['mean'] for r in p], 'o-', color=col,
                label=f'optimal wnext R={R:g}')
        ax.fill_between(tau, [r['lo'] for r in p], [r['hi'] for r in p],
                        color=col, alpha=0.15)
        for r in p:
            if r['nflag']:
                ax.annotate(f"{r['nflag']}/{len(r['seeds'])}!", (r['tau_ms'], r['mean']),
                            fontsize=7, textcoords='offset points', xytext=(3, 3))
    # 76/ reference (OptGrid refine=False, same noise model)
    ax.plot([0, 5, 10, 20, 40], [-14.2, -1.0, 19.8, 12.5, -9.3], 'x:',
            color='gray', label='76/ SS wnext refine=False (ref)')
    ax.axhline(PROP_CLEAN, color='tab:green', lw=0.8, ls=':', label='prop-W noise-free')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('LPF time constant  $\\tau$  [ms]')
    ax.set_ylabel('CLred [%]')
    ax.set_title('B) band-limited 5% preview noise — filter lag trade-off')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    savefig(fig, 'fig_B_clred_vs_tau.png')

# ---------------------------------------------------------------------------
# C) structured errors
# ---------------------------------------------------------------------------
Cx = load('C_struct.npz')
if Cx:
    rows = []
    for r in [x for x in Cx if x.get('kind') == 'point' and x['R'] == 3e-4]:
        if r['arm'] == 'bias':
            rows.append((f"bias {r['beta']:+.0%} W0", r))
        elif r['arm'] == 'scale':
            rows.append((f"scale x{r['scale']:g}", r))
        elif r['arm'] == 'latency':
            rows.append(("latency (Wc=W(t))", r))
        elif r['arm'] == 'jitter':
            rows.append(("jitter k~U{0,1,2}", r))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(rows))
    means = [r['mean'] for _, r in rows]
    ax.barh(y, means, color=['tab:red' if m < PROP_CLEAN else 'tab:blue' for m in means])
    for yi, (lab, r) in zip(y, rows):
        ax.plot([r['lo'], r['hi']], [yi, yi], 'k-', lw=1)
        if r['nflag']:
            ax.annotate(f" {r['nflag']}/{len(r['seeds'])}!", (max(r['hi'], r['mean']), yi),
                        fontsize=7, va='center')
    ax.set_yticks(y); ax.set_yticklabels([lab for lab, _ in rows], fontsize=8)
    ax.axvline(76.6, color='k', ls='--', lw=0.8, label='clean +76.6%')
    ax.axvline(PROP_CLEAN, color='tab:green', ls=':', lw=0.8, label='prop-W +32%')
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('CLred [%] (R=3e-4)')
    ax.set_title('C) structured preview errors — W30/Tg0.4')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x')
    savefig(fig, 'fig_C_struct.png')

# ---------------------------------------------------------------------------
# D) cell dependence
# ---------------------------------------------------------------------------
D = load('D_cells.npz')
if D:
    cells = [(10.0, 0.7), (30.0, 0.7)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xw = 0.35
    for ic, (W0c, Tgc) in enumerate(cells):
        for iR, (R, col) in enumerate(((3e-4, 'tab:blue'), (1e-4, 'tab:orange'))):
            p = sorted(pts(D, arm='optimal', W0=W0c, Tg=Tgc, R=R),
                       key=lambda r: r['frac'])
            xs = ic * 4 + np.arange(len(p)) + iR * xw
            ax.bar(xs, [r['mean'] for r in p], width=xw, color=col,
                   label=f'R={R:g}' if ic == 0 else None)
            for x, r in zip(xs, p):
                ax.plot([x, x], [r['lo'], r['hi']], 'k-', lw=1)
                if r['nflag']:
                    ax.annotate(f"{r['nflag']}!", (x, r['hi']), fontsize=7,
                                ha='center', xytext=(0, 2), textcoords='offset points')
    ticks, labels = [], []
    for ic, (W0c, Tgc) in enumerate(cells):
        p = sorted(pts(D, arm='optimal', W0=W0c, Tg=Tgc, R=3e-4),
                   key=lambda r: r['frac'])
        for j, r in enumerate(p):
            ticks.append(ic * 4 + j + xw / 2)
            labels.append(f"W{W0c:g}/Tg{Tgc:g}\n{r['frac']*100:.0f}%")
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel('CLred [%]')
    ax.set_title('D) preview-noise sensitivity by gust cell')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    savefig(fig, 'fig_D_cells.png')

# ---------------------------------------------------------------------------
# E) mitigation bar chart (+ prop/mpc references)
# ---------------------------------------------------------------------------
E = load('E_mitigation.npz')
if E:
    for frac in (0.02, 0.05):
        rows = [(f"{r['arm']} {r.get('detail','')}".strip(), r)
                for r in sorted([x for x in E if x.get('kind') == 'point'
                                 and x['frac'] == frac],
                                key=lambda r: r['mean'])]
        if A:
            for r in pts(A, arm='prop', frac=frac):
                rows.insert(0, ('prop-W (ref)', r))
        fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1.5))
        y = np.arange(len(rows))
        ax.barh(y, [r['mean'] for _, r in rows],
                color=['tab:gray' if lab.startswith(('none', 'prop', 'mpc'))
                       else 'tab:blue' for lab, _ in rows])
        for yi, (lab, r) in zip(y, rows):
            ax.plot([r['lo'], r['hi']], [yi, yi], 'k-', lw=1)
            if r['nflag']:
                ax.annotate(f" {r['nflag']}/{len(r['seeds'])}!", (max(r['hi'], r['mean']), yi),
                            fontsize=7, va='center', color='tab:red')
        ax.set_yticks(y); ax.set_yticklabels([lab for lab, _ in rows], fontsize=8)
        ax.axvline(76.6, color='k', ls='--', lw=0.8, label='clean +76.6%')
        ax.axvline(PROP_CLEAN, color='tab:green', ls=':', lw=0.8, label='prop-W clean +32%')
        ax.axvline(0, color='k', lw=0.6)
        ax.set_xlabel('CLred [%]')
        ax.set_title(f'E) mitigations at sigma={frac*100:.0f}% white preview noise')
        ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3, axis='x')
        savefig(fig, f'fig_E_mitigation_f{frac:g}.png')

# ---------------------------------------------------------------------------
# Money plot: worst unmitigated sigma=2% seed vs best mitigated (+ clean ref)
# ---------------------------------------------------------------------------
if E and A:
    none02 = trajs(E, 'E_none_f0.02')
    best02 = trajs(E, 'E_best_f0.02')
    clean = trajs(A, 'A_opt_clean_R1e-4')
    openrec = next((r for r in A if r.get('kind') == 'open'), None)
    if none02 and best02:
        worst = min(none02, key=lambda r: r['clred'])
        best = max(best02, key=lambda r: r['clred'])
        curves = [(f"noisy 2%, unmitigated (seed {worst['seed']}, "
                   f"{worst['clred']:+.0f}%{' ' + worst['flag'] if worst['flag'] else ''})",
                   worst, 'tab:red'),
                  (f"best mitigated: {best.get('arm','?')} {best.get('detail','')} "
                   f"(seed {best['seed']}, {best['clred']:+.0f}%)", best, 'tab:blue')]
        if clean:
            curves.insert(0, (f"clean preview R=1e-4 ({clean[0]['clred']:+.0f}%)",
                              clean[0], 'tab:gray'))
        fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        tmax = 1.5
        for lab, r, col in curves:
            mw = r['t'] <= tmax
            axs[0].plot(r['t'][mw], r['CL'][mw], color=col, lw=1.2, label=lab)
            axs[1].plot(r['t'][mw], r['de'][mw], color=col, lw=1.2)
            axs[2].plot(r['t'][mw], r['Wc'][mw], color=col, lw=0.8, alpha=0.8)
            axs[3].plot(r['t'][mw], np.degrees(r['ad'][mw]), color=col, lw=1.2)
        if openrec is not None:
            mw = openrec['t'] <= tmax
            axs[0].plot(openrec['t'][mw], openrec['CL'][mw], 'k:', lw=1,
                        label=f"open loop (cex0={openrec['cex0']:.3f})")
        r0 = curves[-1][1]; mw = r0['t'] <= tmax
        axs[2].plot(r0['t'][mw], r0['W'][mw], 'k-', lw=1.5, label='true W(t)')
        axs[0].set_ylabel('$C_L$'); axs[0].legend(fontsize=7)
        axs[1].set_ylabel('$\\delta$ [deg]')
        axs[2].set_ylabel('W [m/s]'); axs[2].legend(fontsize=7)
        axs[3].set_ylabel('$\\dot\\alpha$ [deg/s]'); axs[3].set_xlabel('t [s]')
        for a in axs:
            a.grid(alpha=0.3)
        axs[0].set_title('Money plot — sigma=2% white preview noise, W30/Tg0.4')
        savefig(fig, 'fig_money_f0.02.png')

print("# PLOTS DONE", flush=True)

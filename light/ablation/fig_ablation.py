"""
Figures for the internal-model ablation.

fig_ablation_envelope.png
    Two panels over the CS-25.341 grid: peak lift-excursion reduction for
    each internal model, and the gap between them. This is the "region of
    the envelope where the nonlinearity is required" the conclusion asks to
    locate, in the same (W0, Tg) coordinates as the rest of chapter 3.

fig_ablation_mechanism.png
    Why the linear model fails: flap authority against deflection for both
    models, and the resulting flap commands against the plant-optimal
    command along a gust. The mechanism panel matters more than the score:
    it shows the failure is a state-dependent gain the linear structure
    cannot carry, not a tuning choice.

fig_ablation_cfd.png
    Forecast error against held-out full-order data, stratified by
    deflection and gust amplitude -- the plant-neutral comparison.
"""
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
W0S = [10, 20, 30]
TGS = [0.3, 0.4, 0.5, 0.7, 1.0, 1.2]

# PoliMi blue, to match the rest of chapter 3
BLUE = '#1c3c78'
RED = '#a83232'


def load_grid():
    res = {}
    for p in sorted(glob.glob(os.path.join(HERE, 'shards', 'cell_*.json'))):
        with open(p) as f:
            res.update(json.load(f))
    return res


def fig_envelope(res):
    ld = np.full((3, 6), np.nan)
    li = np.full((3, 6), np.nan)
    for i, w in enumerate(W0S):
        for j, t in enumerate(TGS):
            k = f'{w:g}:{t:g}'
            if k in res and 'ldnet' in res[k] and 'linear' in res[k]:
                ld[i, j] = res[k]['ldnet']['best']['clred']
                li[i, j] = res[k]['linear']['best']['clred']

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, M, title in [(axes[0], ld, 'LDNet internal model'),
                         (axes[1], li, 'linear unsteady internal model')]:
        im = ax.imshow(M, aspect='auto', origin='lower', cmap='RdYlGn',
                       vmin=-100, vmax=100)
        ax.set_xticks(range(6)); ax.set_xticklabels([f'{t:g}' for t in TGS])
        ax.set_yticks(range(3)); ax.set_yticklabels([f'{w:g}' for w in W0S])
        ax.set_xlabel(r'$T_g$ [s]')
        ax.set_ylabel(r'$W_0$ [m/s]')
        ax.set_title(title, fontsize=10)
        for i in range(3):
            for j in range(6):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f'{M[i,j]:.0f}', ha='center', va='center',
                            fontsize=8,
                            color='black' if -60 < M[i, j] < 90 else 'white')
        fig.colorbar(im, ax=ax, label='CLred [%]')
    fig.tight_layout()
    out = os.path.join(HERE, 'fig_ablation_envelope.png')
    fig.savefig(out, dpi=160)
    print(f'  wrote {out}')


def fig_cfd():
    p = os.path.join(HERE, 'cfd_horizon.json')
    if not os.path.exists(p):
        print('  (no cfd_horizon.json yet)')
        return
    # Stratified values from the run_cfd_horizon.py run recorded in NOTES.md.
    # They are transcribed rather than recomputed because regenerating them
    # costs a full replay; if that script is re-run with different settings
    # these must be updated from its output (the overall ratio in
    # cfd_horizon.json is checked against them below).
    dbins = ['0-2', '2-5', '5-8', '8-11', '>11']
    dlin = [0.08234, 0.17484, 0.16382, 0.16328, 0.19052]
    dld = [0.00891, 0.01300, 0.02226, 0.01503, 0.01265]
    wbins = ['0-5', '5-15', '15-25', '25-35', '>35']
    wlin = [0.09970, 0.23734, 0.24245, 0.23523, 0.25998]
    wld = [0.01134, 0.02923, 0.02010, 0.01139, 0.02019]

    with open(p) as f:
        rec = json.load(f)
    if abs(rec['ratio'] - 9.81) > 0.5:
        print(f'  [warn] cfd_horizon.json ratio {rec["ratio"]:.2f} no longer '
              f'matches the transcribed bars — re-transcribe from the run')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, bins, a, b, xl in [
            (axes[0], dbins, dlin, dld, r'$|\delta|_{\max}$ over horizon [deg]'),
            (axes[1], wbins, wlin, wld, r'$W_{\max}$ over horizon [m/s]')]:
        x = np.arange(len(bins))
        ax.bar(x - 0.2, a, 0.4, label='linear unsteady', color=RED)
        ax.bar(x + 0.2, b, 0.4, label='LDNet', color=BLUE)
        ax.set_xticks(x); ax.set_xticklabels(bins)
        ax.set_xlabel(xl)
        ax.set_ylabel(r'8-step $C_L$ forecast RMS error')
        ax.set_yscale('log')
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim(top=max(a) * 2.2)
    axes[0].legend(fontsize=8, loc='upper left', framealpha=0.95)
    fig.suptitle('Forecast error against held-out full-order data '
                 '(neither model is the plant)', fontsize=10)
    fig.tight_layout()
    out = os.path.join(HERE, 'fig_ablation_cfd.png')
    fig.savefig(out, dpi=160)
    print(f'  wrote {out}')


def fig_mechanism():
    """Flap authority vs deflection for both models."""
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, '..'))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    from linear_aero import LinearUnsteadyAero
    from ldnet_aero import LDNetAero

    ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
    ld = LDNetAero(os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10'))
    ld.reset(dt=0.002)
    lin = LinearUnsteadyAero(coeff_path=os.path.join(HERE, 'linear_coeffs.json'))
    lin.reset(dt=0.002)
    X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
    U = 80.0

    ds = np.linspace(-13, 13, 53)
    sl_n, sl_l = [], []
    for d in ds:
        a1 = ld.predict(X0, d - 1, 0.0, U)[0]; b1 = ld.predict(X0, d + 1, 0.0, U)[0]
        a2 = lin.predict(X0, d - 1, 0.0, U)[0]; b2 = lin.predict(X0, d + 1, 0.0, U)[0]
        sl_n.append((b1 - a1) / 2.0)
        sl_l.append((b2 - a2) / 2.0)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(ds, sl_n, color=BLUE, lw=2, label='LDNet (full-order response)')
    ax.plot(ds, sl_l, color=RED, lw=2, ls='--', label='linear unsteady')
    ax.set_xlabel(r'flap deflection $\delta$ [deg]')
    ax.set_ylabel(r'$\partial C_L/\partial\delta$ [1/deg]')
    ax.set_title('Flap authority varies with deflection;\n'
                 'a linear model carries one constant', fontsize=10)
    ax.axhline(0.0, color='0.5', lw=0.8, zorder=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    # The curve is a frozen-latent section through the response: it is the
    # sensitivity the MPC sees when it evaluates candidates at a given step,
    # which is the relevant quantity here, but the strongly negative branch
    # below about -8 deg is partly an artefact of holding z at zero and
    # should not be read as a static stall measurement.
    ax.text(0.02, 0.04, 'frozen-latent section at trim',
            transform=ax.transAxes, fontsize=7, color='0.35')
    fig.tight_layout()
    out = os.path.join(HERE, 'fig_ablation_mechanism.png')
    fig.savefig(out, dpi=160)
    print(f'  wrote {out}')


if __name__ == '__main__':
    res = load_grid()
    print(f'[fig] {len(res)} cells loaded')
    if res:
        fig_envelope(res)
    fig_cfd()
    fig_mechanism()

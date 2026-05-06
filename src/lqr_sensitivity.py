#!/usr/bin/env python3
"""
LQR weight sensitivity analysis for GLA.

For each of 4 weights {Q_CL, Q_CM, Q_w, R}, varies the value over a
logarithmic grid of 6 values (×0.1 to ×10 of baseline) keeping all others
fixed. For each variation runs a closed-loop simulation and plots C_L(t) and
C_M(t) as families of curves — one curve per weight value.

Output (all in results/lqr_sensitivity/):
  fig_sens_Q_CL.png
  fig_sens_Q_CM.png
  fig_sens_Q_w.png
  fig_sens_R.png
  sensitivity_peaks.csv   — peak |C_L|, peak |C_M| for every case
"""

import sys
import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── simulation constants (must match LQR_weight_optimizer) ───────────────────
U_INF    = 75.0
T_END    = 3.0
DT       = 0.01
LAMBDA_W = 0.98

GUST_W0  = 60.0
GUST_TG  = 1.0

# ── baseline weights ──────────────────────────────────────────────────────────
Q_H_BASE  = 1.0  / 0.00428**2
Q_A_BASE  = 10.0 / 0.00823**2
Q_CL_BASE = 1.0  / 0.0484**2
Q_CM_BASE = 1.0  / 0.04**2
R_BASE    = 1.0  / 2.0**2
Q_W_BASE  = 1.0  / 10.0**2

# ── sensitivity grid: 6 multiplicative factors on log scale ──────────────────
# covers ×0.1 … ×10 of baseline
FACTORS = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])

# ── parameters to sweep ───────────────────────────────────────────────────────
# Each entry: (param_name, baseline_value, label_for_plot)
SWEEP_PARAMS = [
    ('Q_CL', Q_CL_BASE, r'$Q_{C_L}$'),
    ('Q_CM', Q_CM_BASE, r'$Q_{C_M}$'),
    ('Q_w',  Q_W_BASE,  r'$Q_w$'),
    ('R',    R_BASE,    r'$R$'),
]

# ── output ────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / 'results' / 'lqr_sensitivity'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── plot style ────────────────────────────────────────────────────────────────
_MARKERS  = ['+', 'o', '*', 's', 'x', 'D']
_CMAP     = plt.cm.viridis
_MARKEVERY = 25


def make_gust(W0, T_g):
    def g(t):
        if 0.0 <= t <= T_g:
            return (W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / T_g))
        return 0.0
    return g


def run_case(aero_model, z_trim, CL_trim, CM_trim, xi_trim,
             A_s, B_s, JAC, A_ekf, C_ekf,
             Q_CL, Q_CM, Q_w, R):
    """Run one LQR+EKF simulation with given weights. Returns result dict."""
    from control.lqr           import LQRController
    from control.ekf_augmented import NonlinearEKF
    from control.run_controller import run_simulation

    Q_lqr = np.diag([Q_H_BASE, 0., Q_A_BASE, 0., 0.])
    Q_y   = np.diag([Q_CL, Q_CM])
    lqr   = LQRController(
        aero_model, U_INF, DT,
        x_trim=np.zeros(4), z_trim=z_trim,
        Q_lqr=Q_lqr, R_lqr=np.array([[R]]),
        CL_trim=CL_trim, CM_trim=CM_trim,
        Q_y=Q_y, Q_w=Q_w, lambda_w=LAMBDA_W, delta_max=20.0,
        precomputed_jacobians=JAC)

    ekf = NonlinearEKF(aero_model, U_INF, DT, xi_trim, lambda_w=LAMBDA_W,
                       A_trim=A_ekf, C_trim=C_ekf)
    ekf.reset(xi_trim)

    return run_simulation(
        U_INF, T_END, DT, aero_model, lqr, A_s, B_s,
        gust_profile=make_gust(GUST_W0, GUST_TG),
        observer='ekf_ad', kalman_filter=ekf)


def sensitivity_fig(t, results_list, factors, param_label, fname):
    """
    Two-panel figure: C_L(t) and C_M(t) families.
    results_list: list of result dicts, one per factor value.
    """
    colors = [_CMAP(v) for v in np.linspace(0.1, 0.9, len(factors))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    fig.suptitle(f'Sensitivity to {param_label}  '
                 f'(baseline × factor,  $W_0={GUST_W0}$ m/s, '
                 f'$T_g={GUST_TG}$ s)', fontsize=12)

    for i, (res, fac) in enumerate(zip(results_list, factors)):
        lbl = f'×{fac:g}'
        mk  = _MARKERS[i % len(_MARKERS)]
        kw  = dict(color=colors[i], lw=1.4, marker=mk,
                   markevery=_MARKEVERY, label=lbl)
        axes[0].plot(t, res['C_L'], **kw)
        axes[1].plot(t, res['C_M'], **kw)

    for ax, ylabel, title in [
            (axes[0], '$C_L$',  'Lift coefficient'),
            (axes[1], '$C_M$',  'Pitching moment coefficient')]:
        ax.axhline(0, color='k', ls=':', lw=0.6)
        ax.set_xlabel('t [s]')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, title=param_label)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  [OK] {fname.name}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── load model & trim ────────────────────────────────────────────────────
    from aerodynamics.model    import LDNetModel
    from structural.smd        import get_space_state_matrices
    from control.lqr           import compute_jacobians
    from control.ekf_augmented import _compute_ekf_jacobians
    from control.run_controller import run_simulation

    print("Loading aerodynamic model...")
    _MODELS_DIR = Path(__file__).parent.parent / 'models'
    aero_model  = LDNetModel(str(_MODELS_DIR))

    print("Computing trim state...")
    z_trim = np.zeros(aero_model.num_latent_states)
    for _ in range(200):
        z_trim, CL_trim, CM_trim = aero_model.step(
            z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)
    CL_trim = float(CL_trim); CM_trim = float(CM_trim)
    xi_trim = np.concatenate([np.zeros(4), z_trim, [0.0]])
    print(f"  Trim: CL={CL_trim:.5f}  CM={CM_trim:.5f}")

    A_s, B_s, _, _ = get_space_state_matrices()

    print("Pre-computing trim Jacobians...")
    JAC            = compute_jacobians(aero_model, np.zeros(4), z_trim,
                                       U_INF, DT, CL_trim, CM_trim)
    A_ekf, C_ekf   = _compute_ekf_jacobians(aero_model, xi_trim, U_INF, DT, LAMBDA_W)
    print("  Done.")

    # ── open-loop reference ───────────────────────────────────────────────────
    print("\nRunning open-loop reference...")
    res_ol = run_simulation(
        U_INF, T_END, DT, aero_model, None, A_s, B_s,
        gust_profile=make_gust(GUST_W0, GUST_TG),
        observer='true_state')
    t = res_ol['t']

    # ── sensitivity sweep ─────────────────────────────────────────────────────
    rows = []

    for param_name, baseline, param_label in SWEEP_PARAMS:
        print(f"\nSweep: {param_name}  baseline={baseline:.4g}")
        results_list = []

        for fac in FACTORS:
            val = baseline * fac

            # build weight dict for this case
            w = dict(Q_CL=Q_CL_BASE, Q_CM=Q_CM_BASE,
                     Q_w=Q_W_BASE,   R=R_BASE)
            w[param_name] = val

            print(f"  {param_name}={val:.4g} (×{fac:g})...", end=' ', flush=True)
            try:
                res = run_case(aero_model, z_trim, CL_trim, CM_trim, xi_trim,
                               A_s, B_s, JAC, A_ekf, C_ekf, **w)
                pk_CL = float(np.abs(res['C_L']).max())
                pk_CM = float(np.abs(res['C_M']).max())
                print(f"pk_CL={pk_CL:.4f}  pk_CM={pk_CM:.4f}")
                results_list.append(res)
            except Exception as e:
                print(f"FAILED: {e}")
                results_list.append(None)
                pk_CL = pk_CM = float('nan')

            rows.append(dict(param=param_name, factor=fac, value=val,
                             peak_CL=pk_CL, peak_CM=pk_CM))

        # replace None with open-loop so plot doesn't crash
        clean = [r if r is not None else res_ol for r in results_list]
        sensitivity_fig(t, clean, FACTORS, param_label,
                        OUT_DIR / f'fig_sens_{param_name}.png')

    # ── save peak table ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'sensitivity_peaks.csv', index=False)
    print(f"\n[OK] sensitivity_peaks.csv  ({len(df)} rows)")

    # ── summary figure: peak C_L and peak C_M vs factor for all params ────────
    print("Generating summary figure...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Peak response vs. weight scaling factor', fontsize=12)
    colors_p = plt.cm.tab10(np.linspace(0, 0.6, len(SWEEP_PARAMS)))

    for i, (param_name, _, param_label) in enumerate(SWEEP_PARAMS):
        sub = df[df['param'] == param_name]
        kw  = dict(color=colors_p[i], marker='o', lw=1.5, label=param_label)
        axes[0].plot(sub['factor'], sub['peak_CL'], **kw)
        axes[1].plot(sub['factor'], sub['peak_CM'], **kw)

    # open-loop reference lines
    pk_CL_ol = float(np.abs(res_ol['C_L']).max())
    pk_CM_ol = float(np.abs(res_ol['C_M']).max())
    for ax, pk_ol in [(axes[0], pk_CL_ol), (axes[1], pk_CM_ol)]:
        ax.axhline(pk_ol, color='steelblue', ls='--', lw=1.2, label='Open loop')
        ax.set_xscale('log')
        ax.set_xlabel('Scale factor (×baseline)')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('peak $|C_L|$')
    axes[1].set_ylabel('peak $|C_M|$')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_sens_summary.png', dpi=150)
    plt.close(fig)
    print("[OK] fig_sens_summary.png")

    print(f"\nDone. Results in: {OUT_DIR}")

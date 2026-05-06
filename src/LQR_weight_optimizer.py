#!/usr/bin/env python3
"""
LQR weight optimizer for Gust Load Alleviation (GLA) via CMA-ES.

Sections:
  0 — Imports, constants, math printout + confirmation guard
  1 — Simulation helpers (trim, gust factory, zero-controller)
  2 — Objective function + parallel worker
  3 — CMA-ES optimization loop + save weights  (parallelized, 4 workers)
  4 — Sweep helpers (run_sweep_case, extract_metrics)
  5 — Amplitude sweep  W0 ∈ {20, 40, 60} m/s  (T_g = 1.0 s fixed)
  6 — Duration sweep   T_g ∈ {0.5, 0.75, 1.0, 1.5, 2.0} s  (W0 = 60 m/s fixed)
  7 — Set A: summary figures (peak metrics vs. sweep parameter)
  8 — Set B: per-case time-series figures (5 panels each)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — Imports, constants, math printout
# ═══════════════════════════════════════════════════════════════════════════════

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
import concurrent.futures
import multiprocessing

# ── flag: set False to skip interactive pause (required for batch/HPC runs) ──
CONFIRM_MATH = False

# ── simulation constants ─────────────────────────────────────────────────────
U_INF    = 75.0
T_END    = 3.0
DT       = 0.01
LAMBDA_W = 0.98

# ── fixed Q_w (W state in DARE, not optimized) ───────────────────────────────
Q_W_FIXED = 1.0 / 10.0**2   # 0.01

# ── baseline weights (starting point for CMA-ES) ─────────────────────────────
Q_H_BASE   = 1.0  / 0.00428**2
Q_A_BASE   = 10.0 / 0.00823**2
Q_CL_BASE  = 1.0  / 0.0484**2
Q_CM_BASE  = 1.0  / 0.04**2
R_BASE     = 1.0  / 2.0**2

# ── objective coefficients ────────────────────────────────────────────────────
ALPHA_OBJ   = 0.6
BETA_OBJ    = 0.4

# ── nominal gust for optimization ────────────────────────────────────────────
GUST_NOM_W0  = 60.0
GUST_NOM_TG  = 1.0
EVAL_WINDOW  = GUST_NOM_TG + 0.5   # 1.5 s

# ── CMA-ES settings ───────────────────────────────────────────────────────────
CMA_SIGMA0   = 0.3
CMA_MAXITER  = 300
CMA_POPSIZE  = 8
CMA_N_WORKERS = 4   # matches ncpus=4 in PBS; set to 1 to disable parallelism

# ── plot style ────────────────────────────────────────────────────────────────
C_BL   = dict(color='steelblue', lw=1.5, alpha=0.85, label='Open loop')
C_BASE = dict(color='darkcyan',  lw=1.5, alpha=0.85, label='LQR baseline')
C_OPT  = dict(color='crimson',   lw=1.5, alpha=0.85, label='LQR optimized')
DDELTA_SAT = 200.0

# ── output directories ────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / 'results'
TS_DIR      = RESULTS_DIR / 'sweep_timeseries'

# ── math printout ─────────────────────────────────────────────────────────────
_MATH_BLOCK = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              OPTIMIZATION PROBLEM FORMULATION                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Decision variables (5 scalars, log-space):                                ║
║    θ = [log Q_H,  log Q_A,  log Q_CL,  log Q_CM,  log R]                  ║
║    Q_w = 0.01  fixed (W-state weight in DARE, not optimized)               ║
║                                                                            ║
║  Objective (nominal gust W₀=60 m/s, T_g=1.0 s):                           ║
║    Evaluation window: [0, T_g + 0.5 s] = [0, 1.5 s]                       ║
║                                                                            ║
║    J(θ) = α · peak|C_L| + β · peak|C_L_2nd|                               ║
║      α = 0.6,  β = 0.4                                                    ║
║                                                                            ║
║  Soft constraints (additive penalties if violated):                        ║
║    peak|δ(t)|  > 20°         →  J += 5.0                                  ║
║    peak|δ̇(t)| > 200°/s      →  J += 2.0                                  ║
║    divergence detected        →  J  = 10.0                                 ║
║                                                                            ║
║  Algorithm: CMA-ES,  σ₀=0.3,  maxiter=300,  popsize=8,  workers=4        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Simulation helpers  (importable by worker processes)
# ═══════════════════════════════════════════════════════════════════════════════

class _ZeroController:
    def __init__(self, z_trim):
        self.x_trim = np.zeros(4)
        self.z_trim = z_trim.copy()
    def solve(self, x_hat, z_hat, W_hat=0.0):
        return 0.0


def compute_trim(aero_model):
    z_trim = np.zeros(aero_model.num_latent_states)
    for _ in range(200):
        z_trim, CL_trim, CM_trim = aero_model.step(
            z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)
    return z_trim, float(CL_trim), float(CM_trim), \
           np.concatenate([np.zeros(4), z_trim, [0.0]])


def make_gust(W0, T_g):
    def g(t):
        if 0.0 <= t <= T_g:
            return (W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / T_g))
        return 0.0
    return g


def build_lqr(theta, aero_model, z_trim, CL_trim, CM_trim, jac=None):
    from control.lqr import LQRController
    Q_H, Q_A, Q_CL, Q_CM, R = np.exp(theta)
    return LQRController(
        aero_model, U_INF, DT,
        x_trim=np.zeros(4), z_trim=z_trim,
        Q_lqr=np.diag([Q_H, 0., Q_A, 0., 0.]),
        R_lqr=np.array([[R]]),
        CL_trim=CL_trim, CM_trim=CM_trim,
        Q_y=np.diag([Q_CL, Q_CM]),
        Q_w=Q_W_FIXED, lambda_w=LAMBDA_W, delta_max=20.0,
        precomputed_jacobians=jac)


def build_ekf(aero_model, xi_trim, A_trim=None, C_trim=None):
    from control.ekf_augmented import NonlinearEKF
    return NonlinearEKF(aero_model, U_INF, DT, xi_trim, lambda_w=LAMBDA_W,
                        A_trim=A_trim, C_trim=C_trim)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Objective function + parallel worker
# ═══════════════════════════════════════════════════════════════════════════════

def _second_peak(CL_abs, t_arr, pk1_t):
    mask2 = t_arr > pk1_t + 0.3
    return float(CL_abs[mask2].max()) if mask2.any() else 0.0


# Module-level variable: holds the aero model once loaded per worker process
_worker_aero = None

def _worker_init(models_dir):
    """Called once per worker process at pool startup — loads TF + model."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from aerodynamics.model import LDNetModel
    global _worker_aero
    _worker_aero = LDNetModel(str(models_dir))


def _eval_worker(args):
    """Worker called once per CMA-ES candidate. Model already loaded."""
    import numpy as np
    from control.mpc import run_mpc_simulation

    (theta, z_trim, CL_trim, CM_trim, xi_trim,
     JAC, A_ekf, C_ekf, A_s, B_s) = args

    try:
        aero = _worker_aero
        lqr  = build_lqr(theta, aero, z_trim, CL_trim, CM_trim, jac=JAC)
        ekf  = build_ekf(aero, xi_trim, A_trim=A_ekf, C_trim=C_ekf)
        ekf.reset(xi_trim)
        res  = run_mpc_simulation(
            U_INF, T_END, DT, aero, lqr, A_s, B_s,
            gust_profile=make_gust(GUST_NOM_W0, GUST_NOM_TG),
            observer='ekf_ad', kalman_filter=ekf)
    except Exception:
        return 10.0, float('nan'), float('nan'), float('nan')

    t      = res['t']
    CL_abs = np.abs(res['C_L'])
    delta  = res['delta']

    if not np.isfinite(CL_abs).all() or CL_abs.max() > 5.0:
        return 10.0, float('nan'), float('nan'), float('nan')

    win  = t <= EVAL_WINDOW
    CLw  = CL_abs[win];  tw = t[win]
    pk1i = int(np.argmax(CLw))
    pk1  = float(CLw[pk1i]);  pk1t = float(tw[pk1i])
    pk2  = _second_peak(CLw, tw, pk1t)

    J = ALPHA_OBJ * pk1 + BETA_OBJ * pk2
    pd_ = float(np.abs(delta[win]).max())
    if pd_ > 20.0:           J += 0.5
    if np.abs(np.diff(delta) / DT).max() > 200.0:  J += 0.2
    return J, pk1, pk2, pd_


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Sweep helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _peak_second(arr_abs, t_arr, T_g):
    win = t_arr <= (T_g + 0.5)
    a = arr_abs[win];  t = t_arr[win]
    pk1_t = float(t[np.argmax(a)])
    m2 = t > pk1_t + 0.3
    return float(a[m2].max()) if m2.any() else 0.0


def extract_metrics(res, W0, T_g, label):
    t      = res['t'];  win = t <= (T_g + 0.5)
    delta  = res['delta'];  dr = np.abs(np.diff(delta) / DT)
    return dict(
        W0=W0, T_g=T_g, controller=label,
        peak_CL      = float(np.abs(res['C_L'][win]).max()),
        peak_CL2     = _peak_second(np.abs(res['C_L']), t, T_g),
        peak_CM      = float(np.abs(res['C_M'][win]).max()),
        peak_h_ddot  = float(np.abs(res['h_ddot'][win]).max()),
        amp_h        = float((res['h'][win].max() - res['h'][win].min()) / 2.0),
        peak_alpha_deg = float(np.rad2deg(np.abs(res['a'][win]).max())),
        RMS_delta    = float(np.sqrt(np.mean(delta[win]**2))),
        peak_ddelta  = float(dr.max()),
        peak_delta   = float(np.abs(delta[win]).max()),
    )


def run_sweep_case(label, theta, aero_model, z_trim, CL_trim, CM_trim,
                   xi_trim, A_s, B_s, W0, T_g, JAC, A_ekf, C_ekf):
    from control.mpc import run_mpc_simulation
    gust_fn = make_gust(W0, T_g)
    if label == 'open_loop':
        ctrl, observer, ekf = _ZeroController(z_trim), 'true_state', None
    else:
        ctrl     = build_lqr(theta, aero_model, z_trim, CL_trim, CM_trim, jac=JAC)
        observer = 'ekf_ad'
        ekf      = build_ekf(aero_model, xi_trim, A_trim=A_ekf, C_trim=C_ekf)
        ekf.reset(xi_trim)
    return run_mpc_simulation(
        U_INF, T_END, DT, aero_model, ctrl, A_s, B_s,
        gust_profile=gust_fn, observer=observer, kalman_filter=ekf)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Set A figures
# ═══════════════════════════════════════════════════════════════════════════════

def _summary_fig(df, x_col, x_label, fname):
    metrics = [('peak_CL',    'peak $|C_L|$'),
               ('peak_CM',    'peak $|C_M|$'),
               ('peak_h_ddot','peak $|\\ddot{h}|$ [m/s²]'),
               ('RMS_delta',  'RMS $\\delta$ [°]')]
    x_vals = sorted(df[x_col].unique())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Sweep summary — {x_label}', fontsize=13)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for lbl, sty in [('open_loop', C_BL), ('lqr_base', C_BASE), ('lqr_opt', C_OPT)]:
            sub = df[df['controller'] == lbl]
            y = [float(sub[sub[x_col] == xv][metric].values[0]) for xv in x_vals]
            ax.plot(x_vals, y, marker='o', **sty)
        ax.set_xlabel(x_label); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)
    print(f"[OK] {fname.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Set B figures
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(ax, ylabel, title=None, legend=True):
    ax.set_xlabel('t [s]'); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.25)
    if title:  ax.set_title(title, fontsize=10)
    if legend: ax.legend(fontsize=8)


def save_ts_figures(t, r_bl, r_base, r_opt, out_dir, title_suffix=''):
    out_dir.mkdir(parents=True, exist_ok=True)
    d_base = -r_base['delta'];  d_opt = -r_opt['delta']

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    fig.suptitle(f'Structural state{title_suffix}', fontsize=13)
    for r, s in [(r_bl, C_BL), (r_base, C_BASE), (r_opt, C_OPT)]:
        axes[0,0].plot(t, r['h'],              **s)
        axes[0,1].plot(t, r['hd'],             **s)
        axes[1,0].plot(t, np.rad2deg(r['a']),  **s)
        axes[1,1].plot(t, np.rad2deg(r['ad']), **s)
    _fmt(axes[0,0], 'h [m]',    'Heave displacement')
    _fmt(axes[0,1], 'ḣ [m/s]',  'Heave velocity')
    _fmt(axes[1,0], 'α [°]',    'Pitch angle')
    _fmt(axes[1,1], 'α̇ [°/s]', 'Pitch rate')
    fig.tight_layout(); fig.savefig(out_dir / 'fig1_state.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    fig.suptitle(f'Structural accelerations{title_suffix}', fontsize=13)
    for r, s in [(r_bl, C_BL), (r_base, C_BASE), (r_opt, C_OPT)]:
        axes[0].plot(t, r['h_ddot'],             **s)
        axes[1].plot(t, np.rad2deg(r['a_ddot']), **s)
    _fmt(axes[0], 'ḧ [m/s²]', 'Heave acceleration')
    _fmt(axes[1], 'α̈ [°/s²]', 'Pitch acceleration')
    fig.tight_layout(); fig.savefig(out_dir / 'fig2_accels.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    fig.suptitle(f'Control input{title_suffix}', fontsize=13)
    axes[0].plot(t, d_base, **C_BASE); axes[0].plot(t, d_opt, **C_OPT)
    axes[0].axhline( 20, color='k', ls=':', lw=0.8, label='±20° sat.')
    axes[0].axhline(-20, color='k', ls=':', lw=0.8)
    _fmt(axes[0], 'δ [°]  (+ = flap down)', 'Flap deflection')
    axes[1].plot(t, np.gradient(d_base, t), **C_BASE)
    axes[1].plot(t, np.gradient(d_opt,  t), **C_OPT)
    axes[1].axhline( DDELTA_SAT, color='k', ls=':', lw=0.8, label=f'±{DDELTA_SAT:.0f}°/s')
    axes[1].axhline(-DDELTA_SAT, color='k', ls=':', lw=0.8)
    _fmt(axes[1], 'δ̇ [°/s]', 'Flap rate')
    fig.tight_layout(); fig.savefig(out_dir / 'fig3_control.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    fig.suptitle(f'Aerodynamic coefficients{title_suffix}', fontsize=13)
    for r, s in [(r_bl, C_BL), (r_base, C_BASE), (r_opt, C_OPT)]:
        axes[0].plot(t, r['C_L'], **s); axes[1].plot(t, r['C_M'], **s)
    axes[0].axhline(0, color='k', ls=':', lw=0.6)
    axes[1].axhline(0, color='k', ls=':', lw=0.6)
    _fmt(axes[0], '$C_L$', 'Lift coefficient')
    _fmt(axes[1], '$C_M$', 'Pitching moment coefficient')
    fig.tight_layout(); fig.savefig(out_dir / 'fig4_aero.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, r_bl['W_gust'], label='W true', color='k', lw=2.0, ls='--')
    ax.plot(t, r_base['W_hat'], **C_BASE); ax.plot(t, r_opt['W_hat'], **C_OPT)
    _fmt(ax, 'W [m/s]', 'Gust estimation — NonlinEKF')
    fig.suptitle(f'Gust estimation{title_suffix}', fontsize=13)
    fig.tight_layout(); fig.savefig(out_dir / 'fig5_W_estimation.png', dpi=150); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — guarded so spawn workers don't re-execute this block
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print(_MATH_BLOCK)
    if CONFIRM_MATH:
        input("Press Enter to proceed...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load model & trim ────────────────────────────────────────────────────
    print("=" * 70)
    print("Loading aerodynamic model...")
    from aerodynamics.model    import LDNetModel
    from structural.smd        import get_space_state_matrices
    from control.lqr           import compute_jacobians
    from control.ekf_augmented import _compute_ekf_jacobians

    _MODELS_DIR = Path(__file__).parent.parent / 'models'
    aero_model  = LDNetModel(str(_MODELS_DIR))

    print("Computing trim state...")
    z_trim, CL_trim, CM_trim, xi_trim = compute_trim(aero_model)
    print(f"  Trim: CL={CL_trim:.5f}  CM={CM_trim:.5f}")

    A_s, B_s, _, _ = get_space_state_matrices()

    print("Pre-computing trim Jacobians...")
    _JAC   = compute_jacobians(aero_model, np.zeros(4), z_trim, U_INF, DT, CL_trim, CM_trim)
    _A_ekf, _C_ekf = _compute_ekf_jacobians(aero_model, xi_trim, U_INF, DT, LAMBDA_W)
    print("  Done.")

    # ── CMA-ES ───────────────────────────────────────────────────────────────
    theta0 = np.log([Q_H_BASE, Q_A_BASE, Q_CL_BASE, Q_CM_BASE, R_BASE])
    print(f"\nBaseline: Q_H={Q_H_BASE:.0f}  Q_A={Q_A_BASE:.0f}  "
          f"Q_CL={Q_CL_BASE:.0f}  Q_CM={Q_CM_BASE:.0f}  R={R_BASE:.4f}")
    print(f"CMA-ES: σ₀={CMA_SIGMA0}  maxiter={CMA_MAXITER}  "
          f"popsize={CMA_POPSIZE}  workers={CMA_N_WORKERS}\n")

    try:
        import cma
    except ImportError:
        print("ERROR: pip install cma"); sys.exit(1)

    es = cma.CMAEvolutionStrategy(
        theta0, CMA_SIGMA0,
        {'maxiter': CMA_MAXITER, 'popsize': CMA_POPSIZE, 'verbose': -9})

    best_J = np.inf;  best_theta = theta0.copy()
    best_pk1 = best_pk2 = best_pd = float('nan')
    iter_num = 0

    _ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=CMA_N_WORKERS, mp_context=_ctx,
            initializer=_worker_init,
            initargs=(_MODELS_DIR,)) as pool:

        while not es.stop():
            solutions = es.ask()
            args_list = [
                (th, z_trim, CL_trim, CM_trim, xi_trim,
                 _JAC, _A_ekf, _C_ekf, A_s, B_s)
                for th in solutions
            ]
            results   = list(pool.map(_eval_worker, args_list))
            fitnesses = [r[0] for r in results]

            for r, th in zip(results, solutions):
                if r[0] < best_J:
                    best_J, best_theta = r[0], th.copy()
                    best_pk1, best_pk2, best_pd = r[1], r[2], r[3]

            es.tell(solutions, fitnesses)
            iter_num += 1
            w = np.exp(best_theta)
            print(f"  iter {iter_num:3d} | J={best_J:.5f} | "
                  f"pk_CL1={best_pk1:.4f} | pk_CL2={best_pk2:.4f} | "
                  f"pk_δ={best_pd:.2f}° | "
                  f"Q_H={w[0]:.0f} Q_A={w[1]:.0f} Q_CL={w[2]:.0f} "
                  f"Q_CM={w[3]:.0f} R={w[4]:.4f}", flush=True)

    Q_H_OPT, Q_A_OPT, Q_CL_OPT, Q_CM_OPT, R_OPT = np.exp(best_theta)
    weights_path = RESULTS_DIR / 'lqr_optimized_weights.txt'
    with open(weights_path, 'w') as _f:
        _f.write(f"Q_H   = {Q_H_OPT:.6g}\n")
        _f.write(f"Q_A   = {Q_A_OPT:.6g}\n")
        _f.write(f"Q_CL  = {Q_CL_OPT:.6g}\n")
        _f.write(f"Q_CM  = {Q_CM_OPT:.6g}\n")
        _f.write(f"R     = {R_OPT:.6g}\n")
        _f.write(f"Q_w   = {Q_W_FIXED:.6g}  (fixed)\n")
        _f.write(f"J_opt = {best_J:.6f}\n")
    print(f"\n[OK] {weights_path}")
    print(f"     Q_H={Q_H_OPT:.1f}  Q_A={Q_A_OPT:.1f}  "
          f"Q_CL={Q_CL_OPT:.1f}  Q_CM={Q_CM_OPT:.1f}  R={R_OPT:.4f}")

    # ── amplitude sweep  (parallelized with ThreadPoolExecutor) ──────────────
    W0_LIST = [20, 40, 60];  T_G_FIX = 1.0
    TG_LIST = [0.5, 0.75, 1.0, 1.5, 2.0];  W0_FIX = 60.0

    _KEYS = ['C_L', 'C_M', 'h_ddot', 'a_ddot', 'delta',
             'W_hat', 'W_gust', 'h', 'hd', 'a', 'ad']

    def _do_sweep(label_theta_W0_Tg):
        label, theta, W0, T_g = label_theta_W0_Tg
        res = run_sweep_case(label, theta, aero_model, z_trim, CL_trim, CM_trim,
                             xi_trim, A_s, B_s, W0, T_g, _JAC, _A_ekf, _C_ekf)
        return label, theta, W0, T_g, res

    print("\n" + "=" * 70)
    print("Amplitude sweep  W0 ∈ {20,40,60} m/s")
    amp_tasks = [(lbl, th, W0, T_G_FIX)
                 for W0 in W0_LIST
                 for lbl, th in [('open_loop', theta0),
                                  ('lqr_base',  theta0),
                                  ('lqr_opt',   best_theta)]]
    rows_amp = [];  ts_amp = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as tex:
        for label, _, W0, T_g, res in tex.map(_do_sweep, amp_tasks):
            print(f"  [done] W0={W0}  {label}", flush=True)
            rows_amp.append(extract_metrics(res, W0, T_g, label))
            for k in _KEYS:
                ts_amp[f'W{W0}_{label}_{k}'] = res[k]
    pd.DataFrame(rows_amp).to_csv(RESULTS_DIR / 'sweep_amplitude.csv', index=False)
    np.savez(RESULTS_DIR / 'sweep_amplitude_timeseries.npz', **ts_amp)
    print(f"[OK] sweep_amplitude.csv  ({len(rows_amp)} rows)")

    # ── duration sweep ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Duration sweep  T_g ∈ {0.5,0.75,1.0,1.5,2.0} s")
    dur_tasks = [(lbl, th, W0_FIX, T_g)
                 for T_g in TG_LIST
                 for lbl, th in [('open_loop', theta0),
                                  ('lqr_base',  theta0),
                                  ('lqr_opt',   best_theta)]]
    rows_dur = [];  ts_dur = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as tex:
        for label, _, W0, T_g, res in tex.map(_do_sweep, dur_tasks):
            Tg_tag = int(round(T_g * 100))
            print(f"  [done] T_g={T_g}  {label}", flush=True)
            rows_dur.append(extract_metrics(res, W0, T_g, label))
            for k in _KEYS:
                ts_dur[f'Tg{Tg_tag}_{label}_{k}'] = res[k]
    pd.DataFrame(rows_dur).to_csv(RESULTS_DIR / 'sweep_duration.csv', index=False)
    np.savez(RESULTS_DIR / 'sweep_duration_timeseries.npz', **ts_dur)
    print(f"[OK] sweep_duration.csv  ({len(rows_dur)} rows)")

    # ── Set A figures ─────────────────────────────────────────────────────────
    print("\nGenerating Set A summary figures...")
    _summary_fig(pd.read_csv(RESULTS_DIR / 'sweep_amplitude.csv'),
                 'W0', '$W_0$ [m/s]', RESULTS_DIR / 'fig_sweep_amplitude_summary.png')
    _summary_fig(pd.read_csv(RESULTS_DIR / 'sweep_duration.csv'),
                 'T_g', '$T_g$ [s]', RESULTS_DIR / 'fig_sweep_duration_summary.png')

    # ── Set B figures ─────────────────────────────────────────────────────────
    print("\nGenerating Set B time-series figures...")
    _t = np.linspace(0.0, T_END, int(T_END / DT) + 1)

    npz_amp = np.load(RESULTS_DIR / 'sweep_amplitude_timeseries.npz')
    for W0 in W0_LIST:
        Tg_tag = int(round(T_G_FIX * 100))
        r_bl   = {k: npz_amp[f'W{W0}_open_loop_{k}'] for k in _KEYS}
        r_base = {k: npz_amp[f'W{W0}_lqr_base_{k}']  for k in _KEYS}
        r_opt  = {k: npz_amp[f'W{W0}_lqr_opt_{k}']   for k in _KEYS}
        save_ts_figures(_t, r_bl, r_base, r_opt,
                        TS_DIR / f'W{W0}_Tg{Tg_tag}',
                        f'  (W₀={W0} m/s, Tg={T_G_FIX} s)')
        print(f"  [OK] W{W0}_Tg{Tg_tag}")

    npz_dur = np.load(RESULTS_DIR / 'sweep_duration_timeseries.npz')
    for T_g in TG_LIST:
        Tg_tag = int(round(T_g * 100))
        r_bl   = {k: npz_dur[f'Tg{Tg_tag}_open_loop_{k}'] for k in _KEYS}
        r_base = {k: npz_dur[f'Tg{Tg_tag}_lqr_base_{k}']  for k in _KEYS}
        r_opt  = {k: npz_dur[f'Tg{Tg_tag}_lqr_opt_{k}']   for k in _KEYS}
        save_ts_figures(_t, r_bl, r_base, r_opt,
                        TS_DIR / f'W{int(W0_FIX)}_Tg{Tg_tag}',
                        f'  (W₀={W0_FIX:.0f} m/s, Tg={T_g} s)')
        print(f"  [OK] W{int(W0_FIX)}_Tg{Tg_tag}")

    print("\n" + "=" * 70)
    print("Done. Results in:", RESULTS_DIR)
    print("=" * 70)

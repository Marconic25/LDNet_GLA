#!/usr/bin/env python3
"""
Numerical tuning of LQR and MPC weights.

Objective (both controllers):
    minimise   peak |C_L|   over gust window [0, T_g + 0.5 s]

Constraint:
    |C_M(t) - CM_trim| < CM_DRIFT_TOL  for all t in [0, T_END]
    (no steady-state drift allowed)

Search variables:
    LQR: log10(Q_CL_lqr), log10(Q_CM_lqr)   — Q_H, Q_A fixed at lqg values
    MPC: log10(Q_CL_mpc), log10(Q_CM_mpc)   — Q_H, Q_A same as LQR, Q_dCL fixed

Uses scipy.optimize.differential_evolution (gradient-free, robust to non-convex
landscapes) with a penalty method for the C_M constraint.
"""
import sys, numpy as np, json
from pathlib import Path
from scipy.optimize import differential_evolution

SRC_DIR    = Path(__file__).parent
MODELS_DIR = SRC_DIR.parent / 'models'
sys.path.insert(0, str(SRC_DIR))

from aerodynamics.model    import LDNetModel
from control.lqr           import LQRController, compute_jacobians
from control.mpc           import MPCController, run_mpc_simulation
from control.ekf_augmented import NonlinearEKF
from structural.smd        import get_space_state_matrices

# ── fixed parameters ──────────────────────────────────────────────────────────
U_INF      = 75.0
T_END      = 3.0
DT         = 0.01
LAMBDA_W   = 0.98
GUST_W0    = 60.0
GUST_TG    = 1.0
GUST_WIN   = GUST_TG + 0.5   # evaluation window for peak C_L

# Fixed structural weights (same for both)
Q_H  = 1.0  / 0.00428**2
Q_A  = 10.0 / 0.00823**2
R    = 1.0  / 2.0**2
Q_w  = 1.0  / 10.0**2

# MPC-only fixed
Q_dCL      = 1.0  / 0.005**2
R_du       = 2.0  / 2.0**2
N_HOR      = 40
DDELTA_MAX = 150.0

# Constraint: C_M must not drift more than this from trim
CM_DRIFT_TOL = 0.010   # [—]  peak |C_M(t) - CM_trim| over full simulation

# Penalty weight for constraint violation
PENALTY = 1e4

# ── setup (done once) ─────────────────────────────────────────────────────────
print("Loading model...")
aero_model = LDNetModel(str(MODELS_DIR))
z_trim = np.zeros(aero_model.num_latent_states)
for _ in range(200):
    z_trim, CL_trim, CM_trim = aero_model.step(
        z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)
CL_trim = float(CL_trim); CM_trim = float(CM_trim)
print(f"  Trim: CL={CL_trim:.4f}  CM={CM_trim:.4f}")

A_s, B_s, _, _ = get_space_state_matrices()
xi_trim = np.concatenate([np.zeros(4), z_trim, [0.0]])

def gust(t):
    if 0.0 <= t <= GUST_TG:
        return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / GUST_TG))
    return 0.0


def eval_lqr(log_Q_CL, log_Q_CM):
    """Run one LQR+EKF simulation, return (peak_CL, max_CM_drift)."""
    Q_CL = 10**log_Q_CL
    Q_CM = 10**log_Q_CM
    try:
        lqr = LQRController(
            aero_model, U_INF, DT,
            x_trim=np.zeros(4), z_trim=z_trim,
            Q_lqr=np.diag([Q_H, 0., Q_A, 0., 0.]),
            R_lqr=np.array([[R]]),
            CL_trim=CL_trim, CM_trim=CM_trim,
            Q_y=np.diag([Q_CL, Q_CM]),
            Q_w=Q_w, lambda_w=LAMBDA_W, delta_max=20.0)
        ekf = NonlinearEKF(aero_model, U_INF, DT, xi_trim, lambda_w=LAMBDA_W)
        res = run_mpc_simulation(
            U_INF, T_END, DT, aero_model, lqr, A_s, B_s,
            gust_profile=gust, observer='ekf_ad', kalman_filter=ekf)
        t    = res['t']
        mask = t <= GUST_WIN
        peak_CL  = np.max(np.abs(res['C_L'][mask]))
        CM_drift = np.max(np.abs(res['C_M'] - CM_trim))
        return peak_CL, CM_drift
    except Exception:
        return 1.0, 1.0


def eval_mpc(log_Q_CL, log_Q_CM):
    """Run one MPC+EKF simulation, return (peak_CL, max_CM_drift)."""
    Q_CL = 10**log_Q_CL
    Q_CM = 10**log_Q_CM
    try:
        mpc = MPCController(
            aero_model, U_INF, DT,
            Q_CL=Q_CL, Q_CM=Q_CM,
            Q_h=Q_H, Q_a=Q_A, Q_dCL=Q_dCL,
            R=R, R_du=R_du, N=N_HOR,
            delta_max=20.0, CL_trim=CL_trim, CM_trim=CM_trim,
            use_tf_solver=True, ddelta_max=DDELTA_MAX)
        ekf = NonlinearEKF(aero_model, U_INF, DT, xi_trim, lambda_w=LAMBDA_W)
        res = run_mpc_simulation(
            U_INF, T_END, DT, aero_model, mpc, A_s, B_s,
            gust_profile=gust, observer='ekf_ad', kalman_filter=ekf)
        t    = res['t']
        mask = t <= GUST_WIN
        peak_CL  = np.max(np.abs(res['C_L'][mask]))
        CM_drift = np.max(np.abs(res['C_M'] - CM_trim))
        return peak_CL, CM_drift
    except Exception:
        return 1.0, 1.0


_eval_count = [0]

def make_objective(eval_fn, label):
    def objective(x):
        log_Q_CL, log_Q_CM = x
        peak_CL, CM_drift = eval_fn(log_Q_CL, log_Q_CM)
        violation = max(0.0, CM_drift - CM_DRIFT_TOL)
        cost = peak_CL + PENALTY * violation**2
        _eval_count[0] += 1
        print(f"  [{label}] eval#{_eval_count[0]:3d}  "
              f"log_QCL={log_Q_CL:.2f} log_QCM={log_Q_CM:.2f}  "
              f"peak_CL={peak_CL:.4f}  CM_drift={CM_drift:.4f}  cost={cost:.4f}")
        return cost
    return objective


# Search bounds: log10(Q) in [1, 6]  →  Q in [10, 1e6]
BOUNDS = [(1.0, 6.0), (1.0, 6.0)]

# ── LQR tuning ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  TUNING LQR")
print("="*60)
_eval_count[0] = 0

res_lqr_opt = differential_evolution(
    make_objective(eval_lqr, 'LQR'),
    bounds=BOUNDS,
    maxiter=20,
    popsize=6,
    tol=1e-3,
    seed=42,
    workers=1,
    disp=False)

log_QCL_lqr, log_QCM_lqr = res_lqr_opt.x
peak_CL_lqr, CM_drift_lqr = eval_lqr(log_QCL_lqr, log_QCM_lqr)
print(f"\n  LQR best:  Q_CL=10^{log_QCL_lqr:.3f}={10**log_QCL_lqr:.1f}"
      f"  Q_CM=10^{log_QCM_lqr:.3f}={10**log_QCM_lqr:.1f}")
print(f"  peak C_L = {peak_CL_lqr:.4f}   CM drift = {CM_drift_lqr:.4f}")

# ── MPC tuning ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  TUNING MPC")
print("="*60)
_eval_count[0] = 0

res_mpc_opt = differential_evolution(
    make_objective(eval_mpc, 'MPC'),
    bounds=BOUNDS,
    maxiter=20,
    popsize=6,
    tol=1e-3,
    seed=42,
    workers=1,
    disp=False)

log_QCL_mpc, log_QCM_mpc = res_mpc_opt.x
peak_CL_mpc, CM_drift_mpc = eval_mpc(log_QCL_mpc, log_QCM_mpc)
print(f"\n  MPC best:  Q_CL=10^{log_QCL_mpc:.3f}={10**log_QCL_mpc:.1f}"
      f"  Q_CM=10^{log_QCM_mpc:.3f}={10**log_QCM_mpc:.1f}")
print(f"  peak C_L = {peak_CL_mpc:.4f}   CM drift = {CM_drift_mpc:.4f}")

# ── save results ──────────────────────────────────────────────────────────────
result = {
    'lqr': {'log_Q_CL': log_QCL_lqr, 'log_Q_CM': log_QCM_lqr,
            'Q_CL': 10**log_QCL_lqr, 'Q_CM': 10**log_QCM_lqr,
            'peak_CL': peak_CL_lqr, 'CM_drift': CM_drift_lqr},
    'mpc': {'log_Q_CL': log_QCL_mpc, 'log_Q_CM': log_QCM_mpc,
            'Q_CL': 10**log_QCL_mpc, 'Q_CM': 10**log_QCM_mpc,
            'peak_CL': peak_CL_mpc, 'CM_drift': CM_drift_mpc},
    'CM_DRIFT_TOL': CM_DRIFT_TOL,
    'Q_H': Q_H, 'Q_A': Q_A, 'R': R,
}
out = SRC_DIR.parent / 'results' / 'tuning_weights.json'
out.parent.mkdir(exist_ok=True)
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  Results saved to {out}")

print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  {'':12s}  {'Q_CL':>10}  {'Q_CM':>10}  {'peak C_L':>10}  {'CM drift':>10}")
print(f"  {'LQR':12s}  {10**log_QCL_lqr:>10.1f}  {10**log_QCM_lqr:>10.1f}"
      f"  {peak_CL_lqr:>10.4f}  {CM_drift_lqr:>10.4f}")
print(f"  {'MPC':12s}  {10**log_QCL_mpc:>10.1f}  {10**log_QCM_mpc:>10.1f}"
      f"  {peak_CL_mpc:>10.4f}  {CM_drift_mpc:>10.4f}")
print(f"  {'Open loop':12s}  {'—':>10}  {'—':>10}  {0.0936:>10.4f}  {'—':>10}")

#!/usr/bin/env python3
"""
Numerical optimisation of LQR and MPC weight vectors to minimise the
L2 norm of C_L deviation from trim over a closed-loop gust simulation.

Objective (same for both controllers):
    f(θ) = Σ_k (C_L_k - CL_trim)² · Δt  +  λ_δ · Σ_k δ_k² · Δt

Weight parameterisation (log-scale to handle orders-of-magnitude differences):
    θ_LQR = [log Q_CL, log Q_CM, log Q_H, log Q_A, log R]         (5 params)
    θ_MPC = [log Q_CL, log Q_CM, log Q_H, log Q_A, log R, log R_du] (6 params)

Fixed (not optimised):
    LQR: Q_w, lambda_w
    MPC: Q_dCL, ddelta_max, N_HOR

Usage:
    python optimize_weights.py --target lqr
    python optimize_weights.py --target mpc
    python optimize_weights.py --target both   (sequential: LQR first, then MPC)
"""
import sys, argparse, json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

SRC_DIR    = Path(__file__).parent
MODELS_DIR = SRC_DIR.parent / 'models'
OUT_DIR    = SRC_DIR.parent / 'results' / 'weight_optim'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from aerodynamics.model    import LDNetModel
from control.lqr           import LQRController
from control.mpc           import MPCController, run_mpc_simulation
from control.ekf_augmented import NonlinearEKF
from structural.smd        import get_space_state_matrices

# ── Fixed simulation parameters ───────────────────────────────────────────────
U_INF      = 75.0
T_END      = 3.0
DT         = 0.01
LAMBDA_W   = 0.98
GUST_W0    = 60.0
GUST_TG    = 1.0
Q_w        = 1.0 / 10.0**2   # fixed W-state weight in LQR DARE
Q_dCL      = 1.0              # fixed MPC smoothing term
DDELTA_MAX = 150.0            # fixed MPC rate limit [°/s]
N_HOR      = 80               # fixed MPC horizon

# Penalty on control effort in the outer objective (keeps actuator sane)
LAMBDA_DELTA = 1e-4

# ── Bounds in log-space ───────────────────────────────────────────────────────
# Each entry: (log_lower, log_upper)
BOUNDS_LQR = [
    (np.log(10),   np.log(5000)),   # Q_CL
    (np.log(10),   np.log(5000)),   # Q_CM
    (np.log(1e3),  np.log(5e5)),    # Q_H
    (np.log(1e3),  np.log(5e5)),    # Q_A
    (np.log(0.01), np.log(10)),     # R
]

BOUNDS_MPC = [
    (np.log(10),   np.log(5000)),   # Q_CL
    (np.log(10),   np.log(5000)),   # Q_CM
    (np.log(1e3),  np.log(5e5)),    # Q_H
    (np.log(1e3),  np.log(5e5)),    # Q_A
    (np.log(0.01), np.log(10)),     # R
    (np.log(0.01), np.log(10)),     # R_du
]

# ── Current best weights (starting point = values from mpc_observer_comparison) ─
THETA0_LQR = np.array([
    np.log(1.0  / 0.0484**2),   # Q_CL ≈ 427
    np.log(1.0  / 0.04**2),     # Q_CM = 625
    np.log(1.0  / 0.00428**2),  # Q_H  ≈ 54k
    np.log(10.0 / 0.00823**2),  # Q_A  ≈ 148k
    np.log(1.0  / 2.0**2),      # R    = 0.25
])

THETA0_MPC = np.array([
    np.log(1.0  / 0.0484**2),   # Q_CL
    np.log(1.0  / 0.04**2),     # Q_CM
    np.log(1.0  / 0.00428**2),  # Q_H
    np.log(10.0 / 0.00823**2),  # Q_A
    np.log(1.0  / 2.0**2),      # R
    np.log(2.0  / 2.0**2),      # R_du = 0.5
])


def gust(t):
    if 0.0 <= t <= GUST_TG:
        return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / GUST_TG))
    return 0.0


def unpack_lqr(theta):
    Q_CL, Q_CM, Q_H, Q_A, R = np.exp(theta)
    return Q_CL, Q_CM, Q_H, Q_A, R


def unpack_mpc(theta):
    Q_CL, Q_CM, Q_H, Q_A, R, R_du = np.exp(theta)
    return Q_CL, Q_CM, Q_H, Q_A, R, R_du


def objective(C_L_arr, delta_arr, CL_trim, dt):
    """
    L2 cost: integral of (C_L - CL_trim)^2 + lambda * delta^2
    """
    cl_dev = C_L_arr - CL_trim
    return (np.sum(cl_dev**2) + LAMBDA_DELTA * np.sum(delta_arr**2)) * dt


# ── LQR objective ─────────────────────────────────────────────────────────────

def make_lqr_objective(aero_model, z_trim, CL_trim, CM_trim, A_s, B_s, xi_trim):

    eval_count = [0]

    def f(theta):
        eval_count[0] += 1
        Q_CL, Q_CM, Q_H, Q_A, R = unpack_lqr(theta)

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
                gust_profile=gust,
                observer='ekf_ad',
                kalman_filter=ekf)

            cost = objective(res['C_L'], res['delta'], CL_trim, DT)

        except Exception as e:
            print(f"    [WARN] eval {eval_count[0]}: {e}")
            cost = 1e6

        print(f"  LQR eval {eval_count[0]:3d}  cost={cost:.6f}  "
              f"Q_CL={Q_CL:.0f}  Q_CM={Q_CM:.0f}  "
              f"Q_H={Q_H:.0f}  Q_A={Q_A:.0f}  R={R:.4f}")
        return cost

    return f


# ── MPC objective ─────────────────────────────────────────────────────────────

def make_mpc_objective(aero_model, CL_trim, CM_trim, A_s, B_s, xi_trim):

    eval_count = [0]

    def f(theta):
        eval_count[0] += 1
        Q_CL, Q_CM, Q_H, Q_A, R, R_du = unpack_mpc(theta)

        try:
            mpc = MPCController(
                aero_model, U_INF, DT,
                Q_CL=Q_CL, Q_CM=Q_CM,
                Q_h=Q_H,   Q_a=Q_A,
                Q_dCL=Q_dCL,
                R=R, R_du=R_du, N=N_HOR,
                delta_max=20.0, CL_trim=CL_trim, CM_trim=CM_trim,
                use_tf_solver=True, ddelta_max=DDELTA_MAX)

            ekf = NonlinearEKF(aero_model, U_INF, DT, xi_trim, lambda_w=LAMBDA_W)

            res = run_mpc_simulation(
                U_INF, T_END, DT, aero_model, mpc, A_s, B_s,
                gust_profile=gust,
                observer='ekf_ad',
                kalman_filter=ekf)

            cost = objective(res['C_L'], res['delta'], CL_trim, DT)

        except Exception as e:
            print(f"    [WARN] eval {eval_count[0]}: {e}")
            cost = 1e6

        print(f"  MPC eval {eval_count[0]:3d}  cost={cost:.6f}  "
              f"Q_CL={Q_CL:.0f}  Q_CM={Q_CM:.0f}  "
              f"Q_H={Q_H:.0f}  Q_A={Q_A:.0f}  "
              f"R={R:.4f}  R_du={R_du:.4f}")
        return cost

    return f


# ── Nelder-Mead wrapper ───────────────────────────────────────────────────────

def run_nelder_mead(f, theta0, bounds, label, max_iter=300):
    print(f"\n{'='*60}")
    print(f"  Nelder-Mead optimisation — {label}")
    print(f"  n_params={len(theta0)}  max_iter={max_iter}")
    print(f"{'='*60}")

    result = minimize(
        f, theta0,
        method='Nelder-Mead',
        bounds=bounds,
        options={
            'maxiter':  max_iter,
            'xatol':    1e-3,
            'fatol':    1e-5,
            'adaptive': True,   # scales simplex to n_params automatically
            'disp':     True,
        }
    )

    print(f"\n  Converged: {result.success}  ({result.message})")
    print(f"  Final cost: {result.fun:.6f}")
    return result


def print_weights(label, theta, unpack_fn):
    vals = np.exp(theta)
    names_lqr = ['Q_CL', 'Q_CM', 'Q_H', 'Q_A', 'R']
    names_mpc = ['Q_CL', 'Q_CM', 'Q_H', 'Q_A', 'R', 'R_du']
    names = names_mpc if len(theta) == 6 else names_lqr
    print(f"\n  Optimal {label} weights:")
    for n, v in zip(names, vals):
        print(f"    {n:<8} = {v:.4g}")


def save_result(label, theta, cost):
    vals = np.exp(theta).tolist()
    names_lqr = ['Q_CL', 'Q_CM', 'Q_H', 'Q_A', 'R']
    names_mpc = ['Q_CL', 'Q_CM', 'Q_H', 'Q_A', 'R', 'R_du']
    names = names_mpc if len(theta) == 6 else names_lqr
    out = {'cost': cost, 'weights': dict(zip(names, vals)), 'theta_log': theta.tolist()}
    path = OUT_DIR / f'optimal_{label.lower()}.json'
    path.write_text(json.dumps(out, indent=2))
    print(f"  Saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['lqr', 'mpc', 'both'],
                        default='both', help='Which controller to optimise')
    parser.add_argument('--max-iter', type=int, default=300,
                        help='Max Nelder-Mead iterations per controller')
    parser.add_argument('--resume-lqr', type=str, default=None,
                        help='Path to JSON with previous LQR result (warm start)')
    parser.add_argument('--resume-mpc', type=str, default=None,
                        help='Path to JSON with previous MPC result (warm start)')
    args = parser.parse_args()

    print("Loading model and computing trim...")
    aero_model = LDNetModel(str(MODELS_DIR))
    z_trim = np.zeros(aero_model.num_latent_states)
    for _ in range(200):
        z_trim, CL_trim, CM_trim = aero_model.step(
            z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)
    CL_trim = float(CL_trim)
    CM_trim = float(CM_trim)
    print(f"  Trim: CL={CL_trim:.4f}  CM={CM_trim:.4f}")

    A_s, B_s, _, _ = get_space_state_matrices()
    xi_trim = np.concatenate([np.zeros(4), z_trim, [0.0]])

    # Warm-start from previous run if provided
    theta0_lqr = THETA0_LQR.copy()
    theta0_mpc = THETA0_MPC.copy()

    if args.resume_lqr:
        data = json.loads(Path(args.resume_lqr).read_text())
        theta0_lqr = np.array(data['theta_log'])
        print(f"  LQR warm-start from {args.resume_lqr}  (cost={data['cost']:.6f})")

    if args.resume_mpc:
        data = json.loads(Path(args.resume_mpc).read_text())
        theta0_mpc = np.array(data['theta_log'])
        print(f"  MPC warm-start from {args.resume_mpc}  (cost={data['cost']:.6f})")

    if args.target in ('lqr', 'both'):
        f_lqr = make_lqr_objective(aero_model, z_trim, CL_trim, CM_trim,
                                    A_s, B_s, xi_trim)
        res_lqr = run_nelder_mead(f_lqr, theta0_lqr, BOUNDS_LQR,
                                   'LQR', args.max_iter)
        print_weights('LQR', res_lqr.x, unpack_lqr)
        save_result('lqr', res_lqr.x, res_lqr.fun)

    if args.target in ('mpc', 'both'):
        f_mpc = make_mpc_objective(aero_model, CL_trim, CM_trim,
                                    A_s, B_s, xi_trim)
        res_mpc = run_nelder_mead(f_mpc, theta0_mpc, BOUNDS_MPC,
                                   'MPC', args.max_iter)
        print_weights('MPC', res_mpc.x, unpack_mpc)
        save_result('mpc', res_mpc.x, res_mpc.fun)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
MPC weight sweep — find optimal weights for NonlinearEKF observer.

Both LQR and MPC optimise structural state (h, α). The comparison metric
is peak |C_L| and peak |C_M| — these emerge from the controlled dynamics,
not from direct penalisation. This makes the comparison fair: LQR uses a
linearised model, MPC uses the full nonlinear LDNet rollout.

Sweep: log10(Q_h) × log10(Q_a), with Q_CL=Q_CM=0 (no direct aero penalty).
R and R_du fixed to match LQR.

Constraint: max |C_M(t) - CM_trim| < CM_DRIFT_TOL (no steady-state drift).
Objective:  minimise peak |C_L| over [0, GUST_WIN].

Results saved to results/mpc_weight_sweep.json.
"""
import sys, numpy as np, json
from pathlib import Path

SRC_DIR    = Path(__file__).parent
MODELS_DIR = SRC_DIR.parent / 'models'
OUT_DIR    = SRC_DIR.parent / 'results'
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from aerodynamics.model    import LDNetModel
from control.mpc           import MPCController, run_mpc_simulation
from control.ekf_augmented import NonlinearEKF
from structural.smd        import get_space_state_matrices

# ── fixed ─────────────────────────────────────────────────────────────────────
U_INF  = 75.0; T_END = 3.0; DT = 0.01; LAMBDA_W = 0.98
GUST_W0 = 60.0; GUST_TG = 1.0; GUST_WIN = 1.5
CM_DRIFT_TOL = 0.010

# Same as LQR
R    = 1.0 / 4.0
R_du = 0.5

# MPC-only fixed
Q_CL      = 0.0   # no direct C_L penalty — emerges from state control
Q_CM      = 0.0   # no direct C_M penalty
Q_dCL     = 0.0
N_HOR     = 40
DDELTA_MAX = 150.0

# ── sweep grid ────────────────────────────────────────────────────────────────
# LQR optimal: Q_H = 1/0.00428² ≈ 54k, Q_A = 10/0.00823² ≈ 148k
# Sweep around these values
QH_EXPS = [3.5, 4.0, 4.5, 4.75, 5.0]    # log10(Q_h)
QA_EXPS = [4.5, 5.0, 5.17, 5.5, 6.0]    # log10(Q_a), 5.17 ≈ log10(148k)

# ── setup ─────────────────────────────────────────────────────────────────────
print("Loading model...")
aero_model = LDNetModel(str(MODELS_DIR))
z_trim = np.zeros(aero_model.num_latent_states)
for _ in range(200):
    z_trim, CL_trim, CM_trim = aero_model.step(
        z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)
CL_trim = float(CL_trim); CM_trim = float(CM_trim)
print(f"  Trim: CL={CL_trim:.4f}  CM={CM_trim:.4f}")
print(f"  LQR reference: Q_H=10^4.74={1./0.00428**2:.0f}  "
      f"Q_A=10^5.17={10./0.00823**2:.0f}")

A_s, B_s, _, _ = get_space_state_matrices()
xi_trim = np.concatenate([np.zeros(4), z_trim, [0.0]])

def gust(t):
    if 0.0 <= t <= GUST_TG:
        return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / GUST_TG))
    return 0.0

# ── sweep ─────────────────────────────────────────────────────────────────────
results = []
n_total = len(QH_EXPS) * len(QA_EXPS)
n_done  = 0

print(f"\nSweeping {n_total} combinations...\n")
print(f"  {'Q_h':>10}  {'Q_a':>12}  {'peak_CL':>10}  {'peak_CM':>10}  "
      f"{'CM_drift':>10}  {'peak_h':>8}  {'peak_a':>8}  status")
print("  " + "-"*85)

for qh_exp in QH_EXPS:
    for qa_exp in QA_EXPS:
        Q_h = 10**qh_exp
        Q_a = 10**qa_exp
        try:
            mpc = MPCController(
                aero_model, U_INF, DT,
                Q_CL=Q_CL, Q_CM=Q_CM, Q_h=Q_h, Q_a=Q_a, Q_dCL=Q_dCL,
                R=R, R_du=R_du, N=N_HOR,
                delta_max=20.0, CL_trim=CL_trim, CM_trim=CM_trim,
                use_tf_solver=True, ddelta_max=DDELTA_MAX)
            ekf = NonlinearEKF(aero_model, U_INF, DT, xi_trim, lambda_w=LAMBDA_W)
            res = run_mpc_simulation(
                U_INF, T_END, DT, aero_model, mpc, A_s, B_s,
                gust_profile=gust, observer='ekf_ad', kalman_filter=ekf)

            t    = res['t']
            mask = t <= GUST_WIN
            peak_CL  = float(np.max(np.abs(res['C_L'][mask])))
            peak_CM  = float(np.max(np.abs(res['C_M'][mask])))
            CM_drift = float(np.max(np.abs(res['C_M'] - CM_trim)))
            peak_h   = float(np.max(np.abs(res['h_ddot'][mask])))
            peak_a   = float(np.max(np.rad2deg(np.abs(res['a'][mask]))))
            status   = 'OK' if CM_drift < CM_DRIFT_TOL else 'DRIFT'

        except Exception as e:
            peak_CL = peak_CM = CM_drift = peak_h = peak_a = 999.0
            status = f'ERR:{e}'

        n_done += 1
        print(f"  {Q_h:>10.0f}  {Q_a:>12.0f}  {peak_CL:>10.4f}  {peak_CM:>10.4f}  "
              f"{CM_drift:>10.4f}  {peak_h:>8.3f}  {peak_a:>8.3f}  "
              f"{status}  [{n_done}/{n_total}]", flush=True)

        results.append({
            'log_Q_h': qh_exp, 'log_Q_a': qa_exp,
            'Q_h': Q_h, 'Q_a': Q_a,
            'peak_CL': peak_CL, 'peak_CM': peak_CM,
            'CM_drift': CM_drift, 'peak_h': peak_h, 'peak_a': peak_a,
            'status': status,
        })

# ── save ──────────────────────────────────────────────────────────────────────
out_file = OUT_DIR / 'mpc_weight_sweep.json'
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_file}")

# ── summary ───────────────────────────────────────────────────────────────────
feasible = [r for r in results if r['status'] == 'OK']
if feasible:
    best = min(feasible, key=lambda r: r['peak_CL'])
    print(f"\nBest feasible MPC (min peak C_L):")
    print(f"  Q_h  = {best['Q_h']:.0f}  (10^{best['log_Q_h']:.2f})")
    print(f"  Q_a  = {best['Q_a']:.0f}  (10^{best['log_Q_a']:.2f})")
    print(f"  peak C_L = {best['peak_CL']:.4f}")
    print(f"  peak C_M = {best['peak_CM']:.4f}")
    print(f"  CM drift = {best['CM_drift']:.4f}")
    print(f"  peak ḧ   = {best['peak_h']:.4f}")
    print(f"  peak α   = {best['peak_a']:.4f} °")

    # Also show best on peak_h
    best_h = min(feasible, key=lambda r: r['peak_h'])
    if best_h is not best:
        print(f"\nBest feasible MPC (min peak ḧ):")
        print(f"  Q_h={best_h['Q_h']:.0f}  Q_a={best_h['Q_a']:.0f}  "
              f"peak_CL={best_h['peak_CL']:.4f}  peak_h={best_h['peak_h']:.4f}")
else:
    print("\nNo feasible point — relax CM_DRIFT_TOL or expand grid.")

print("\nLQR reference (from lqg_diagnostics):")
print("  Q_H=54590  Q_A=147639  peak_CL=0.0806  peak_h=0.448  peak_a=0.585°")

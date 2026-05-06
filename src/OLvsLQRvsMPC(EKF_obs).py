#!/usr/bin/env python3
"""
Closed-loop comparison — 3 cases:
  1. Open loop
  2. LQR  + NonlinearEKF
  3. MPC  + NonlinearEKF

Shared aerodynamic objective (both controllers):
  J_aero = Q_CL * C_L²  +  Q_CM * C_M²

LQR: implemented via output-weight Q_y on the linearised Jacobian C_y,
     plus structural weights Q_H, Q_A on (h, α).
MPC: implemented via explicit terms in the nonlinear rollout cost,
     plus same structural weights Q_H, Q_A to prevent steady-state drift.

The C_M drift in the MPC was caused by the absence of structural weights
(Q_h=Q_a=0): without an incentive to return to trim, the flap stayed
deflected after the gust.  Adding Q_H and Q_A (same as LQR) fixes this.

Weight alignment:
  Q_CL  = 1/0.0484²  ≈  427   (shared)
  Q_CM  = 1/0.04²    =  625   (shared — same tolerance as used in LQR Q_y)
  Q_H   = 1/0.00428² ≈  54k   (shared structural weight)
  Q_A   = 10/0.00823²≈ 148k   (shared structural weight)
  R     = 1/2²       = 0.25   (shared control effort)

MPC-only extras:
  Q_dCL = 1/0.005²   = 40k    (smooth C_L rate — no LQR equivalent)
  R_du  = 2/2²       = 0.5    (smooth δ rate)
  ddelta_max = 150 °/s        (hard rate limit)
"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SRC_DIR    = Path(__file__).parent
MODELS_DIR = SRC_DIR.parent / 'models'
OUT_DIR    = SRC_DIR.parent / 'results' / 'mpc_observer_comparison'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from aerodynamics.model        import LDNetModel
from control.lqr               import LQRController, compute_jacobians
from control.mpc               import MPCController
from control.run_controller    import run_simulation
from control.ekf_augmented     import NonlinearEKF
from structural.smd            import get_space_state_matrices

# ── shared weights ────────────────────────────────────────────────────────────
# LQR optimal weights (found manually — Q_CL small so DARE is dominated by
# structural weights Q_H/Q_A, with Q_CM stabilising C_M drift)
Q_CL  = 1.0  / 0.0484**2    # ≈  427
Q_CM  = 1.0  / 0.04**2      # =  625
Q_H   = 1.0  / 0.00428**2   # ≈  54k
Q_A   = 10.0 / 0.00823**2   # ≈ 148k
R     = 1.0  / 2.0**2       # =  0.25

# MPC-only
Q_dCL      = 1.0  / 0.005**2   # = 40k
R_du       = 2.0  / 2.0**2     # =  0.5
N_HOR      = 40
DDELTA_MAX = 150.0              # [°/s]
Q_w        = 1.0  / 10.0**2    # W state weight in LQR DARE

# Simulation
U_INF    = 75.0
T_END    = 3.0
DT       = 0.01
LAMBDA_W = 0.98
GUST_W0  = 60.0
GUST_TG  = 1.0

# ── plot style ────────────────────────────────────────────────────────────────
C_BL  = dict(color='steelblue', lw=1.5, alpha=0.85, label='Open loop')
C_LQR = dict(color='darkcyan',  lw=1.5, alpha=0.85, label='LQR + NonlinEKF')
C_MPC = dict(color='crimson',   lw=1.5, alpha=0.85, label='MPC + NonlinEKF')


def fmt(ax, ylabel, title=None, legend=True):
    ax.set_xlabel('t [s]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(fontsize=8)


def gust(t):
    if 0.0 <= t <= GUST_TG:
        return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / GUST_TG))
    return 0.0


def make_lqr(aero_model, z_trim, CL_trim, CM_trim, A_s, B_s):
    """Build LQR with structural + C_L output weights."""
    Q_lqr = np.diag([Q_H, 0., Q_A, 0., 0.])
    Q_y   = np.diag([Q_CL, Q_CM])
    return LQRController(
        aero_model, U_INF, DT,
        x_trim=np.zeros(4), z_trim=z_trim,
        Q_lqr=Q_lqr, R_lqr=np.array([[R]]),
        CL_trim=CL_trim, CM_trim=CM_trim,
        Q_y=Q_y, Q_w=Q_w, lambda_w=LAMBDA_W, delta_max=20.0)


def make_mpc(aero_model, CL_trim, CM_trim):
    """Build MPC with identical structural weights to LQR.
    Q_CL=Q_CM=0: no direct aero penalty, same objective as LQR.
    The advantage of MPC over LQR comes from the nonlinear model rollout."""
    return MPCController(
        aero_model, U_INF, DT,
        Q_CL=Q_CL, Q_CM=Q_CM,
        Q_h=Q_H, Q_a=Q_A,
        Q_dCL=Q_dCL,
        R=R, R_du=R_du, N=80,
        delta_max=20.0, CL_trim=CL_trim, CM_trim=CM_trim,
        use_tf_solver=True, ddelta_max=DDELTA_MAX)


def make_ekf(aero_model, xi_trim):
    return NonlinearEKF(aero_model, U_INF, DT, xi_trim, lambda_w=LAMBDA_W)


def run_case(label, aero_model, controller, A_s, B_s, ekf, xi_trim,
             observer_mode):
    """Run one simulation case, reset EKF beforehand."""
    if ekf is not None:
        ekf.reset(xi_trim)
    print(f"  {label}...")
    return run_simulation(
        U_INF, T_END, DT, aero_model, controller, A_s, B_s,
        gust_profile=gust,
        observer=observer_mode,
        kalman_filter=ekf)


def print_table(res_bl, res_lqr, res_mpc, t, mask):
    def amp(a, m):  return (a[m].max() - a[m].min()) / 2.0
    def peak(a, m): return np.max(np.abs(a[m]))
    def rms(a, m):  return np.sqrt(np.mean(a[m]**2))

    for r in (res_bl, res_lqr, res_mpc):
        r['a_deg']  = np.rad2deg(np.abs(r['a']))
        r['ddelta'] = np.abs(np.gradient(r['delta'], t))

    print(f"\n{'─'*65}")
    print(f"  PERFORMANCE TABLE  (gust window 0–{GUST_TG+0.5:.1f}s)")
    print(f"{'─'*65}")
    print(f"  {'Metric':<22}  {'Open loop':>10}  {'LQR+EKF':>10}  {'MPC+EKF':>10}")
    print(f"  {'─'*60}")
    for mname, key, fn in [
        ('peak |ḧ|  [m/s²]',  'h_ddot', peak),
        ('amp  h    [m]',      'h',      amp),
        ('peak |α|  [°]',      'a_deg',  peak),
        ('peak |C_L|',         'C_L',    peak),
        ('peak |C_M|',         'C_M',    peak),
        ('RMS  δ    [°]',      'delta',  rms),
        ('peak |δ̇|  [°/s]',   'ddelta', peak),
    ]:
        vals = [fn(r[key], mask) for r in (res_bl, res_lqr, res_mpc)]
        print(f"  {mname:<22}  " + "  ".join(f"{v:>10.4f}" for v in vals))


def save_figures(res_bl, res_lqr, res_mpc, t):
    # δ in aeronautic convention (negate LDNet sign)
    d_lqr = -res_lqr['delta']
    d_mpc = -res_mpc['delta']

    # Fig 1 — structural state
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    fig.suptitle('Structural state  —  open loop / LQR / MPC  (NonlinEKF)',
                 fontsize=13)
    axes[0,0].plot(t, res_bl['h'],            **C_BL)
    axes[0,0].plot(t, res_lqr['h'],           **C_LQR)
    axes[0,0].plot(t, res_mpc['h'],           **C_MPC)
    fmt(axes[0,0], 'h [m]', 'Heave displacement')
    axes[0,1].plot(t, res_bl['hd'],           **C_BL)
    axes[0,1].plot(t, res_lqr['hd'],          **C_LQR)
    axes[0,1].plot(t, res_mpc['hd'],          **C_MPC)
    fmt(axes[0,1], 'ḣ [m/s]', 'Heave velocity')
    axes[1,0].plot(t, np.rad2deg(res_bl['a']),  **C_BL)
    axes[1,0].plot(t, np.rad2deg(res_lqr['a']), **C_LQR)
    axes[1,0].plot(t, np.rad2deg(res_mpc['a']), **C_MPC)
    fmt(axes[1,0], 'α [°]', 'Pitch angle')
    axes[1,1].plot(t, np.rad2deg(res_bl['ad']),  **C_BL)
    axes[1,1].plot(t, np.rad2deg(res_lqr['ad']), **C_LQR)
    axes[1,1].plot(t, np.rad2deg(res_mpc['ad']), **C_MPC)
    fmt(axes[1,1], 'α̇ [°/s]', 'Pitch rate')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig1_state.png', dpi=150); plt.close(fig)
    print(f"  [OK] fig1_state.png")

    # Fig 2 — accelerations
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    fig.suptitle('Structural accelerations  —  open loop / LQR / MPC', fontsize=13)
    axes[0].plot(t, res_bl['h_ddot'],   **C_BL)
    axes[0].plot(t, res_lqr['h_ddot'],  **C_LQR)
    axes[0].plot(t, res_mpc['h_ddot'],  **C_MPC)
    fmt(axes[0], 'ḧ [m/s²]', 'Heave acceleration')
    axes[1].plot(t, np.rad2deg(res_bl['a_ddot']),   **C_BL)
    axes[1].plot(t, np.rad2deg(res_lqr['a_ddot']),  **C_LQR)
    axes[1].plot(t, np.rad2deg(res_mpc['a_ddot']),  **C_MPC)
    fmt(axes[1], 'α̈ [°/s²]', 'Pitch acceleration')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig2_accels.png', dpi=150); plt.close(fig)
    print(f"  [OK] fig2_accels.png")

    # Fig 3 — control
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    fig.suptitle('Control input  —  LQR / MPC', fontsize=13)
    axes[0].plot(t, d_lqr, **C_LQR)
    axes[0].plot(t, d_mpc, **C_MPC)
    axes[0].axhline( 20, color='k', ls=':', lw=0.8, label='±20° sat.')
    axes[0].axhline(-20, color='k', ls=':', lw=0.8)
    fmt(axes[0], 'δ [°]  (+ = flap down)', 'Flap deflection')
    axes[1].plot(t, np.gradient(d_lqr, t), **C_LQR)
    axes[1].plot(t, np.gradient(d_mpc, t), **C_MPC)
    axes[1].axhline( DDELTA_MAX, color='k', ls=':', lw=0.8,
                    label=f'±{DDELTA_MAX:.0f}°/s')
    axes[1].axhline(-DDELTA_MAX, color='k', ls=':', lw=0.8)
    fmt(axes[1], 'δ̇ [°/s]', 'Flap rate')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig3_control.png', dpi=150); plt.close(fig)
    print(f"  [OK] fig3_control.png")

    # Fig 4 — aero coefficients
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    fig.suptitle('Aerodynamic coefficients  —  open loop / LQR / MPC', fontsize=13)
    axes[0].plot(t, res_bl['C_L'],   **C_BL)
    axes[0].plot(t, res_lqr['C_L'],  **C_LQR)
    axes[0].plot(t, res_mpc['C_L'],  **C_MPC)
    axes[0].axhline(0, color='k', ls=':', lw=0.6)
    fmt(axes[0], '$C_L$', 'Lift coefficient')
    axes[1].plot(t, res_bl['C_M'],   **C_BL)
    axes[1].plot(t, res_lqr['C_M'],  **C_LQR)
    axes[1].plot(t, res_mpc['C_M'],  **C_MPC)
    axes[1].axhline(0, color='k', ls=':', lw=0.6)
    fmt(axes[1], '$C_M$', 'Pitching moment coefficient')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig4_aero.png', dpi=150); plt.close(fig)
    print(f"  [OK] fig4_aero.png")

    # Fig 5 — W estimation
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, res_bl['W_gust'], label='W true', color='k', lw=2.0, ls='--')
    ax.plot(t, res_lqr['W_hat'], **C_LQR)
    ax.plot(t, res_mpc['W_hat'], **C_MPC)
    fmt(ax, 'W [m/s]', 'Gust estimation — NonlinEKF')
    fig.suptitle('Gust estimation', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig5_W_estimation.png', dpi=150); plt.close(fig)
    print(f"  [OK] fig5_W_estimation.png")


# ── main ──────────────────────────────────────────────────────────────────────
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

print(f"\nBuilding LQR  (Q_H={Q_H:.0f}, Q_A={Q_A:.0f}, Q_CL={Q_CL:.0f}, Q_CM={Q_CM:.0f})...")
lqr = make_lqr(aero_model, z_trim, CL_trim, CM_trim, A_s, B_s)

print(f"\nBuilding MPC  (same Q_H, Q_A, Q_CL, Q_CM + Q_dCL={Q_dCL:.0f})...")
mpc = make_mpc(aero_model, CL_trim, CM_trim)

print("\nBuilding NonlinearEKF observers...")
ekf_lqr = make_ekf(aero_model, xi_trim)
ekf_mpc = make_ekf(aero_model, xi_trim)

print("\nRunning simulations...")
res_bl  = run_case('Open loop',       aero_model, None, A_s, B_s,
                   None, xi_trim, 'true_state')
res_lqr = run_case('LQR + NonlinEKF', aero_model, lqr,  A_s, B_s,
                   ekf_lqr, xi_trim, 'ekf_ad')
res_mpc = run_case('MPC + NonlinEKF', aero_model, mpc,  A_s, B_s,
                   ekf_mpc, xi_trim, 'ekf_ad')

t    = res_bl['t']
mask = (t >= 0.0) & (t <= GUST_TG + 0.5)

print_table(res_bl, res_lqr, res_mpc, t, mask)

print("\nSaving figures...")
save_figures(res_bl, res_lqr, res_mpc, t)

print(f"\nDone. Figures in: {OUT_DIR}")

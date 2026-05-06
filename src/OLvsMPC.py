#!/usr/bin/env python3
"""
Open loop vs MPC (NonlinearEKF observer).
Single 1-cosine gust, duration 1s, T_END=3s.
"""
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SRC_DIR    = Path(__file__).parent
MODELS_DIR = SRC_DIR.parent / 'models'
OUT_DIR    = SRC_DIR.parent / 'results' / 'ol_vs_mpc'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from aerodynamics.model     import LDNetModel
from control.mpc            import MPCController
from control.run_controller import run_simulation
from control.ekf_augmented  import NonlinearEKF
from structural.smd         import get_space_state_matrices

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
U_INF      = 75.0
T_END      = 3.0
DT         = 0.01
LAMBDA_W   = 0.98
GUST_W0    = 60.0
GUST_DUR   = 1.0

Q_CL       = 1.0  / 0.0484**2   # ≈  427
Q_CM       = 1.0  / 0.04**2     # =  625
Q_H        = 1.0  / 0.00428**2  # ≈  54k
Q_A        = 10.0 / 0.00823**2  # ≈ 148k
R          = 1.0  / 2.0**2      # =  0.25
Q_dCL      = 1.0  / 0.005**2    # = 40k
R_du       = 2.0  / 2.0**2      # =  0.5
N_HOR      = 80
DDELTA_MAX = 150.0               # [°/s]

# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────
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

print(f"\nBuilding MPC  (Q_CL={Q_CL:.0f}, Q_CM={Q_CM:.0f}, Q_H={Q_H:.0f}, Q_A={Q_A:.0f})...")
mpc = MPCController(
    aero_model, U_INF, DT,
    Q_CL=Q_CL, Q_CM=Q_CM,
    Q_h=Q_H, Q_a=Q_A,
    Q_dCL=Q_dCL,
    R=R, R_du=R_du, N=N_HOR,
    delta_max=20.0, CL_trim=CL_trim, CM_trim=CM_trim,
    use_tf_solver=True, ddelta_max=DDELTA_MAX)

print("Building NonlinearEKF observer...")
ekf = NonlinearEKF(aero_model, U_INF, DT, xi_trim, lambda_w=LAMBDA_W)

def single_gust(t):
    if 0.0 <= t <= GUST_DUR:
        return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / GUST_DUR))
    return 0.0

# ─────────────────────────────────────────────────────────────
# SIMULATIONS
# ─────────────────────────────────────────────────────────────
print("\nRunning BASELINE (no control)...")
res_b = run_simulation(U_INF, T_END, DT, aero_model, None, A_s, B_s,
                       observer='heuristic', gust_profile=single_gust)

print("Running MPC (EKF observer)...")
ekf.reset(xi_trim)
res_mpc = run_simulation(U_INF, T_END, DT, aero_model, mpc, A_s, B_s,
                         observer='ekf_clinv', kalman_filter=ekf,
                         gust_profile=single_gust)

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def amplitude(arr):
    return (arr.max() - arr.min()) / 2.0

_W_END = GUST_DUR + 0.5

def amp_window(res, t_start, t_end):
    mask = (res['t'] >= t_start) & (res['t'] <= t_end)
    return {k: amplitude(res[k][mask]) for k in ('h', 'a', 'h_ddot', 'a_ddot', 'C_L', 'C_M')}

gb   = amp_window(res_b,   0.0, _W_END)
gmpc = amp_window(res_mpc, 0.0, _W_END)

print(f"\n── Gust window [0 – {_W_END:.1f}s] ──────────────────────────────────────────")
print(f"{'':12s}  {'Baseline':>10s}  {'MPC':>10s}  {'Red%':>7s}")
for name in ('h', 'a', 'h_ddot', 'a_ddot', 'C_L', 'C_M'):
    b, m = gb[name], gmpc[name]
    print(f"  {name:<10s}  {b:10.5f}  {m:10.5f}  {(b-m)/b*100 if b > 0 else 0:+6.1f}%")

if np.any(np.isnan(res_mpc['h'])):
    print("\n[WARNING] NaN detected!")
else:
    print("\n[OK] No NaNs detected")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
t_b   = res_b['t']
t_mpc = res_mpc['t']

BL  = dict(color='steelblue', lw=1.5, alpha=0.85, label='Open loop')
MPC = dict(color='crimson',   lw=1.5, alpha=0.85, label='MPC + EKF')

def fmt(ax, ylabel, title=None, legend=True):
    ax.set_xlabel('t [s]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if title: ax.set_title(title, fontsize=10)
    if legend: ax.legend(fontsize=8)

# Figure 1 — Structural state
fig1, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
fig1.suptitle('Structural State  —  Open loop vs MPC', fontsize=13)
axes[0,0].plot(t_b, res_b['h'],              **BL)
axes[0,0].plot(t_mpc, res_mpc['h'],          **MPC)
fmt(axes[0,0], 'h [m]', 'Heave displacement')
axes[0,1].plot(t_b, res_b['hd'],             **BL)
axes[0,1].plot(t_mpc, res_mpc['hd'],         **MPC)
fmt(axes[0,1], 'ḣ [m/s]', 'Heave velocity')
axes[1,0].plot(t_b, np.rad2deg(res_b['a']),   **BL)
axes[1,0].plot(t_mpc, np.rad2deg(res_mpc['a']), **MPC)
fmt(axes[1,0], 'α [°]', 'Pitch angle')
axes[1,1].plot(t_b, np.rad2deg(res_b['ad']),  **BL)
axes[1,1].plot(t_mpc, np.rad2deg(res_mpc['ad']), **MPC)
fmt(axes[1,1], 'α̇ [°/s]', 'Pitch rate')
fig1.tight_layout()
fig1.savefig(OUT_DIR / 'fig1_state.png', dpi=150); plt.close(fig1)
print("[OK] fig1_state.png")

# Figure 2 — Accelerations
fig2, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig2.suptitle('Structural Accelerations  —  Open loop vs MPC', fontsize=13)
axes[0].plot(t_b, res_b['h_ddot'],        **BL)
axes[0].plot(t_mpc, res_mpc['h_ddot'],    **MPC)
fmt(axes[0], 'ḧ [m/s²]', 'Heave acceleration')
axes[1].plot(t_b, np.rad2deg(res_b['a_ddot']),   **BL)
axes[1].plot(t_mpc, np.rad2deg(res_mpc['a_ddot']), **MPC)
fmt(axes[1], 'α̈ [°/s²]', 'Pitch acceleration')
fig2.tight_layout()
fig2.savefig(OUT_DIR / 'fig2_accels.png', dpi=150); plt.close(fig2)
print("[OK] fig2_accels.png")

# Figure 3 — Control input
delta_plot = -res_mpc['delta']
fig3, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig3.suptitle('Control Input  —  MPC', fontsize=13)
axes[0].plot(t_mpc, delta_plot, **MPC)
axes[0].axhline( 20, color='k', ls=':', lw=0.8, label='±20° sat.')
axes[0].axhline(-20, color='k', ls=':', lw=0.8)
fmt(axes[0], 'δ [°]  (+ = flap down)', 'Flap deflection')
axes[1].plot(t_mpc, np.gradient(delta_plot, t_mpc), **MPC)
axes[1].axhline( DDELTA_MAX, color='k', ls=':', lw=0.8, label=f'±{DDELTA_MAX:.0f}°/s')
axes[1].axhline(-DDELTA_MAX, color='k', ls=':', lw=0.8)
fmt(axes[1], 'δ̇ [°/s]', 'Flap rate')
fig3.tight_layout()
fig3.savefig(OUT_DIR / 'fig3_control.png', dpi=150); plt.close(fig3)
print("[OK] fig3_control.png")

# Figure 4 — Aerodynamic coefficients
fig4, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig4.suptitle('Aerodynamic Coefficients  —  Open loop vs MPC', fontsize=13)
axes[0].plot(t_b, res_b['C_L'],        **BL)
axes[0].plot(t_mpc, res_mpc['C_L'],    **MPC)
fmt(axes[0], '$C_L$', 'Lift coefficient')
axes[1].plot(t_b, res_b['C_M'],        **BL)
axes[1].plot(t_mpc, res_mpc['C_M'],    **MPC)
fmt(axes[1], '$C_M$', 'Pitching moment coefficient')
fig4.tight_layout()
fig4.savefig(OUT_DIR / 'fig4_aero.png', dpi=150); plt.close(fig4)
print("[OK] fig4_aero.png")

# Figure 5 — Gust estimation
fig5, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_mpc, res_mpc['W_gust'], label='W true', color='k', lw=2.0, ls='--')
ax.plot(t_mpc, res_mpc['W_hat'],  **MPC)
fmt(ax, 'W [m/s]', 'Gust estimation — NonlinEKF')
fig5.suptitle('Gust estimation', fontsize=13)
fig5.tight_layout()
fig5.savefig(OUT_DIR / 'fig5_W_estimation.png', dpi=150); plt.close(fig5)
print("[OK] fig5_W_estimation.png")

print(f"\nDone. Figures in: {OUT_DIR}")

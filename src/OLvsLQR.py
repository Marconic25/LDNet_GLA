#!/usr/bin/env python3
"""
Test script: Baseline vs LQR (linearized LDNet).
Single 1-cosine gust, duration 1s, T_END=3s.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from structural.smd import get_space_state_matrices
from control.run_controller import run_simulation
from control.lqr import LQRController
from control.ekf_augmented import NonlinearEKF
from aerodynamics.model import LDNetModel as AeroModel
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
U_INF  = 75.0
T_END  = 3.0
DT     = 0.01

# Q_a=10 minimizes C_L peak (0.0793 vs 0.0936 baseline)
Q_h   = 1.0  / 0.00428**2
Q_a   = 10.0 / 0.00823**2
R_lqr = np.array([[1.0 / 2.0**2]])

# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────
print("Loading structural state-space matrices...")
A_s, B_s, C_s, D_s = get_space_state_matrices()

print("Loading aerodynamic model...")
models_dir = Path(__file__).parent.parent / 'models'
if not models_dir.exists():
    models_dir = Path(__file__).parent.parent.parent / 'models'
aero_model = AeroModel(str(models_dir))
print(f"  LDNet: {aero_model.num_latent_states} latent states")

print("Computing trim...")
_z_trim = np.zeros(aero_model.num_latent_states)
for _ in range(200): #200 iteration to converge to equilibrium of LDNet
    _z_trim, _CL_trim, _CM_trim = aero_model.step(_z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)
print(f"  C_L_trim = {float(_CL_trim):.5f},  C_M_trim = {float(_CM_trim):.5f}")

print("Initializing LQR...")
Q_lqr = np.diag([Q_h, 0.0, Q_a, 0.0, 0.0])
lqr = LQRController(aero_model, U_INF, DT,
                    x_trim=np.zeros(4), z_trim=_z_trim,
                    Q_lqr=Q_lqr, R_lqr=R_lqr,
                    CL_trim=float(_CL_trim), CM_trim=float(_CM_trim),
                    Q_w=1.0 / 10.0**2, lambda_w=0.98, delta_max=20.0)

print("Initializing EKF...")
xi_trim = np.concatenate([np.zeros(4), _z_trim, [0.0]])
ekf = NonlinearEKF(aero_model, U_INF, DT, xi_trim=xi_trim, lambda_w=0.98)

# ─────────────────────────────────────────────────────────────
# GUST PROFILE — single 1-cosine gust at t=0
# ─────────────────────────────────────────────────────────────
_GUST_W0  = 60.0
_GUST_DUR = 1.0

def single_gust(t):
    if 0.0 <= t <= _GUST_DUR:
        return (_GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / _GUST_DUR))
    return 0.0

# ─────────────────────────────────────────────────────────────
# SIMULATIONS
# ─────────────────────────────────────────────────────────────
print(f"\nRunning BASELINE (no control)...")
res_b = run_simulation(U_INF, T_END, DT, aero_model, None, A_s, B_s,
                       observer='heuristic', gust_profile=single_gust)

print(f"Running LQR (EKF observer)...")
res_lqr = run_simulation(U_INF, T_END, DT, aero_model, lqr, A_s, B_s,
                         observer='ekf_clinv', kalman_filter=ekf,
                         gust_profile=single_gust)

# ─────────────────────────────────────────────────────────────
# METRICS — gust window [0, 1.5s]
# ─────────────────────────────────────────────────────────────
def amplitude(arr):
    return (arr.max() - arr.min()) / 2.0

_W_END = _GUST_DUR + 0.5

def amp_window(res, t_start, t_end):
    mask = (res['t'] >= t_start) & (res['t'] <= t_end)
    return {k: amplitude(res[k][mask]) for k in ('h','a','h_ddot','a_ddot','C_L','C_M')}

gb   = amp_window(res_b,   0.0, _W_END)
glqr = amp_window(res_lqr, 0.0, _W_END)

print(f"\n── Gust window [0 – {_W_END:.1f}s] ──────────────────────────────────────────")
print(f"{'':12s}  {'Baseline':>10s}  {'LQR':>10s}  {'Red%':>7s}")
for name in ('h','a','h_ddot','a_ddot','C_L','C_M'):
    b, l = gb[name], glqr[name]
    print(f"  {name:<10s}  {b:10.5f}  {l:10.5f}  {(b-l)/b*100 if b>0 else 0:+6.1f}%")

if np.any(np.isnan(res_lqr['h'])):
    print("\n[WARNING] NaN detected!")
else:
    print("\n[OK] No NaNs detected")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
t_b   = res_b['t']
t_lqr = res_lqr['t']

BL  = dict(color='steelblue', lw=1.5, alpha=0.85, label='Open loop')
LQR = dict(color='darkcyan',  lw=1.5, alpha=0.85, label='LQR')

def fmt(ax, ylabel, title=None, legend=True):
    ax.set_xlabel('t [s]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if title: ax.set_title(title, fontsize=10)
    if legend: ax.legend(fontsize=8)

# Figure 1 — Structural state
fig1, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
fig1.suptitle('Structural State  —  Open loop vs LQR', fontsize=13)
axes[0,0].plot(t_b, res_b['h'], **BL); axes[0,0].plot(t_lqr, res_lqr['h'], **LQR)
fmt(axes[0,0], 'h [m]', 'Heave displacement')
axes[0,1].plot(t_b, res_b['hd'], **BL); axes[0,1].plot(t_lqr, res_lqr['hd'], **LQR)
fmt(axes[0,1], 'ḣ [m/s]', 'Heave velocity')
axes[1,0].plot(t_b, np.rad2deg(res_b['a']), **BL)
axes[1,0].plot(t_lqr, np.rad2deg(res_lqr['a']), **LQR)
fmt(axes[1,0], 'α [°]', 'Pitch angle')
axes[1,1].plot(t_b, np.rad2deg(res_b['ad']), **BL)
axes[1,1].plot(t_lqr, np.rad2deg(res_lqr['ad']), **LQR)
fmt(axes[1,1], 'α̇ [°/s]', 'Pitch rate')
fig1.tight_layout(); fig1.savefig('mpc_fig1_state.png', dpi=150)
print("[OK] mpc_fig1_state.png")

# Figure 2 — Accelerations
fig2, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig2.suptitle('Structural Accelerations  —  Open loop vs LQR', fontsize=13)
axes[0].plot(t_b, res_b['h_ddot'], **BL); axes[0].plot(t_lqr, res_lqr['h_ddot'], **LQR)
fmt(axes[0], 'ḧ [m/s²]', 'Heave acceleration')
axes[1].plot(t_b, np.rad2deg(res_b['a_ddot']), **BL)
axes[1].plot(t_lqr, np.rad2deg(res_lqr['a_ddot']), **LQR)
fmt(axes[1], 'α̈ [°/s²]', 'Pitch acceleration')
fig2.tight_layout(); fig2.savefig('mpc_fig2_accels.png', dpi=150)
print("[OK] mpc_fig2_accels.png")

# Figure 3 — Control input
# Note: LDNet convention is delta<0 = flap down = CL increases.
# We negate for display so that positive = flap down (standard aeronautic convention).
delta_plot = -res_lqr['delta']
fig3, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig3.suptitle('Control Input  —  LQR', fontsize=13)
axes[0].plot(t_lqr, delta_plot, **LQR)
axes[0].axhline( 20, color='k', ls=':', lw=0.8, label='±20° sat.')
axes[0].axhline(-20, color='k', ls=':', lw=0.8)
fmt(axes[0], 'δ [°]  (+ = flap down)', 'Flap deflection')
axes[1].plot(t_lqr, np.gradient(delta_plot, t_lqr), **LQR)
fmt(axes[1], 'δ̇ [°/s]', 'Flap rate')
fig3.tight_layout(); fig3.savefig('mpc_fig3_control.png', dpi=150)
print("[OK] mpc_fig3_control.png")

# Figure 4 — Aerodynamic coefficients
fig4, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig4.suptitle('Aerodynamic Coefficients  —  Open loop vs LQR', fontsize=13)
axes[0].plot(t_b, res_b['C_L'], **BL); axes[0].plot(t_lqr, res_lqr['C_L'], **LQR)
fmt(axes[0], '$C_L$', 'Lift coefficient')
axes[1].plot(t_b, res_b['C_M'], **BL); axes[1].plot(t_lqr, res_lqr['C_M'], **LQR)
fmt(axes[1], '$C_M$', 'Pitching moment coefficient')
fig4.tight_layout(); fig4.savefig('mpc_fig4_aero.png', dpi=150)
print("[OK] mpc_fig4_aero.png")

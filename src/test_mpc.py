#!/usr/bin/env python3
"""
Test script: Baseline vs MPC (TF N-D GradientTape solver).
Single 1-cosine gust, duration 1s, T_END=2s.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from structural.smd import get_space_state_matrices
from control.mpc import MPCController, run_mpc_simulation
from aerodynamics.model import LDNetModel as AeroModel
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
U_INF       = 75.0
T_END       = 3.0     # 1s gust + 2s tail
DT          = 0.01
MPC_HORIZON = 10      # 0.1s horizon

# GLA weights — baseline amplitudes at U=75:
#   C_L=0.0484, C_M=0.00511, h=0.00428m, a=0.00823rad
Q_CL  = 1.0 / 0.0484**2
Q_CM  = 10.0 / 0.00511**2
Q_h   = 1.0 / 0.00428**2
Q_a   = 5.0 / 0.00823**2
Q_hd  = 0.0
Q_ad  = 0.0
R_mpc = 1.0 / 2.0**2
R_du  = 2.0 / 2.0**2   # rate cost — smooth δ sequence, less structural excitation

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

print("Computing trim aerodynamic coefficients...")
_z_trim = np.zeros(aero_model.num_latent_states)
for _ in range(200):
    _z_trim, _CL_trim, _CM_trim = aero_model.step(_z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)
print(f"  C_L_trim = {float(_CL_trim):.5f},  C_M_trim = {float(_CM_trim):.5f}")

print("Initializing MPC (TF N-D solver)...")
mpc_tf = MPCController(aero_model, U_INF, DT,
                       Q_CL=Q_CL, Q_CM=Q_CM,
                       Q_h=Q_h,   Q_a=Q_a,
                       Q_hd=Q_hd, Q_ad=Q_ad,
                       R=R_mpc, R_du=R_du, N=MPC_HORIZON, delta_max=20.0,
                       CL_trim=float(_CL_trim), CM_trim=float(_CM_trim),
                       use_tf_solver=True)

# ─────────────────────────────────────────────────────────────
# GUST PROFILE — single 1-cosine, duration 1s
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
res_b = run_mpc_simulation(U_INF, T_END, DT, aero_model, None, A_s, B_s,
                           use_ekf=False, gust_profile=single_gust)

print(f"Running MPC TF (N-D GradientTape, C_L observer)...")
res_tf = run_mpc_simulation(U_INF, T_END, DT, aero_model, mpc_tf, A_s, B_s,
                            use_ekf=True, gust_profile=single_gust)

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def amplitude(arr):
    return (arr.max() - arr.min()) / 2.0

def amp_window(res, t_start, t_end):
    mask = (res['t'] >= t_start) & (res['t'] <= t_end)
    return {k: amplitude(res[k][mask]) for k in ('h','a','h_ddot','a_ddot','C_L','C_M')}

_W_END = _GUST_DUR + 0.5   # [0, 1.5s]

print("\n── Gust window  [0 – {:.1f}s] ──────────────────────────────────────────".format(_W_END))
gb  = amp_window(res_b,  0.0, _W_END)
gtf = amp_window(res_tf, 0.0, _W_END)
print(f"{'':12s}  {'Baseline':>10s}  {'MPC-TF':>10s}  {'Red%':>7s}")
for name in ('h','a','h_ddot','a_ddot','C_L','C_M'):
    b, tf_ = gb[name], gtf[name]
    print(f"  {name:<10s}  {b:10.5f}  {tf_:10.5f}  {(b-tf_)/b*100 if b>0 else 0:+6.1f}%")

if np.any(np.isnan(res_tf['h'])):
    print("\n[WARNING] NaN detected!")
else:
    print("\n[OK] No NaNs detected")

# ─────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────
t_b  = res_b['t']
t_tf = res_tf['t']

BL   = dict(color='steelblue',  lw=1.5, alpha=0.85, label='Baseline')
TFMPC= dict(color='tomato',     lw=1.5, alpha=0.85, label='MPC TF (N-D)')
TRU  = dict(color='steelblue',  lw=1.5,             label='True $W_{gust}$')
HAT  = dict(color='darkorange', lw=1.5, ls='--',    label='$\\hat{W}$ (C_L inversion)')

def fmt(ax, ylabel, title=None, legend=True):
    ax.set_xlabel('t [s]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(fontsize=8)

# ─────────────────────────────────────────────────────────────
# FIGURE 1 — Structural state
# ─────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
fig1.suptitle('Structural State  —  Baseline vs MPC TF', fontsize=13)

axes[0,0].plot(t_b,  res_b['h'],  **BL)
axes[0,0].plot(t_tf, res_tf['h'], **TFMPC)
fmt(axes[0,0], 'h [m]', 'Heave displacement')

axes[0,1].plot(t_b,  res_b['hd'],  **BL)
axes[0,1].plot(t_tf, res_tf['hd'], **TFMPC)
fmt(axes[0,1], 'ḣ [m/s]', 'Heave velocity')

axes[1,0].plot(t_b,  np.rad2deg(res_b['a']),  **BL)
axes[1,0].plot(t_tf, np.rad2deg(res_tf['a']), **TFMPC)
fmt(axes[1,0], 'α [°]', 'Pitch angle')

axes[1,1].plot(t_b,  np.rad2deg(res_b['ad']),  **BL)
axes[1,1].plot(t_tf, np.rad2deg(res_tf['ad']), **TFMPC)
fmt(axes[1,1], 'α̇ [°/s]', 'Pitch rate')

fig1.tight_layout()
fig1.savefig('mpc_fig1_state.png', dpi=150)
print("[OK] mpc_fig1_state.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 2 — Structural accelerations
# ─────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig2.suptitle('Structural Accelerations  —  Baseline vs MPC TF', fontsize=13)

axes[0].plot(t_b,  res_b['h_ddot'],  **BL)
axes[0].plot(t_tf, res_tf['h_ddot'], **TFMPC)
fmt(axes[0], 'ḧ [m/s²]', 'Heave acceleration')

axes[1].plot(t_b,  np.rad2deg(res_b['a_ddot']),  **BL)
axes[1].plot(t_tf, np.rad2deg(res_tf['a_ddot']), **TFMPC)
fmt(axes[1], 'α̈ [°/s²]', 'Pitch acceleration')

fig2.tight_layout()
fig2.savefig('mpc_fig2_accels.png', dpi=150)
print("[OK] mpc_fig2_accels.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 3 — Gust estimate vs true
# ─────────────────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(10, 4))
fig3.suptitle('Gust Velocity  —  True vs $C_L$-inversion estimate', fontsize=13)
ax.plot(t_tf, res_tf['W_gust'], **TRU)
ax.plot(t_tf, res_tf['W_hat'],  **HAT)
fmt(ax, '$W_{gust}$ [m/s]')
fig3.tight_layout()
fig3.savefig('mpc_fig3_gust.png', dpi=150)
print("[OK] mpc_fig3_gust.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 4 — Control input
# ─────────────────────────────────────────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig4.suptitle('Control Input  —  MPC TF', fontsize=13)

delta_tf = res_tf['delta']
axes[0].plot(t_tf, delta_tf, **TFMPC)
axes[0].axhline( 20, color='k', ls=':', lw=0.8, label='±20° sat.')
axes[0].axhline(-20, color='k', ls=':', lw=0.8)
fmt(axes[0], 'δ [°]', 'Flap deflection')

axes[1].plot(t_tf, np.gradient(delta_tf, t_tf), **TFMPC)
fmt(axes[1], 'δ̇ [°/s]', 'Flap rate')

fig4.tight_layout()
fig4.savefig('mpc_fig4_control.png', dpi=150)
print("[OK] mpc_fig4_control.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 5 — Aerodynamic coefficients
# ─────────────────────────────────────────────────────────────
fig5, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig5.suptitle('Aerodynamic Coefficients  —  Baseline vs MPC TF', fontsize=13)

axes[0].plot(t_b,  res_b['C_L'],  **BL)
axes[0].plot(t_tf, res_tf['C_L'], **TFMPC)
fmt(axes[0], '$C_L$', 'Lift coefficient')

axes[1].plot(t_b,  res_b['C_M'],  **BL)
axes[1].plot(t_tf, res_tf['C_M'], **TFMPC)
fmt(axes[1], '$C_M$', 'Pitching moment coefficient')

fig5.tight_layout()
fig5.savefig('mpc_fig5_aero.png', dpi=150)
print("[OK] mpc_fig5_aero.png")

#!/usr/bin/env python3
"""
Test script for MPC closed-loop controller.
Compares baseline (no control) vs MPC controlled response.

Plots:
  Figure 1 — Structural state: h, hd, a, ad  (baseline vs MPC)
  Figure 2 — Structural accelerations: h_ddot, a_ddot  (baseline vs MPC)
  Figure 3 — Gust: true W_gust vs EKF estimate W_hat
  Figure 4 — Control: delta, delta_dot
  Figure 5 — Aerodynamic coefficients: C_L, C_M  (baseline vs MPC)
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
U_INF = 80.0
T_END = 1.0
DT = 0.01
MPC_HORIZON = 10       # prediction horizon steps (0.1 s)

# GLA cost: W_hat now available in horizon via C_L inversion.
# State cost drives the optimizer (strong gradient); W_hat makes the
# predicted trajectory accurate (x evolves correctly under the gust).
# Baseline amplitudes: C_L~0.17, C_M~0.011, h~0.006m, a~0.012rad
# Scaling analysis: to cancel W=30 gust needs ~13° flap.
# State cost at baseline amplitude = 1.0 (normalized).
# Control cost must be << 1 to allow meaningful flap action.
# delta_max=20°, so R ~ 1/delta_max² = 1/400 gives unit cost at saturation.
# R=1/20² was too small → flap active but pitch blows up.
# Add strong Q_CM and Q_a to prevent pitch amplification.
# R = 1/10² is intermediate — allows ~10° deflection at unit cost.
# Weights: normalized so unit cost = baseline amplitude of each channel.
# R scaled so unit cost = 5° deflection (moderate authority).
# Q_CM and Q_a boosted to prevent pitch amplification seen at low R.
Q_CL  = 1.0 / 0.17**2        # lift load
Q_CM  = 10.0 / 0.011**2      # moment load (10×)
Q_h   = 1.0 / 0.006**2       # heave displacement
Q_a   = 5.0 / 0.012**2       # pitch angle (5× heavier)
Q_hd  = 0.0
Q_ad  = 0.0
R_mpc = 1.0 / 5.0**2         # unit cost at 5° — intermediate authority

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

print("Initializing MPC...")
mpc = MPCController(aero_model, U_INF, DT,
                    Q_CL=Q_CL, Q_CM=Q_CM,
                    Q_h=Q_h,   Q_a=Q_a,
                    Q_hd=Q_hd, Q_ad=Q_ad,
                    R=R_mpc, N=MPC_HORIZON, delta_max=20.0)

# ─────────────────────────────────────────────────────────────
# SIMULATIONS
# ─────────────────────────────────────────────────────────────
print(f"\nRunning BASELINE (no control)...")
res_b = run_mpc_simulation(U_INF, T_END, DT, aero_model, None, A_s, B_s, use_ekf=False)

print(f"Running MPC ORACLE (true W, true state — theoretical upper bound)...")
res_o = run_mpc_simulation(U_INF, T_END, DT, aero_model, mpc, A_s, B_s, use_ekf=False)

print(f"Running MPC (C_L inversion observer, W_hat in horizon)...")
res_m = run_mpc_simulation(U_INF, T_END, DT, aero_model, mpc, A_s, B_s, use_ekf=True)

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def amplitude(arr):
    return (arr.max() - arr.min()) / 2.0

metrics = {
    'h':      (amplitude(res_b['h']),      amplitude(res_o['h']),      amplitude(res_m['h'])),
    'a':      (amplitude(res_b['a']),      amplitude(res_o['a']),      amplitude(res_m['a'])),
    'h_ddot': (amplitude(res_b['h_ddot']), amplitude(res_o['h_ddot']), amplitude(res_m['h_ddot'])),
    'a_ddot': (amplitude(res_b['a_ddot']), amplitude(res_o['a_ddot']), amplitude(res_m['a_ddot'])),
    'C_L':    (amplitude(res_b['C_L']),    amplitude(res_o['C_L']),    amplitude(res_m['C_L'])),
    'C_M':    (amplitude(res_b['C_M']),    amplitude(res_o['C_M']),    amplitude(res_m['C_M'])),
}

print("\n── Amplitude comparison ──────────────────────────────────────────────")
print(f"{'':12s}  {'Baseline':>10s}  {'Oracle':>10s}  {'Red%':>7s}  {'MPC+Obs':>10s}  {'Red%':>7s}")
for name, (b, o, m) in metrics.items():
    red_o = (b - o) / b * 100 if b > 0 else 0.0
    red_m = (b - m) / b * 100 if b > 0 else 0.0
    print(f"  {name:<10s}  {b:10.5f}  {o:10.5f}  {red_o:+6.1f}%  {m:10.5f}  {red_m:+6.1f}%")

if np.any(np.isnan(res_m['h'])) or np.any(np.isnan(res_o['h'])):
    print("\n[WARNING] NaN detected in MPC result!")
else:
    print("\n[OK] No NaNs detected")

# ─────────────────────────────────────────────────────────────
# DERIVED QUANTITIES
# ─────────────────────────────────────────────────────────────
t   = res_m['t']
t_b = res_b['t']
t_o = res_o['t']
delta     = res_m['delta']
delta_dot = np.gradient(delta, t)
delta_o   = res_o['delta']

# ─────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────
BL  = dict(color='steelblue',  lw=1.5, alpha=0.85, label='Baseline')
ORC = dict(color='forestgreen',lw=1.5, alpha=0.85, label='MPC Oracle')
MPC = dict(color='tomato',     lw=1.5, alpha=0.85, label='MPC + $C_L$ observer')
TRU = dict(color='steelblue',  lw=1.5,             label='True $W_{gust}$')
HAT = dict(color='darkorange', lw=1.5, ls='--',    label='$\\hat{W}$ (C_L inversion)')

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
fig1.suptitle('Structural State  —  Baseline vs MPC', fontsize=13)

axes[0,0].plot(t_b, res_b['h'],  **BL)
axes[0,0].plot(t_o, res_o['h'],  **ORC)
axes[0,0].plot(t,   res_m['h'],  **MPC)
fmt(axes[0,0], 'h [m]', 'Heave displacement')

axes[0,1].plot(t_b, res_b['hd'], **BL)
axes[0,1].plot(t_o, res_o['hd'], **ORC)
axes[0,1].plot(t,   res_m['hd'], **MPC)
fmt(axes[0,1], 'ḣ [m/s]', 'Heave velocity')

axes[1,0].plot(t_b, np.rad2deg(res_b['a']),  **BL)
axes[1,0].plot(t_o, np.rad2deg(res_o['a']),  **ORC)
axes[1,0].plot(t,   np.rad2deg(res_m['a']),  **MPC)
fmt(axes[1,0], 'α [°]', 'Pitch angle')

axes[1,1].plot(t_b, np.rad2deg(res_b['ad']), **BL)
axes[1,1].plot(t_o, np.rad2deg(res_o['ad']), **ORC)
axes[1,1].plot(t,   np.rad2deg(res_m['ad']), **MPC)
fmt(axes[1,1], 'α̇ [°/s]', 'Pitch rate')

fig1.tight_layout()
fig1.savefig('mpc_fig1_state.png', dpi=150)
print("[OK] mpc_fig1_state.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 2 — Structural accelerations
# ─────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig2.suptitle('Structural Accelerations  —  Baseline vs MPC', fontsize=13)

axes[0].plot(t_b, res_b['h_ddot'], **BL)
axes[0].plot(t_o, res_o['h_ddot'], **ORC)
axes[0].plot(t,   res_m['h_ddot'], **MPC)
fmt(axes[0], 'ḧ [m/s²]', 'Heave acceleration')

axes[1].plot(t_b, np.rad2deg(res_b['a_ddot']), **BL)
axes[1].plot(t_o, np.rad2deg(res_o['a_ddot']), **ORC)
axes[1].plot(t,   np.rad2deg(res_m['a_ddot']), **MPC)
fmt(axes[1], 'α̈ [°/s²]', 'Pitch acceleration')

fig2.tight_layout()
fig2.savefig('mpc_fig2_accels.png', dpi=150)
print("[OK] mpc_fig2_accels.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 3 — Gust estimate vs true
# ─────────────────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(10, 4))
fig3.suptitle('Gust Velocity  —  True vs $C_L$-inversion estimate', fontsize=13)

ax.plot(t, res_m['W_gust'], **TRU)
ax.plot(t, res_m['W_hat'],  **HAT)
fmt(ax, '$W_{gust}$ [m/s]', legend=True)

fig3.tight_layout()
fig3.savefig('mpc_fig3_gust.png', dpi=150)
print("[OK] mpc_fig3_gust.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 4 — Control input
# ─────────────────────────────────────────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig4.suptitle('Control Input', fontsize=13)

axes[0].plot(t_o, delta_o, **ORC)
axes[0].plot(t,   delta,   **MPC)
axes[0].axhline( 20, color='k', ls=':', lw=0.8, label='±20° sat.')
axes[0].axhline(-20, color='k', ls=':', lw=0.8)
fmt(axes[0], 'δ [°]', 'Flap deflection', legend=True)

axes[1].plot(t_o, np.gradient(delta_o, t_o), **ORC)
axes[1].plot(t,   delta_dot,                  **MPC)
fmt(axes[1], 'δ̇ [°/s]', 'Flap deflection rate', legend=True)

fig4.tight_layout()
fig4.savefig('mpc_fig4_control.png', dpi=150)
print("[OK] mpc_fig4_control.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 5 — Aerodynamic coefficients
# ─────────────────────────────────────────────────────────────
fig5, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig5.suptitle('Aerodynamic Coefficients  —  Baseline vs MPC', fontsize=13)

axes[0].plot(t_b, res_b['C_L'], **BL)
axes[0].plot(t_o, res_o['C_L'], **ORC)
axes[0].plot(t,   res_m['C_L'], **MPC)
fmt(axes[0], '$C_L$', 'Lift coefficient')

axes[1].plot(t_b, res_b['C_M'], **BL)
axes[1].plot(t_o, res_o['C_M'], **ORC)
axes[1].plot(t,   res_m['C_M'], **MPC)
fmt(axes[1], '$C_M$', 'Pitching moment coefficient')

fig5.tight_layout()
fig5.savefig('mpc_fig5_aero.png', dpi=150)
print("[OK] mpc_fig5_aero.png")

#plt.show()

#!/usr/bin/env python3
"""
Test script for LQG closed-loop controller.
"""
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from structural.smd import get_space_state_matrices
from control.lqg import design_lqr, design_Kalman, run_lqg_simulation
from aerodynamics.model import LDNetModel as AeroModel

# Test parameters
U_INF = 80.0
T_END = 10  # short test
DT = 0.01

# Load structural model
print("Loading structural state-space matrices...")
A_s, B_s, C_s, D_s = get_space_state_matrices()
print(f"A_s shape: {A_s.shape}")
print(f"B_s shape: {B_s.shape}")
print(f"C_s shape: {C_s.shape}")

# Design LQR
print("\nDesigning LQR...")
Q = np.diag([100.0, 1.0, 100.0, 1.0])  # penalize positions more than velocities
R = np.array([[1.0]])  # penalize control effort
K = design_lqr(A_s, B_s, Q, R)
print(f"LQR gain K: {K}")

# Design Kalman
print("\nDesigning Kalman filter...")
Q_noise = np.eye(4) * 0.01   # process noise covariance (4×4)
R_noise = np.eye(2) * 0.1    # measurement noise covariance (2×2)
L = design_Kalman(A_s, Q_noise, R_noise)
print(f"Kalman gain L shape: {L.shape}")

# Load aerodynamic model
print("\nLoading aerodynamic model...")
models_dir = Path(__file__).parent.parent / 'models'
print(f"Looking for models in: {models_dir}")
print(f"Exists: {models_dir.exists()}")
if not models_dir.exists():
    # Try alternative path (models is outside src)
    models_dir = Path(__file__).parent.parent.parent / 'models'
    print(f"Trying alternative: {models_dir}")
    print(f"Exists: {models_dir.exists()}")
aero_model = AeroModel(str(models_dir))
print(f"LDNet loaded with {aero_model.num_latent_states} latent states")

# Run simulation
print(f"\nRunning LQG simulation (T_END={T_END}s, DT={DT}s)...")
result = run_lqg_simulation(U_INF, T_END, DT, aero_model, K, L, A_s, B_s)

print(f"Simulation complete!")
print(f"  h (heave):     min={result['h'].min():.6f}, max={result['h'].max():.6f} m")
print(f"  a (pitch):     min={result['a'].min():.6f}, max={result['a'].max():.6f} rad")
print(f"  delta (flap):  min={result['delta'].min():.2f}, max={result['delta'].max():.2f}°")
print(f"  C_L (lift):    min={result['C_L'].min():.6f}, max={result['C_L'].max():.6f}")

# Verify no NaNs
if np.any(np.isnan(result['h'])):
    print("\n[WARNING] NaN detected in heave displacement!")
else:
    print("\n[OK] No NaNs detected")

# Plot results
print("\nGenerating plots...")
import matplotlib.pyplot as plt

t = result['t']
h = result['h']
hd = result['hd']
a = result['a']
ad = result['ad']
delta = result['delta']
C_L = result['C_L']
C_M = result['C_M']
h_ddot = result['h_ddot']
a_ddot = result['a_ddot']

# Compute delta derivatives numerically
delta_dot = np.gradient(delta, t)
delta_ddot = np.gradient(delta_dot, t)

fig, axes = plt.subplots(4, 2, figsize=(14, 10))
fig.suptitle(f'LQG Control Results (U_INF={U_INF} m/s, T_END={T_END}s, r=0.01)', fontsize=14)

# Row 0: C_L and C_M
axes[0, 0].plot(t, C_L, 'b-', linewidth=1.5)
axes[0, 0].set_ylabel('$C_L$', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('Lift Coefficient')

axes[0, 1].plot(t, C_M, 'r-', linewidth=1.5)
axes[0, 1].set_ylabel('$C_M$', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('Moment Coefficient')

# Row 1: h and hd
axes[1, 0].plot(t, h, 'g-', linewidth=1.5)
axes[1, 0].set_ylabel('h [m]', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('Heave Displacement')

axes[1, 1].plot(t, hd, 'm-', linewidth=1.5)
axes[1, 1].set_ylabel(r'$\dot{h}$ [m/s]', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('Heave Velocity')

# Row 2: a and ad
axes[2, 0].plot(t, a, 'c-', linewidth=1.5)
axes[2, 0].set_ylabel('α [rad]', fontsize=11)
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_title('Pitch Angle')

axes[2, 1].plot(t, ad, 'orange', linewidth=1.5)
axes[2, 1].set_ylabel(r'$\dot{\alpha}$ [rad/s]', fontsize=11)
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].set_title('Pitch Angular Velocity')

# Row 3: delta and delta_dot
axes[3, 0].plot(t, delta, 'k-', linewidth=1.5)
axes[3, 0].axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Limit ±20°')
axes[3, 0].axhline(y=-20, color='r', linestyle='--', alpha=0.5)
axes[3, 0].set_ylabel('δ [°]', fontsize=11)
axes[3, 0].set_xlabel('Time [s]', fontsize=11)
axes[3, 0].grid(True, alpha=0.3)
axes[3, 0].set_title('Flap Deflection')
axes[3, 0].legend()

axes[3, 1].plot(t, delta_dot, 'purple', linewidth=1.5)
axes[3, 1].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Limit ±100°/s')
axes[3, 1].axhline(y=-100, color='r', linestyle='--', alpha=0.5)
axes[3, 1].set_ylabel(r'$\dot{\delta}$ [deg/s]', fontsize=11)
axes[3, 1].set_xlabel('Time [s]', fontsize=11)
axes[3, 1].grid(True, alpha=0.3)
axes[3, 1].set_title('Flap Deflection Rate')
axes[3, 1].legend()

plt.tight_layout()
plt.savefig('lqg_results.png', dpi=150)
print("[OK] Plot saved as 'lqg_results.png'")

import numpy as np
import matplotlib.pyplot as plt
from control.gla import optimize_gla

# Esegui ottimizzazione
print("Running GLA optimization...")
delta_opt, C_L_arr, C_M_arr, h_traj, alpha_traj, U_INF, T_END, DT = optimize_gla()

# Crea array dei tempi
t_win = np.linspace(0.0, T_END, int(T_END/DT))

# Calcola derivate numeriche di delta
delta_dot = np.zeros_like(delta_opt)
delta_ddot = np.zeros_like(delta_opt)

for i in range(1, len(delta_opt)):
    delta_dot[i] = (delta_opt[i] - delta_opt[i-1]) / DT

for i in range(1, len(delta_dot)):
    delta_ddot[i] = (delta_dot[i] - delta_dot[i-1]) / DT

# Estrai h_dot e alpha_dot da h_traj e alpha_traj (se sono array 2D con [pos, vel])
if h_traj.ndim == 2:
    h_vals = h_traj[0, :]
    h_dot_vals = h_traj[1, :]
else:
    h_vals = h_traj
    h_dot_vals = np.gradient(h_traj, DT)

if alpha_traj.ndim == 2:
    alpha_vals = alpha_traj[0, :]
    alpha_dot_vals = alpha_traj[1, :]
else:
    alpha_vals = alpha_traj
    alpha_dot_vals = np.gradient(alpha_traj, DT)

# Crea figura con subplots
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle(f'GLA Control Results (U_INF={U_INF} m/s, T_END={T_END} s, DT={DT} s)', fontsize=14)

# 1. C_L vs time
axes[0, 0].plot(t_win, C_L_arr, 'b-', linewidth=1.5)
axes[0, 0].set_xlabel('Time [s]')
axes[0, 0].set_ylabel('C_L')
axes[0, 0].set_title('Lift Coefficient')
axes[0, 0].grid(True, alpha=0.3)

# 2. C_M vs time
axes[0, 1].plot(t_win, C_M_arr, 'r-', linewidth=1.5)
axes[0, 1].set_xlabel('Time [s]')
axes[0, 1].set_ylabel('C_M')
axes[0, 1].set_title('Pitching Moment Coefficient')
axes[0, 1].grid(True, alpha=0.3)

# 3. h (heave) vs time
axes[1, 0].plot(t_win, h_vals, 'g-', linewidth=1.5)
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_ylabel('h [m]')
axes[1, 0].set_title('Heave Displacement')
axes[1, 0].grid(True, alpha=0.3)

# 4. h_dot (heave velocity) vs time
axes[1, 1].plot(t_win, h_dot_vals, 'm-', linewidth=1.5)
axes[1, 1].set_xlabel('Time [s]')
axes[1, 1].set_ylabel('h_dot [m/s]')
axes[1, 1].set_title('Heave Velocity')
axes[1, 1].grid(True, alpha=0.3)

# 5. alpha (pitch angle) vs time
axes[2, 0].plot(t_win, alpha_vals, 'c-', linewidth=1.5)
axes[2, 0].set_xlabel('Time [s]')
axes[2, 0].set_ylabel('α [rad]')
axes[2, 0].set_title('Pitch Angle')
axes[2, 0].grid(True, alpha=0.3)

# 6. alpha_dot (pitch angular velocity) vs time
axes[2, 1].plot(t_win, alpha_dot_vals, 'orange', linewidth=1.5)
axes[2, 1].set_xlabel('Time [s]')
axes[2, 1].set_ylabel('α_dot [rad/s]')
axes[2, 1].set_title('Pitch Angular Velocity')
axes[2, 1].grid(True, alpha=0.3)

# 7. delta (flap deflection) vs time
axes[3, 0].plot(t_win, delta_opt, 'k-', linewidth=1.5, label='δ')
axes[3, 0].axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Max limit')
axes[3, 0].axhline(y=-20, color='r', linestyle='--', alpha=0.5, label='Min limit')
axes[3, 0].set_xlabel('Time [s]')
axes[3, 0].set_ylabel('δ [°]')
axes[3, 0].set_title('Flap Deflection')
axes[3, 0].legend()
axes[3, 0].grid(True, alpha=0.3)

# 8. delta_dot (flap velocity) and delta_ddot (flap acceleration) vs time
axes[3, 1].plot(t_win, delta_dot, 'b-', linewidth=1.5, label='δ_dot')
axes[3, 1].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Max rate limit')
axes[3, 1].axhline(y=-100, color='r', linestyle='--', alpha=0.5, label='Min rate limit')
axes[3, 1].set_xlabel('Time [s]')
axes[3, 1].set_ylabel('δ_dot [°/s]')
axes[3, 1].set_title('Flap Deflection Rate')
axes[3, 1].legend()
axes[3, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gla_results.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'gla_results.png'")
plt.show()

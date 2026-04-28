#!/usr/bin/env python3
"""
LQG diagnostics — points 5 and 6.

Point 5: comparison of observer estimation quality and closed-loop
         performance for the reference 1-cosine gust
         (W0=60 m/s, T_g=1 s, U_inf=75 m/s).

Point 6: sensitivity of W estimation to Q_kf[W,W] / Q_kf[struct].

Figures are saved to src/ (same directory as test_mpc.py figures).
Style matches test_mpc.py: steelblue/darkcyan palette, lw=1.5, alpha=0.85,
grid alpha=0.25, legend fontsize=8, suptitle fontsize=13.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aerodynamics.model import LDNetModel
from control.lqr import LQRController, compute_jacobians
from control.kalman import ExtendedKalmanFilter, build_C_obs, \
                           _Q_KF_DIAG_DEFAULT, _R_KF_DIAG_DEFAULT
from control.mpc import run_mpc_simulation
from structural.smd import get_space_state_matrices

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
U_INF    = 75.0
T_END    = 3.0
DT       = 0.01
LAMBDA_W = 0.98
GUST_W0  = 60.0
GUST_TG  = 1.0

# LQR weights (same as test_mpc.py)
Q_H   = 1.0  / 0.00428**2
Q_A   = 10.0 / 0.00823**2
R_LQR = np.array([[1.0 / 2.0**2]])

# Plot style — matches test_mpc.py
C_BL   = dict(color='steelblue',  lw=1.5, alpha=0.85)   # baseline (open-loop)
C_TRUE = dict(color='steelblue',  lw=1.5, alpha=0.85)   # LQR + true_state
C_HEUR = dict(color='darkcyan',   lw=1.5, alpha=0.85)   # LQR + heuristic
C_EKF  = dict(color='crimson',    lw=1.5, alpha=0.85)   # LQR + EKF


def fmt(ax, ylabel, title=None, legend=True):
    ax.set_xlabel('t [s]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(fontsize=8)


# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────
print("Loading model...")
models_dir = Path(__file__).parent.parent / 'models'
aero_model = LDNetModel(str(models_dir))

z_trim = np.zeros(aero_model.num_latent_states)
for _ in range(200):
    z_trim, CL_trim, CM_trim = aero_model.step(
        z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)
CL_trim = float(CL_trim)
CM_trim = float(CM_trim)

A_s, B_s, _, _ = get_space_state_matrices()

print("Building LQR...")
Q_lqr = np.diag([Q_H, 0., Q_A, 0., 0.])
lqr = LQRController(aero_model, U_INF, DT,
                    x_trim=np.zeros(4), z_trim=z_trim,
                    Q_lqr=Q_lqr, R_lqr=R_LQR,
                    CL_trim=CL_trim, CM_trim=CM_trim,
                    Q_w=1. / 10.**2, lambda_w=LAMBDA_W, delta_max=20.0)

print("Building EKF...")
A_d, B_d, C_y, D_y, B_w = compute_jacobians(
    aero_model, np.zeros(4), z_trim, U_INF, DT, CL_trim, CM_trim)
A_aug = np.block([[A_d,              B_w.reshape(5, 1)        ],
                  [np.zeros((1, 5)), np.array([[LAMBDA_W]])  ]])
C_obs = build_C_obs(C_y, U_INF)
xi_trim = np.concatenate([np.zeros(4), z_trim, [0.0]])

ekf = ExtendedKalmanFilter.from_augmented(
    aero_model, U_INF, DT, C_obs, A_aug, xi_trim, lambda_w=LAMBDA_W)


def gust(t):
    if 0.0 <= t <= GUST_TG:
        return (GUST_W0 / 2.) * (1. - np.cos(2. * np.pi * t / GUST_TG))
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATIONS
# ──────────────────────────────────────────────────────────────────────────────
print("\nRunning simulations...")

res_bl = run_mpc_simulation(          # open-loop baseline
    U_INF, T_END, DT, aero_model, None, A_s, B_s,
    gust_profile=gust, observer='true_state')

res_true = run_mpc_simulation(        # LQR + oracle
    U_INF, T_END, DT, aero_model, lqr, A_s, B_s,
    gust_profile=gust, observer='true_state')

res_heur = run_mpc_simulation(        # LQR + heuristic
    U_INF, T_END, DT, aero_model, lqr, A_s, B_s,
    gust_profile=gust, observer='heuristic', use_aoa_sensor=True)

res_ekf = run_mpc_simulation(         # LQR + EKF
    U_INF, T_END, DT, aero_model, lqr, A_s, B_s,
    gust_profile=gust, observer='ekf', kalman_filter=ekf)

t = res_true['t']
mask = t <= (GUST_TG + 0.5)

# ──────────────────────────────────────────────────────────────────────────────
# FIG 1 — State estimation: truth vs heuristic vs EKF
# ──────────────────────────────────────────────────────────────────────────────
state_keys   = ['h',    'hd',     'a',    'ad',       'z_hat', 'W_hat']
state_labels = ['h [m]', 'ḣ [m/s]', 'α [rad]', 'α̇ [rad/s]', 'z [—]', 'W [m/s]']
state_titles = ['Heave', 'Heave velocity', 'Pitch', 'Pitch rate',
                'Latent state z', 'Gust velocity W']

fig1, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
fig1.suptitle('State estimation  —  heuristic vs EKF', fontsize=13)

for ax, key, ylabel, title in zip(axes.ravel(), state_keys, state_labels, state_titles):
    truth_arr = res_true['W_gust'] if key == 'W_hat' else res_true[key]

    if key in ('h', 'hd', 'a', 'ad'):
        heur_est = res_heur[key + '_hat']
        ekf_est  = res_ekf[key + '_hat']
    else:
        heur_est = res_heur[key]
        ekf_est  = res_ekf[key]

    ax.plot(t, truth_arr, label='True',      **C_TRUE)
    ax.plot(t, heur_est,  label='Heuristic', **C_HEUR)
    ax.plot(t, ekf_est,   label='EKF',       **C_EKF)
    fmt(ax, ylabel, title)

fig1.tight_layout()
fig1.savefig('lqg_fig1_estimation.png', dpi=150)
print("[OK] lqg_fig1_estimation.png")

# ──────────────────────────────────────────────────────────────────────────────
# TABLE — estimation error
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 66)
print("  ESTIMATION ERROR TABLE  (gust window 0 – %.1fs)" % (GUST_TG + 0.5))
print("=" * 66)

truth_map = {'h': res_true['h'], 'hd': res_true['hd'],
             'a': res_true['a'], 'ad': res_true['ad'],
             'z_hat': res_true['z_hat'], 'W_hat': res_true['W_gust']}
heur_map  = {'h': res_heur['h_hat'], 'hd': res_heur['hd_hat'],
             'a': res_heur['a_hat'], 'ad': res_heur['ad_hat'],
             'z_hat': res_heur['z_hat'], 'W_hat': res_heur['W_hat']}
ekf_map   = {'h': res_ekf['h_hat'],  'hd': res_ekf['hd_hat'],
             'a': res_ekf['a_hat'],  'ad': res_ekf['ad_hat'],
             'z_hat': res_ekf['z_hat'],  'W_hat': res_ekf['W_hat']}

row_names = ['h [m]', 'hd [m/s]', 'a [rad]', 'ad [rad/s]', 'z [—]', 'W [m/s]']
keys_ord  = ['h', 'hd', 'a', 'ad', 'z_hat', 'W_hat']

print(f"  {'State':<12}  {'Heur RMS':>10}  {'Heur peak':>10}  "
      f"{'EKF RMS':>10}  {'EKF peak':>10}")
print("  " + "-" * 58)
for name, key in zip(row_names, keys_ord):
    tr = truth_map[key][mask]
    eh = heur_map[key][mask] - tr
    ee = ekf_map[key][mask]  - tr
    print(f"  {name:<12}  "
          f"{np.sqrt(np.mean(eh**2)):>10.4e}  {np.max(np.abs(eh)):>10.4e}  "
          f"{np.sqrt(np.mean(ee**2)):>10.4e}  {np.max(np.abs(ee)):>10.4e}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# FIG 2 — Closed-loop performance: baseline / true / heuristic / EKF
# ──────────────────────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
fig2.suptitle('Closed-loop performance  —  baseline / LQR observers', fontsize=13)

delta_bl   = np.zeros(len(t))          # open-loop: δ=0
delta_true = -res_true['delta']
delta_heur = -res_heur['delta']
delta_ekf  = -res_ekf['delta']

# h_ddot
axes[0, 0].plot(t, res_bl['h_ddot'],   label='Baseline',       **C_BL)
axes[0, 0].plot(t, res_true['h_ddot'], label='LQR+true_state', **C_TRUE)
axes[0, 0].plot(t, res_heur['h_ddot'], label='LQR+heuristic',  **C_HEUR)
axes[0, 0].plot(t, res_ekf['h_ddot'],  label='LQR+EKF',        **C_EKF)
fmt(axes[0, 0], 'ḧ [m/s²]', 'Heave acceleration')

# alpha
axes[0, 1].plot(t, np.rad2deg(res_bl['a']),   label='Baseline',       **C_BL)
axes[0, 1].plot(t, np.rad2deg(res_true['a']), label='LQR+true_state', **C_TRUE)
axes[0, 1].plot(t, np.rad2deg(res_heur['a']), label='LQR+heuristic',  **C_HEUR)
axes[0, 1].plot(t, np.rad2deg(res_ekf['a']),  label='LQR+EKF',        **C_EKF)
fmt(axes[0, 1], 'α [°]', 'Pitch angle')

# delta
axes[1, 0].plot(t, delta_true, label='LQR+true_state', **C_TRUE)
axes[1, 0].plot(t, delta_heur, label='LQR+heuristic',  **C_HEUR)
axes[1, 0].plot(t, delta_ekf,  label='LQR+EKF',        **C_EKF)
axes[1, 0].axhline( 20, color='k', ls=':', lw=0.8, label='±20° sat.')
axes[1, 0].axhline(-20, color='k', ls=':', lw=0.8)
fmt(axes[1, 0], 'δ [°]  (+ = flap down)', 'Flap deflection')

# delta_dot
ddt_true = np.gradient(delta_true, t)
ddt_heur = np.gradient(delta_heur, t)
ddt_ekf  = np.gradient(delta_ekf,  t)
axes[1, 1].plot(t, ddt_true, **C_TRUE)
axes[1, 1].plot(t, ddt_heur, **C_HEUR)
axes[1, 1].plot(t, ddt_ekf,  **C_EKF)
fmt(axes[1, 1], 'δ̇ [°/s]', 'Flap rate', legend=False)

fig2.tight_layout()
fig2.savefig('lqg_fig2_performance.png', dpi=150)
print("[OK] lqg_fig2_performance.png")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 3 — Aerodynamic coefficients: baseline / heuristic / EKF
# ──────────────────────────────────────────────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig3.suptitle('Aerodynamic Coefficients  —  baseline / LQR observers', fontsize=13)

axes[0].plot(t, res_bl['C_L'],   label='Baseline',       **C_BL)
axes[0].plot(t, res_true['C_L'], label='LQR+true_state', **C_TRUE)
axes[0].plot(t, res_heur['C_L'], label='LQR+heuristic',  **C_HEUR)
axes[0].plot(t, res_ekf['C_L'],  label='LQR+EKF',        **C_EKF)
fmt(axes[0], '$C_L$', 'Lift coefficient')

axes[1].plot(t, res_bl['C_M'],   label='Baseline',       **C_BL)
axes[1].plot(t, res_true['C_M'], label='LQR+true_state', **C_TRUE)
axes[1].plot(t, res_heur['C_M'], label='LQR+heuristic',  **C_HEUR)
axes[1].plot(t, res_ekf['C_M'],  label='LQR+EKF',        **C_EKF)
fmt(axes[1], '$C_M$', 'Pitching moment coefficient')

fig3.tight_layout()
fig3.savefig('lqg_fig3_aero.png', dpi=150)
print("[OK] lqg_fig3_aero.png")

# ──────────────────────────────────────────────────────────────────────────────
# TABLE — closed-loop performance
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 66)
print("  CLOSED-LOOP PERFORMANCE TABLE  (gust window 0 – %.1fs)" % (GUST_TG + 0.5))
print("=" * 66)
print(f"  {'Metric':<22}  {'baseline':>10}  {'true_state':>10}  "
      f"{'heuristic':>10}  {'EKF':>10}")
print("  " + "-" * 66)


def peak(arr, mask): return np.max(np.abs(arr[mask]))
def rms(arr, mask):  return np.sqrt(np.mean(arr[mask]**2))


for res in (res_bl, res_true, res_heur, res_ekf):
    res['a_deg']  = np.rad2deg(np.abs(res['a']))
    res['ddelta'] = np.abs(np.gradient(-res['delta'], t))

metrics = [
    ('peak |ḧ| [m/s²]', 'h_ddot', peak),
    ('peak |α| [°]',     'a_deg',  peak),
    ('peak |C_L|',       'C_L',    peak),
    ('peak |C_M|',       'C_M',    peak),
    ('RMS δ [°]',        'delta',  rms),
    ('peak |δ̇| [°/s]',  'ddelta', peak),
]

for mname, key, fn in metrics:
    vbl = fn(res_bl[key],   mask)
    vt  = fn(res_true[key], mask)
    vh  = fn(res_heur[key], mask)
    ve  = fn(res_ekf[key],  mask)
    print(f"  {mname:<22}  {vbl:>10.4f}  {vt:>10.4f}  {vh:>10.4f}  {ve:>10.4f}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# FIG 4 — Sensitivity: Q_kf[W,W] sweep (EKF, W estimation)
# ──────────────────────────────────────────────────────────────────────────────
print("Running Q_kf[W,W] sensitivity sweep (EKF)...")

Q_struct_base = _Q_KF_DIAG_DEFAULT[0]
ratios    = np.logspace(-2, 2, 7)
Q_w_vals  = Q_struct_base * ratios
palette   = plt.cm.viridis(np.linspace(0.1, 0.9, len(Q_w_vals)))

fig4, ax4 = plt.subplots(figsize=(10, 5))
fig4.suptitle('Sensitivity: Q_kf[W,W] / Q_kf[struct]  —  gust estimation (EKF)',
              fontsize=13)

for q_w, color in zip(Q_w_vals, palette):
    Q_diag = _Q_KF_DIAG_DEFAULT.copy()
    Q_diag[5] = q_w
    ekf_s = ExtendedKalmanFilter.from_augmented(
        aero_model, U_INF, DT, C_obs, A_aug, xi_trim,
        Q_kf=np.diag(Q_diag),
        R_kf=np.diag(_R_KF_DIAG_DEFAULT),
        lambda_w=LAMBDA_W)
    res_s = run_mpc_simulation(
        U_INF, T_END, DT, aero_model, lqr, A_s, B_s,
        gust_profile=gust, observer='ekf', kalman_filter=ekf_s)
    ax4.plot(t, res_s['W_hat'],
             color=color, lw=1.3, alpha=0.85,
             label=f'ratio={q_w / Q_struct_base:.2g}')

ax4.plot(t, res_true['W_gust'], color='k', lw=2.0, ls='--', label='True W')
ax4.set_xlabel('t [s]')
ax4.set_ylabel('Estimated W [m/s]')
ax4.set_title('Higher ratio → more process noise on W → faster tracking',
              fontsize=9)
ax4.grid(True, alpha=0.25)
ax4.legend(fontsize=7, ncol=2)

sm = plt.cm.ScalarMappable(
    cmap='viridis',
    norm=plt.Normalize(vmin=np.log10(ratios[0]), vmax=np.log10(ratios[-1])))
sm.set_array([])
cbar = fig4.colorbar(sm, ax=ax4, pad=0.01)
cbar.set_label('log₁₀(Q_w / Q_struct)', fontsize=9)

fig4.tight_layout()
fig4.savefig('lqg_fig4_sensitivity.png', dpi=150)
print("[OK] lqg_fig4_sensitivity.png")

print("\nAll diagnostics complete.")

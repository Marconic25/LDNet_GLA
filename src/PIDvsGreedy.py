#!/usr/bin/env python3
"""
Comparison script: Open Loop / PID / Greedy-N1.

Phase 1: uses linear Theodorsen model as both the true system and inside the
         Greedy controller.  observer='true_state' (ideal sensors).
         No NN / TensorFlow dependency in this phase.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from structural.smd import get_space_state_matrices
from control.run_controller import run_simulation
from control.pid import PIDController
from control.greedy import GreedyN1Controller
from control.linear_aero import predict as linear_predict


class LinearAeroModel:
    """
    Wrapper that exposes the quasi-steady Theodorsen linear aerodynamic model
    through the same interface as LDNetModel, so it can be used as the 'true
    system' in run_simulation() without any neural-network dependency.

    The linear model has no latent state (num_latent_states = 0).
    """
    num_latent_states = 0

    def step(self, z, h, hd, a, ad, delta, W_gust, U_inf, dt):
        """
        Stateless step: ignores z/dt, returns (z_unchanged, C_L, C_M).

        Parameters match LDNetModel.step signature exactly.
        """
        x = [h, hd, a, ad]
        C_L, C_M = linear_predict(x, delta, W_gust, U_inf)
        return z, float(C_L), float(C_M)


# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
U_INF = 80.0
T_END = 3.0
DT    = 0.01

GUST_W0  = 60.0   # peak gust [m/s]
GUST_DUR = 1.0    # gust duration [s]

DELTA_MAX     = 20.0    # [°]
DELTA_DOT_MAX = 100.0   # [°/s]

# ─────────────────────────────────────────────────────────────
# OBJECTIVE (PID only):
#   Minimise peak(|h|) in gust window [0, GUST_DUR+0.5s]
#   Subject to:
#     - peak(|α|) <= OL peak (soft penalty if exceeded)
#     - No growing oscillations after gust (stability over full T_END)
# ─────────────────────────────────────────────────────────────
_OPT_T_END = GUST_DUR + 0.5   # gust evaluation window

# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────
print("Using linear Theodorsen model as true aerodynamic system (Phase 1).")
aero_model = LinearAeroModel()
print(f"  Linear aero: {aero_model.num_latent_states} latent state(s) (stateless)")

A_s, B_s, _, _ = get_space_state_matrices()

def gust_profile(t):
    if 0.0 <= t <= GUST_DUR:
        return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / GUST_DUR))
    return 0.0

# Open-loop reference — computed over full T_END to capture post-gust peaks too
print("Running Open Loop for normalisation...")
_res_ol_ref = run_simulation(U_INF, T_END, DT, aero_model, None, A_s, B_s,
                             gust_profile=gust_profile, observer='true_state')
_gust_mask  = (_res_ol_ref['t'] >= 0.0) & (_res_ol_ref['t'] <= _OPT_T_END)
_post_mask  = (_res_ol_ref['t'] > _OPT_T_END)
_OL_PEAK_H      = max(np.max(np.abs(_res_ol_ref['h'][_gust_mask])), 1e-8)
_OL_PEAK_A      = max(np.max(np.abs(_res_ol_ref['a'][_gust_mask])), 1e-8)
_OL_PEAK_H_POST = max(np.max(np.abs(_res_ol_ref['h'][_post_mask])), 1e-8) if _post_mask.any() else _OL_PEAK_H
_OL_PEAK_A_POST = max(np.max(np.abs(_res_ol_ref['a'][_post_mask])), 1e-8) if _post_mask.any() else _OL_PEAK_A


def _pid_objective(res):
    """
    Minimise peak(|h|) in gust window.
    Soft penalty:  peak(|α|) exceeds OL during gust      → controller weakened α damping
    Stability penalty: post-gust h or α exceed OL peaks  → controller excited oscillations
    """
    gust_mask = (res['t'] >= 0.0) & (res['t'] <= _OPT_T_END)
    post_mask = (res['t'] > _OPT_T_END)

    peak_h      = np.max(np.abs(res['h'][gust_mask]))
    peak_a_gust = np.max(np.abs(res['a'][gust_mask]))
    peak_h_post = np.max(np.abs(res['h'][post_mask])) if post_mask.any() else 0.0
    peak_a_post = np.max(np.abs(res['a'][post_mask])) if post_mask.any() else 0.0

    obj = peak_h / _OL_PEAK_H

    if peak_a_gust > _OL_PEAK_A:
        obj += 10.0 * (peak_a_gust / _OL_PEAK_A - 1.0)

    if peak_h_post > _OL_PEAK_H_POST:
        obj += 10.0 * (peak_h_post / _OL_PEAK_H_POST - 1.0)

    if peak_a_post > _OL_PEAK_A_POST:
        obj += 10.0 * (peak_a_post / _OL_PEAK_A_POST - 1.0)

    return obj

# ─────────────────────────────────────────────────────────────
# CONTROLLERS — gain-optimised for minimum peak(h) + peak(α)
# ─────────────────────────────────────────────────────────────
from scipy.optimize import minimize

def _run_pid(params):
    # params = [Kp_h, Kd_h, log(Kp_a), log(Kd_a)]
    # h-gains are free (can be negative — needed by sign convention)
    # α-gains stay positive (log-space)
    Kp_h, Kd_h = params[0], params[1]
    Kp_a, Kd_a = np.exp(params[2]), np.exp(params[3])
    ctrl = PIDController(Kp_h=Kp_h, Kd_h=Kd_h, Kp_a=Kp_a, Kd_a=Kd_a,
                         delta_max=DELTA_MAX, delta_dot_max=DELTA_DOT_MAX, DT=DT)
    res = run_simulation(U_INF, T_END, DT, aero_model, ctrl, A_s, B_s,
                         gust_profile=gust_profile, observer='true_state')
    return _pid_objective(res)

def _run_greedy(log_params):
    Q_h, Q_a, R = np.exp(log_params)
    ctrl = GreedyN1Controller(linear_predict, U_INF=U_INF, DT=DT,
                              Q_h=Q_h, Q_a=Q_a, R=R,
                              delta_max=DELTA_MAX, delta_dot_max=DELTA_DOT_MAX)
    res = run_simulation(U_INF, T_END, DT, aero_model, ctrl, A_s, B_s,
                         gust_profile=gust_profile, observer='true_state')
    return _pid_objective(res)

# h-gains free (signed), α-gains in log-space (positive)
# x0: Kp_h=-50 (negative: h<0→δ>0), Kd_h=-10, log(Kp_a)=log(30), log(Kd_a)=log(5)
print("\nOptimising PID gains...")
_x0_pid = np.array([-50.0, -10.0, np.log(30.0), np.log(5.0)])
pid_opt = minimize(_run_pid, x0=_x0_pid,
                   method='Nelder-Mead',
                   options={'maxiter': 800, 'xatol': 0.05, 'fatol': 1e-5})
Kp_h_opt, Kd_h_opt = pid_opt.x[0], pid_opt.x[1]
Kp_a_opt, Kd_a_opt = np.exp(pid_opt.x[2]), np.exp(pid_opt.x[3])
print(f"  Kp_h={Kp_h_opt:.1f}  Kd_h={Kd_h_opt:.1f}  Kp_a={Kp_a_opt:.1f}  Kd_a={Kd_a_opt:.1f}  obj={pid_opt.fun:.4f}")

print("Optimising Greedy weights...")
greedy_opt = minimize(_run_greedy,
                      x0=np.log([1/0.004**2, 1/0.008**2, 1/10.0**2]),
                      method='Nelder-Mead',
                      options={'maxiter': 600, 'xatol': 0.05, 'fatol': 1e-5})
Q_h_opt, Q_a_opt, R_opt = np.exp(greedy_opt.x)
print(f"  Q_h={Q_h_opt:.1f}  Q_a={Q_a_opt:.1f}  R={R_opt:.5f}  obj={greedy_opt.fun:.4f}")

pid = PIDController(Kp_h=Kp_h_opt, Kd_h=Kd_h_opt,
                    Kp_a=Kp_a_opt, Kd_a=Kd_a_opt,
                    delta_max=DELTA_MAX, delta_dot_max=DELTA_DOT_MAX, DT=DT)

greedy = GreedyN1Controller(linear_predict, U_INF=U_INF, DT=DT,
                             Q_h=Q_h_opt, Q_a=Q_a_opt, R=R_opt,
                             delta_max=DELTA_MAX, delta_dot_max=DELTA_DOT_MAX)

# ─────────────────────────────────────────────────────────────
# SIMULATIONS
# ─────────────────────────────────────────────────────────────
print("\nRunning Open Loop...")
res_ol = run_simulation(U_INF, T_END, DT, aero_model, None, A_s, B_s,
                        gust_profile=gust_profile, observer='true_state')

print("Running PID...")
pid.reset()
res_pid = run_simulation(U_INF, T_END, DT, aero_model, pid, A_s, B_s,
                         gust_profile=gust_profile, observer='true_state')

print("Running Greedy-N1...")
greedy.reset()
res_g = run_simulation(U_INF, T_END, DT, aero_model, greedy, A_s, B_s,
                       gust_profile=gust_profile, observer='true_state')

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def amplitude(arr):
    return (arr.max() - arr.min()) / 2.0

_W_END = GUST_DUR + 0.5

def amp_window(res, t_end):
    mask = (res['t'] >= 0.0) & (res['t'] <= t_end)
    return {k: amplitude(res[k][mask])
            for k in ('h', 'a', 'h_ddot', 'a_ddot', 'C_L', 'C_M')}

def actuation_energy(res):
    return float(np.sum(res['delta']**2) * DT)

g_ol  = amp_window(res_ol,  _W_END)
g_pid = amp_window(res_pid, _W_END)
g_g   = amp_window(res_g,   _W_END)

print(f"\n── Gust window [0 – {_W_END:.1f}s] ─────────────────────────────────────────")
print(f"{'':12s}  {'OL':>10s}  {'PID':>10s}  {'PID Red%':>9s}  {'Greedy':>10s}  {'Gr Red%':>8s}")
for name in ('h', 'a', 'h_ddot', 'a_ddot', 'C_L', 'C_M'):
    ol, p, gr = g_ol[name], g_pid[name], g_g[name]
    r_pid = (ol - p)  / ol * 100 if ol > 0 else 0.0
    r_gr  = (ol - gr) / ol * 100 if ol > 0 else 0.0
    print(f"  {name:<10s}  {ol:10.5f}  {p:10.5f}  {r_pid:+8.1f}%  {gr:10.5f}  {r_gr:+7.1f}%")

print(f"\n  Actuation energy (Σδ²·dt):")
print(f"    OL:     {actuation_energy(res_ol):.4f}")
print(f"    PID:    {actuation_energy(res_pid):.4f}")
print(f"    Greedy: {actuation_energy(res_g):.4f}")

# Sanity checks
for name, res in [('PID', res_pid), ('Greedy', res_g)]:
    assert np.all(np.abs(res['delta']) <= DELTA_MAX + 1e-6), f"{name}: delta exceeds {DELTA_MAX}°"
    print(f"  [{name}] max |δ| = {np.max(np.abs(res['delta'])):.2f}°  ✓")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / 'results' / 'pid_vs_greedy'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OL  = dict(color='steelblue',  lw=1.5, alpha=0.85, label='Open loop')
PID = dict(color='darkorange',  lw=1.5, alpha=0.85, label='PID')
GR  = dict(color='crimson',    lw=1.5, alpha=0.85, label='Greedy-N1')

def fmt(ax, ylabel, title=None):
    ax.set_xlabel('t [s]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

t = res_ol['t']

# Fig 1 — Structural state
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
fig.suptitle('Structural State  —  OL / PID / Greedy-N1', fontsize=13)
axes[0,0].plot(t, res_ol['h'],  **OL)
axes[0,0].plot(t, res_pid['h'], **PID)
axes[0,0].plot(t, res_g['h'],   **GR)
fmt(axes[0,0], 'h [m]', 'Heave displacement')
axes[0,1].plot(t, res_ol['hd'],  **OL)
axes[0,1].plot(t, res_pid['hd'], **PID)
axes[0,1].plot(t, res_g['hd'],   **GR)
fmt(axes[0,1], 'ḣ [m/s]', 'Heave velocity')
axes[1,0].plot(t, np.rad2deg(res_ol['a']),  **OL)
axes[1,0].plot(t, np.rad2deg(res_pid['a']), **PID)
axes[1,0].plot(t, np.rad2deg(res_g['a']),   **GR)
fmt(axes[1,0], 'α [°]', 'Pitch angle')
axes[1,1].plot(t, np.rad2deg(res_ol['ad']),  **OL)
axes[1,1].plot(t, np.rad2deg(res_pid['ad']), **PID)
axes[1,1].plot(t, np.rad2deg(res_g['ad']),   **GR)
fmt(axes[1,1], 'α̇ [°/s]', 'Pitch rate')
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig1_state.png', dpi=150)
print(f"\n[OK] {OUT_DIR / 'fig1_state.png'}")

# Fig 2 — Control input
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig.suptitle('Control Input  —  OL / PID / Greedy-N1', fontsize=13)
axes[0].plot(t, res_ol['delta'],  **OL)
axes[0].plot(t, res_pid['delta'], **PID)
axes[0].plot(t, res_g['delta'],   **GR)
axes[0].axhline( DELTA_MAX, color='k', ls=':', lw=0.8, label=f'±{DELTA_MAX}°')
axes[0].axhline(-DELTA_MAX, color='k', ls=':', lw=0.8)
fmt(axes[0], 'δ [°]', 'Flap deflection')
delta_rate_pid = np.gradient(res_pid['delta'], t)
delta_rate_g   = np.gradient(res_g['delta'],   t)
axes[1].plot(t, np.zeros_like(t), **OL)
axes[1].plot(t, delta_rate_pid,   **PID)
axes[1].plot(t, delta_rate_g,     **GR)
fmt(axes[1], 'δ̇ [°/s]', 'Flap rate')
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig2_control.png', dpi=150)
print(f"[OK] {OUT_DIR / 'fig2_control.png'}")

# Fig 3 — Aerodynamic coefficients
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig.suptitle('Aerodynamic Coefficients  —  OL / PID / Greedy-N1', fontsize=13)
axes[0].plot(t, res_ol['C_L'],  **OL)
axes[0].plot(t, res_pid['C_L'], **PID)
axes[0].plot(t, res_g['C_L'],   **GR)
fmt(axes[0], '$C_L$', 'Lift coefficient')
axes[1].plot(t, res_ol['C_M'],  **OL)
axes[1].plot(t, res_pid['C_M'], **PID)
axes[1].plot(t, res_g['C_M'],   **GR)
fmt(axes[1], '$C_M$', 'Pitching moment coefficient')
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig3_aero.png', dpi=150)
print(f"[OK] {OUT_DIR / 'fig3_aero.png'}")

# Fig 4 — Accelerations
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig.suptitle('Structural Accelerations  —  OL / PID / Greedy-N1', fontsize=13)
axes[0].plot(t, res_ol['h_ddot'],  **OL)
axes[0].plot(t, res_pid['h_ddot'], **PID)
axes[0].plot(t, res_g['h_ddot'],   **GR)
fmt(axes[0], 'ḧ [m/s²]', 'Heave acceleration')
axes[1].plot(t, np.rad2deg(res_ol['a_ddot']),  **OL)
axes[1].plot(t, np.rad2deg(res_pid['a_ddot']), **PID)
axes[1].plot(t, np.rad2deg(res_g['a_ddot']),   **GR)
fmt(axes[1], 'α̈ [°/s²]', 'Pitch acceleration')
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig4_accels.png', dpi=150)
print(f"[OK] {OUT_DIR / 'fig4_accels.png'}")

plt.show()

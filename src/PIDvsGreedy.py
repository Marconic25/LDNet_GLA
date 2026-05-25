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

PID_KP_H = 50.0    # proportional gain on h
PID_KD_H = 10.0    # derivative gain on ḣ
PID_KP_A = 30.0    # proportional gain on α
PID_KD_A = 5.0     # derivative gain on α̇

GREEDY_Q_H = 1.0 / 0.004**2   # weight on h  (normalised by expected peak)
GREEDY_Q_A = 1.0 / 0.008**2   # weight on α
GREEDY_R   = 1.0 / 20.0**2    # weight on δ — lower R → more actuation

DELTA_MAX     = 20.0    # [°]
DELTA_DOT_MAX = 100.0   # [°/s]

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

# ─────────────────────────────────────────────────────────────
# CONTROLLERS
# ─────────────────────────────────────────────────────────────
pid = PIDController(
    Kp_h=PID_KP_H, Kd_h=PID_KD_H,
    Kp_a=PID_KP_A, Kd_a=PID_KD_A,
    delta_max=DELTA_MAX, delta_dot_max=DELTA_DOT_MAX, DT=DT
)

greedy = GreedyN1Controller(
    aero_predict=linear_predict,
    U_INF=U_INF, DT=DT,
    Q_h=GREEDY_Q_H, Q_a=GREEDY_Q_A, R=GREEDY_R,
    delta_max=DELTA_MAX, delta_dot_max=DELTA_DOT_MAX
)

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

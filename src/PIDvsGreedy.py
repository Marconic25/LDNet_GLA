#!/usr/bin/env python3
"""
Gust Load Alleviation — Phase 1 comparison: Open Loop vs PD baseline vs Greedy-N1.

Baseline controller (offline, no aero model, no gust feedforward):
    δ(t) = Kp_h*h + Kd_h*ḣ + Kp_α*α + Kd_α*α̇

Gains optimised offline (Nelder-Mead) to minimise peak C_L + damp post-gust
oscillations. The baseline has NO knowledge of W — it can only react after
the structure moves.

Greedy-N1: uses W_hat from heuristic observer + linear aero model to predict
C_L one step ahead and choose δ optimally.

Convention: δ > 0 = TE down = C_L increases.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))

from structural.smd import get_space_state_matrices
from control.run_controller import run_simulation
from control.linear_aero import predict as linear_predict


# ─────────────────────────────────────────────────────────────
# TRUE AERO SYSTEM  (linear Theodorsen, Phase 1)
# ─────────────────────────────────────────────────────────────
class LinearAeroModel:
    num_latent_states = 0

    def step(self, z, h, hd, a, ad, delta, W_gust, U_inf, dt):
        C_L, C_M = linear_predict([h, hd, a, ad], delta, W_gust, U_inf)
        return z, float(C_L), float(C_M)


# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
U_INF         = 80.0    # freestream [m/s]
T_END         = 5.0     # simulation duration [s]
DT            = 0.01    # timestep [s]
GUST_W0       = 10.0    # peak gust [m/s]  — well below saturation
GUST_DUR      = 1.0     # gust duration [s]
DELTA_MAX         = 20.0   # flap saturation [°]
DELTA_DOT_MAX     = 100.0  # flap rate limit [°/s]  — used by Greedy
PD_DELTA_DOT_MAX  = 10.0   # slower rate for PD baseline — avoids exciting structural resonance

def gust_profile(t):
    if 0.0 <= t <= GUST_DUR:
        return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / GUST_DUR))
    return 0.0




# ─────────────────────────────────────────────────────────────
# BASELINE CONTROLLER  (offline — no aero model, no gust feedforward)
#   δ = Kp_h*h + Kd_h*ḣ + Kp_α*α + Kd_α*α̇
# Pure structural state feedback; W_hat is accepted but ignored.
# ─────────────────────────────────────────────────────────────
class PDController:
    def __init__(self, Kp_h=0., Kd_h=0., Kp_a=0., Kd_a=0.,
                 delta_max=20., delta_dot_max=100., DT=0.01):
        self.Kp_h = float(Kp_h);  self.Kd_h = float(Kd_h)
        self.Kp_a = float(Kp_a);  self.Kd_a = float(Kd_a)
        self.delta_max     = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)
        self.DT            = float(DT)
        self._delta_prev   = 0.0

    def solve(self, x_hat, z_hat, W_hat=0.0):  # W_hat ignored — no aero model
        h, hd, a, ad = float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), float(x_hat[3])
        delta_raw = (self.Kp_h * h  + self.Kd_h * hd
                   + self.Kp_a * a  + self.Kd_a * ad)

        delta_dot = (delta_raw - self._delta_prev) / self.DT
        if abs(delta_dot) > self.delta_dot_max:
            delta_raw = self._delta_prev + np.sign(delta_dot) * self.delta_dot_max * self.DT

        delta = float(np.clip(delta_raw, -self.delta_max, self.delta_max))
        self._delta_prev = delta
        return delta

    def reset(self):
        self._delta_prev = 0.0


# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────
aero_model = LinearAeroModel()
A_s, B_s, _, _ = get_space_state_matrices()

# Open-loop reference
print("\nRunning Open Loop reference...")
res_ol = run_simulation(U_INF, T_END, DT, aero_model, None, A_s, B_s,
                        gust_profile=gust_profile, observer='heuristic')

_gust_mask = (res_ol['t'] >= 0.0) & (res_ol['t'] <= GUST_DUR + 0.5)
_post_mask = (res_ol['t'] > GUST_DUR + 0.5)
_OL_PEAK_CL     = np.max(res_ol['C_L'][_gust_mask])
_OL_PEAK_H      = np.max(np.abs(res_ol['h'][_gust_mask]))
_OL_PEAK_A      = np.max(np.abs(res_ol['a'][_gust_mask]))
_OL_PEAK_H_POST = np.max(np.abs(res_ol['h'][_post_mask])) if _post_mask.any() else _OL_PEAK_H
_OL_PEAK_A_POST = np.max(np.abs(res_ol['a'][_post_mask])) if _post_mask.any() else _OL_PEAK_A
_OL_RMS_H_POST  = (np.sqrt(np.mean(res_ol['h'][_post_mask]**2))
                   if _post_mask.any() else _OL_PEAK_H) or 1e-6
_OL_RMS_A_POST  = (np.sqrt(np.mean(res_ol['a'][_post_mask]**2))
                   if _post_mask.any() else _OL_PEAK_A) or 1e-6


# ─────────────────────────────────────────────────────────────
# OBJECTIVE: minimise peak C_L + damp post-gust oscillations
# ─────────────────────────────────────────────────────────────
def _objective(res):
    gust_mask = (res['t'] >= 0.0) & (res['t'] <= GUST_DUR + 0.5)
    post_mask = (res['t'] > GUST_DUR + 0.5)

    peak_CL = np.max(res['C_L'][gust_mask])
    obj = peak_CL / _OL_PEAK_CL

    if post_mask.any():
        rms_h_post = np.sqrt(np.mean(res['h'][post_mask]**2))
        rms_a_post = np.sqrt(np.mean(res['a'][post_mask]**2))

        # Penalise post-gust energy exceeding open-loop level
        if rms_h_post > _OL_RMS_H_POST:
            obj += 20.0 * (rms_h_post / _OL_RMS_H_POST - 1.0)
        if rms_a_post > _OL_RMS_A_POST:
            obj += 20.0 * (rms_a_post / _OL_RMS_A_POST - 1.0)

    return obj


def _run(params):
    # params = [Kp_h, Kd_h, Kp_a, Kd_a]  — all free (signed)
    Kp_h, Kd_h, Kp_a, Kd_a = params
    ctrl = PDController(Kp_h=Kp_h, Kd_h=Kd_h, Kp_a=Kp_a, Kd_a=Kd_a,
                        delta_max=DELTA_MAX, delta_dot_max=PD_DELTA_DOT_MAX, DT=DT)
    res = run_simulation(U_INF, T_END, DT, aero_model, ctrl, A_s, B_s,
                         gust_profile=gust_profile, observer='heuristic')
    return _objective(res)


# ─────────────────────────────────────────────────────────────
# OPTIMISE PD GAINS  [Kp_h, Kd_h, Kp_a, Kd_a]
# ─────────────────────────────────────────────────────────────
# Physics: during gust h<0, hd<0 → need δ<0 (TE up) → Kp_h>0, Kd_h>0
# Keep gains conservative to avoid exciting structural resonance (~2 Hz, ζ≈2%).
# RMS post-gust penalty prevents the optimizer from finding unstable high-gain solutions.
print("Optimising PD gains...")
opt = minimize(_run, x0=[50., 10., 0., 0.],
               method='Nelder-Mead',
               options={'maxiter': 2000, 'xatol': 0.005, 'fatol': 1e-5,
                        'initial_simplex': np.array([
                            [ 50.,  10.,   0.,   0. ],
                            [100.,  10.,   0.,   0. ],
                            [ 50.,  20.,   0.,   0. ],
                            [ 50.,  10.,   5.,   0. ],
                            [ 50.,  10.,   0.,   1. ],
                        ])})
Kp_h_opt, Kd_h_opt, Kp_a_opt, Kd_a_opt = opt.x
print(f"  Kp_h={Kp_h_opt:.2f}  Kd_h={Kd_h_opt:.2f}  "
      f"Kp_α={Kp_a_opt:.2f}  Kd_α={Kd_a_opt:.2f}  obj={opt.fun:.4f}")


# ─────────────────────────────────────────────────────────────
# FINAL SIMULATIONS
# ─────────────────────────────────────────────────────────────
pd_ctrl = PDController(Kp_h=Kp_h_opt, Kd_h=Kd_h_opt,
                       Kp_a=Kp_a_opt, Kd_a=Kd_a_opt,
                       delta_max=DELTA_MAX, delta_dot_max=PD_DELTA_DOT_MAX, DT=DT)

print("Running PD baseline...")
pd_ctrl.reset()
res_pd = run_simulation(U_INF, T_END, DT, aero_model, pd_ctrl, A_s, B_s,
                        gust_profile=gust_profile, observer='heuristic')

# ─────────────────────────────────────────────────────────────
# GREEDY-N1 CONTROLLER  (online, uses aero model + W_hat)
# cost: Q_CL*C_L(k+1)² + Q_h*h(k+1)² + Q_a*α(k+1)² + R*δ²
# Tuned on same objective as baseline for fair comparison.
# ─────────────────────────────────────────────────────────────
from control.greedy import GreedyN1Controller
from control.linear_aero import predict as linear_predict

def _run_greedy(log_params):
    Q_CL, Q_h, Q_a, R = np.exp(log_params)
    ctrl = GreedyN1Controller(linear_predict, U_INF=U_INF, DT=DT,
                              Q_h=Q_h, Q_a=Q_a, Q_CL=Q_CL, R=R,
                              delta_max=DELTA_MAX, delta_dot_max=DELTA_DOT_MAX)
    res = run_simulation(U_INF, T_END, DT, aero_model, ctrl, A_s, B_s,
                         gust_profile=gust_profile, observer='heuristic')
    return _objective(res)

print("Optimising Greedy weights...")
greedy_opt = minimize(_run_greedy,
                      x0=np.log([1e3, 1e3, 1e3, 1.0]),
                      method='Nelder-Mead',
                      options={'maxiter': 1500, 'xatol': 0.05, 'fatol': 1e-5})
Q_CL_opt, Q_h_opt, Q_a_opt, R_opt = np.exp(greedy_opt.x)
print(f"  Q_CL={Q_CL_opt:.1f}  Q_h={Q_h_opt:.1f}  Q_a={Q_a_opt:.1f}  "
      f"R={R_opt:.4f}  obj={greedy_opt.fun:.4f}")

greedy_ctrl = GreedyN1Controller(linear_predict, U_INF=U_INF, DT=DT,
                                  Q_h=Q_h_opt, Q_a=Q_a_opt, Q_CL=Q_CL_opt, R=R_opt,
                                  delta_max=DELTA_MAX, delta_dot_max=DELTA_DOT_MAX)
print("Running Greedy-N1...")
greedy_ctrl.reset()
res_gr = run_simulation(U_INF, T_END, DT, aero_model, greedy_ctrl, A_s, B_s,
                        gust_profile=gust_profile, observer='heuristic')


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def peak_window(res, t_end):
    mask = (res['t'] >= 0.0) & (res['t'] <= t_end)
    return {k: np.max(np.abs(res[k][mask]))
            for k in ('h', 'a', 'h_ddot', 'C_L', 'C_M')}

_W_END = GUST_DUR + 0.5
p_ol = peak_window(res_ol, _W_END)
p_pd = peak_window(res_pd, _W_END)
p_gr = peak_window(res_gr, _W_END)

print(f"\n── Peak in gust window [0–{_W_END:.1f}s] ──────────────────────────────────────")
print(f"{'':10s}  {'OL':>8s}  {'PD base':>8s}  {'Red%':>7s}  {'Greedy':>8s}  {'Red%':>7s}")
for name in ('C_L', 'h_ddot', 'h', 'a', 'C_M'):
    ol = p_ol[name]; pd = p_pd[name]; gr = p_gr[name]
    r_pd = (ol - pd) / ol * 100 if ol > 0 else 0.
    r_gr = (ol - gr) / ol * 100 if ol > 0 else 0.
    print(f"  {name:<8s}  {ol:8.4f}  {pd:8.4f}  {r_pd:+6.1f}%  {gr:8.4f}  {r_gr:+6.1f}%")

print(f"  [PD base] max |δ| = {np.max(np.abs(res_pd['delta'])):.2f}°")
print(f"  [Greedy]  max |δ| = {np.max(np.abs(res_gr['delta'])):.2f}°")


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / 'results' / 'pid_vs_greedy'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OL = dict(color='steelblue',  lw=1.5, alpha=0.85, label='Open loop')
PD = dict(color='darkorange', lw=1.5, alpha=0.85, label='PD baseline')
GR = dict(color='crimson',    lw=1.5, alpha=0.85, label='Greedy-N1')

def fmt(ax, ylabel, title=None):
    ax.set_xlabel('t [s]'); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    if title: ax.set_title(title, fontsize=10)

t = res_ol['t']

# Fig 1 — Primary load metrics
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig.suptitle(f'GLA — OL / PD baseline / Greedy-N1   (W0={GUST_W0} m/s, U={U_INF} m/s)', fontsize=13)
axes[0].plot(t, res_ol['C_L'],   **OL)
axes[0].plot(t, res_pd['C_L'],   **PD)
axes[0].plot(t, res_gr['C_L'],   **GR)
fmt(axes[0], '$C_L$ [-]', 'Lift coefficient')
axes[1].plot(t, res_ol['h_ddot'],  **OL)
axes[1].plot(t, res_pd['h_ddot'],  **PD)
axes[1].plot(t, res_gr['h_ddot'],  **GR)
fmt(axes[1], 'ḧ [m/s²]', 'Heave acceleration')
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig1_loads.png', dpi=150)
print(f"\n[OK] {OUT_DIR / 'fig1_loads.png'}")

# Fig 2 — Structural state
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
fig.suptitle('Structural State', fontsize=13)
axes[0,0].plot(t, res_ol['h'],              **OL)
axes[0,0].plot(t, res_pd['h'],              **PD)
axes[0,0].plot(t, res_gr['h'],              **GR)
fmt(axes[0,0], 'h [m]', 'Heave')
axes[0,1].plot(t, res_ol['hd'],             **OL)
axes[0,1].plot(t, res_pd['hd'],             **PD)
axes[0,1].plot(t, res_gr['hd'],             **GR)
fmt(axes[0,1], 'ḣ [m/s]', 'Heave velocity')
axes[1,0].plot(t, np.rad2deg(res_ol['a']),  **OL)
axes[1,0].plot(t, np.rad2deg(res_pd['a']),  **PD)
axes[1,0].plot(t, np.rad2deg(res_gr['a']),  **GR)
fmt(axes[1,0], 'α [°]', 'Pitch')
axes[1,1].plot(t, np.rad2deg(res_ol['ad']),  **OL)
axes[1,1].plot(t, np.rad2deg(res_pd['ad']),  **PD)
axes[1,1].plot(t, np.rad2deg(res_gr['ad']),  **GR)
fmt(axes[1,1], 'α̇ [°/s]', 'Pitch rate')
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig2_state.png', dpi=150)
print(f"[OK] {OUT_DIR / 'fig2_state.png'}")

# Fig 3 — Control + gust estimation
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig.suptitle('Control Input & Gust Estimation', fontsize=13)
axes[0].plot(t, res_pd['delta'], **PD)
axes[0].plot(t, res_gr['delta'], **GR)
axes[0].axhline( DELTA_MAX, color='k', ls=':', lw=0.8, label=f'±{DELTA_MAX}°')
axes[0].axhline(-DELTA_MAX, color='k', ls=':', lw=0.8)
fmt(axes[0], 'δ [°]', 'Flap deflection')
axes[1].plot(t, res_ol['W_gust'], color='gray',    lw=1.5, label='W gust (true)')
axes[1].plot(t, res_gr['W_hat'],  color='crimson', lw=1.5, ls='--', label='W_hat (Greedy observer)')
fmt(axes[1], 'W [m/s]', 'Gust vs W_hat (Greedy only)')
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig3_control.png', dpi=150)
print(f"[OK] {OUT_DIR / 'fig3_control.png'}")

plt.show()

#!/usr/bin/env python3
"""
Monte Carlo robustness analysis: LQR vs open loop.
Varies gust amplitude W0, duration T, and freestream U_INF.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from structural.smd import get_space_state_matrices
from control.mpc import run_mpc_simulation
from control.lqr import LQRController
from aerodynamics.model import LDNetModel as AeroModel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────
# FIXED PARAMETERS
# ─────────────────────────────────────────────────────────────
U_INF_DESIGN = 75.0   # LQR linearization point
DT           = 0.01
T_END        = 3.0
N_SAMPLES    = 200
RNG_SEED     = 42

# LQR weights (same as test_mpc.py)
Q_h   = 1.0  / 0.00428**2
Q_a   = 10.0 / 0.00823**2
R_lqr = np.array([[1.0 / 2.0**2]])

# Monte Carlo ranges
W0_MIN, W0_MAX   = 20.0,  80.0   # gust amplitude [m/s]
T_MIN,  T_MAX    = 0.5,   2.0    # gust duration [s]
U_MIN,  U_MAX    = 60.0,  90.0   # freestream [m/s]

# ─────────────────────────────────────────────────────────────
# SETUP — load model and design LQR at trim U_INF_DESIGN
# ─────────────────────────────────────────────────────────────
print("Loading model...")
A_s, B_s, C_s, D_s = get_space_state_matrices()
models_dir = Path(__file__).parent.parent / 'models'
if not models_dir.exists():
    models_dir = Path(__file__).parent.parent.parent / 'models'
aero_model = AeroModel(str(models_dir))

print("Computing trim at design point...")
_z_trim = np.zeros(aero_model.num_latent_states)
for _ in range(200):
    _z_trim, _CL_trim, _CM_trim = aero_model.step(
        _z_trim, 0., 0., 0., 0., 0., 0., U_INF_DESIGN, DT)
print(f"  C_L_trim={float(_CL_trim):.5f}, C_M_trim={float(_CM_trim):.5f}")

print("Designing LQR...")
Q_lqr = np.diag([Q_h, 0.0, Q_a, 0.0, 0.0])
lqr = LQRController(aero_model, U_INF_DESIGN, DT,
                    x_trim=np.zeros(4), z_trim=_z_trim,
                    Q_lqr=Q_lqr, R_lqr=R_lqr,
                    CL_trim=float(_CL_trim), CM_trim=float(_CM_trim),
                    Q_w=1.0/10.0**2, lambda_w=0.98, delta_max=20.0)

# ─────────────────────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)
W0_samples  = rng.uniform(W0_MIN, W0_MAX, N_SAMPLES)
T_samples   = rng.uniform(T_MIN,  T_MAX,  N_SAMPLES)
U_samples   = rng.uniform(U_MIN,  U_MAX,  N_SAMPLES)

def amp_peak(res, t_end):
    mask = res['t'] <= t_end
    def amp(x): return (x[mask].max() - x[mask].min()) / 2.0
    return {k: amp(res[k]) for k in ('h_ddot', 'a_ddot', 'C_L', 'C_M')}

results = []
print(f"\nRunning {N_SAMPLES} Monte Carlo samples...")
for s in range(N_SAMPLES):
    W0 = W0_samples[s]
    T  = T_samples[s]
    U  = U_samples[s]
    t_window = T + 0.5

    def gust(t, _W0=W0, _T=T):
        if 0.0 <= t <= _T:
            return (_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / _T))
        return 0.0

    rb = run_mpc_simulation(U, T_END, DT, aero_model, None, A_s, B_s,
                            use_ekf=False, gust_profile=gust)
    rl = run_mpc_simulation(U, T_END, DT, aero_model, lqr, A_s, B_s,
                            use_ekf=True, gust_profile=gust, use_aoa_sensor=True)

    ab = amp_peak(rb, t_window)
    al = amp_peak(rl, t_window)

    def red(b, l): return (b - l) / b * 100 if b > 1e-10 else 0.0

    results.append({
        'W0': W0, 'T': T, 'U': U,
        'r_hddot': red(ab['h_ddot'], al['h_ddot']),
        'r_addot': red(ab['a_ddot'], al['a_ddot']),
        'r_CL':    red(ab['C_L'],    al['C_L']),
        'r_CM':    red(ab['C_M'],    al['C_M']),
        'CL_peak_b':  rb['C_L'][rb['t'] <= t_window].max(),
        'CL_peak_l':  rl['C_L'][rl['t'] <= t_window].max(),
    })

    if (s + 1) % 20 == 0:
        r_hdd = np.mean([r['r_hddot'] for r in results])
        r_cl  = np.mean([r['r_CL']    for r in results])
        print(f"  [{s+1:3d}/{N_SAMPLES}]  mean h_ddot red={r_hdd:+.1f}%  CL red={r_cl:+.1f}%")

# ─────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────
r_hddot = np.array([r['r_hddot'] for r in results])
r_addot = np.array([r['r_addot'] for r in results])
r_CL    = np.array([r['r_CL']    for r in results])
r_CM    = np.array([r['r_CM']    for r in results])
W0_arr  = np.array([r['W0']      for r in results])
T_arr   = np.array([r['T']       for r in results])
U_arr   = np.array([r['U']       for r in results])

print(f"\n── Monte Carlo Summary ({N_SAMPLES} samples) ────────────────────────────────")
print(f"{'Metric':12s}  {'Mean':>8s}  {'Std':>8s}  {'5th pct':>8s}  {'95th pct':>8s}  {'% >0':>6s}")
for name, arr in [('h_ddot', r_hddot), ('α̈', r_addot), ('C_L', r_CL), ('C_M', r_CM)]:
    print(f"  {name:<10s}  {np.mean(arr):>+7.1f}%  {np.std(arr):>7.1f}%  "
          f"{np.percentile(arr,5):>+7.1f}%  {np.percentile(arr,95):>+7.1f}%  "
          f"{np.mean(arr>0)*100:>5.0f}%")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f'Monte Carlo Robustness — LQR vs Open Loop  (N={N_SAMPLES})', fontsize=13)

def scatter_with_fit(ax, x, y, xlabel, ylabel, title):
    sc = ax.scatter(x, y, c=y, cmap='RdYlGn', vmin=-100, vmax=100,
                    s=15, alpha=0.7)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.axhline(np.mean(y), color='navy', lw=1.2, ls='-', label=f'mean={np.mean(y):+.1f}%')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    return sc

scatter_with_fit(axes[0,0], W0_arr, r_hddot, 'W₀ [m/s]', 'Red% ḧ',    'ḧ reduction vs W₀')
scatter_with_fit(axes[0,1], T_arr,  r_hddot, 'T [s]',    'Red% ḧ',    'ḧ reduction vs T')
scatter_with_fit(axes[0,2], U_arr,  r_hddot, 'U [m/s]',  'Red% ḧ',    'ḧ reduction vs U')
scatter_with_fit(axes[1,0], W0_arr, r_CL,    'W₀ [m/s]', 'Red% C_L',  'C_L reduction vs W₀')
scatter_with_fit(axes[1,1], T_arr,  r_CL,    'T [s]',    'Red% C_L',  'C_L reduction vs T')
scatter_with_fit(axes[1,2], U_arr,  r_CL,    'U [m/s]',  'Red% C_L',  'C_L reduction vs U')

fig.tight_layout()
fig.savefig('mc_scatter.png', dpi=150)
print("[OK] mc_scatter.png")

# Violin plots
fig2, axes2 = plt.subplots(1, 4, figsize=(14, 5))
fig2.suptitle('Distribution of Load Reduction — Monte Carlo', fontsize=13)
for ax, arr, label in zip(axes2,
                          [r_hddot, r_addot, r_CL, r_CM],
                          ['ḧ red%', 'α̈ red%', 'C_L red%', 'C_M red%']):
    vp = ax.violinplot(arr, showmedians=True, showextrema=True)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_ylabel('Reduction [%]')
    ax.set_title(label, fontsize=10)
    ax.set_xticks([])
    ax.grid(True, alpha=0.2, axis='y')
    ax.text(1.0, np.mean(arr), f'  mean={np.mean(arr):+.1f}%', va='center', fontsize=8)
    for pc in vp['bodies']:
        pc.set_facecolor('steelblue'); pc.set_alpha(0.6)
fig2.tight_layout()
fig2.savefig('mc_violin.png', dpi=150)
print("[OK] mc_violin.png")

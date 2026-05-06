#!/usr/bin/env python3
"""
Observer upgrade diagnostics — Tasks 6 and 7.

Compares three observer modes on the GLA closed-loop system:
  'legacy'    : existing kalman.py DARE-gain EKF + C_L bisection (W≥0)
  'ekf'       : NonlinearEKF (ekf_augmented.py) without C_L inversion
  'ekf_clinv' : NonlinearEKF + C_L inversion pseudo-measure (full scheme)

Task 6  —  nominal 1-cosine gust  W₀=60 m/s, T_g=1 s
  · fig1: state estimation series (6 panels, 3 curves each: true/legacy/new)
  · fig2: closed-loop performance (2×2: ḧ, α, δ, δ̇)
  · fig3: aerodynamic coefficients C_L, C_M
  · fig4: W estimation comparison — 4 curves overlaid
  · estimation error table and performance table

Task 7  —  stress tests (no retuning)
  · W₀ ∈ {30, 90, 120} m/s, T_g=1 s
  · T_g ∈ {0.5, 2.0} s, W₀=60 m/s
  · Ramp gust: 0→60 m/s in 0.5 s, hold 0.5 s, 60→0 in 0.5 s

Style: identical to lqg_diagnostics.py — steelblue/darkcyan/crimson/purple,
       lw=1.5, alpha=0.85, grid alpha=0.25, legend fontsize=8, suptitle fontsize=13.

All figures saved to  results/observer_upgrade/
"""
import sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
SRC_DIR    = Path(__file__).parent
MODELS_DIR = SRC_DIR.parent / 'models'
OUT_DIR    = SRC_DIR.parent / 'results' / 'observer_upgrade'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from aerodynamics.model    import LDNetModel
from control.lqr           import LQRController, compute_jacobians
from control.kalman        import ExtendedKalmanFilter, build_C_obs
from control.ekf_augmented import NonlinearEKF
from control.cl_inversion  import CLInversionFusion
from control.mpc           import run_mpc_simulation
from structural.smd        import get_space_state_matrices

# ── simulation parameters ─────────────────────────────────────────────────────
U_INF    = 75.0
T_END    = 3.0
DT       = 0.01
LAMBDA_W = 0.98
GUST_W0  = 60.0
GUST_TG  = 1.0

Q_H   = 1.0  / 0.00428**2
Q_A   = 10.0 / 0.00823**2
R_LQR = np.array([[1.0 / 2.0**2]])

# ── plot style (identical to lqg_diagnostics.py) ──────────────────────────────
C_BL     = dict(color='steelblue',  lw=1.5, alpha=0.85)   # open-loop baseline
C_TRUE   = dict(color='steelblue',  lw=1.5, alpha=0.85)   # LQR + true state
C_LEGACY = dict(color='darkcyan',   lw=1.5, alpha=0.85)   # legacy DARE-EKF
C_EKF    = dict(color='crimson',    lw=1.5, alpha=0.85)   # NonlinearEKF
C_CLINV  = dict(color='purple',     lw=1.5, alpha=0.85)   # NonlinearEKF+CLInv


def fmt(ax, ylabel, title=None, legend=True):
    ax.set_xlabel('t [s]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(fontsize=8)


def peak(arr, mask): return np.max(np.abs(arr[mask]))
def rms(arr, mask):  return np.sqrt(np.mean(arr[mask]**2))


# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  Observer Upgrade Diagnostics")
print("=" * 70)

print("\nLoading LDNet model...")
aero_model = LDNetModel(str(MODELS_DIR))

z_trim = np.zeros(aero_model.num_latent_states)
for _ in range(200):
    z_trim, CL_trim, CM_trim = aero_model.step(
        z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)
CL_trim = float(CL_trim); CM_trim = float(CM_trim)
print(f"  Trim: z={z_trim}, CL={CL_trim:.4f}, CM={CM_trim:.4f}")

A_s, B_s, _, _ = get_space_state_matrices()

print("\nBuilding LQR...")
Q_lqr = np.diag([Q_H, 0., Q_A, 0., 0.])
lqr   = LQRController(
    aero_model, U_INF, DT,
    x_trim=np.zeros(4), z_trim=z_trim,
    Q_lqr=Q_lqr, R_lqr=R_LQR,
    CL_trim=CL_trim, CM_trim=CM_trim,
    Q_w=1./10.**2, lambda_w=LAMBDA_W, delta_max=20.0)

print("\nBuilding legacy EKF (DARE gain)...")
A_d, B_d, C_y, D_y, B_w = compute_jacobians(
    aero_model, np.zeros(4), z_trim, U_INF, DT, CL_trim, CM_trim)
A_aug   = np.block([[A_d,              B_w.reshape(5, 1)       ],
                    [np.zeros((1, 5)), np.array([[LAMBDA_W]])  ]])
C_obs   = build_C_obs(C_y, U_INF)
xi_trim = np.concatenate([np.zeros(4), z_trim, [0.0]])
ekf_legacy = ExtendedKalmanFilter.from_augmented(
    aero_model, U_INF, DT, C_obs, A_aug, xi_trim, lambda_w=LAMBDA_W)

print("\nBuilding NonlinearEKF (online AD Jacobians)...")
nl_ekf = NonlinearEKF(aero_model, U_INF, DT, xi_trim, lambda_w=LAMBDA_W)

print("\nBuilding NonlinearEKF + CLInversionFusion...")
nl_ekf_clinv = NonlinearEKF(aero_model, U_INF, DT, xi_trim, lambda_w=LAMBDA_W)
fusion = CLInversionFusion(aero_model, U_INF, DT)
nl_ekf_clinv._fusion = fusion   # run_mpc_simulation finds fusion via _fusion


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def cosine_gust(W0, Tg):
    def g(t):
        if 0.0 <= t <= Tg:
            return (W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / Tg))
        return 0.0
    return g


def ramp_gust(W0=60.0, t_ramp=0.5, t_hold=0.5, t_ramp_down=0.5):
    """Linear ramp up, hold, linear ramp down."""
    t1 = t_ramp; t2 = t_ramp + t_hold; t3 = t2 + t_ramp_down
    def g(t):
        if   t < 0:   return 0.0
        elif t <= t1: return W0 * t / t_ramp
        elif t <= t2: return W0
        elif t <= t3: return W0 * (1.0 - (t - t2) / t_ramp_down)
        return 0.0
    return g


def run_all_modes(gust_fn, tag=''):
    print(f"  [legacy]{tag}...")
    ekf_legacy.reset(xi_trim)
    res_leg = run_mpc_simulation(
        U_INF, T_END, DT, aero_model, lqr, A_s, B_s,
        gust_profile=gust_fn, observer='ekf', kalman_filter=ekf_legacy)

    print(f"  [ekf_ad]{tag}...")
    nl_ekf.reset(xi_trim)
    res_ekf = run_mpc_simulation(
        U_INF, T_END, DT, aero_model, lqr, A_s, B_s,
        gust_profile=gust_fn, observer='ekf_ad', kalman_filter=nl_ekf)

    print(f"  [ekf_clinv]{tag}...")
    nl_ekf_clinv.reset(xi_trim)
    fusion.reset()
    res_cli = run_mpc_simulation(
        U_INF, T_END, DT, aero_model, lqr, A_s, B_s,
        gust_profile=gust_fn, observer='ekf_clinv', kalman_filter=nl_ekf_clinv)

    return res_leg, res_ekf, res_cli


# ──────────────────────────────────────────────────────────────────────────────
# TASK 6 — Nominal gust
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TASK 6 — Nominal gust  W₀=60 m/s, T_g=1 s")
print("=" * 70)

gust_nom = cosine_gust(GUST_W0, GUST_TG)
res_leg, res_ekf, res_cli = run_all_modes(gust_nom)

print("  [true_state]...")
res_true = run_mpc_simulation(
    U_INF, T_END, DT, aero_model, lqr, A_s, B_s,
    gust_profile=gust_nom, observer='true_state')

print("  [baseline open-loop]...")
res_bl = run_mpc_simulation(
    U_INF, T_END, DT, aero_model, None, A_s, B_s,
    gust_profile=gust_nom, observer='true_state')

t    = res_true['t']
mask = (t >= 0.0) & (t <= GUST_TG + 0.5)

# ── Fig 1 — State estimation (3×2, one figure, 4 curves per panel) ────────────
# Layout mirrors lqg_fig1: rows=(h,α,z), cols=(pos/vel, pos/vel, z/W)
# truth / legacy / ekf_ad / ekf_clinv
state_panel = [
    # (truth_key, est_key_suffix, ylabel, title)
    ('h',     'h',    'h [m]',       'Heave'),
    ('hd',    'hd',   'ḣ [m/s]',     'Heave velocity'),
    ('a',     'a',    'α [rad]',      'Pitch'),
    ('ad',    'ad',   'α̇ [rad/s]',   'Pitch rate'),
    ('z_hat', 'z',    'z [—]',        'Latent state z'),
    ('W_gust','W',    'W [m/s]',      'Gust velocity W'),
]

fig1, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
fig1.suptitle('State estimation  —  Legacy EKF / NonlinEKF / NonlinEKF+CLInv',
              fontsize=13)

for ax, (tk, ek, ylabel, title) in zip(axes.ravel(), state_panel):
    if tk == 'W_gust':
        truth = res_true['W_gust']
        e_leg = res_leg['W_hat']
        e_ekf = res_ekf['W_hat']
        e_cli = res_cli['W_hat']
    elif tk == 'z_hat':
        truth = res_true['z_hat']
        e_leg = res_leg['z_hat']
        e_ekf = res_ekf['z_hat']
        e_cli = res_cli['z_hat']
    else:
        truth = res_true[tk]
        e_leg = res_leg[ek + '_hat']
        e_ekf = res_ekf[ek + '_hat']
        e_cli = res_cli[ek + '_hat']

    ax.plot(t, truth, label='True',             **C_TRUE)
    ax.plot(t, e_leg, label='Legacy EKF',       **C_LEGACY)
    ax.plot(t, e_ekf, label='NonlinEKF',        **C_EKF)
    ax.plot(t, e_cli, label='NonlinEKF+CLInv',  **C_CLINV)
    fmt(ax, ylabel, title)

fig1.tight_layout()
p = OUT_DIR / 'fig1_estimation.png'
fig1.savefig(p, dpi=150); plt.close(fig1)
print(f"  [OK] {p.name}")

# ── Fig 2 — Closed-loop performance (2×2) ────────────────────────────────────
delta_true = res_true['delta']
delta_leg  = res_leg['delta']
delta_ekf  = res_ekf['delta']
delta_cli  = res_cli['delta']

ddelta_true = np.gradient(delta_true, t)
ddelta_leg  = np.gradient(delta_leg,  t)
ddelta_ekf  = np.gradient(delta_ekf,  t)
ddelta_cli  = np.gradient(delta_cli,  t)

fig2, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
fig2.suptitle('Closed-loop performance  —  baseline / observers', fontsize=13)

axes[0, 0].plot(t, res_bl['h_ddot'],   label='Baseline',          **C_BL)
axes[0, 0].plot(t, res_true['h_ddot'], label='LQR+true_state',    **C_TRUE)
axes[0, 0].plot(t, res_leg['h_ddot'],  label='LQR+legacy EKF',    **C_LEGACY)
axes[0, 0].plot(t, res_ekf['h_ddot'],  label='LQR+NonlinEKF',     **C_EKF)
axes[0, 0].plot(t, res_cli['h_ddot'],  label='LQR+NonlinEKF+CLInv', **C_CLINV)
fmt(axes[0, 0], 'ḧ [m/s²]', 'Heave acceleration')

axes[0, 1].plot(t, np.rad2deg(res_bl['a']),   label='Baseline',          **C_BL)
axes[0, 1].plot(t, np.rad2deg(res_true['a']), label='LQR+true_state',    **C_TRUE)
axes[0, 1].plot(t, np.rad2deg(res_leg['a']),  label='LQR+legacy EKF',    **C_LEGACY)
axes[0, 1].plot(t, np.rad2deg(res_ekf['a']),  label='LQR+NonlinEKF',     **C_EKF)
axes[0, 1].plot(t, np.rad2deg(res_cli['a']),  label='LQR+NonlinEKF+CLInv', **C_CLINV)
fmt(axes[0, 1], 'α [°]', 'Pitch angle')

axes[1, 0].plot(t, delta_true, label='LQR+true_state',    **C_TRUE)
axes[1, 0].plot(t, delta_leg,  label='LQR+legacy EKF',    **C_LEGACY)
axes[1, 0].plot(t, delta_ekf,  label='LQR+NonlinEKF',     **C_EKF)
axes[1, 0].plot(t, delta_cli,  label='LQR+NonlinEKF+CLInv', **C_CLINV)
axes[1, 0].axhline( 20, color='k', ls=':', lw=0.8, label='±20° sat.')
axes[1, 0].axhline(-20, color='k', ls=':', lw=0.8)
fmt(axes[1, 0], 'δ [°]  (+ = flap down)', 'Flap deflection')

axes[1, 1].plot(t, ddelta_true, label='LQR+true_state',    **C_TRUE)
axes[1, 1].plot(t, ddelta_leg,  label='LQR+legacy EKF',    **C_LEGACY)
axes[1, 1].plot(t, ddelta_ekf,  label='LQR+NonlinEKF',     **C_EKF)
axes[1, 1].plot(t, ddelta_cli,  label='LQR+NonlinEKF+CLInv', **C_CLINV)
fmt(axes[1, 1], 'δ̇ [°/s]', 'Flap rate', legend=False)

fig2.tight_layout()
p = OUT_DIR / 'fig2_performance.png'
fig2.savefig(p, dpi=150); plt.close(fig2)
print(f"  [OK] {p.name}")

# ── Fig 3 — Aerodynamic coefficients (1×2) ────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig3.suptitle('Aerodynamic coefficients  —  baseline / observers', fontsize=13)

axes[0].plot(t, res_bl['C_L'],   label='Baseline',          **C_BL)
axes[0].plot(t, res_true['C_L'], label='LQR+true_state',    **C_TRUE)
axes[0].plot(t, res_leg['C_L'],  label='LQR+legacy EKF',    **C_LEGACY)
axes[0].plot(t, res_ekf['C_L'],  label='LQR+NonlinEKF',     **C_EKF)
axes[0].plot(t, res_cli['C_L'],  label='LQR+NonlinEKF+CLInv', **C_CLINV)
fmt(axes[0], '$C_L$', 'Lift coefficient')

axes[1].plot(t, res_bl['C_M'],   label='Baseline',          **C_BL)
axes[1].plot(t, res_true['C_M'], label='LQR+true_state',    **C_TRUE)
axes[1].plot(t, res_leg['C_M'],  label='LQR+legacy EKF',    **C_LEGACY)
axes[1].plot(t, res_ekf['C_M'],  label='LQR+NonlinEKF',     **C_EKF)
axes[1].plot(t, res_cli['C_M'],  label='LQR+NonlinEKF+CLInv', **C_CLINV)
fmt(axes[1], '$C_M$', 'Pitching moment coefficient')

fig3.tight_layout()
p = OUT_DIR / 'fig3_aero.png'
fig3.savefig(p, dpi=150); plt.close(fig3)
print(f"  [OK] {p.name}")

# ── Fig 4 — W estimation comparison ───────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(t, res_true['W_gust'], label='W true',              color='k',  lw=2.0, ls='--')
ax4.plot(t, res_leg['W_hat'],   label='Legacy EKF',          **C_LEGACY)
ax4.plot(t, res_ekf['W_hat'],   label='NonlinEKF',           **C_EKF)
ax4.plot(t, res_cli['W_hat'],   label='NonlinEKF+CLInv',     **C_CLINV)
fmt(ax4, 'W [m/s]', 'Gust estimation  (W₀=60, T_g=1 s)')
fig4.suptitle('W estimation — all observers', fontsize=13)
fig4.tight_layout()
p = OUT_DIR / 'fig4_W_comparison.png'
fig4.savefig(p, dpi=150); plt.close(fig4)
print(f"  [OK] {p.name}")

# ── Estimation error table ────────────────────────────────────────────────────
print()
print("=" * 90)
print("  TASK 6 — ESTIMATION ERROR  (gust window 0–%.1fs)" % (GUST_TG + 0.5))
print("=" * 90)

truth_map = {'h': res_true['h'],    'hd': res_true['hd'],
             'a': res_true['a'],    'ad': res_true['ad'],
             'z': res_true['z_hat'],'W':  res_true['W_gust']}
leg_map   = {'h': res_leg['h_hat'], 'hd': res_leg['hd_hat'],
             'a': res_leg['a_hat'], 'ad': res_leg['ad_hat'],
             'z': res_leg['z_hat'], 'W':  res_leg['W_hat']}
ekf_map   = {'h': res_ekf['h_hat'], 'hd': res_ekf['hd_hat'],
             'a': res_ekf['a_hat'], 'ad': res_ekf['ad_hat'],
             'z': res_ekf['z_hat'], 'W':  res_ekf['W_hat']}
cli_map   = {'h': res_cli['h_hat'], 'hd': res_cli['hd_hat'],
             'a': res_cli['a_hat'], 'ad': res_cli['ad_hat'],
             'z': res_cli['z_hat'], 'W':  res_cli['W_hat']}

row_defs = [('h [m]',      'h'), ('ḣ [m/s]', 'hd'),
            ('α [rad]',    'a'), ('α̇ [rad/s]','ad'),
            ('z [—]',      'z'), ('W [m/s]',  'W')]

hdr = (f"  {'State':<12}  "
       f"{'Leg RMS':>10}  {'Leg peak':>10}  "
       f"{'EKF RMS':>10}  {'EKF peak':>10}  "
       f"{'Clinv RMS':>10}  {'Clinv peak':>10}")
print(hdr); print("  " + "-"*(len(hdr)-2))

for name, k in row_defs:
    tr = truth_map[k][mask]
    el = leg_map[k][mask]  - tr
    ee = ekf_map[k][mask]  - tr
    ec = cli_map[k][mask]  - tr
    print(f"  {name:<12}  "
          f"{np.sqrt(np.mean(el**2)):>10.3e}  {np.max(np.abs(el)):>10.3e}  "
          f"{np.sqrt(np.mean(ee**2)):>10.3e}  {np.max(np.abs(ee)):>10.3e}  "
          f"{np.sqrt(np.mean(ec**2)):>10.3e}  {np.max(np.abs(ec)):>10.3e}")

# ── Closed-loop performance table ─────────────────────────────────────────────
print()
print("=" * 90)
print("  TASK 6 — CLOSED-LOOP PERFORMANCE  (gust window 0–%.1fs)" % (GUST_TG + 0.5))
print("=" * 90)

for res in (res_bl, res_true, res_leg, res_ekf, res_cli):
    res['a_deg']  = np.rad2deg(np.abs(res['a']))
    res['ddelta'] = np.abs(np.gradient(res['delta'], t))

metrics = [
    ('peak |ḧ| [m/s²]',  'h_ddot', peak),
    ('peak |α| [°]',      'a_deg',  peak),
    ('peak |C_L|',        'C_L',    peak),
    ('peak |C_M|',        'C_M',    peak),
    ('RMS δ [°]',         'delta',  rms),
    ('peak |δ̇| [°/s]',   'ddelta', peak),
]
print(f"  {'Metric':<22}  {'Baseline':>10}  {'True':>10}  "
      f"{'Legacy':>10}  {'EKF_ad':>10}  {'EKF+CLInv':>10}")
print("  " + "-"*80)
for mname, key, fn in metrics:
    vals = [fn(r[key], mask) for r in (res_bl, res_true, res_leg, res_ekf, res_cli)]
    print(f"  {mname:<22}  " + "  ".join(f"{v:>10.4f}" for v in vals))


# ──────────────────────────────────────────────────────────────────────────────
# TASK 7 — Stress tests
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TASK 7 — Stress tests")
print("=" * 70)

stress_cases = [
    ('W0=30 Tg=1',   cosine_gust(30.,  1.0)),
    ('W0=60 Tg=1',   cosine_gust(60.,  1.0)),
    ('W0=90 Tg=1',   cosine_gust(90.,  1.0)),
    ('W0=120 Tg=1',  cosine_gust(120., 1.0)),
    ('W0=60 Tg=0.5', cosine_gust(60.,  0.5)),
    ('W0=60 Tg=2.0', cosine_gust(60.,  2.0)),
    ('Ramp 0→60→0',  ramp_gust(60., 0.5, 0.5, 0.5)),
]

print()
print("=" * 110)
print("  TASK 7 — STRESS TEST TABLE")
print("=" * 110)
hdr7 = (f"  {'Case':<20}  "
        f"{'ḧ_leg':>10}  {'ḧ_ekf':>10}  {'ḧ_cli':>10}  "
        f"{'ΔW_leg':>10}  {'ΔW_ekf':>10}  {'ΔW_cli':>10}")
print(hdr7); print("  " + "-"*(len(hdr7)-2))

for label, gust_fn in stress_cases:
    print(f"\n  Case: {label}")
    rl, re, rc = run_all_modes(gust_fn, tag=f' [{label}]')
    r_ref = run_mpc_simulation(
        U_INF, T_END, DT, aero_model, lqr, A_s, B_s,
        gust_profile=gust_fn, observer='true_state')

    W_true = r_ref['W_gust']
    ph_l = np.max(np.abs(rl['h_ddot']))
    ph_e = np.max(np.abs(re['h_ddot']))
    ph_c = np.max(np.abs(rc['h_ddot']))
    rw_l = np.sqrt(np.mean((rl['W_hat'] - W_true)**2))
    rw_e = np.sqrt(np.mean((re['W_hat'] - W_true)**2))
    rw_c = np.sqrt(np.mean((rc['W_hat'] - W_true)**2))

    print(f"  {label:<20}  "
          f"{ph_l:>10.4f}  {ph_e:>10.4f}  {ph_c:>10.4f}  "
          f"{rw_l:>10.4f}  {rw_e:>10.4f}  {rw_c:>10.4f}")

    # W estimation figure per stress case (same style as fig4)
    t_s = r_ref['t']
    fig_s, ax_s = plt.subplots(figsize=(10, 4))
    ax_s.plot(t_s, W_true,      label='W true',            color='k', lw=2.0, ls='--')
    ax_s.plot(t_s, rl['W_hat'], label='Legacy EKF',        **C_LEGACY)
    ax_s.plot(t_s, re['W_hat'], label='NonlinEKF',         **C_EKF)
    ax_s.plot(t_s, rc['W_hat'], label='NonlinEKF+CLInv',   **C_CLINV)
    fmt(ax_s, 'W [m/s]', f'W estimation — {label}')
    fig_s.suptitle(f'Stress test: {label}', fontsize=13)
    fig_s.tight_layout()
    safe = label.replace('=','').replace(' ','_').replace('→','_')
    p_s  = OUT_DIR / f'fig7_W_{safe}.png'
    fig_s.savefig(p_s, dpi=150); plt.close(fig_s)
    print(f"  [OK] {p_s.name}")

    # Performance figure per stress case (ḧ and α, 1×2)
    fig_p, axp = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    fig_p.suptitle(f'Performance — {label}', fontsize=13)
    axp[0].plot(t_s, r_ref['h_ddot'], label='LQR+true',         **C_TRUE)
    axp[0].plot(t_s, rl['h_ddot'],    label='LQR+legacy EKF',   **C_LEGACY)
    axp[0].plot(t_s, re['h_ddot'],    label='LQR+NonlinEKF',    **C_EKF)
    axp[0].plot(t_s, rc['h_ddot'],    label='LQR+NonlinEKF+CLInv', **C_CLINV)
    fmt(axp[0], 'ḧ [m/s²]', 'Heave acceleration')
    axp[1].plot(t_s, np.rad2deg(r_ref['a']), label='LQR+true',         **C_TRUE)
    axp[1].plot(t_s, np.rad2deg(rl['a']),    label='LQR+legacy EKF',   **C_LEGACY)
    axp[1].plot(t_s, np.rad2deg(re['a']),    label='LQR+NonlinEKF',    **C_EKF)
    axp[1].plot(t_s, np.rad2deg(rc['a']),    label='LQR+NonlinEKF+CLInv', **C_CLINV)
    fmt(axp[1], 'α [°]', 'Pitch angle')
    fig_p.tight_layout()
    p_p = OUT_DIR / f'fig7_perf_{safe}.png'
    fig_p.savefig(p_p, dpi=150); plt.close(fig_p)
    print(f"  [OK] {p_p.name}")

# ── Final report ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FIGURES SAVED")
print("=" * 70)
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p}")
print("\nDone.")

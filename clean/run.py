"""
Gust load alleviation — open loop vs model-based optimal controller.

Two simulations are run back-to-back:
  - Open loop   : flap fixed at zero, no control action.
  - Closed loop : controller computes the optimal flap deflection at each step
                  using one-step-ahead predictions from the aerodynamic model.

The observer estimates structural states and gust velocity from sensors
(accelerometer, pitch encoder, rate gyro) and feeds them to the controller.

Peak values and percentage improvements are printed, and three figures are
saved to the results/ subfolder.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import argparse
import structure
from observer import Observer
from controller import Controller, ProportionalController

_parser = argparse.ArgumentParser(description='GLA simulation — linear or LDNet aero')
_parser.add_argument('--model-dir', default=None,
                     help='Path to LDNet weights directory. '
                          'Omit to use the linear Theodorsen model.')
_parser.add_argument('--warmup-csv', default=None,
                     help='Path to structural_trajectory.csv used to pre-load LDNet latent state.')
_parser.add_argument('--controller', default='proportional', choices=['proportional','optimal'],
                     help='Closed-loop controller: proportional C_L feedback (default) or model-based optimal.')
_parser.add_argument('--gain', type=float, default=10.0,
                     help='Proportional controller gain G (deg per unit C_L deviation).')
_args = _parser.parse_args()

U_INF         = 80.0   # freestream velocity [m/s]
RHO           = 1.225  # air density [kg/m³]
S_REF         = 0.05   # reference area [m²]
C_REF         = 1.0    # reference chord [m]

T_END         = 5.0    # simulation duration [s]
DT            = 0.002  # time step [s] (matches training dt_ref)

GUST_W0       = 10.0   # peak gust velocity [m/s]
GUST_DURATION = 1.0    # gust duration [s]

DELTA_MAX     = 20.0   # flap deflection limit [deg]
DELTA_DOT_MAX = 100.0  # flap rate limit [deg/s]

# ── Aerodynamic model selection ────────────────────────────────────────────────

if _args.model_dir:
    from ldnet_aero import LDNetAero
    _aero_module = LDNetAero(_args.model_dir)
    print(f"Using LDNet model from: {_args.model_dir}  (latent_dim={_aero_module._num_z})")
else:
    import aero as _aero_module
    print("Using linear Theodorsen model")

# Controller cost weights
Q_H         = 1e4   # heave displacement penalty
Q_ALPHA     = 1e4   # pitch angle penalty
Q_ALPHA_DOT = 1e4   # pitch rate penalty
Q_CL        = 1e3   # lift coefficient penalty
R           = 1.0   # control effort penalty


# ── Gust profile ───────────────────────────────────────────────────────────────

def gust(t):
    """1-cosine gust: smooth ramp-up and ramp-down over GUST_DURATION seconds."""
    if 0.0 <= t <= GUST_DURATION:
        return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / GUST_DURATION))
    return 0.0


# ── Aeroelastic simulation loop ────────────────────────────────────────────────

TRIM_CL = TRIM_FY = TRIM_MZ = 0.0   # set by simulate() for the LDNet model


def simulate(ctrl=None):
    """
    Run one aeroelastic simulation.

    At each time step:
      1. The controller (if present) reads the observer output and commands δ.
      2. The true aerodynamic forces are computed with the real gust velocity.
      3. The structural state is advanced one RK4 step.
      4. The observer updates its estimates from the new sensor readings.

    Parameters
    ----------
    ctrl : Controller or None
        Pass a Controller instance for closed-loop, None for open loop.

    Returns
    -------
    dict with time histories: t, h, hd, a, ad, C_L, C_M, h_ddot, delta,
                              W_hat, W_true.
    """
    q_dyn = 0.5 * RHO * U_INF**2 * S_REF

    t_arr  = np.arange(0.0, T_END + DT, DT)
    N      = len(t_arr)
    W_true = np.array([gust(t) for t in t_arr])

    h_arr     = np.zeros(N)
    hd_arr    = np.zeros(N)
    a_arr     = np.zeros(N)
    ad_arr    = np.zeros(N)
    CL_arr    = np.zeros(N)
    CM_arr    = np.zeros(N)
    hddot_arr = np.zeros(N)
    delta_arr = np.zeros(N)
    W_hat_arr = np.zeros(N)

    if hasattr(_aero_module, 'reset'):
        _aero_module.reset(dt=DT, warmup_csv=_args.warmup_csv if hasattr(_args, 'warmup_csv') else None)

    # Compute trim aero loads (W=0, delta=0) to subtract as static offset.
    # The structural integrator only sees perturbations from trim equilibrium.
    if hasattr(_aero_module, 'reset'):
        from scipy.optimize import fsolve

        def _trim_residual(x_eq):
            C_L_eq, C_M_eq = _aero_module.predict(
                np.array([x_eq[0], 0., x_eq[1], 0.]), 0., 0., U_INF)
            dFy = q_dyn * C_L_eq
            dMz = q_dyn * C_M_eq * C_REF
            return [-dFy - structure.K_H * x_eq[0],
                     dMz - structure.K_ALPHA * x_eq[1]]

        x_eq = fsolve(_trim_residual, [0., 0.])
        x0   = np.array([x_eq[0], 0., x_eq[1], 0.])
        C_L_trim, C_M_trim = _aero_module.predict(x0, 0., 0., U_INF)
        Fy_trim = q_dyn * C_L_trim
        Mz_trim = q_dyn * C_M_trim * C_REF
    else:
        x0 = np.zeros(4)
        C_L_trim = 0.0
        Fy_trim, Mz_trim = 0.0, 0.0

    global TRIM_CL, TRIM_FY, TRIM_MZ
    TRIM_CL, TRIM_FY, TRIM_MZ = float(C_L_trim), float(Fy_trim), float(Mz_trim)

    x   = x0.copy()
    obs = Observer(dt=DT, U=U_INF, aero_module=_aero_module)

    if ctrl is not None:
        ctrl.reset()

    x_hat = x0.copy()
    W_hat = 0.0
    C_L_prev = C_L_trim
    # The proportional controller needs no state/gust estimate, so skip the
    # (expensive) observer unless the model-based optimal controller is in use.
    _use_observer = ctrl is not None and not isinstance(ctrl, ProportionalController)

    for i, t in enumerate(t_arr):

        h_arr[i], hd_arr[i], a_arr[i], ad_arr[i] = x
        W_hat_arr[i] = W_hat

        if ctrl is not None:
            if isinstance(ctrl, ProportionalController):
                delta = ctrl.compute(C_L_prev)        # pure C_L feedback (prev step)
            else:
                delta = ctrl.compute(x_hat, W_hat)
        else:
            delta = 0.0
        delta_arr[i] = delta

        C_L, C_M = _aero_module.predict(x, delta, W_true[i], U_INF)
        if hasattr(_aero_module, 'advance'):
            _aero_module.advance(x, delta, W_true[i], U_INF, DT)
        CL_arr[i] = C_L
        CM_arr[i] = C_M
        C_L_prev  = C_L

        Fy = q_dyn * C_L - Fy_trim
        Mz = q_dyn * C_M * C_REF - Mz_trim

        _, hddot, _, _ = structure.rhs(x, Fy, Mz)
        hddot_arr[i]   = hddot

        x = structure.step_rk4(x, Fy, Mz, DT)

        # Sensors: h_ddot from accelerometer, α and α̇ from encoder/gyro.
        if _use_observer:
            x_hat, W_hat = obs.update(
                h_ddot    = hddot,
                alpha     = x[2],
                alpha_dot = x[3],
                delta     = delta,
                C_L_meas  = C_L,
            )

    return {
        't':      t_arr,
        'h':      h_arr,
        'hd':     hd_arr,
        'a':      a_arr,
        'ad':     ad_arr,
        'C_L':    CL_arr,
        'C_M':    CM_arr,
        'h_ddot': hddot_arr,
        'delta':  delta_arr,
        'W_hat':  W_hat_arr,
        'W_true': W_true,
    }


# ── Run both simulations ───────────────────────────────────────────────────────

print("Running open loop...")
res_ol = simulate(ctrl=None)

if _args.controller == 'proportional':
    print(f"Running closed loop (proportional C_L feedback, gain={_args.gain})...")
    ctrl = ProportionalController(
        C_L_trim      = TRIM_CL,
        gain          = _args.gain,
        dt            = DT,
        delta_max     = DELTA_MAX,
        delta_dot_max = DELTA_DOT_MAX,
    )
else:
    print("Running closed loop (model-based optimal controller)...")
    ctrl = Controller(
        aero_predict  = _aero_module.predict,
        U             = U_INF,
        dt            = DT,
        # Tuned config that beats the proportional law on C_L without exciting pitch:
        Q_h           = 0.0,
        Q_alpha       = 0.0,
        Q_alpha_dot   = 3e5,     # penalize pitch rate -> protects alpha_dot AND sharpens C_L
        Q_CL          = 1e3,
        R             = 0.3,
        delta_max     = DELTA_MAX,
        delta_dot_max = DELTA_DOT_MAX,
        C_L_trim      = TRIM_CL,
        Fy_trim       = TRIM_FY,
        Mz_trim       = TRIM_MZ,
        global_search = True,    # non-convex one-step cost on LDNet -> need global search
        causal_basin  = True,    # restrict to the causal (stabilizing) flap-sign basin
        n_grid        = 15,
        target_lpf    = 0.95,    # low-pass the optimal delta -> kills chatter (C_L peak, pitch)
    )
res_cl = simulate(ctrl=ctrl)


# ── Peak metrics ───────────────────────────────────────────────────────────────

gust_win = (res_ol['t'] >= 0.0) & (res_ol['t'] <= GUST_DURATION + 0.5)

def peak(res, key):
    return np.max(np.abs(res[key][gust_win]))

metrics = ['C_L', 'h_ddot', 'h', 'a']
labels  = ['C_L [-]', 'h_ddot [m/s²]', 'h [m]', 'α [rad]']

print(f"\n{'':12s}  {'Open loop':>10s}  {'Controlled':>10s}  {'Reduction':>10s}")
for key, label in zip(metrics, labels):
    ol  = peak(res_ol, key)
    cl  = peak(res_cl, key)
    red = (ol - cl) / ol * 100 if ol > 0 else 0.0
    print(f"  {label:<12s}  {ol:10.4f}  {cl:10.4f}  {red:+9.1f} %")

print(f"\n  Max flap deflection: {np.max(np.abs(res_cl['delta'])):.1f} deg")


# ── Figures ────────────────────────────────────────────────────────────────────

OUT = Path(__file__).parent / 'results'
OUT.mkdir(exist_ok=True)

OL  = dict(color='steelblue', lw=1.8, label='Open loop')
CL  = dict(color='crimson',   lw=1.8, label='Model-based optimal controller')

t = res_ol['t']

# Figure 1 — primary load metrics
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'Gust load alleviation  —  W₀ = {GUST_W0} m/s,  U = {U_INF} m/s',
             fontsize=13)

axes[0].plot(t, res_ol['C_L'],    **OL)
axes[0].plot(t, res_cl['C_L'],    **CL)
axes[0].set_xlabel('t [s]');  axes[0].set_ylabel('$C_L$ [-]')
axes[0].set_title('Lift coefficient');  axes[0].grid(alpha=0.3);  axes[0].legend()

axes[1].plot(t, res_ol['h_ddot'], **OL)
axes[1].plot(t, res_cl['h_ddot'], **CL)
axes[1].set_xlabel('t [s]');  axes[1].set_ylabel('$\\ddot{h}$ [m/s²]')
axes[1].set_title('Heave acceleration');  axes[1].grid(alpha=0.3);  axes[1].legend()

fig.tight_layout()
fig.savefig(OUT / 'fig1_loads.png', dpi=150)
print(f"\nSaved {OUT / 'fig1_loads.png'}")

# Figure 2 — structural states
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle('Structural response', fontsize=13)

axes[0, 0].plot(t, res_ol['h'],              **OL)
axes[0, 0].plot(t, res_cl['h'],              **CL)
axes[0, 0].set_ylabel('h [m]');  axes[0, 0].set_title('Heave')

axes[0, 1].plot(t, res_ol['hd'],             **OL)
axes[0, 1].plot(t, res_cl['hd'],             **CL)
axes[0, 1].set_ylabel('$\\dot{h}$ [m/s]');  axes[0, 1].set_title('Heave velocity')

axes[1, 0].plot(t, np.rad2deg(res_ol['a']),  **OL)
axes[1, 0].plot(t, np.rad2deg(res_cl['a']),  **CL)
axes[1, 0].set_ylabel('α [°]');  axes[1, 0].set_title('Pitch angle')

axes[1, 1].plot(t, np.rad2deg(res_ol['ad']), **OL)
axes[1, 1].plot(t, np.rad2deg(res_cl['ad']), **CL)
axes[1, 1].set_ylabel('$\\dot{\\alpha}$ [°/s]');  axes[1, 1].set_title('Pitch rate')

for ax in axes.flat:
    ax.set_xlabel('t [s]');  ax.grid(alpha=0.3);  ax.legend()

fig.tight_layout()
fig.savefig(OUT / 'fig2_states.png', dpi=150)
print(f"Saved {OUT / 'fig2_states.png'}")

# Figure 3 — control input and gust estimation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Control input and gust observer', fontsize=13)

axes[0].plot(t, res_cl['delta'], **CL)
axes[0].axhline( DELTA_MAX, color='k', ls=':', lw=0.8)
axes[0].axhline(-DELTA_MAX, color='k', ls=':', lw=0.8, label=f'±{DELTA_MAX}° limit')
axes[0].set_xlabel('t [s]');  axes[0].set_ylabel('δ [deg]')
axes[0].set_title('Flap deflection');  axes[0].grid(alpha=0.3);  axes[0].legend()

axes[1].plot(t, res_ol['W_true'], color='gray',    lw=1.8, label='True gust W')
axes[1].plot(t, res_cl['W_hat'],  color='crimson', lw=1.8, ls='--', label='Estimated Ŵ')
axes[1].set_xlabel('t [s]');  axes[1].set_ylabel('W [m/s]')
axes[1].set_title('Gust estimation');  axes[1].grid(alpha=0.3);  axes[1].legend()

fig.tight_layout()
fig.savefig(OUT / 'fig3_control.png', dpi=150)
print(f"Saved {OUT / 'fig3_control.png'}")

plt.show()

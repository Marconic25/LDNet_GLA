"""
Closed-loop simulation loop for aeroelastic GLA.

Decoupled from the controller implementations (lqr.py, mpc.py) so that
any controller — LQR, MPC, or None (open loop) — can be driven by the
same loop without importing unused dependencies.

Public API
----------
run_simulation(U_INF, T_END, DT, aero_model, controller, A_s, B_s, *, ...)
    Run one closed-loop simulation and return a result dict.

_estimate_W_from_CL(...)
    Bisection helper used by the heuristic observer.
"""
import numpy as np
from structural.smd import structural_rhs, M_WING, M_FLAP, I_WING, I_FLAP_EA, \
                            D_H, D_ALPHA, K_H, K_ALPHA, _D_X


def _estimate_W_from_CL(aero_model, z_hat, x_hat, delta, C_L_meas, U_INF, DT,
                         W_lo=0.0, W_hi=80.0, tol=0.5):
    """Estimate W_gust by bisection on C_L(W) = C_L_meas."""
    def CL_pred(W):
        _, CL, _ = aero_model.step(z_hat, x_hat[0], x_hat[1], x_hat[2], x_hat[3],
                                    delta, W, U_INF, DT)
        return float(CL)

    flo = CL_pred(W_lo) - C_L_meas
    fhi = CL_pred(W_hi) - C_L_meas

    if flo * fhi > 0:
        return W_lo if abs(flo) < abs(fhi) else W_hi

    for _ in range(20):
        W_mid = 0.5 * (W_lo + W_hi)
        if (W_hi - W_lo) < tol:
            break
        fmid = CL_pred(W_mid) - C_L_meas
        if flo * fmid <= 0:
            W_hi = W_mid; fhi = fmid
        else:
            W_lo = W_mid; flo = fmid

    return 0.5 * (W_lo + W_hi)


def run_simulation(U_INF, T_END, DT, aero_model, controller, A_s, B_s,
                   gust_profile=None, observer='heuristic', kalman_filter=None):
    """
    Closed-loop simulation with selectable observer.

    Parameters
    ----------
    U_INF, T_END, DT : float
        Freestream velocity [m/s], end time [s], timestep [s].
    aero_model : LDNetModel
    controller : LQRController | MPCController | None
        None runs open-loop (delta=0 at every step).
    A_s, B_s : ndarray
        Structural state-space matrices (passed for legacy compatibility).
    gust_profile : callable(t) -> float, optional
        Returns gust velocity [m/s] at time t.  Defaults to a single
        1-cosine gust of 30 m/s over 0.8 s.
    observer : {'heuristic', 'ekf', 'ekf_ad', 'ekf_clinv', 'true_state'}
        - 'heuristic'  : leaky kinematic integrator + C_L bisection.
        - 'ekf'        : EKF with constant DARE gain; requires kalman_filter.
        - 'ekf_ad'     : NonlinearEKF with AD Jacobians; requires kalman_filter.
        - 'ekf_clinv'  : NonlinearEKF + C_L inversion; requires kalman_filter.
        - 'true_state' : oracle — true state and gust (upper bound).
    kalman_filter : NonlinearEKF | ExtendedKalmanFilter | None
        Required for ekf / ekf_ad / ekf_clinv observers.

    Returns
    -------
    dict with keys: t, h, hd, a, ad, delta, C_L, C_M, h_ddot, a_ddot,
                    W_hat, W_gust, h_hat, hd_hat, a_hat, ad_hat, z_hat.
    """
    _VALID_OBSERVERS = ('heuristic', 'ekf', 'ekf_ad', 'ekf_clinv', 'true_state')
    if observer not in _VALID_OBSERVERS:
        raise ValueError(
            f"observer must be one of {_VALID_OBSERVERS}; got {observer!r}")
    if observer in ('ekf', 'ekf_ad', 'ekf_clinv') and kalman_filter is None:
        raise ValueError(f"observer={observer!r} requires a kalman_filter instance")

    q_dyn    = 0.5 * 1.225 * U_INF**2 * 0.05
    M_hh     = M_WING + M_FLAP
    M_aa     = I_WING + I_FLAP_EA
    M_ha     = M_FLAP * _D_X
    tau_leak = 5.0

    t_win = np.linspace(0.0, T_END, int(T_END / DT) + 1)
    N     = len(t_win)

    if gust_profile is None:
        def gust_profile(t):
            if 0.0 <= t <= 0.8:
                return 30.0 * (1.0 - np.cos(2.0 * np.pi * t / 0.8))
            return 0.0

    W_gust_arr = np.array([gust_profile(t) for t in t_win])

    # trim latent state
    _z_trim = np.zeros(aero_model.num_latent_states)
    for _ in range(200):
        _z_trim, _, _ = aero_model.step(_z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)

    x     = np.zeros(4)
    z     = _z_trim.copy()
    x_hat = np.zeros(4)
    z_hat = _z_trim.copy()

    if observer in ('ekf', 'ekf_ad', 'ekf_clinv'):
        xi0 = np.concatenate([x_hat, _z_trim, [0.0]])
        kalman_filter.reset(xi0)

    h_hist      = np.zeros(N)
    hd_hist     = np.zeros(N)
    a_hist      = np.zeros(N)
    ad_hist     = np.zeros(N)
    delta_hist  = np.zeros(N)
    C_L_hist    = np.zeros(N)
    C_M_hist    = np.zeros(N)
    h_ddot_hist = np.zeros(N)
    a_ddot_hist = np.zeros(N)
    W_hat_hist  = np.zeros(N)
    h_hat_hist  = np.zeros(N)
    hd_hat_hist = np.zeros(N)
    a_hat_hist  = np.zeros(N)
    ad_hat_hist = np.zeros(N)
    z_hat_hist  = np.zeros(N)

    delta_prev = 0.0

    for i, t in enumerate(t_win):
        h_hist[i] = x[0]; hd_hist[i] = x[1]
        a_hist[i] = x[2]; ad_hist[i] = x[3]

        # ── Control ──────────────────────────────────────────────────────────
        if controller is not None:
            if hasattr(controller, 'solve_tf'):
                # MPC path
                if observer != 'true_state':
                    W_now = W_hat_hist[i]
                    steps = np.arange(controller.N, dtype=np.float64)
                    W_gust_seq = W_now * (0.98 ** steps)
                else:
                    horizon_idx = np.arange(i, min(i + controller.N, N))
                    W_gust_seq  = W_gust_arr[horizon_idx]
                    if len(W_gust_seq) < controller.N:
                        W_gust_seq = np.pad(W_gust_seq,
                                            (0, controller.N - len(W_gust_seq)))
                delta, _ = controller.solve_tf(x_hat, z_hat, W_gust_seq,
                                               CL_meas=float(C_L_hist[i]))
            else:
                # LQR path
                delta = controller.solve(x_hat, z_hat, W_hat=W_hat_hist[i])
        else:
            delta = 0.0
        delta_hist[i] = delta

        # ── True system step ──────────────────────────────────────────────────
        z, C_L, C_M = aero_model.step(z, x[0], x[1], x[2], x[3],
                                       delta, W_gust_arr[i], U_INF, DT)
        C_L_hist[i] = C_L
        C_M_hist[i] = C_M

        Fy = q_dyn * C_L
        Mz = q_dyn * C_M

        def _srhs(s):
            return np.array(structural_rhs(t, s, Fy, Mz, 0.0, 0.0))

        k1 = _srhs(x); k2 = _srhs(x + 0.5*DT*k1)
        k3 = _srhs(x + 0.5*DT*k2); k4 = _srhs(x + DT*k3)
        x  = x + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        _, h_ddot, _, a_ddot = _srhs(x)
        h_ddot_hist[i] = h_ddot
        a_ddot_hist[i] = a_ddot

        # ── Observer ─────────────────────────────────────────────────────────
        if observer == 'heuristic':
            leak   = 1.0 - DT / tau_leak
            hd_hat = leak * x_hat[1] + h_ddot * DT
            ad_hat = leak * x_hat[3] + a_ddot * DT
            h_hat  = leak * x_hat[0] + hd_hat * DT
            a_hat  = leak * x_hat[2] + ad_hat * DT
            x_hat  = np.array([h_hat, hd_hat, a_hat, ad_hat])

            W_hat = _estimate_W_from_CL(aero_model, z_hat, x_hat, delta,
                                         C_L_hist[i], U_INF, DT)
            z_hat, _, _ = aero_model.step(
                z_hat, x_hat[0], x_hat[1], x_hat[2], x_hat[3],
                delta, W_hat, U_INF, DT)

            if i + 1 < N:
                W_hat_hist[i + 1] = W_hat

        elif observer == 'ekf':
            y_meas = np.array([h_ddot, x[2], x[3]])
            W_bisect = _estimate_W_from_CL(
                aero_model, kalman_filter.xi_hat[4:5],
                kalman_filter.xi_hat[:4], delta,
                C_L_hist[i], U_INF, DT)
            xi_hat = kalman_filter.step(u=delta_prev, y=y_meas,
                                        W_bisect=W_bisect)
            x_hat  = xi_hat[:4]
            z_hat  = xi_hat[4:5]
            W_hat  = float(xi_hat[5])
            if i + 1 < N:
                W_hat_hist[i + 1] = W_hat

        elif observer in ('ekf_ad', 'ekf_clinv'):
            y_meas  = np.array([h_ddot, x[2]])
            C_L_arg = float(C_L_hist[i]) if observer == 'ekf_clinv' else None
            xi_hat  = kalman_filter.step(u=delta_prev, y_meas=y_meas,
                                         C_L_meas=C_L_arg)
            x_hat  = xi_hat[:4]
            z_hat  = xi_hat[4:5]
            W_hat  = float(xi_hat[5])
            W_hat_hist[i] = W_hat
            if i + 1 < N:
                W_hat_hist[i + 1] = W_hat

        else:  # 'true_state'
            x_hat = x.copy()
            z_hat = z.copy()
            W_hat_hist[i] = W_gust_arr[i]
            W_hat = W_gust_arr[i]

        h_hat_hist[i]  = x_hat[0]
        hd_hat_hist[i] = x_hat[1]
        a_hat_hist[i]  = x_hat[2]
        ad_hat_hist[i] = x_hat[3]
        z_hat_hist[i]  = float(z_hat[0]) if hasattr(z_hat, '__len__') else float(z_hat)

        delta_prev = delta

    return {
        't':      t_win,
        'h':      h_hist,
        'hd':     hd_hist,
        'a':      a_hist,
        'ad':     ad_hist,
        'delta':  delta_hist,
        'C_L':    C_L_hist,
        'C_M':    C_M_hist,
        'h_ddot': h_ddot_hist,
        'a_ddot': a_ddot_hist,
        'W_hat':  W_hat_hist,
        'W_gust': W_gust_arr,
        'h_hat':  h_hat_hist,
        'hd_hat': hd_hat_hist,
        'a_hat':  a_hat_hist,
        'ad_hat': ad_hat_hist,
        'z_hat':  z_hat_hist,
    }

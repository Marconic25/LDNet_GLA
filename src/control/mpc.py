import numpy as np
import tensorflow as tf
from structural.smd import structural_rhs, M_WING, M_FLAP, I_WING, I_FLAP_EA, D_H, D_ALPHA, K_H, K_ALPHA, _D_X


class MPCController:
    """
    MPC for aeroelastic gust load alleviation.

    Cost: J = Σ [ Q_CL·C_L² + Q_CM·C_M² + Q_h·h² + Q_a·α² + Q_dCL·(ΔC_L)² + R·u² + R_du·(Δu)² ]
    Solved via Adam on the full N-step sequence [u_0,...,u_{N-1}] using TF GradientTape.
    """

    def __init__(self, aero_model, U_INF, DT,
                 Q_CL=0.0, Q_CM=0.0, Q_h=0.0, Q_a=0.0,
                 R=1.0, R_du=0.0, N=10, delta_max=20.0,
                 CL_trim=0.0, CM_trim=0.0, Q_dCL=0.0,
                 use_tf_solver=False, ddelta_max=None):
        self.aero_model = aero_model
        self.U_INF      = U_INF
        self.DT         = DT
        self.Q_CL       = Q_CL
        self.Q_CM       = Q_CM
        self.Q_h        = Q_h
        self.Q_a        = Q_a
        self.Q_dCL      = Q_dCL
        self.R          = R
        self.R_du       = R_du
        self.N          = N
        self.delta_max  = delta_max
        self.CL_trim    = CL_trim
        self.CM_trim    = CM_trim

        self.use_tf_solver = use_tf_solver
        self.k_prev        = 0.0
        self.u_prev        = np.zeros(N)
        self.delta_applied = 0.0
        # Hard rate limit: max flap speed [deg/s]. None = no hard limit.
        self.ddelta_max    = float(ddelta_max) if ddelta_max is not None else None

        if use_tf_solver:
            self._tf_u_var     = tf.Variable(np.zeros(N, dtype=np.float64), trainable=True)
            self._tf_opt       = tf.keras.optimizers.Adam(learning_rate=0.5)
            self._tf_adam_step = self._build_tf_step()

    def _build_tf_step(self):
        dtype  = tf.float64
        dm     = tf.constant(self.delta_max, dtype=dtype)
        # ddm: max |Δu| per step = ddelta_max * DT. If None, use 2*dm (no effective limit).
        _ddm_val = (self.ddelta_max * self.DT) if self.ddelta_max is not None else 2.0 * self.delta_max
        ddm    = tf.constant(_ddm_val, dtype=dtype)
        U_INF  = tf.constant(self.U_INF, dtype=dtype)
        DT     = tf.constant(self.DT, dtype=dtype)
        q_dyn  = tf.constant(0.5 * 1.225 * self.U_INF**2 * 0.05, dtype=dtype)
        CL_tr  = tf.constant(float(self.CL_trim), dtype=dtype)
        CM_tr  = tf.constant(float(self.CM_trim), dtype=dtype)

        M_hh = tf.constant(float(M_WING + M_FLAP), dtype=dtype)
        M_aa = tf.constant(float(I_WING + I_FLAP_EA), dtype=dtype)
        M_ha = tf.constant(float(M_FLAP * _D_X), dtype=dtype)
        det  = M_hh * M_aa - M_ha * M_ha
        dH   = tf.constant(float(D_H), dtype=dtype)
        kH   = tf.constant(float(K_H), dtype=dtype)
        dA   = tf.constant(float(D_ALPHA), dtype=dtype)
        kA   = tf.constant(float(K_ALPHA), dtype=dtype)

        aero  = self.aero_model
        u_var = self._tf_u_var
        opt   = self._tf_opt
        N     = self.N

        def rk4(x, Fy, Mz):
            def rhs(s):
                RHS_h = -Fy - dH*s[1] - kH*s[0]
                RHS_a =  Mz - dA*s[3] - kA*s[2]
                return tf.stack([s[1],
                                 (M_aa*RHS_h - M_ha*RHS_a) / det,
                                 s[3],
                                 (M_hh*RHS_a - M_ha*RHS_h) / det])
            k1 = rhs(x); k2 = rhs(x + 0.5*DT*k1)
            k3 = rhs(x + 0.5*DT*k2); k4 = rhs(x + DT*k3)
            return x + (DT/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        @tf.function
        def adam_step(x0, z0, W_s, u0_prev, CL_prev,
                      Q_CL, Q_CM, Q_h, Q_a, Q_dCL, R, R_du):
            with tf.GradientTape() as tape:
                u_cl = tf.clip_by_value(u_var, -dm, dm)
                x = x0; z = z0
                J   = tf.zeros((), dtype=dtype)
                u_p = u0_prev
                CL_p = CL_prev
                for i in tf.range(N):
                    # Hard rate clip: u_i ∈ [u_p - ddm, u_p + ddm]
                    u_i = tf.clip_by_value(u_cl[i], u_p - ddm, u_p + ddm)
                    z, C_L, C_M = aero.step_tf(
                        z, x[0], x[1], x[2], x[3],
                        u_i, W_s[i], U_INF, DT)
                    x   = rk4(x, q_dyn*C_L, q_dyn*C_M)
                    dCL = C_L - CL_p
                    J   = J + (Q_CL*C_L**2 + Q_CM*(C_M - CM_tr)**2
                               + Q_h*x[0]**2 + Q_a*x[2]**2
                               + Q_dCL*dCL**2
                               + R*u_i**2 + R_du*(u_i - u_p)**2)
                    u_p  = u_i
                    CL_p = C_L
            opt.apply_gradients([(tape.gradient(J, u_var), u_var)])

        return adam_step

    def solve_tf(self, x_hat, z_hat, W_gust_seq, CL_meas=0.0,
                 gust_phase=True, n_steps=15):
        """Adam optimisation on full u-sequence via @tf.function rollout."""
        dtype = tf.float64
        dm    = float(self.delta_max)

        self._tf_u_var.assign(np.clip(self.u_prev, -dm, dm).astype(np.float64))

        x0      = tf.constant(x_hat,      dtype=dtype)
        z0      = tf.constant(z_hat,      dtype=dtype)
        W_s     = tf.constant(W_gust_seq, dtype=dtype)
        u0_prev = tf.constant(float(self.delta_applied), dtype=dtype)
        CL_prev = tf.constant(float(CL_meas),            dtype=dtype)

        Q_CL  = tf.constant(float(self.Q_CL),  dtype=dtype)
        Q_CM  = tf.constant(float(self.Q_CM),  dtype=dtype)
        Q_h   = tf.constant(float(self.Q_h)  if gust_phase else 0.0, dtype=dtype)
        Q_a   = tf.constant(float(self.Q_a)  if gust_phase else 0.0, dtype=dtype)
        Q_dCL = tf.constant(float(self.Q_dCL), dtype=dtype)
        R     = tf.constant(float(self.R),     dtype=dtype)
        R_du  = tf.constant(float(self.R_du),  dtype=dtype)

        for _ in range(n_steps):
            self._tf_adam_step(x0, z0, W_s, u0_prev, CL_prev,
                               Q_CL, Q_CM, Q_h, Q_a, Q_dCL, R, R_du)

        u_opt = np.clip(self._tf_u_var.numpy(), -dm, dm)
        # Apply hard rate limit to first element (the one actually sent to plant)
        if self.ddelta_max is not None:
            max_step = self.ddelta_max * self.DT
            u_opt[0] = float(np.clip(u_opt[0],
                                     self.delta_applied - max_step,
                                     self.delta_applied + max_step))
        self.u_prev        = u_opt
        self.delta_applied = float(u_opt[0])
        self.k_prev        = float(u_opt[0]) / dm if dm > 0 else 0.0
        return float(u_opt[0]), u_opt


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


def run_mpc_simulation(U_INF, T_END, DT, aero_model, mpc_controller, A_s, B_s,
                       use_ekf=True, gust_profile=None, use_aoa_sensor=False,
                       observer='heuristic', kalman_filter=None):
    """
    Closed-loop MPC/LQR simulation with selectable observer.

    Parameters
    ----------
    observer : {'heuristic', 'ekf', 'true_state'}
        - 'heuristic'  : leaky kinematic integrator + C_L bisection.
        - 'ekf'        : Extended Kalman Filter; requires kalman_filter arg.
        - 'true_state' : oracle — true state and gust (upper bound).
    kalman_filter : ExtendedKalmanFilter or None
        Required when observer='ekf'.

    Returns
    -------
    dict with keys: t, h, hd, a, ad, delta, C_L, C_M, h_ddot, a_ddot,
                    W_hat, W_gust, h_hat, hd_hat, a_hat, ad_hat, z_hat.
    """
    from aerodynamics.model import LDNetModel
    from structural.smd import structural_rhs, M_WING, M_FLAP, I_WING, I_FLAP_EA, \
                                D_H, D_ALPHA, K_H, K_ALPHA, _D_X

    _VALID_OBSERVERS = ('heuristic', 'ekf', 'ekf_ad', 'ekf_clinv', 'true_state')
    if observer not in _VALID_OBSERVERS:
        raise ValueError(
            f"observer must be one of {_VALID_OBSERVERS}; got {observer!r}")
    if observer == 'ekf' and kalman_filter is None:
        raise ValueError("observer='ekf' requires a kalman_filter instance")
    if observer in ('ekf_ad', 'ekf_clinv') and kalman_filter is None:
        raise ValueError(f"observer={observer!r} requires a kalman_filter (NonlinearEKF) instance")

    q_dyn    = 0.5 * 1.225 * U_INF**2 * 0.05
    M_hh     = M_WING + M_FLAP
    M_aa     = I_WING + I_FLAP_EA
    M_ha     = M_FLAP * _D_X
    tau_leak = 5.0

    t_win      = np.linspace(0.0, T_END, int(T_END / DT) + 1)
    N          = len(t_win)

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

    # Reset EKF if provided (absolute coords: start at trim, W=0)
    if observer == 'ekf':
        xi0 = np.concatenate([x_hat, _z_trim, [0.0]])
        kalman_filter.reset(xi0)
    elif observer in ('ekf_ad', 'ekf_clinv'):
        xi0 = np.concatenate([x_hat, _z_trim, [0.0]])
        kalman_filter.reset(xi0)
        # CLInversionFusion instance expected in kalman_filter._fusion if present
        _fusion = getattr(kalman_filter, '_fusion', None)
        if _fusion is not None:
            _fusion.reset()

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
    # Estimate trajectories (for diagnostics)
    h_hat_hist  = np.zeros(N)
    hd_hat_hist = np.zeros(N)
    a_hat_hist  = np.zeros(N)
    ad_hat_hist = np.zeros(N)
    z_hat_hist  = np.zeros(N)

    delta_prev = 0.0   # used as Kalman input u_{k-1}

    for i, t in enumerate(t_win):
        h_hist[i] = x[0]; hd_hist[i] = x[1]
        a_hist[i] = x[2]; ad_hist[i] = x[3]

        # ── Control ──────────────────────────────────────────────────
        if mpc_controller is not None:
            if hasattr(mpc_controller, 'solve_tf'):
                # MPC: build W_gust_seq over horizon, then optimise
                if observer != 'true_state':
                    W_now  = W_hat_hist[i]
                    # Exponential decay W_{k+j} = W_now * λ_w^j — consistent
                    # with the EKF model W_{k+1} = λ_w * W_k used by the observer.
                    steps  = np.arange(mpc_controller.N, dtype=np.float64)
                    W_gust_seq = W_now * (0.98 ** steps)
                else:
                    horizon_idx = np.arange(i, min(i + mpc_controller.N, N))
                    W_gust_seq  = W_gust_arr[horizon_idx]
                    if len(W_gust_seq) < mpc_controller.N:
                        W_gust_seq = np.pad(W_gust_seq,
                                            (0, mpc_controller.N - len(W_gust_seq)))
                delta, _ = mpc_controller.solve_tf(x_hat, z_hat, W_gust_seq,
                                                   CL_meas=float(C_L_hist[i]))
            else:
                # LQR: state-feedback + W_hat feed-forward
                delta = mpc_controller.solve(x_hat, z_hat, W_hat=W_hat_hist[i])
        else:
            delta = 0.0
        delta_hist[i] = delta

        # ── True system step ──────────────────────────────────────────
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

        # ── Observer ─────────────────────────────────────────────────
        if observer == 'heuristic':
            leak   = 1.0 - DT / tau_leak
            hd_hat = leak * x_hat[1] + h_ddot * DT
            ad_hat = leak * x_hat[3] + a_ddot * DT
            h_hat  = leak * x_hat[0] + hd_hat * DT
            a_hat  = leak * x_hat[2] + ad_hat * DT
            x_hat  = np.array([h_hat, hd_hat, a_hat, ad_hat])

            if use_aoa_sensor:
                x_hat = np.array([h_hat, hd_hat, x[2], x[3]])

            W_hat = _estimate_W_from_CL(aero_model, z_hat, x_hat, delta,
                                         C_L_hist[i], U_INF, DT)
            z_hat, _, _ = aero_model.step(
                z_hat, x_hat[0], x_hat[1], x_hat[2], x_hat[3],
                delta, W_hat, U_INF, DT)

            if i + 1 < N:
                W_hat_hist[i + 1] = W_hat

        elif observer == 'ekf':
            # EKF: nonlinear prediction + structural update;
            # W estimated by C_L bisection (same as heuristic observer)
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
            # NonlinearEKF with time-varying AD Jacobians.
            # Measurements: y = [h_ddot, alpha]  (2-vector, NOT alpha_dot)
            # h_ddot is the measured vertical acceleration (from the true system
            # after structural RK4); alpha is read directly from the AoA sensor.
            y_meas = np.array([h_ddot, x[2]])   # [ḧ_meas, α_meas]

            fusion = getattr(kalman_filter, '_fusion', None)
            W_inv_arg = None
            R_W_arg   = None

            if observer == 'ekf_clinv' and fusion is not None:
                # Provide z_{k} from current EKF estimate (before update) to fusion
                # so it can use z_{k+1} after update below.
                z_now = kalman_filter.xi_hat[4:5].copy()
                # Retrieve inversion result computed at previous step
                # (fusion.update is called after push_state with previous xi_hat)
                W_inv_arg, R_W_arg, _valid = fusion.update(z_now)
                if not _valid:
                    W_inv_arg = None

            xi_hat = kalman_filter.step(u=delta_prev, y_meas=y_meas,
                                        W_inv=W_inv_arg, R_W=R_W_arg)
            x_hat  = xi_hat[:4]
            z_hat  = xi_hat[4:5]
            W_hat  = float(xi_hat[5])
            W_hat_hist[i] = W_hat
            if i + 1 < N:
                W_hat_hist[i + 1] = W_hat

            # Register current state for next-step inversion (1-step delay)
            if observer == 'ekf_clinv' and fusion is not None:
                fusion.push_state(xi_hat, delta)

        else:  # 'true_state'
            x_hat = x.copy()
            z_hat = z.copy()
            W_hat_hist[i] = W_gust_arr[i]
            W_hat = W_gust_arr[i]

        # Store estimates
        h_hat_hist[i]  = x_hat[0]
        hd_hat_hist[i] = x_hat[1]
        a_hat_hist[i]  = x_hat[2]
        ad_hat_hist[i] = x_hat[3]
        z_hat_hist[i]  = float(z_hat[0]) if hasattr(z_hat, '__len__') else float(z_hat)

        delta_prev = delta

    return {
        't':        t_win,
        'h':        h_hist,
        'hd':       hd_hist,
        'a':        a_hist,
        'ad':       ad_hist,
        'delta':    delta_hist,
        'C_L':      C_L_hist,
        'C_M':      C_M_hist,
        'h_ddot':   h_ddot_hist,
        'a_ddot':   a_ddot_hist,
        'W_hat':    W_hat_hist,
        'W_gust':   W_gust_arr,
        # Observer estimates (for diagnostics)
        'h_hat':    h_hat_hist,
        'hd_hat':   hd_hat_hist,
        'a_hat':    a_hat_hist,
        'ad_hat':   ad_hat_hist,
        'z_hat':    z_hat_hist,
    }

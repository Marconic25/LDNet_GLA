import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from structural.smd import structural_rhs, M_WING, M_FLAP, I_WING, I_FLAP_EA, D_H, D_ALPHA, K_H, K_ALPHA, _D_X

"""
Model Predictive Control (MPC) for aeroelastic flutter suppression.
Uses nonlinear LDNet model + RK4 structural integration.
"""

class MPCController:
    """
    MPC controller that solves an optimal control problem at each time step.

    Minimizes (GLA formulation):
        J = sum_{i=0}^{N-1} [ Q_CL·C_L_i² + Q_CM·C_M_i² + Q_h·h_i² + Q_a·a_i² + R·u_i² ]

    The structural state terms (h², α²) provide the reaction signal: with W=0 in the
    horizon, the MPC can't see future gust, but the current x_hat carries the integrated
    gust response. The load terms (C_L², C_M²) tell the optimizer which direction to
    deflect the flap — to cancel aerodynamic load, not just push the wing back.

    Subject to:
        x_{i+1} = f(x_i, u_i, W_gust_i=0)  (nonlinear, with LDNet, no gust model)
        -delta_max <= u_i <= delta_max

    Where:
        x = [h, hd, a, ad] is the state vector
        u = delta is the flap deflection [°]
        C_L, C_M are the aerodynamic coefficients from LDNet
        R is the control cost (scalar)
        delta_max is the maximum flap deflection (saturation)
    """

    def __init__(self, aero_model, U_INF, DT, Q_CL=0.0, Q_CM=0.0,
                 Q_h=0.0, Q_a=0.0, Q_hd=0.0, Q_ad=0.0,
                 R=1.0, R_du=0.0, N=10, delta_max=20.0, CL_trim=0.0, CM_trim=0.0,
                 use_tf_solver=False):
        self.aero_model = aero_model
        self.U_INF = U_INF
        self.DT = DT
        self.Q_CL = Q_CL
        self.Q_CM = Q_CM
        self.Q_h  = Q_h
        self.Q_a  = Q_a
        self.Q_hd = Q_hd
        self.Q_ad = Q_ad
        self.R    = R
        self.R_du = R_du   # rate cost on Δu — penalizes step changes that excite structural modes
        self.N = N
        self.delta_max = delta_max
        self.CL_trim = CL_trim
        self.CM_trim = CM_trim

        self.use_tf_solver = use_tf_solver
        self.k_prev        = 0.0
        self.u_prev        = np.zeros(N)
        self.delta_applied = 0.0

        if use_tf_solver:
            self._tf_u_var     = tf.Variable(np.zeros(N, dtype=np.float64), trainable=True)
            self._tf_opt       = tf.keras.optimizers.Adam(learning_rate=0.5)
            self._tf_adam_step = self._build_tf_step()

    def _rk4_step(self, x, Fy, Mz, DT):
        """RK4 integration for structural dynamics. Given current state x and aerodynamic forces Fy, Mz, compute next state."""
        def struct_rhs_wrapper(state):
            """Wrapper to compute structural RHS given state and current aerodynamic forces.Freezes the rest of the arguments (t, Fy, Mz) for RK4."""
            return np.array(structural_rhs(0.0, state, Fy, Mz, 0.0, 0.0))

        k1 = struct_rhs_wrapper(x)
        k2 = struct_rhs_wrapper(x + 0.5 * DT * k1)
        k3 = struct_rhs_wrapper(x + 0.5 * DT * k2)
        k4 = struct_rhs_wrapper(x + DT * k3)

        return x + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _predict_trajectory(self, x_hat, z_hat, W_gust_seq, u_seq, gust_phase=True, u_prev_applied=0.0):
        """
        Predict trajectory and compute GLA cost.

        The cost penalizes aerodynamic loads directly:
            J = Σ [ Q_CL·C_L_i² + Q_CM·C_M_i² + R·u_i² ]

        This is the standard GLA formulation: the flap cancels gust-induced
        lift increments rather than reacting to structural displacements.

        Args:
            x_hat: estimated state [h, hd, a, ad]
            z_hat: estimated LDNet latent state
            W_gust_seq: gust velocities for next N steps (zeros — no gust model)
            u_seq: control sequence [u_0, ..., u_{N-1}]

        Returns:
            J: total cost
        """
        x = x_hat.copy()
        z = z_hat.copy()
        J = 0.0
        u_prev_i = u_prev_applied

        for i in range(self.N):
            u = np.clip(u_seq[i], -self.delta_max, self.delta_max)

            z_new, C_L, C_M = self.aero_model.step(
                z, x[0], x[1], x[2], x[3],
                u, W_gust_seq[i], self.U_INF, self.DT
            )

            Fy = 0.5 * 1.225 * self.U_INF**2 * 0.05 * C_L
            Mz = 0.5 * 1.225 * self.U_INF**2 * 0.05 * 1.0 * C_M

            x_new = self._rk4_step(x, Fy, Mz, self.DT)

            load_cost    = float(float(C_L)**2 * self.Q_CL +
                                 float(C_M)**2 * self.Q_CM)
            s_h  = self.Q_h  if gust_phase else 0.0
            s_a  = self.Q_a  if gust_phase else 0.0
            s_hd = self.Q_hd if gust_phase else 0.0
            s_ad = self.Q_ad if gust_phase else 0.0
            state_cost   = float(x[0]**2 * s_h + x[2]**2 * s_a +
                                 x[1]**2 * s_hd + x[3]**2 * s_ad)
            control_cost = float(u**2 * self.R)
            rate_cost    = float((u - u_prev_i)**2 * self.R_du)

            J += load_cost + state_cost + control_cost + rate_cost

            x = x_new
            z = z_new
            u_prev_i = u

        return J

    def solve(self, x_hat, z_hat, W_gust_seq, gust_phase=True):
        """
        Solve MPC optimization problem.

        Args:
            x_hat: estimated state [h, hd, a, ad]
            z_hat: estimated LDNet latent state
            W_gust_seq: gust sequence for next N steps (zeros — no gust model)

        Returns:
            u_opt[0]: optimal control input for this step
            u_opt: full optimal sequence (for warm-start next iteration)
        """
        # Warm-start from previous solution (shift and zero-pad)
        u_warm = np.roll(self.u_prev, -1)
        u_warm[-1] = 0.0

        # 1D scalar search: find optimal gain k such that u = k * delta_max
        # This reduces the N-D problem to 1D — solvable in ~20 LDNet evaluations.
        # Rationale: during a gust, the optimal sequence is approximately constant
        # (same sign, similar magnitude), so a scalar gain captures most of the benefit.
        from scipy.optimize import minimize_scalar
        u_applied = self.delta_applied

        def objective_1d(k):
            u_seq = np.clip(np.full(self.N, k * self.delta_max),
                            -self.delta_max, self.delta_max)
            return self._predict_trajectory(x_hat, z_hat, W_gust_seq, u_seq, gust_phase,
                                            u_prev_applied=u_applied)

        res = minimize_scalar(objective_1d, bounds=(-1.0, 1.0), method='bounded',
                              options={'xatol': 0.02, 'maxiter': 20})
        k_opt = res.x
        self.k_prev = k_opt
        u_opt = np.clip(np.full(self.N, k_opt * self.delta_max),
                        -self.delta_max, self.delta_max)

        self.u_prev = u_opt
        self.delta_applied = float(u_opt[0])
        return u_opt[0], u_opt

    def _build_tf_step(self):
        """Build and return a @tf.function-compiled Adam step for the rollout.
        Called once from __init__ after all constants are known."""
        dtype  = tf.float64
        dm     = tf.constant(self.delta_max, dtype=dtype)
        U_INF  = tf.constant(self.U_INF, dtype=dtype)
        DT     = tf.constant(self.DT, dtype=dtype)
        q_dyn  = tf.constant(0.5 * 1.225 * self.U_INF**2 * 0.05, dtype=dtype)

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
        def adam_step(x0, z0, W_s, u0_prev,
                      Q_CL, Q_CM, Q_h, Q_a, R, R_du):
            with tf.GradientTape() as tape:
                u_cl = tf.clip_by_value(u_var, -dm, dm)
                x = x0; z = z0
                J = tf.zeros((), dtype=dtype)
                u_p = u0_prev
                for i in tf.range(N):
                    u_i = u_cl[i]
                    z, C_L, C_M = aero.step_tf(
                        z, x[0], x[1], x[2], x[3],
                        u_i, W_s[i], U_INF, DT)
                    x = rk4(x, q_dyn*C_L, q_dyn*C_M)
                    J = J + (Q_CL*C_L**2 + Q_CM*C_M**2
                             + Q_h*x[0]**2 + Q_a*x[2]**2
                             + R*u_i**2 + R_du*(u_i - u_p)**2)
                    u_p = u_i
            opt.apply_gradients([(tape.gradient(J, u_var), u_var)])

        return adam_step

    def solve_tf(self, x_hat, z_hat, W_gust_seq, gust_phase=True,
                 n_steps=15):
        """N-D solver: Adam on full u-sequence via @tf.function rollout."""
        dtype = tf.float64
        dm    = float(self.delta_max)

        self._tf_u_var.assign(np.clip(self.u_prev, -dm, dm).astype(np.float64))

        x0      = tf.constant(x_hat,      dtype=dtype)
        z0      = tf.constant(z_hat,      dtype=dtype)
        W_s     = tf.constant(W_gust_seq, dtype=dtype)
        u0_prev = tf.constant(float(self.delta_applied), dtype=dtype)

        Q_CL = tf.constant(float(self.Q_CL), dtype=dtype)
        Q_CM = tf.constant(float(self.Q_CM), dtype=dtype)
        Q_h  = tf.constant(float(self.Q_h) if gust_phase else 0.0, dtype=dtype)
        Q_a  = tf.constant(float(self.Q_a) if gust_phase else 0.0, dtype=dtype)
        R    = tf.constant(float(self.R),    dtype=dtype)
        R_du = tf.constant(float(self.R_du), dtype=dtype)

        for _ in range(n_steps):
            self._tf_adam_step(x0, z0, W_s, u0_prev,
                               Q_CL, Q_CM, Q_h, Q_a, R, R_du)

        u_opt = np.clip(self._tf_u_var.numpy(), -dm, dm)
        self.u_prev = u_opt
        self.delta_applied = float(u_opt[0])
        self.k_prev = float(u_opt[0]) / dm if dm > 0 else 0.0
        return float(u_opt[0]), u_opt


def _estimate_W_from_CL(aero_model, z_hat, x_hat, delta, C_L_meas, U_INF, DT,
                         W_lo=0.0, W_hi=80.0, tol=0.5):
    """
    Estimate W_gust by inverting C_L = LDNet(z_hat, x_hat, delta, W).

    C_L(W) is monotonically increasing for W >= 0 (verified numerically).
    Uses bisection — cheap (< 20 LDNet evaluations).

    Args:
        aero_model: LDNetModel
        z_hat, x_hat: current observer state
        delta: current flap deflection [deg]
        C_L_meas: measured C_L from real system
        U_INF, DT: flight condition
        W_lo, W_hi: search bracket [m/s]
        tol: convergence tolerance on W [m/s]

    Returns:
        W_hat: estimated gust velocity [m/s]
    """
    def CL_pred(W):
        _, CL, _ = aero_model.step(z_hat, x_hat[0], x_hat[1], x_hat[2], x_hat[3],
                                    delta, W, U_INF, DT)
        return float(CL)

    flo = CL_pred(W_lo) - C_L_meas
    fhi = CL_pred(W_hi) - C_L_meas

    # If measurement is outside bracket, clamp
    if flo * fhi > 0:
        return W_lo if abs(flo) < abs(fhi) else W_hi

    for _ in range(20):
        W_mid = 0.5 * (W_lo + W_hi)
        if (W_hi - W_lo) < tol:
            break
        fmid = CL_pred(W_mid) - C_L_meas
        if flo * fmid <= 0:
            W_hi = W_mid
            fhi  = fmid
        else:
            W_lo = W_mid
            flo  = fmid

    return 0.5 * (W_lo + W_hi)


def run_mpc_simulation(U_INF, T_END, DT, aero_model, mpc_controller, A_s, B_s,
                       use_ekf=True, Q_noise=None, R_noise=None,
                       L_g=50.0, sigma_w=5.0, W_obs_gain=0.3,
                       gust_profile=None, gust_gate=False):
    """
    Closed-loop MPC simulation with C_L-inversion gust observer.

    Observer (use_ekf=True):
      - W_hat: estimated each step by inverting C_L_meas = LDNet(z_hat, x_hat, δ, W).
        C_L(W) is monotone for W>=0, so bisection gives a unique solution.
      - Structural state [h, hd, a, ad]: leaky kinematic double-integrator.
      - LDNet latent z_hat: advanced with W_hat (not W=0).
      - W_gust_seq passed to MPC: [W_hat, W_hat, ..., W_hat] (constant over horizon).

    Args:
        U_INF: freestream velocity [m/s]
        T_END: simulation end time [s]
        DT: time step [s]
        aero_model: LDNetModel instance
        mpc_controller: MPCController instance (None → baseline)
        A_s, B_s: structural state/input matrices
        use_ekf: if True, use C_L observer; if False, use true state (oracle)
        Q_noise, R_noise, L_g, sigma_w, W_obs_gain: kept for API compatibility (unused)

    Returns:
        dict with time history
    """
    from aerodynamics.model import LDNetModel
    from structural.smd import structural_rhs, M_WING, M_FLAP, I_WING, I_FLAP_EA, \
                                D_H, D_ALPHA, K_H, K_ALPHA, _D_X

    rho_inf = 1.225
    S_ref   = 0.05
    c       = 1.0
    q_dyn   = 0.5 * rho_inf * U_INF**2 * S_ref   # dynamic pressure × ref area [N]

    M_hh     = M_WING + M_FLAP
    M_aa     = I_WING + I_FLAP_EA
    M_ha     = M_FLAP * _D_X
    det_mass = M_hh * M_aa - M_ha**2

    # Time grid
    t_win = np.linspace(0.0, T_END, int(T_END / DT) + 1)
    N = len(t_win)

    # ── Gust profile (1-cosine default, or caller-supplied callable) ─────────
    if gust_profile is None:
        def gust_profile(t):
            GUST_W0 = 60.0; GUST_T_END = 0.8
            if 0.0 <= t <= GUST_T_END:
                return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / GUST_T_END))
            return 0.0

    W_gust_arr = np.array([gust_profile(t) for t in t_win])

    # ── Trim latent state (computed once; used to init z and as observer baseline) ──
    _z_trim = np.zeros(aero_model.num_latent_states)
    for _ in range(200):
        _z_trim, _, _ = aero_model.step(_z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)

    # ── True system state ──────────────────────────────────────────────────
    x = np.zeros(4)                                    # [h, hd, a, ad]
    z = _z_trim.copy()                                 # start at trim latent

    # ── Observer state ─────────────────────────────────────────────────────
    x_hat = np.zeros(4)      # [h, hd, a, ad] — structural state estimate
    z_hat = _z_trim.copy()   # LDNet latent state estimate (starts at trim)

    # Leaky integrator time constant — prevents drift while tracking oscillations
    tau_leak = 5.0   # [s]

    # History arrays
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

    # MPC always active — unified GLA + flutter suppression cost.
    # No on/off switching: the controller is conservative at rest (R dominates)
    # and aggressive during gusts (Q_CL, Q_CM, Q_h, Q_a rise with gust response).
    # W_hat drives the gust feed-forward; structural velocities provide damping.

    # Simulation loop
    for i, t in enumerate(t_win):
        # Save state
        h_hist[i]     = x[0]
        hd_hist[i]    = x[1]
        a_hist[i]     = x[2]
        ad_hist[i]    = x[3]

        if gust_gate:
            mpc_enabled = bool(W_gust_arr[i] > 0.0)
            gust_phase  = mpc_enabled
        else:
            mpc_enabled = True
            gust_phase  = True

        # ─────────────────────────────────────────────────────────────
        # MPC CONTROL (or baseline: delta = 0)
        # ─────────────────────────────────────────────────────────────
        if mpc_controller is not None:
            if use_ekf:
                # C_L observer: constant W_hat over horizon (zero-order hold)
                W_gust_seq = np.full(mpc_controller.N, W_hat_hist[i])
            else:
                # Oracle: true future gust over horizon
                horizon_idx = np.arange(i, min(i + mpc_controller.N, N))
                W_gust_seq = W_gust_arr[horizon_idx]
                if len(W_gust_seq) < mpc_controller.N:
                    W_gust_seq = np.pad(W_gust_seq,
                                        (0, mpc_controller.N - len(W_gust_seq)))
            if mpc_enabled:
                if mpc_controller.use_tf_solver:
                    delta, _ = mpc_controller.solve_tf(x_hat, z_hat, W_gust_seq,
                                                        gust_phase=gust_phase)
                else:
                    delta, _ = mpc_controller.solve(x_hat, z_hat, W_gust_seq,
                                                     gust_phase=gust_phase)
            else:
                delta = 0.0
                mpc_controller.k_prev = 0.0
                mpc_controller.delta_applied = 0.0
            delta_hist[i] = delta
        else:
            delta = 0.0
            delta_hist[i] = delta

        # ─────────────────────────────────────────────────────────────
        # REAL SYSTEM STEP
        # ─────────────────────────────────────────────────────────────
        z, C_L, C_M = aero_model.step(z, x[0], x[1], x[2], x[3],
                                       delta, W_gust_arr[i], U_INF, DT)
        C_L_hist[i] = C_L
        C_M_hist[i] = C_M

        Fy = q_dyn * C_L
        Mz = q_dyn * c * C_M

        def _srhs(s):
            return np.array(structural_rhs(t, s, Fy, Mz, 0.0, 0.0))

        k1 = _srhs(x)
        k2 = _srhs(x + 0.5*DT*k1)
        k3 = _srhs(x + 0.5*DT*k2)
        k4 = _srhs(x + DT*k3)
        x  = x + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        _, h_ddot, _, a_ddot = _srhs(x)
        h_ddot_hist[i] = h_ddot
        a_ddot_hist[i] = a_ddot

        # ─────────────────────────────────────────────────────────────
        # OBSERVER
        # ─────────────────────────────────────────────────────────────
        if use_ekf:
            # 1. Structural state: leaky kinematic integrator from accelerometers
            leak   = 1.0 - DT / tau_leak
            hd_hat = leak * x_hat[1] + h_ddot * DT
            ad_hat = leak * x_hat[3] + a_ddot * DT
            h_hat  = leak * x_hat[0] + hd_hat * DT
            a_hat  = leak * x_hat[2] + ad_hat * DT
            x_hat  = np.array([h_hat, hd_hat, a_hat, ad_hat])

            # 2. Gust estimate: only when MPC is active (gust window)
            #    Outside gust window W_hat=0 — avoids spurious estimates
            #    from structural oscillations saturating the bisection
            if mpc_enabled:
                W_hat = _estimate_W_from_CL(aero_model, z_hat, x_hat, delta,
                                             C_L_hist[i], U_INF, DT)
            else:
                W_hat = 0.0

            # 3. Advance z_hat with W_hat (gust-corrected aero history)
            z_hat, _, _ = aero_model.step(
                z_hat, x_hat[0], x_hat[1], x_hat[2], x_hat[3],
                delta, W_hat, U_INF, DT)

            # Store W_hat for next MPC call
            if i + 1 < N:
                W_hat_hist[i + 1] = W_hat
        else:
            # Oracle mode: use true state and true gust
            x_hat = x.copy()
            z_hat = z.copy()
            W_hat_hist[i] = W_gust_arr[i]

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
    }
import numpy as np
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
                 R=1.0, N=10, delta_max=20.0):
        """
        Initialize MPC controller.

        Args:
            aero_model: LDNetModel instance
            U_INF: freestream velocity [m/s]
            DT: time step [s]
            Q_CL: cost weight on C_L² (lift load — only useful with W known)
            Q_CM: cost weight on C_M² (pitching moment load)
            Q_h:  cost weight on h²   (heave displacement)
            Q_a:  cost weight on α²   (pitch angle)
            Q_hd: cost weight on ḣ²   (heave velocity — damping signal)
            Q_ad: cost weight on α̇²   (pitch rate — damping signal)
            R: control cost (scalar)
            N: prediction horizon (number of steps)
            delta_max: control input saturation [°]
        """
        self.aero_model = aero_model
        self.U_INF = U_INF
        self.DT = DT
        self.Q_CL = Q_CL
        self.Q_CM = Q_CM
        self.Q_h  = Q_h
        self.Q_a  = Q_a
        self.Q_hd = Q_hd
        self.Q_ad = Q_ad
        self.R = R
        self.N = N
        self.delta_max = delta_max

        # Previous optimal control sequence (for warm-start)
        self.u_prev = np.zeros(N)

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

    def _predict_trajectory(self, x_hat, z_hat, W_gust_seq, u_seq):
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

        for i in range(self.N):
            u = np.clip(u_seq[i], -self.delta_max, self.delta_max)

            z_new, C_L, C_M = self.aero_model.step(
                z, x[0], x[1], x[2], x[3],
                u, W_gust_seq[i], self.U_INF, self.DT
            )

            Fy = 0.5 * 1.225 * self.U_INF**2 * 0.05 * C_L
            Mz = 0.5 * 1.225 * self.U_INF**2 * 0.05 * 1.0 * C_M

            x_new = self._rk4_step(x, Fy, Mz, self.DT)

            load_cost    = float(C_L**2 * self.Q_CL + C_M**2 * self.Q_CM)
            state_cost   = float(x[0]**2 * self.Q_h  + x[2]**2 * self.Q_a +
                                 x[1]**2 * self.Q_hd + x[3]**2 * self.Q_ad)
            control_cost = float(u**2 * self.R)

            J += load_cost + state_cost + control_cost

            x = x_new
            z = z_new

        return J

    def solve(self, x_hat, z_hat, W_gust_seq):
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
        def objective_1d(k):
            u_seq = np.clip(np.full(self.N, k * self.delta_max),
                            -self.delta_max, self.delta_max)
            return self._predict_trajectory(x_hat, z_hat, W_gust_seq, u_seq)

        res = minimize_scalar(objective_1d, bounds=(-1.0, 1.0), method='bounded',
                              options={'xatol': 0.02, 'maxiter': 20})
        k_opt = res.x
        u_opt = np.clip(np.full(self.N, k_opt * self.delta_max),
                        -self.delta_max, self.delta_max)

        self.u_prev = u_opt
        return u_opt[0], u_opt


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
                       L_g=50.0, sigma_w=5.0, W_obs_gain=0.3):
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

    # ── Gust profile (1-cosine) ────────────────────────────────────────────
    def gust_velocity(t):
        GUST_W0 = 60.0; GUST_T_END = 0.8
        if 0.0 <= t <= GUST_T_END:
            return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / GUST_T_END))
        return 0.0

    W_gust_arr = np.array([gust_velocity(t) for t in t_win])

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

    # Simulation loop
    for i, t in enumerate(t_win):
        # Save state
        h_hist[i]     = x[0]
        hd_hist[i]    = x[1]
        a_hist[i]     = x[2]
        ad_hist[i]    = x[3]

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
            delta, _   = mpc_controller.solve(x_hat, z_hat, W_gust_seq)
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

            # 2. Gust estimate: invert C_L_meas = LDNet(z_hat, x_hat, delta, W)
            #    C_L(W) is monotone for W>=0 → bisection gives unique W_hat
            W_hat = _estimate_W_from_CL(aero_model, z_hat, x_hat, delta,
                                         C_L_hist[i], U_INF, DT)

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
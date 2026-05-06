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
                 use_tf_solver=False, ddelta_max=None,
                 Q_CL_terminal=0.0, Q_h_terminal=0.0, Q_a_terminal=0.0):
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
        self.N              = N
        self.delta_max      = delta_max
        self.CL_trim        = CL_trim
        self.CM_trim        = CM_trim
        self.Q_CL_terminal  = Q_CL_terminal
        self.Q_h_terminal   = Q_h_terminal
        self.Q_a_terminal   = Q_a_terminal

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
        CL_tr  = tf.constant(float(self.CL_trim),       dtype=dtype)
        CM_tr  = tf.constant(float(self.CM_trim),       dtype=dtype)
        Q_CLf  = tf.constant(float(self.Q_CL_terminal), dtype=dtype)
        Q_hf   = tf.constant(float(self.Q_h_terminal),  dtype=dtype)
        Q_af   = tf.constant(float(self.Q_a_terminal),  dtype=dtype)

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
                # Terminal cost on final state
                J = J + Q_CLf*C_L**2 + Q_hf*x[0]**2 + Q_af*x[2]**2
            opt.apply_gradients([(tape.gradient(J, u_var), u_var)])

        return adam_step

    def solve_tf(self, x_hat, z_hat, W_gust_seq, CL_meas=0.0,
                 gust_phase=True, n_steps=30):
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



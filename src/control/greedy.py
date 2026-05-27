"""
Greedy one-step-ahead controller (N=1 MPC) for aeroelastic GLA.

At each timestep solves:
    min_{δ}  Q_h * h(k+1)² + Q_a * a(k+1)² + Q_CL * C_L(k+1)² + R * δ²
subject to |δ| ≤ delta_max

where x(k+1) is predicted by:
    1. aero_model.predict(x_hat, δ, W_hat, U) → C_L, C_M
    2. one RK4 step of structural_rhs

Phase 1: aero_model is LinearAeroModel (Theodorsen).
Phase 2: swap for LDNetModel — only the predict() call changes.

Interface: .solve(x_hat, z_hat, W_hat) -> delta [degrees]
"""
import numpy as np
from scipy.optimize import minimize_scalar
from structural.smd import structural_rhs


# Aero → force conversion constants (same as run_simulation)
_RHO     = 1.225   # [kg/m³]
_S_REF   = 0.05    # [m²]
_C_REF   = 1.0     # [m]


class GreedyN1Controller:
    """
    One-step greedy optimal controller.

    Parameters
    ----------
    aero_predict : callable(x, delta_deg, W, U) -> (C_L, C_M)
        Aerodynamic prediction function. Use LinearAeroModel.predict for Phase 1,
        or wrap LDNetModel.step for Phase 2.
    U_INF   : float  — freestream velocity [m/s]
    DT      : float  — timestep [s]
    Q_h     : float  — cost weight on heave h [m]
    Q_a     : float  — cost weight on pitch α [rad]
    Q_CL    : float  — cost weight on lift coefficient C_L (0 = disabled)
    R       : float  — cost weight on control effort δ [deg]
    delta_max      : float  — saturation limit [degrees]
    delta_dot_max  : float  — rate limit [degrees/s]
    """

    def __init__(self, aero_predict, U_INF=80.0, DT=0.01,
                 Q_h=1e4, Q_a=1e4, Q_CL=0.0, R=1.0,
                 delta_max=20.0, delta_dot_max=100.0):
        self.aero_predict  = aero_predict
        self.U_INF         = float(U_INF)
        self.DT            = float(DT)
        self.Q_h           = float(Q_h)
        self.Q_a           = float(Q_a)
        self.Q_CL          = float(Q_CL)
        self.R             = float(R)
        self.delta_max     = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)
        self._delta_prev   = 0.0

        q_dyn = 0.5 * _RHO * U_INF**2 * _S_REF
        self._q_dyn = q_dyn

    def _predict_next(self, x_hat, delta_deg, W_hat):
        """One RK4 step; returns (x_next, C_L_next)."""
        C_L, C_M = self.aero_predict(x_hat, delta_deg, W_hat, self.U_INF)
        Fy = self._q_dyn * C_L
        Mz = self._q_dyn * C_M * _C_REF

        t = 0.0
        def rhs(s):
            return np.array(structural_rhs(t, s, Fy, Mz, 0.0, 0.0))

        x = np.array(x_hat, dtype=float)
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * self.DT * k1)
        k3 = rhs(x + 0.5 * self.DT * k2)
        k4 = rhs(x + self.DT * k3)
        x_next = x + (self.DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next, float(C_L)

    def _cost(self, delta_deg, x_hat, W_hat):
        x_next, C_L = self._predict_next(x_hat, delta_deg, W_hat)
        h_next = x_next[0]
        a_next = x_next[2]
        return (self.Q_h  * h_next**2
              + self.Q_a  * a_next**2
              + self.Q_CL * C_L**2
              + self.R    * delta_deg**2)

    def solve(self, x_hat, z_hat=None, W_hat=0.0):
        """
        Compute optimal one-step control.

        Parameters
        ----------
        x_hat : array-like, shape (4,)  — [h, ḣ, α, α̇]
        z_hat : ignored in Phase 1 (linear aero has no latent state)
        W_hat : float  — estimated gust velocity [m/s]

        Returns
        -------
        delta : float  — flap deflection [degrees]
        """
        # Apply rate limit to search bounds
        dot_limit = self.delta_dot_max * self.DT
        lb = max(-self.delta_max, self._delta_prev - dot_limit)
        ub = min( self.delta_max, self._delta_prev + dot_limit)

        if lb >= ub:
            return float(np.clip(self._delta_prev, -self.delta_max, self.delta_max))

        result = minimize_scalar(
            self._cost,
            bounds=(lb, ub),
            method='bounded',
            args=(x_hat, float(W_hat)),
            options={'xatol': 0.01}
        )
        if not np.isfinite(result.fun):
            return float(np.clip(self._delta_prev, -self.delta_max, self.delta_max))
        delta = float(np.clip(result.x, -self.delta_max, self.delta_max))
        self._delta_prev = delta
        return delta

    def reset(self):
        self._delta_prev = 0.0

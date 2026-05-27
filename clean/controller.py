"""
One-step optimal controller for aeroelastic gust load alleviation.

At each time step the controller solves the scalar optimisation:

    min_{δ}  Q_h · h(k+1)²  +  Q_α · α(k+1)²  +  Q_CL · C_L(k+1)²  +  R · δ²

subject to   |δ| ≤ δ_max   and   |δ - δ_prev| / dt ≤ δ̇_max

The next state x(k+1) and lift coefficient C_L(k+1) are predicted by:
  1. calling the aerodynamic model with the candidate δ and the current
     gust estimate W_hat;
  2. advancing the structural state one RK4 step with the resulting forces.

Because the cost is smooth and the search is over a scalar bounded interval,
scipy.optimize.minimize_scalar with method='bounded' is fast and reliable.

This controller is model-agnostic: swapping `aero_predict` for an LDNet
wrapper gives the neural-network version with no other changes.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from structure import rhs as structural_rhs

# Air and geometry constants (must match the values used in the simulation)
RHO   = 1.225   # air density [kg/m³]
S_REF = 0.05    # reference wing area [m²]
C_REF = 1.0     # reference chord [m]


class Controller:
    """
    One-step optimal controller.

    Parameters
    ----------
    aero_predict : callable(state, delta_deg, W, U) -> (C_L, C_M)
        Aerodynamic prediction function.  Pass `aero.predict` for the linear
        model or a wrapper around LDNet for the neural-network version.
    U            : float   freestream velocity [m/s]
    dt           : float   simulation time step [s]
    Q_h          : float   cost weight on heave displacement h [m]
    Q_alpha      : float   cost weight on pitch angle α [rad]
    Q_alpha_dot  : float   cost weight on pitch rate α̇ [rad/s] — damps pitch oscillations
    Q_CL         : float   cost weight on lift coefficient C_L
    R            : float   cost weight on control effort δ [deg]
    delta_max    : float   flap deflection limit [deg]
    delta_dot_max: float   flap rate limit [deg/s]
    """

    def __init__(self, aero_predict, U=80.0, dt=0.01,
                 Q_h=1e4, Q_alpha=1e4, Q_alpha_dot=1e4, Q_CL=1e3, R=1.0,
                 delta_max=20.0, delta_dot_max=100.0):
        self.aero_predict  = aero_predict
        self.U             = float(U)
        self.dt            = float(dt)
        self.Q_h           = float(Q_h)
        self.Q_alpha       = float(Q_alpha)
        self.Q_alpha_dot   = float(Q_alpha_dot)
        self.Q_CL          = float(Q_CL)
        self.R             = float(R)
        self.delta_max     = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)

        self._q_dyn    = 0.5 * RHO * U**2 * S_REF
        self._delta_prev = 0.0

    # ------------------------------------------------------------------
    def _predict_next(self, state, delta_deg, W_hat):
        """
        Predict the state and C_L one step ahead for a given δ.

        Uses the aerodynamic model to get forces, then integrates
        the structural equations with RK4.
        """
        C_L, C_M = self.aero_predict(state, delta_deg, W_hat, self.U)

        Fy = self._q_dyn * C_L
        Mz = self._q_dyn * C_M * C_REF

        x = np.asarray(state, dtype=float)
        dt = self.dt

        def f(s):
            return np.array(structural_rhs(s, Fy, Mz))

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return x_next, float(C_L)

    # ------------------------------------------------------------------
    def _cost(self, delta_deg, state, W_hat):
        """Objective function evaluated at a candidate δ."""
        x_next, C_L = self._predict_next(state, delta_deg, W_hat)
        return (  self.Q_h         * x_next[0]**2
                + self.Q_alpha     * x_next[2]**2
                + self.Q_alpha_dot * x_next[3]**2
                + self.Q_CL        * C_L**2
                + self.R           * delta_deg**2)

    # ------------------------------------------------------------------
    def compute(self, state, W_hat=0.0):
        """
        Compute the optimal flap deflection for the current time step.

        Parameters
        ----------
        state : array-like (h, ḣ, α, α̇)
        W_hat : float   estimated gust velocity [m/s] (from observer)

        Returns
        -------
        delta : float   flap deflection command [degrees]
        """
        # Rate limit: constrain search interval to reachable δ values
        reach = self.delta_dot_max * self.dt
        lb = max(-self.delta_max, self._delta_prev - reach)
        ub = min( self.delta_max, self._delta_prev + reach)

        if lb >= ub:
            delta = float(np.clip(self._delta_prev, -self.delta_max, self.delta_max))
            self._delta_prev = delta
            return delta

        result = minimize_scalar(
            self._cost,
            bounds=(lb, ub),
            method='bounded',
            args=(state, float(W_hat)),
            options={'xatol': 0.01}
        )

        # result.x is guaranteed within [lb, ub] ⊆ [-delta_max, delta_max]
        delta = float(result.x)
        self._delta_prev = delta
        return delta

    def reset(self):
        """Reset internal state (call before each new simulation run)."""
        self._delta_prev = 0.0

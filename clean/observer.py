"""
State and gust observer for the aeroelastic GLA system.

Assumed sensor suite:
  - heave accelerometer  → h_ddot (measured)
  - pitch encoder        → α (measured directly)
  - pitch rate gyro      → α̇ (measured directly)

Heave displacement h and heave velocity ḣ are not directly measurable,
so they are reconstructed by integrating h_ddot with a leaky integrator.
The leak (time constant TAU_LEAK) prevents unbounded drift from sensor bias.

Gust velocity W is not measured.  It is estimated by inverting the
aerodynamic model: given the current state estimate and flap angle,
find the W that matches the measured C_L (bisection on a scalar equation).
"""
import numpy as np
import aero as _default_aero


TAU_LEAK = 5.0   # leaky integrator time constant [s]


class Observer:
    """
    Kinematic observer: integrates accelerometer + reads encoder/gyro.

    Parameters
    ----------
    dt : float   simulation time step [s]
    U  : float   freestream velocity [m/s]  (needed for gust estimation)
    """

    def __init__(self, dt, U, aero_module=None):
        self.dt    = float(dt)
        self.U     = float(U)
        self._aero = aero_module if aero_module is not None else _default_aero
        self._x_hat = np.zeros(4)   # [h, ḣ, α, α̇]
        self._W_hat = 0.0

    def reset(self):
        self._x_hat = np.zeros(4)
        self._W_hat = 0.0

    def update(self, h_ddot, alpha, alpha_dot, delta, C_L_meas):
        """
        Update state and gust estimates from the latest sensor readings.

        Parameters
        ----------
        h_ddot    : float   heave acceleration from accelerometer [m/s²]
        alpha     : float   pitch angle from encoder [rad]
        alpha_dot : float   pitch rate from rate gyro [rad/s]
        delta     : float   current flap deflection [deg]  (needed to invert C_L)
        C_L_meas  : float   measured lift coefficient [-]

        Returns
        -------
        x_hat : np.ndarray (h, ḣ, α, α̇)   structural state estimate
        W_hat : float                        gust velocity estimate [m/s]
        """
        leak = 1.0 - self.dt / TAU_LEAK

        hd_hat = leak * self._x_hat[1] + h_ddot * self.dt
        h_hat  = leak * self._x_hat[0] + hd_hat * self.dt

        # Pitch states come directly from sensors — no integration needed
        self._x_hat = np.array([h_hat, hd_hat, alpha, alpha_dot])

        # Estimate gust velocity by inverting C_L = f(state, delta, W)
        self._W_hat = self._estimate_gust(C_L_meas, delta)

        return self._x_hat.copy(), self._W_hat

    def _estimate_gust(self, C_L_meas, delta, W_lo=0.0, W_hi=80.0, tol=0.5):
        """Recover gust W by minimizing |C_L(W) - C_L_meas| over [W_lo, W_hi].

        Bounded scalar minimization works regardless of monotonicity of C_L(W).
        With LDNet, predict() is a nonlinear function of the latent state z and is
        not guaranteed monotone in W, so the old sign-change bisection could fail
        and spike W_hat to W_hi=80 m/s every non-monotone step.
        """
        from scipy.optimize import minimize_scalar
        def obj(W):
            return (self._aero.predict(self._x_hat, delta, W, self.U)[0] - C_L_meas)**2
        result = minimize_scalar(obj, bounds=(W_lo, W_hi), method='bounded',
                                 options={'xatol': tol})
        return float(result.x)

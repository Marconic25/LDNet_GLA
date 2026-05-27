"""
PID controller for aeroelastic gust load alleviation.

Two-channel PD (no integrator to avoid wind-up) on heave and pitch.
Interface: .solve(x_hat, z_hat, W_hat) -> delta [degrees]
Matches the LQR interface used in run_simulation().
"""
import numpy as np


class PIDController:
    """
    Proportional-derivative controller on heave and pitch,
    with optional gust feedforward K_W * W_hat.

    Parameters
    ----------
    Kp_h, Kd_h : float  — proportional and derivative gains on heave h [m]
    Kp_a, Kd_a : float  — proportional and derivative gains on pitch α [rad]
    K_W         : float  — gust feedforward gain [deg/(m/s)]; δ += K_W * W_hat
    delta_max   : float  — saturation limit [degrees]
    delta_dot_max : float — rate limit [degrees/s]
    DT          : float  — timestep [s], used for rate limiting
    """

    def __init__(self, Kp_h=0.0, Kd_h=0.0, Kp_a=0.0, Kd_a=0.0, K_W=0.0,
                 delta_max=20.0, delta_dot_max=100.0, DT=0.01):
        self.Kp_h = float(Kp_h)
        self.Kd_h = float(Kd_h)
        self.Kp_a = float(Kp_a)
        self.Kd_a = float(Kd_a)
        self.K_W  = float(K_W)
        self.delta_max     = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)
        self.DT = float(DT)
        self._delta_prev = 0.0

    def solve(self, x_hat, z_hat, W_hat=0.0):
        """
        Compute control command.

        Parameters
        ----------
        x_hat : array-like, shape (4,)  — [h, ḣ, α, α̇]
        z_hat : ignored (no aero model)
        W_hat : float  — estimated gust [m/s], used for feedforward term

        Returns
        -------
        delta : float  — flap deflection [degrees]
        """
        h, hd, a, ad = float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), float(x_hat[3])
        delta_raw = (self.Kp_h * h + self.Kd_h * hd +
                     self.Kp_a * a + self.Kd_a * ad +
                     self.K_W  * float(W_hat))

        # Rate limit
        delta_dot = (delta_raw - self._delta_prev) / self.DT
        if abs(delta_dot) > self.delta_dot_max:
            delta_raw = self._delta_prev + np.sign(delta_dot) * self.delta_dot_max * self.DT

        # Saturate
        delta = float(np.clip(delta_raw, -self.delta_max, self.delta_max))
        self._delta_prev = delta
        return delta

    def reset(self):
        self._delta_prev = 0.0

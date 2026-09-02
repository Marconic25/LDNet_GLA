"""
MPC controller for the dagger_fom study porting the FULL validated-stable
cost recipe from clean/controller.py (Q_h, Q_alpha, Q_alpha_dot, Q_CL, R --
documented in clean/CONTROLLER_NOTES.md as "the only model-based controller
STABLE while alleviating on BOTH gust strengths"), combined with
light/optimal.py's N-step gust PREVIEW architecture (clean/controller.py only
sees a single constant W_hat per decision, not a preview vector).

controller_qadot.py (this study's first attempt) only added Q_alpha_dot on
top of light/optimal.py's cost, which uses R=3e-4 and an UNWEIGHTED C_L term
(implicit Q_CL=1) -- completely different relative scaling from
clean/controller.py's validated R=1.0 / Q_CL=1e3 ratio. Bolting Q_alpha_dot=1e4
onto that mismatched cost structure is not the same controller as the one
documented stable, which explains why it only partially helped. This file
uses the SAME absolute magnitudes as clean/controller.py's validated defaults
for all four extra terms, rather than re-deriving a ratio.

Cost per horizon step (accumulated over N preview steps, decision = one
constant delta held for the whole horizon, matching optimal.py's grid-search
mechanics):

    J = R*delta^2 + R_du*(delta-delta_prev)^2
        + sum_k [ Q_h*h_k^2 + Q_alpha*alpha_k^2
                  + Q_alpha_dot*(ad_k - ad_{k-1})^2
                  + Q_CL*(CL_k - C_L_trim)^2 ]
"""
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, '..'))  # light/
from optimal import dp45_batch  # noqa: E402


class MPCPreviewControllerStable:
    """
    Same N-step preview architecture as light/optimal.py::MPCPreviewController,
    with the full clean/controller.py-validated state-cost recipe added.

    Parameters
    ----------
    (aero, U, dt, rho, S, C, C_L_trim, N, G, delta_max, delta_dot_max: as in
    MPCPreviewController)
    Q_h, Q_alpha, Q_alpha_dot, Q_CL, R, R_du : float
        Defaults match clean/controller.py's validated-stable recipe exactly
        (Q_h=1e4, Q_alpha=1e4, Q_alpha_dot=1e4, Q_CL=1e3, R=1.0, R_du=0).
        NOTE: R*/Q_CL from light/results_cs25_combo/summary.md do NOT carry
        over here -- that grid search was calibrated against an unweighted
        (implicit Q_CL=1) cost. Changing Q_CL changes what R means; use these
        validated absolute defaults, not summary.md's R*.
    """

    def __init__(self, aero, U=80.0, dt=0.002, rho=1.225, S=0.05, C=1.0,
                 C_L_trim=0.0, N=8, G=161,
                 delta_max=14.0, delta_dot_max=300.0,
                 Q_h=1e4, Q_alpha=1e4, Q_alpha_dot=1e4, Q_CL=1e3, R=1.0, R_du=0.0):
        self.aero = aero
        self.U = float(U)
        self.dt = float(dt)
        self.q = 0.5 * float(rho) * float(U) ** 2 * float(S)
        self.C = float(C)
        self.lam = float(aero._z_leak)
        self.C_L_trim = float(C_L_trim)
        self.N = int(N)
        self.Q_h = float(Q_h)
        self.Q_alpha = float(Q_alpha)
        self.Q_alpha_dot = float(Q_alpha_dot)
        self.Q_CL = float(Q_CL)
        self.R = float(R)
        self.R_du = float(R_du)
        self.G = int(G)
        self.delta_max = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)
        self._dg = np.linspace(-self.delta_max, self.delta_max, self.G)
        self._delta_prev = 0.0
        self._num_z = int(aero._num_z)
        self._z_ctrl = np.zeros(self._num_z)   # controller's OWN latent

    def compute(self, state, w_seq, w_now):
        """Return optimal constant-flap deflection [deg]."""
        aero = self.aero
        dg = self._dg
        G = self.G
        reach = self.delta_dot_max * self.dt
        ratem = np.abs(dg - self._delta_prev) <= reach + 1e-9

        z_b = np.tile(np.asarray(self._z_ctrl, float).reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        J = self.R * dg ** 2 + self.R_du * (dg - self._delta_prev) ** 2
        ad_prev = np.full(G, float(state[3]))

        for k in range(self.N):
            Wk = float(w_seq[k]) if k < len(w_seq) else 0.0
            CL, CM, z_new = aero.batch_step(z_b, x_b, dg, Wk, self.U, self.dt)
            z_b = z_new - self.lam * z_b
            Fy = self.q * CL
            Mz = self.q * CM * self.C
            x_b = dp45_batch(x_b, Fy, Mz, self.dt)
            ad_now = x_b[:, 3]
            J = J + (self.Q_h * x_b[:, 0] ** 2
                     + self.Q_alpha * x_b[:, 2] ** 2
                     + self.Q_alpha_dot * (ad_now - ad_prev) ** 2
                     + self.Q_CL * (CL - self.C_L_trim) ** 2)
            ad_prev = ad_now

        J = np.where(ratem, J, np.inf)
        d = float(dg[int(np.argmin(J))])
        d = float(np.clip(d, self._delta_prev - reach, self._delta_prev + reach))
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        self._delta_prev = d

        self._z_ctrl = aero.advance_z(self._z_ctrl, state, d,
                                      float(w_now), self.U, self.dt)
        return d

    def reset(self):
        self._delta_prev = 0.0
        self._z_ctrl = np.zeros(self._num_z)

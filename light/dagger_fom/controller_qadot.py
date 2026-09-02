"""
Modified copy of light/optimal.py::MPCPreviewController with an added
pitch-rate move-suppression term in the horizon cost:

    J += Q_alpha_dot * (ad_{k+1} - ad_k)^2   summed over k=0..N-1

Motivation (see light/dagger_fom/NOTES.md): the real-FOM verification on the
W30/Tg0.4 cell (highest reduced frequency in the grid) shows the flap command
oscillating at the wing's pitch natural frequency (~14.4 Hz), while
light/optimal.py's MPCPreviewController has NO state/pitch penalty at all —
only flap-effort (R) and move-suppression on delta itself (R_du, documented
harmful for this exact cell/pipeline in light/noise/NOTES.md). A pitch-rate
penalty mirrors clean/controller.py's Q_alpha_dot term, documented in
clean/CONTROLLER_NOTES.md as "the only model-based controller STABLE while
alleviating on BOTH gust strengths" for an earlier controller variant.

This file does NOT modify light/optimal.py — it is a standalone copy used
only within this study. Everything else (docstring, API, grid-search
mechanics) is unchanged from MPCPreviewController; only the cost accumulation
inside compute() gains the new term, plus one constructor parameter.
"""
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, '..'))  # light/
from optimal import dp45_batch  # noqa: E402


class MPCPreviewControllerQadot:
    """
    Same as light/optimal.py::MPCPreviewController, plus Q_alpha_dot.

    Parameters
    ----------
    (all identical to MPCPreviewController, see light/optimal.py)
    Q_alpha_dot : float  pitch-rate move-suppression weight (default 0 —
                  set >0 to penalize (ad_{k+1}-ad_k)^2 over the horizon)
    """

    def __init__(self, aero, U=80.0, dt=0.002, rho=1.225, S=0.05, C=1.0,
                 C_L_trim=0.0, N=8, R=3e-4, R_du=0.0, Q_alpha_dot=0.0, G=161,
                 delta_max=14.0, delta_dot_max=300.0):
        self.aero = aero
        self.U = float(U)
        self.dt = float(dt)
        self.q = 0.5 * float(rho) * float(U) ** 2 * float(S)
        self.C = float(C)
        self.lam = float(aero._z_leak)
        self.C_L_trim = float(C_L_trim)
        self.N = int(N)
        self.R = float(R)
        self.R_du = float(R_du)
        self.Q_alpha_dot = float(Q_alpha_dot)
        self.G = int(G)
        self.delta_max = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)
        self._dg = np.linspace(-self.delta_max, self.delta_max, self.G)
        self._delta_prev = 0.0
        self._num_z = int(aero._num_z)
        self._z_ctrl = np.zeros(self._num_z)   # controller's OWN latent

    def compute(self, state, w_seq, w_now):
        """Return optimal constant-flap deflection [deg]. See MPCPreviewController."""
        aero = self.aero
        dg = self._dg
        G = self.G
        reach = self.delta_dot_max * self.dt
        ratem = np.abs(dg - self._delta_prev) <= reach + 1e-9

        z_b = np.tile(np.asarray(self._z_ctrl, float).reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        J = self.R * dg ** 2 + self.R_du * (dg - self._delta_prev) ** 2

        for k in range(self.N):
            Wk = float(w_seq[k]) if k < len(w_seq) else 0.0
            CL, CM, z_new = aero.batch_step(z_b, x_b, dg, Wk, self.U, self.dt)
            z_b = z_new - self.lam * z_b
            Fy = self.q * CL
            Mz = self.q * CM * self.C
            ad_before = x_b[:, 3]
            x_b = dp45_batch(x_b, Fy, Mz, self.dt)
            ad_after = x_b[:, 3]
            if self.Q_alpha_dot != 0.0:
                J = J + self.Q_alpha_dot * (ad_after - ad_before) ** 2
            J = J + (CL - self.C_L_trim) ** 2

        J = np.where(ratem, J, np.inf)
        d = float(dg[int(np.argmin(J))])
        d = float(np.clip(d, self._delta_prev - reach, self._delta_prev + reach))
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        self._delta_prev = d

        self._z_ctrl = aero.advance_z(self._z_ctrl, state, d,
                                      float(w_now), self.U, self.dt)
        return d

    def reset(self):
        """Reset flap state and the controller latent — call before each run."""
        self._delta_prev = 0.0
        self._z_ctrl = np.zeros(self._num_z)

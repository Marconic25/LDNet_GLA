"""
Extended Kalman Filter (EKF) for the LQG GLA controller.

State: xi_aug = [h, hd, alpha, alpha_dot, z, W]  in R^6
    - Structural states [h, hd, alpha, alpha_dot]: indices 0-3
    - Latent aerodynamic state z:                  index 4
    - Gust velocity W:                             index 5

Design:
    Prediction  — full nonlinear propagator (aero_model.step + structural RK4).
                  Handles large z excursions that invalidate a linear predictor.
    Update      — constant gain L on (alpha, alpha_dot) only (indices 1-2 of C_obs).
                  h_ddot row of C_obs depends on C_y(trim) and is inaccurate when
                  z deviates far from trim; excluding it avoids spurious corrections.
    W update    — C_L bisection (same as heuristic observer); bypasses the ill-
                  conditioned L[5,:] caused by near-unobservability of W.

Gain L is computed once from the dual DARE at trim using
scipy.linalg.solve_discrete_are (same pattern as lqr.py).
"""
import numpy as np
from scipy.linalg import solve_discrete_are
from structural.smd import (M_WING, M_FLAP, I_WING, I_FLAP_EA,
                             D_H, D_ALPHA, K_H, K_ALPHA, _D_X)

# ──────────────────────────────────────────────────────────────────────────────
# NOISE COVARIANCE DEFAULTS
# Primary tuning knobs — adjust here, not inside the class.
#
# Q_KF (process noise covariance, 6×6 diagonal):
#   h, hd, alpha, alpha_dot : small  — 2-DOF spring-mass well-modelled
#   z                        : tiny  — LDNet dynamics are accurate
#   W                        : used only for DARE gain computation; W itself
#                              is set by bisection at runtime, so this entry
#                              affects L[5,:] only (which is zeroed in step())
#
# R_KF (measurement noise covariance, 3×3 diagonal):
#   Only rows 1-2 (alpha, alpha_dot) are used in the update.
#   Row 0 (h_ddot) is kept for completeness but excluded at runtime.
# ──────────────────────────────────────────────────────────────────────────────

_Q_KF_DIAG_DEFAULT = np.array([
    1e-6,    # h          [m]
    1e-6,    # hd         [m/s]
    1e-6,    # alpha      [rad]
    1e-6,    # alpha_dot  [rad/s]
    1e-8,    # z          [—]    — LDNet latent accurate
    0.1,     # W          [m/s]  — for DARE only; runtime value set by bisection
])

_R_KF_DIAG_DEFAULT = np.array([
    (1e-2)**2,   # h_ddot  [m²/s⁴]  — accelerometer ~10 mg (unused in update)
    (1e-3)**2,   # alpha   [rad²]   — AoA encoder ~1 mrad
    (5e-3)**2,   # alpha_dot [rad²/s²] — AoA rate ~5 mrad/s
])


def build_C_obs(C_y, U_INF):
    """
    Build the linearized output matrix C_obs in R^{3 x 6}.

    Rows: [h_ddot, alpha, alpha_dot].
    Only rows 1-2 (alpha, alpha_dot) are used in the EKF update step;
    row 0 is included so the DARE can be solved on the full 3-output system.

    Parameters
    ----------
    C_y : ndarray, shape (2, 5)
        Aerodynamic output Jacobian [dC_L/dxi; dC_M/dxi] at trim,
        from compute_jacobians().
    U_INF : float
        Freestream velocity [m/s].

    Returns
    -------
    C_obs : ndarray, shape (3, 6)
    """
    q_dyn = 0.5 * 1.225 * U_INF**2 * 0.05

    M_hh = M_WING + M_FLAP
    M_aa = I_WING + I_FLAP_EA
    M_ha = M_FLAP * _D_X
    det  = M_hh * M_aa - M_ha**2

    row5 = (M_aa * (-q_dyn * C_y[0, :]) - M_ha * (q_dyn * C_y[1, :])) / det
    row5[0] += (-M_aa * K_H)     / det
    row5[1] += (-M_aa * D_H)     / det
    row5[2] += ( M_ha * K_ALPHA) / det
    row5[3] += ( M_ha * D_ALPHA) / det

    row_h_ddot    = np.append(row5, 0.0)
    row_alpha     = np.zeros(6); row_alpha[2]    = 1.0
    row_alpha_dot = np.zeros(6); row_alpha_dot[3] = 1.0

    return np.vstack([row_h_ddot, row_alpha, row_alpha_dot])   # (3, 6)


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for the augmented aeroelastic state.

    Parameters
    ----------
    aero_model : LDNetModel
    U_INF : float
    DT : float
    C_obs : ndarray, shape (3, 6)
        From build_C_obs().
    xi_trim : ndarray, shape (6,)
        Absolute trim point [x_trim(4), z_trim(1), 0].
    Q_kf : ndarray, shape (6, 6), optional
        Process noise covariance. Default: diag(_Q_KF_DIAG_DEFAULT).
    R_kf : ndarray, shape (3, 3), optional
        Measurement noise covariance. Default: diag(_R_KF_DIAG_DEFAULT).
    lambda_w : float
        Gust decay rate (default 0.98).
    xi_aug_0 : ndarray, shape (6,), optional
        Initial absolute state. Default: xi_trim.
    """

    def __init__(self, aero_model, U_INF, DT, C_obs, xi_trim,
                 Q_kf=None, R_kf=None, lambda_w=0.98, xi_aug_0=None):
        self.aero    = aero_model
        self.U_INF   = float(U_INF)
        self.DT      = float(DT)
        self.C       = C_obs.copy()       # (3, 6)
        self.lw      = float(lambda_w)
        self.xi_trim = xi_trim.copy()     # (6,)

        Q_kf = np.diag(_Q_KF_DIAG_DEFAULT) if Q_kf is None else Q_kf
        R_kf = np.diag(_R_KF_DIAG_DEFAULT) if R_kf is None else R_kf

        # Build augmented A for DARE (linearized at trim, needed only for gain)
        # A_aug is not available here directly, so we pass it in implicitly via
        # the trim Jacobian stored in C_obs. We use a simple identity-like A
        # to compute a conservative L — the structural dynamics are slow relative
        # to DT so A ≈ I is a reasonable prior for the DARE.
        # Better: caller can pass A_aug explicitly via the optional argument.
        self._Q_kf = Q_kf
        self._R_kf = R_kf
        self.L = None   # computed in _compute_gain()

        # Structural mass matrix constants
        M_hh = M_WING + M_FLAP
        M_aa = I_WING + I_FLAP_EA
        M_ha = M_FLAP * _D_X
        det  = M_hh * M_aa - M_ha**2
        self._Mdet = det
        self._M_aa = M_aa
        self._M_ha = M_ha
        self._M_hh = M_hh
        self.q_dyn = 0.5 * 1.225 * U_INF**2 * 0.05

        xi0 = xi_trim if xi_aug_0 is None else xi_aug_0
        self.xi_hat = xi0.copy()

    @classmethod
    def from_augmented(cls, aero_model, U_INF, DT, C_obs, A_aug, xi_trim,
                       Q_kf=None, R_kf=None, lambda_w=0.98, xi_aug_0=None):
        """
        Construct EKF and compute Kalman gain L from A_aug via dual DARE.

        Parameters
        ----------
        A_aug : ndarray, shape (6, 6)
            Augmented linearized state matrix from LQRController.
        (all other parameters as in __init__)
        """
        obj = cls(aero_model, U_INF, DT, C_obs, xi_trim,
                  Q_kf=Q_kf, R_kf=R_kf, lambda_w=lambda_w, xi_aug_0=xi_aug_0)
        obj._compute_gain(A_aug)
        return obj

    def _compute_gain(self, A_aug):
        """Solve dual DARE and store L."""
        print("  Solving dual DARE for EKF gain (6×6)...")
        P = solve_discrete_are(A_aug.T, self.C.T, self._Q_kf, self._R_kf)
        S = self.C @ P @ self.C.T + self._R_kf
        self.L = P @ self.C.T @ np.linalg.inv(S)   # (6, 3)
        eigs = np.linalg.eigvals(A_aug - self.L @ self.C)
        print(f"  EKF observer eigenvalues |λ|: {np.abs(eigs).round(4)}")

    def reset(self, xi_aug_0=None):
        """Reset to xi_aug_0 (absolute). Default: trim point."""
        self.xi_hat = (self.xi_trim if xi_aug_0 is None else xi_aug_0).copy()

    def _structural_rhs(self, xs, Fy, Mz):
        h, hd, a, ad = xs
        RHS_h = -Fy - D_H * hd - K_H * h
        RHS_a =  Mz - D_ALPHA * ad - K_ALPHA * a
        h_ddot = (self._M_aa * RHS_h - self._M_ha * RHS_a) / self._Mdet
        a_ddot = (self._M_hh * RHS_a - self._M_ha * RHS_h) / self._Mdet
        return np.array([hd, h_ddot, ad, a_ddot])

    def _predict(self, u):
        """Nonlinear one-step prediction using aero_model + structural RK4."""
        h, hd, a, ad = self.xi_hat[:4]
        z = self.xi_hat[4:5]
        W = float(self.xi_hat[5])

        z_new, C_L, C_M = self.aero.step(
            z, h, hd, a, ad, u, W, self.U_INF, self.DT)
        C_L = float(C_L);  C_M = float(C_M)

        Fy = self.q_dyn * C_L
        Mz = self.q_dyn * C_M

        xs = np.array([h, hd, a, ad])
        k1 = self._structural_rhs(xs,                   Fy, Mz)
        k2 = self._structural_rhs(xs + 0.5*self.DT*k1,  Fy, Mz)
        k3 = self._structural_rhs(xs + 0.5*self.DT*k2,  Fy, Mz)
        k4 = self._structural_rhs(xs + self.DT*k3,      Fy, Mz)
        xs_new = xs + (self.DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        W_new = self.lw * W
        return np.concatenate([xs_new, z_new, [W_new]])

    def step(self, u, y, W_bisect=None):
        """
        One EKF predict-update cycle.

        Update uses only (alpha, alpha_dot) measurements — h_ddot is excluded
        because C_obs row 0 depends on C_y evaluated at trim, which is
        inaccurate when z deviates far from z_trim during a strong gust.
        W is set from C_L bisection (W_bisect) if provided.

        Parameters
        ----------
        u : float
            Control input delta [deg] at the previous step.
        y : ndarray, shape (3,)
            Measurements [h_ddot, alpha, alpha_dot] in absolute units.
        W_bisect : float or None
            Gust estimate from C_L bisection; overrides filter W if given.

        Returns
        -------
        xi_hat : ndarray, shape (6,)
            Updated absolute state [h, hd, alpha, alpha_dot, z, W].
        """
        xi_pred = self._predict(u)

        # Update on alpha and alpha_dot only (rows 1-2 of C_obs)
        C_dir = self.C[1:, :]           # (2, 6)
        L_dir = self.L[:, 1:].copy()    # (6, 2)
        L_dir[5, :] = 0.0               # zero W row — set by bisection instead

        y_pred = C_dir @ (xi_pred - self.xi_trim) + C_dir @ self.xi_trim
        innov  = y[1:] - y_pred

        self.xi_hat = xi_pred + L_dir @ innov

        if W_bisect is not None:
            self.xi_hat[5] = float(W_bisect)

        return self.xi_hat.copy()

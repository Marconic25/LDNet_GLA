"""
Nonlinear EKF on the augmented aeroelastic state ξ̃ = [h, ḣ, α, α̇, z, W] ∈ ℝ⁶.

Difference from existing kalman.py:
  - Full time-varying covariance P (not a constant DARE gain L).
    This means the filter adapts its confidence as the state evolves.
  - Jacobians A and C computed via TF AD ONCE at trim in __init__ and then
    reused as fixed matrices (same cost as lqr.py).  EKF_RELINEARIZE_EVERY
    controls how often to recompute them at the current operating point;
    default is very large (≈never) so the trim Jacobians are always used.
  - Measurements: y = [ḧ, α]ᵀ.
  - W row in Kalman gain is active; W is corrected via the C_L-inversion
    pseudo-measure supplied by cl_inversion.py (when mode='ekf_clinv').
  - Joseph form P update for numerical stability.
"""
import numpy as np
import tensorflow as tf
from structural.smd import (M_WING, M_FLAP, I_WING, I_FLAP_EA,
                             D_H, D_ALPHA, K_H, K_ALPHA, _D_X)
from control.cl_inversion import CLInversionFusion

# ──────────────────────────────────────────────────────────────────────────────
# TUNING CONSTANTS
#
# Q_EKF_DIAG  process noise (6×6 diagonal)
#   h, α   : 1e-10 → structural model very accurate
#   ḣ, α̇   : 1e-8
#   z      : 1e-6  → LDNet dynamics mostly captured
#   W      : 50.0  → W_{k+1}=λ_w*W_k is coarse; high Q lets measurements correct W
#
# R_EKF_DIAG  measurement noise (2×2 diagonal)
#   ḧ      : 1e-6  → accelerometer σ ≈ 1e-3 m/s²
#   α      : 1e-8  → inclinometer  σ ≈ 1e-4 rad
#
# P0_DIAG     initial covariance (6×6 diagonal)
#   structural : 1e-6 → start near trim
#   z          : 1e-4 → z_trim estimate has some residual
#   W          : 1e2  → W=0 initial guess essentially unknown
#
# EKF_RELINEARIZE_EVERY
#   Default 999999 → use trim Jacobians for the whole simulation (fast).
#   Set to 1 for full online relinearization (slow, accurate during large excursions).
#
# OBSERVER_MODE  default for run_mpc_simulation
#   'legacy'    → existing kalman.py (DARE constant gain + CL bisection)
#   'ekf'       → NonlinearEKF, no CL inversion
#   'ekf_clinv' → NonlinearEKF + CL inversion pseudo-measure (recommended)
# ──────────────────────────────────────────────────────────────────────────────

Q_EKF_DIAG = np.array([1e-10, 1e-8, 1e-10, 1e-8, 1e-6, 50.0])
R_EKF_DIAG = np.array([1e-6, 1e-8])
P0_DIAG    = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-4, 1e2])

EKF_RELINEARIZE_EVERY = 1        # relinearize at every step (online, no retracing)
OBSERVER_MODE         = 'ekf_clinv'


def _compute_ekf_jacobians(aero_model, xi_op, U_INF, DT, lambda_w=0.98, u_op=0.0):
    """
    Compute Jacobians A (6×6) and C (2×6) at operating point xi_op via TF AD.

    Uses tape.watch() on tf.constant tensors instead of tf.Variable so that
    TF does NOT retrace the graph across repeated calls — the tensor shapes
    and dtypes are identical every call, which is all TF needs to reuse the
    compiled graph.

    A = ∂F/∂ξ̃  (propagation, augmented with W)
    C = ∂h/∂ξ̃  (measurement  h = [ḧ, α]ᵀ)
    """
    dtype = tf.float64
    lw    = tf.constant(lambda_w,  dtype=dtype)
    q_dyn = tf.constant(0.5 * 1.225 * U_INF**2 * 0.05, dtype=dtype)
    U_tf  = tf.constant(float(U_INF), dtype=dtype)
    DT_tf = tf.constant(float(DT),    dtype=dtype)
    u_tf  = tf.constant(float(u_op),  dtype=dtype)

    M_hh_tf = tf.constant(float(M_WING + M_FLAP),   dtype=dtype)
    M_aa_tf = tf.constant(float(I_WING + I_FLAP_EA), dtype=dtype)
    M_ha_tf = tf.constant(float(M_FLAP * _D_X),      dtype=dtype)
    det_tf  = tf.constant(float((M_WING+M_FLAP)*(I_WING+I_FLAP_EA) - (M_FLAP*_D_X)**2), dtype=dtype)
    dH      = tf.constant(float(D_H),     dtype=dtype)
    kH      = tf.constant(float(K_H),     dtype=dtype)
    dA      = tf.constant(float(D_ALPHA), dtype=dtype)
    kA      = tf.constant(float(K_ALPHA), dtype=dtype)

    def rk4_tf(xs, Fy, Mz):
        def rhs(s):
            RHS_h = -Fy - dH*s[1] - kH*s[0]
            RHS_a =  Mz - dA*s[3] - kA*s[2]
            return tf.stack([s[1],
                             (M_aa_tf*RHS_h - M_ha_tf*RHS_a)/det_tf,
                             s[3],
                             (M_hh_tf*RHS_a - M_ha_tf*RHS_h)/det_tf])
        k1 = rhs(xs); k2 = rhs(xs + 0.5*DT_tf*k1)
        k3 = rhs(xs + 0.5*DT_tf*k2); k4 = rhs(xs + DT_tf*k3)
        return xs + (DT_tf/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # Compute Jacobians column-by-column via tape.gradient on each output scalar.
    # This avoids tape.jacobian which internally uses pfor and causes retracing.
    # Cost: 8 forward passes (6 for A rows via vjp, 2 for C rows) — acceptable.

    def _forward(xi_c):
        h  = xi_c[0]; hd = xi_c[1]
        a  = xi_c[2]; ad = xi_c[3]
        z  = xi_c[4:5]; W = xi_c[5]
        z_new, C_L, C_M = aero_model.step_tf(
            z, h, hd, a, ad, u_tf, W, U_tf, float(DT))
        Fy = q_dyn * C_L; Mz = q_dyn * C_M
        xs_new = rk4_tf(xi_c[:4], Fy, Mz)
        W_new  = lw * W
        xi_new = tf.concat([xs_new, tf.reshape(z_new, (-1,)), [W_new]], axis=0)
        RHS_h  = -Fy - dH*hd - kH*h
        RHS_a  =  Mz - dA*ad - kA*a
        h_ddot = (M_aa_tf*RHS_h - M_ha_tf*RHS_a) / det_tf
        y_out  = tf.stack([h_ddot, a])
        return xi_new, y_out

    xi_c = tf.constant(np.asarray(xi_op, dtype=np.float64), dtype=dtype)

    # A rows: ∂xi_new[i]/∂xi_c  for i=0..5  (use gradient of each scalar output)
    A = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        with tf.GradientTape() as tape:
            tape.watch(xi_c)
            xi_new, _ = _forward(xi_c)
            scalar = xi_new[i]
        g = tape.gradient(scalar, xi_c)
        if g is not None:
            A[i] = g.numpy()

    # C rows: ∂y_out[i]/∂xi_c  for i=0,1
    C = np.zeros((2, 6), dtype=np.float64)
    for i in range(2):
        with tf.GradientTape() as tape:
            tape.watch(xi_c)
            _, y_out = _forward(xi_c)
            scalar = y_out[i]
        g = tape.gradient(scalar, xi_c)
        if g is not None:
            C[i] = g.numpy()

    return A, C


class NonlinearEKF:
    """
    Extended Kalman Filter for the augmented aeroelastic system.

    State: ξ̃ = [h, ḣ, α, α̇, z, W]  ∈ ℝ⁶ (absolute coordinates)

    Prediction:
        Uses full nonlinear propagation (aero_model.step + RK4 structural).
        Covariance propagated with Jacobian A (computed at trim once, or
        refreshed every EKF_RELINEARIZE_EVERY steps via AD).

    Measurements:  y = [ḧ, α]ᵀ
        ḧ from structural EOM at predicted state (nonlinear).
        α direct (index 2 of state).

    Parameters
    ----------
    aero_model : LDNetModel
    U_INF      : float
    DT         : float
    xi_trim    : ndarray, shape (6,)
    Q, R, P0   : ndarray or None  (fall back to module-level defaults)
    lambda_w   : float
    relinearize_every : int
        How often (in steps) to recompute A, C via AD.  Default=EKF_RELINEARIZE_EVERY
        (≈never: trim Jacobians reused throughout).
    """

    def __init__(self, aero_model, U_INF, DT, xi_trim,
                 Q=None, R=None, P0=None,
                 lambda_w=0.98, relinearize_every=EKF_RELINEARIZE_EVERY,
                 A_trim=None, C_trim=None,
                 use_cl_inversion=True):
        self.aero   = aero_model
        self.U_INF  = float(U_INF)
        self.DT     = float(DT)
        self.lw     = float(lambda_w)
        self.relin  = int(relinearize_every)
        self.xi_trim = np.asarray(xi_trim, dtype=np.float64).copy()

        self.Q  = np.diag(Q_EKF_DIAG) if Q  is None else np.asarray(Q,  dtype=np.float64)
        self.R  = np.diag(R_EKF_DIAG) if R  is None else np.asarray(R,  dtype=np.float64)
        self.P0 = np.diag(P0_DIAG)    if P0 is None else np.asarray(P0, dtype=np.float64)

        self.q_dyn = 0.5 * 1.225 * U_INF**2 * 0.05

        M_hh = float(M_WING + M_FLAP)
        M_aa = float(I_WING + I_FLAP_EA)
        M_ha = float(M_FLAP * _D_X)
        det  = M_hh * M_aa - M_ha**2
        self._M_hh = M_hh; self._M_aa = M_aa
        self._M_ha = M_ha; self._Mdet = det

        # Use pre-computed Jacobians if provided, else compute once at trim.
        # Passing A_trim/C_trim avoids a second AD call when the caller
        # (e.g. mpc_observer_comparison.py) already has them from lqr.compute_jacobians.
        if A_trim is not None and C_trim is not None:
            self._A = np.asarray(A_trim, dtype=np.float64)
            self._C = np.asarray(C_trim, dtype=np.float64)
            print(f"  [NonlinearEKF] Using provided trim Jacobians  A{self._A.shape} C{self._C.shape}")
        else:
            print("  [NonlinearEKF] Computing trim Jacobians via AD...")
            self._A, self._C = _compute_ekf_jacobians(aero_model, xi_trim, U_INF, DT)
            print(f"  [NonlinearEKF] A{self._A.shape} C{self._C.shape}")

        self.xi_hat      = self.xi_trim.copy()
        self.P           = self.P0.copy()
        self._step_count = 0

        self._fusion = CLInversionFusion(aero_model, U_INF, DT) if use_cl_inversion else None

    # ── structural helpers ────────────────────────────────────────────────────

    def _struct_rhs(self, xs, Fy, Mz):
        h, hd, a, ad = xs
        RHS_h = -Fy - D_H * hd - K_H * h
        RHS_a =  Mz - D_ALPHA * ad - K_ALPHA * a
        h_ddot = (self._M_aa * RHS_h - self._M_ha * RHS_a) / self._Mdet
        a_ddot = (self._M_hh * RHS_a - self._M_ha * RHS_h) / self._Mdet
        return np.array([hd, h_ddot, ad, a_ddot])

    # ── nonlinear propagation ─────────────────────────────────────────────────

    def _f_nonlinear(self, xi, u):
        h, hd, a, ad = xi[:4]
        z = xi[4:5]; W = float(xi[5])

        z_new, C_L, C_M = self.aero.step(z, h, hd, a, ad, u, W, self.U_INF, self.DT)
        C_L = float(C_L); C_M = float(C_M)

        Fy = self.q_dyn * C_L
        Mz = self.q_dyn * C_M

        xs = np.array([h, hd, a, ad])
        k1 = self._struct_rhs(xs,                  Fy, Mz)
        k2 = self._struct_rhs(xs + 0.5*self.DT*k1, Fy, Mz)
        k3 = self._struct_rhs(xs + 0.5*self.DT*k2, Fy, Mz)
        k4 = self._struct_rhs(xs + self.DT*k3,     Fy, Mz)
        xs_new = xs + (self.DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        W_new = self.lw * W
        return np.concatenate([xs_new, z_new.flatten(), [W_new]])

    # ── nonlinear measurement ─────────────────────────────────────────────────

    def _h_nonlinear(self, xi, u):
        """y = [ḧ, α] evaluated at xi (nonlinear, via aero step)."""
        h, hd, a, ad = xi[:4]
        z = xi[4:5]; W = float(xi[5])

        _, C_L, C_M = self.aero.step(z, h, hd, a, ad, u, W, self.U_INF, self.DT)
        Fy = self.q_dyn * float(C_L)
        Mz = self.q_dyn * float(C_M)

        RHS_h  = -Fy - D_H * hd - K_H * h
        RHS_a  =  Mz - D_ALPHA * ad - K_ALPHA * a
        h_ddot = (self._M_aa * RHS_h - self._M_ha * RHS_a) / self._Mdet
        return np.array([h_ddot, a])

    # ── predict / update ──────────────────────────────────────────────────────

    def predict(self, u):
        """
        EKF prediction: nonlinear state propagation + linear covariance propagation.

        Returns
        -------
        xi_pred : ndarray (6,)
        P_pred  : ndarray (6, 6)
        """
        if self._step_count % self.relin == 0:
            self._A, self._C = _compute_ekf_jacobians(
                self.aero, self.xi_hat, self.U_INF, self.DT,
                lambda_w=self.lw, u_op=float(u))
        xi_pred = self._f_nonlinear(self.xi_hat, u)
        P_pred  = self._A @ self.P @ self._A.T + self.Q
        return xi_pred, P_pred

    def update(self, xi_pred, P_pred, y_meas, u,
               W_inv=None, R_W=None):
        """
        EKF update with optional W^inv pseudo-measure.

        Parameters
        ----------
        xi_pred : ndarray (6,)
        P_pred  : ndarray (6, 6)
        y_meas  : ndarray (2,)  — [ḧ_meas, α_meas]
        u       : float
        W_inv   : float or None
        R_W     : float or None

        Returns
        -------
        xi_hat : ndarray (6,)
        """
        C_k = self._C.copy()   # (2, 6)
        R_k = self.R.copy()    # (2, 2)

        if W_inv is not None and R_W is not None:
            C_W = np.zeros((1, 6)); C_W[0, 5] = 1.0
            C_k = np.vstack([C_k, C_W])                                    # (3, 6)
            R_k = np.block([[self.R,          np.zeros((2, 1))],
                             [np.zeros((1, 2)), np.array([[R_W]])]])        # (3, 3)
            y_meas_full = np.append(y_meas, W_inv)                         # (3,)
        else:
            y_meas_full = y_meas                                            # (2,)

        y_pred = self._h_nonlinear(xi_pred, u)
        if W_inv is not None and R_W is not None:
            y_pred = np.append(y_pred, float(xi_pred[5]))

        innov  = y_meas_full - y_pred
        S      = C_k @ P_pred @ C_k.T + R_k
        K_gain = P_pred @ C_k.T @ np.linalg.inv(S)   # (6, n_meas)

        self.xi_hat = xi_pred + K_gain @ innov

        # Joseph form for numerical stability
        IKC      = np.eye(6) - K_gain @ C_k
        self.P   = IKC @ P_pred @ IKC.T + K_gain @ R_k @ K_gain.T

        self._step_count += 1
        return self.xi_hat.copy()

    def _estimate_W(self, C_L_meas, delta):
        """
        Estimate gust velocity W via C_L inversion (1-step delayed).

        Uses the internal CLInversionFusion instance if available.
        Returns (W_inv, R_W, valid) — same signature as CLInversionFusion.update().
        Must be called BEFORE step() so push_state() registers the current estimate.

        Parameters
        ----------
        C_L_meas : float  — measured lift coefficient at current step
        delta    : float  — flap deflection applied at current step

        Returns
        -------
        W_inv : float or None
        R_W   : float
        valid : bool
        """
        if self._fusion is None:
            return None, None, False
        z_now = self.xi_hat[4:5].copy()
        W_inv, R_W, valid = self._fusion.update(z_now)
        self._fusion.push_state(self.xi_hat, delta)
        return W_inv, R_W, valid

    def step(self, u, y_meas, C_L_meas=None, W_inv=None, R_W=None):
        """
        One full EKF predict-update cycle.

        If the internal CLInversionFusion is active (use_cl_inversion=True)
        and C_L_meas is provided, W is estimated internally via _estimate_W()
        and the explicit W_inv/R_W arguments are ignored.

        Parameters
        ----------
        u        : float         — control applied at previous step
        y_meas   : ndarray (2,)  — [ḧ_meas, α_meas]
        C_L_meas : float or None — measured C_L (used by internal fusion)
        W_inv    : float or None — external W pseudo-measure (ignored if fusion active)
        R_W      : float or None — variance of external W_inv

        Returns
        -------
        xi_hat : ndarray (6,)
        """
        if self._fusion is not None and C_L_meas is not None:
            W_inv, R_W, valid = self._estimate_W(C_L_meas, float(u))
            if not valid:
                W_inv = R_W = None

        xi_pred, P_pred = self.predict(u)
        return self.update(xi_pred, P_pred, y_meas, u, W_inv=W_inv, R_W=R_W)

    def reset(self, xi0=None):
        """Reset state and covariance to initial values."""
        self.xi_hat      = (self.xi_trim if xi0 is None
                            else np.asarray(xi0, dtype=np.float64)).copy()
        self.P           = self.P0.copy()
        self._step_count = 0
        if self._fusion is not None:
            self._fusion.reset()

"""
C_L inversion module: recover a pseudo-measurement of W from the latent
state residual in f_dyn.

Rationale:
  f_rec does NOT depend on W directly; it produces C_L from (z, h, ḣ, α, α̇, δ).
  W enters only through f_dyn, which updates z.  Therefore, if we have
  two successive EKF-smoothed estimates of z:
      z_k^obs  (from EKF update at step k)
      z_{k+1}^obs (from EKF update at step k+1)
  we can invert the f_dyn equation for W:
      z_{k+1}^obs = z_k^obs + (Δt/Δt_ref) · f_dyn(z_k, h_k, ḣ_k, α_k, α̇_k, δ_k, W_k)

  NOTE: this inversion runs with a 1-step delay — z_{k+1}^obs is only
  available after the EKF update at step k+1.  This is intentional and
  documented here so callers are aware.

Newton solver uses TF AD for ∂f_dyn/∂W, falls back to bisection.
R_W is auto-tuned from the residual variance over a warm-up window.
"""
import numpy as np
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────────────
# TUNING CONSTANTS
#
# W_INV_CLIP    physical ceiling on |W^inv| [m/s]; results beyond this are
#               rejected as unphysical (training data up to ~50 m/s, add margin)
#
# NEWTON_MAX_IT max Newton iterations before bisection fallback
# NEWTON_TOL    convergence threshold on |r(W)| for Newton
#
# W_BISECT_LO / W_BISECT_HI   bisection search range [m/s]
#               covers both head-on (negative) and tail (positive) gusts
#
# R_W_MIN       floor on auto-tuned R_W variance; prevents near-zero R_W
#               from making the pseudo-measure dominate the EKF update
#
# WARMUP_STEPS  number of steps used to estimate R_W; during warm-up the
#               inversion is computed but NOT fed into the EKF as a measure
# ──────────────────────────────────────────────────────────────────────────────

W_INV_CLIP    = 150.0   # [m/s]
NEWTON_MAX_IT = 10
NEWTON_TOL    = 1e-6
W_BISECT_LO   = -200.0  # [m/s]
W_BISECT_HI   = 200.0   # [m/s]
R_W_MIN       = 1e-2    # [m²/s²]
WARMUP_STEPS  = 50      # [steps]




def _invert_W_internal(aero_model, h, hd, a, ad, z_k, delta_k,
                       z_next_obs, DT, dt_ref, U_INF, W_init=0.0):
    """
    Internal Newton + bisection solver for W.

    Parameters mirror invert_W_from_zdiff but with explicit h, hd, a, ad, z_k
    and U_INF split out for clarity.

    Returns
    -------
    W_inv : float or None
    residual : float
    converged : bool
    """
    dtype  = tf.float64
    h_tf   = tf.constant(h,        dtype=dtype)
    hd_tf  = tf.constant(hd,       dtype=dtype)
    a_tf   = tf.constant(a,        dtype=dtype)
    ad_tf  = tf.constant(ad,       dtype=dtype)
    z_k_tf = tf.constant(np.asarray(z_k, dtype=np.float64).flatten(), dtype=dtype)
    d_tf   = tf.constant(float(delta_k), dtype=dtype)
    U_tf   = tf.constant(float(U_INF),   dtype=dtype)
    z_tgt  = tf.constant(float(np.asarray(z_next_obs).flat[0]), dtype=dtype)

    def residual_and_grad(W_val):
        """Returns (r, dr/dW).  Uses tape.watch on tf.constant to avoid retracing."""
        W_tf = tf.constant(float(W_val), dtype=dtype)
        with tf.GradientTape() as tape:
            tape.watch(W_tf)
            z_new, _, _ = aero_model.step_tf(
                z_k_tf, h_tf, hd_tf, a_tf, ad_tf, d_tf, W_tf, U_tf, float(DT))
            r = z_tgt - tf.reshape(z_new, ())
        dr = tape.gradient(r, W_tf)
        return float(r.numpy()), float(dr.numpy()) if dr is not None else 0.0

    # ── Newton iterations ─────────────────────────────────────────────────────
    W = float(W_init)
    converged = False
    for _ in range(NEWTON_MAX_IT):
        r, dr = residual_and_grad(W)
        if abs(r) < NEWTON_TOL:
            converged = True
            break
        if abs(dr) < 1e-15:
            break
        W = W - r / dr
        # clamp to search range to avoid divergence
        W = float(np.clip(W, W_BISECT_LO, W_BISECT_HI))

    r_final = abs(residual_and_grad(W)[0])
    if converged and r_final < NEWTON_TOL:
        pass
    else:
        # ── Bisection fallback ────────────────────────────────────────────────
        W_lo, W_hi = W_BISECT_LO, W_BISECT_HI
        r_lo = residual_and_grad(W_lo)[0]
        r_hi = residual_and_grad(W_hi)[0]

        if r_lo * r_hi > 0:
            # no sign change — return endpoint with smaller |r|
            W = W_lo if abs(r_lo) < abs(r_hi) else W_hi
        else:
            for _ in range(40):
                W_mid = 0.5 * (W_lo + W_hi)
                r_mid = residual_and_grad(W_mid)[0]
                if abs(r_mid) < NEWTON_TOL or (W_hi - W_lo) < 1e-8:
                    W = W_mid
                    converged = True
                    break
                if r_lo * r_mid <= 0:
                    W_hi = W_mid; r_hi = r_mid
                else:
                    W_lo = W_mid; r_lo = r_mid
            else:
                W = 0.5 * (W_lo + W_hi)

        r_final = abs(residual_and_grad(W)[0])

    # ── Validation ────────────────────────────────────────────────────────────
    if abs(W) > W_INV_CLIP:
        return None, r_final, False

    return float(W), r_final, converged


class CLInversionFusion:
    """
    Manages the 1-step-delayed W inversion and R_W auto-tuning warm-up.

    Usage pattern (per simulation step k):
        1. Call ekf.step(u, y_meas) → xi_hat at step k
        2. Call fusion.push_z(xi_hat[4:5]) to register z_k^obs
        3. After step k+1 completes and we have z_{k+1}^obs in xi_hat:
           W_inv, R_W, valid = fusion.update(...)
           xi_hat = ekf.step(..., W_inv=W_inv if valid else None, R_W=R_W)

    This is encapsulated in run_mpc_simulation — callers need not worry.

    Parameters
    ----------
    aero_model : LDNetModel
    U_INF      : float
    DT         : float
    warmup_steps : int
    """

    def __init__(self, aero_model, U_INF, DT, warmup_steps=WARMUP_STEPS):
        self.aero         = aero_model
        self.U_INF        = float(U_INF)
        self.DT           = float(DT)
        self.dt_ref       = float(aero_model.normalization['time']['time_constant'])
        self.warmup_steps = int(warmup_steps)

        self._z_prev      = None   # z_k^obs from previous step
        self._xi_prev     = None   # full ξ̃_k^obs from previous step
        self._delta_prev  = 0.0    # δ_k applied at previous step

        self._residuals   = []     # accumulates |r| during warm-up
        self._R_W         = float('inf')   # will be set after warm-up
        self._step_count  = 0
        self._warmed_up   = False

    @property
    def is_warmed_up(self):
        return self._warmed_up

    @property
    def R_W(self):
        return self._R_W

    def push_state(self, xi_hat, delta_applied):
        """
        Register the current EKF state estimate and applied control.
        Must be called after each EKF update, before calling update() at the
        next step.

        Parameters
        ----------
        xi_hat        : ndarray, shape (6,)
        delta_applied : float
        """
        self._xi_prev    = xi_hat.copy()
        self._z_prev     = xi_hat[4:5].copy()
        self._delta_prev = float(delta_applied)

    def update(self, z_now_obs):
        """
        Attempt W inversion using stored previous state and current z estimate.

        Parameters
        ----------
        z_now_obs : ndarray, shape (1,)  — z_{k+1}^obs from latest EKF update

        Returns
        -------
        W_inv  : float or None
        R_W    : float
        valid  : bool
        """
        if self._xi_prev is None:
            return None, self._R_W, False

        xi_k = self._xi_prev[:5]   # [h, ḣ, α, α̇, z_k]

        W_inv, residual, converged = _invert_W_internal(
            self.aero,
            float(xi_k[0]), float(xi_k[1]), float(xi_k[2]), float(xi_k[3]),
            xi_k[4:5],
            self._delta_prev,
            z_now_obs,
            self.DT, self.dt_ref, self.U_INF,
            W_init=float(self._xi_prev[5])   # seed Newton with current W estimate
        )

        self._step_count += 1

        # Warm-up: accumulate residuals, do not feed to EKF yet
        if not self._warmed_up:
            if W_inv is not None:
                self._residuals.append(residual)
            if self._step_count >= self.warmup_steps:
                if len(self._residuals) > 0:
                    var_r = float(np.var(self._residuals))
                    self._R_W = max(var_r, R_W_MIN)
                else:
                    self._R_W = R_W_MIN
                self._warmed_up = True
                print(f"  [CLInv] warm-up complete after {self._step_count} steps; "
                      f"R_W = {self._R_W:.4e}")
            return None, self._R_W, False

        valid = (W_inv is not None)
        if not valid:
            import warnings
            warnings.warn(f"[CLInv] W_inv rejected (|W| > {W_INV_CLIP} m/s); "
                          "skipping pseudo-measure this step.")

        return W_inv, self._R_W, valid

    def reset(self):
        """Reset fusion state (call at start of each simulation)."""
        self._z_prev      = None
        self._xi_prev     = None
        self._delta_prev  = 0.0
        self._residuals   = []
        self._R_W         = float('inf')
        self._step_count  = 0
        self._warmed_up   = False

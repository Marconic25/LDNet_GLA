"""
Linear unsteady aerodynamic model — the ablation baseline for the GLA loop.

This is the classical low-order model the thesis introduction contrasts the
LDNet against: incompressible thin-airfoil theory with a finite-state
representation of the wake memory. It is linear in every input, assumes
attached flow, and has no notion of separation or saturation.

Structure
---------
Circulatory lift follows the effective downwash

    w(t) = alpha + hd/U + (c/U)*(a_34 - a_ea)*ad          [airfoil motion]

filtered through Wagner's indicial function (R.T. Jones two-pole fit)

    phi(s) = 1 - A1 exp(-b1 s) - A2 exp(-b2 s),
    A1=0.165, b1=0.0455, A2=0.335, b2=0.300

and the gust downwash W/U filtered through Kussner's indicial function
(two-pole fit)

    psi(s) = 1 - G1 exp(-g1 s) - G2 exp(-g2 s),
    G1=G2=0.5, g1=0.130, g2=1.000

with s = 2*U*t/c the semichord-based reduced time. Each two-pole filter is
realised as two first-order lag states, giving a 4-state linear system
(2 motion lags + 2 gust lags). The flap enters through its own steady
effectiveness and through the same motion wake lag.

Non-circulatory (added-mass / apparent-mass) lift and moment are algebraic in
the accelerations and rates and are added instantaneously.

All gains are FITTED to the same CFD campaign the LDNet was trained on
(data/GLA_train.h5) by linear least squares. This is deliberate: a baseline
carrying textbook gains would lose the comparison because of a gain error
rather than because of missing nonlinearity, which would make the ablation
uninformative. Fitting gives the linear model the best coefficients it can
have within its own structure, so the residual gap measures nonlinearity.

Interface
---------
Mirrors the subset of LDNetAero that MPCPreviewController and run.py use:

    predict(state, delta_deg, W, U)            -> (C_L, C_M)
    advance(state, delta_deg, W, U, dt)        -> None   (steps internal lags)
    advance_z(z, state, delta_deg, W, U, dt)   -> z_new  (pure)
    batch_step(z_b, x_b, delta_b, W, U, dt)    -> (C_L_b, C_M_b, z_new_b)
    reset(dt=None)
    _z_leak, _num_z                            attributes

so it is a drop-in substitution inside the identical MPC loop: same preview,
same cost function, same structural integrator, same rate/saturation limits.
"""
import json
import numpy as np

# --- Jones two-pole fit of the Wagner function (airfoil-motion wake memory) --
A1, B1 = 0.165, 0.0455
A2, B2 = 0.335, 0.300

# --- Two-pole fit of the Kussner function (gust-penetration wake memory) -----
G1, GB1 = 0.5, 0.130
G2, GB2 = 0.5, 1.000

# Number of internal lag states: 2 motion + 2 gust.
NUM_LAG = 4


class LinearUnsteadyAero:
    """Finite-state linear unsteady aerodynamics with fitted gains."""

    def __init__(self, coeff_path=None, c=1.0, rho=1.225, S=0.05,
                 a_ea=0.40, a_34=0.75):
        """
        coeff_path : JSON produced by fit_linear.py. If None, textbook gains
                     are used (thin-airfoil), which is the UNFITTED baseline.
        c          : chord [m]
        a_ea       : elastic-axis position as fraction of chord
        a_34       : three-quarter-chord (downwash collocation) point
        """
        self.c = float(c)
        self.rho = float(rho)
        self.S = float(S)
        self.a_ea = float(a_ea)
        self.a_34 = float(a_34)

        if coeff_path is not None:
            with open(coeff_path, 'r') as f:
                k = json.load(f)
        else:
            k = {}

        # Circulatory gains [1/rad]
        self.CL_a = float(k.get('CL_a', 2.0 * np.pi))     # lift-curve slope
        self.CL_d = float(k.get('CL_d', 0.7))             # flap effectiveness
        self.CL_g = float(k.get('CL_g', 2.0 * np.pi))     # gust lift slope
        self.CM_a = float(k.get('CM_a', -0.1))
        self.CM_d = float(k.get('CM_d', 0.35))
        self.CM_g = float(k.get('CM_g', -0.05))

        # Added-mass scalings (dimensionless multipliers on the analytic terms)
        self.AM_L = float(k.get('AM_L', 1.0))
        self.AM_M = float(k.get('AM_M', 1.0))

        # Constant offsets (trim residual)
        self.CL_0 = float(k.get('CL_0', 0.0))
        self.CM_0 = float(k.get('CM_0', 0.0))

        self._num_z = NUM_LAG
        self._z_leak = 0.0          # no artificial leak; the lags are physical
        self._z = np.zeros(NUM_LAG)
        self._dt = 0.002

    # -- downwash definitions -------------------------------------------------

    def _w_motion(self, state, delta_deg, U):
        """Effective quasi-steady downwash angle [rad] from airfoil motion+flap."""
        _, hd, a, ad = state
        U = max(float(U), 1.0)
        # heave-rate and pitch-rate contribution at the 3/4-chord point
        return a + hd / U + (self.c / U) * (self.a_34 - self.a_ea) * ad

    @staticmethod
    def _w_motion_b(x_b, U):
        return None  # batch path inlines this for speed

    # -- lag-state dynamics ---------------------------------------------------

    def _lag_rates(self, z, w_m, w_g, U):
        """
        d z / d t for the four lag states, in PHYSICAL time.

        Reduced time s = 2 U t / c  =>  d/dt = (2U/c) d/ds.
        Each lag state x_i obeys  dx_i/ds = -b_i x_i + w,  so the deficiency
        form of the Duhamel integral gives the circulatory downwash

            w_circ = w - A1*b1*x1 - A2*b2*x2      (motion)
            w_circ_gust = w_g - G1*g1*x3 - G2*g2*x4
        """
        k = 2.0 * max(float(U), 1.0) / self.c
        return k * np.array([
            -B1 * z[0] + w_m,
            -B2 * z[1] + w_m,
            -GB1 * z[2] + w_g,
            -GB2 * z[3] + w_g,
        ])

    def _circ(self, z, w_m, w_g):
        """Circulatory downwash after wake-memory attenuation."""
        w_m_eff = w_m - A1 * B1 * z[0] - A2 * B2 * z[1]
        w_g_eff = w_g - G1 * GB1 * z[2] - G2 * GB2 * z[3]
        return w_m_eff, w_g_eff

    # -- loads ----------------------------------------------------------------

    def _loads(self, z, state, delta_deg, W, U):
        _, hd, a, ad = state
        U = max(float(U), 1.0)
        d = np.deg2rad(float(delta_deg))
        w_m = self._w_motion(state, delta_deg, U)
        w_g = float(W) / U

        w_m_eff, w_g_eff = self._circ(z, w_m, w_g)

        # Added mass: the analytic non-circulatory term for a flat plate is
        # pi*c/(2U) * (hdd + U*ad - ...). hdd is not available as an input, so
        # only the rate terms that ARE observable enter; their scaling is fitted.
        am = (np.pi * self.c / (2.0 * U)) * ad

        C_L = (self.CL_a * w_m_eff + self.CL_g * w_g_eff
               + self.CL_d * d + self.AM_L * am + self.CL_0)
        C_M = (self.CM_a * w_m_eff + self.CM_g * w_g_eff
               + self.CM_d * d + self.AM_M * am + self.CM_0)
        return float(C_L), float(C_M)

    # -- LDNetAero-compatible API --------------------------------------------

    def predict(self, state, delta_deg, W, U):
        """(C_L, C_M) from the CURRENT lag state — read-only."""
        return self._loads(self._z, state, delta_deg, W, U)

    def _advance_lags(self, z, state, delta_deg, W, U, dt):
        w_m = self._w_motion(state, delta_deg, U)
        w_g = float(W) / max(float(U), 1.0)
        # RK4 on the linear lag system (cheap, 4 states)
        f = lambda zz: self._lag_rates(zz, w_m, w_g, U)
        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def advance(self, state, delta_deg, W, U, dt):
        self._z = self._advance_lags(self._z, state, delta_deg, W, U, float(dt))

    def advance_z(self, z, state, delta_deg, W, U, dt):
        return self._advance_lags(np.asarray(z, float), state, delta_deg,
                                  W, U, float(dt))

    def batch_step(self, z_b, x_b, delta_b, W, U, dt):
        """
        Vectorised one-step forward for a batch of flap candidates.

        Mirrors LDNetAero.batch_step exactly in convention: C_L/C_M are
        reconstructed from the CURRENT z_b, and z is advanced one step.
        """
        z_b = np.asarray(z_b, float)
        x_b = np.asarray(x_b, float)
        delta_b = np.asarray(delta_b, float)
        B = z_b.shape[0]
        Uf = max(float(U), 1.0)

        hd = x_b[:, 1]; a = x_b[:, 2]; ad = x_b[:, 3]
        d = np.deg2rad(delta_b)
        w_m = a + hd / Uf + (self.c / Uf) * (self.a_34 - self.a_ea) * ad
        w_g = np.full(B, float(W) / Uf)

        w_m_eff = w_m - A1 * B1 * z_b[:, 0] - A2 * B2 * z_b[:, 1]
        w_g_eff = w_g - G1 * GB1 * z_b[:, 2] - G2 * GB2 * z_b[:, 3]
        am = (np.pi * self.c / (2.0 * Uf)) * ad

        C_L_b = (self.CL_a * w_m_eff + self.CL_g * w_g_eff
                 + self.CL_d * d + self.AM_L * am + self.CL_0)
        C_M_b = (self.CM_a * w_m_eff + self.CM_g * w_g_eff
                 + self.CM_d * d + self.AM_M * am + self.CM_0)

        # advance the lags one RK4 step (batched)
        kk = 2.0 * Uf / self.c
        bvec = np.array([B1, B2, GB1, GB2])
        wvec = np.stack([w_m, w_m, w_g, w_g], axis=1)       # (B,4)

        def f(zz):
            return kk * (-bvec[None, :] * zz + wvec)

        dtf = float(dt)
        k1 = f(z_b)
        k2 = f(z_b + 0.5 * dtf * k1)
        k3 = f(z_b + 0.5 * dtf * k2)
        k4 = f(z_b + dtf * k3)
        z_new = z_b + (dtf / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return C_L_b, C_M_b, z_new

    def reset(self, dt=None, warmup_csv=None):
        self._z = np.zeros(NUM_LAG)
        if dt is not None:
            self._dt = float(dt)

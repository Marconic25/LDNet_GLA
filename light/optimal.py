"""
One-step optimal GLA controller — final version (wnext + refine).

At each time step it minimises over a 161-point flap grid

    J(delta) = (C_L(delta) - C_L_trim)^2 + R * delta^2

subject to |delta| <= delta_max, |delta - delta_prev|/dt <= delta_dot_max and
a causal sign gate (C_L-reducing half only). C_L(delta) is the frozen-z LDNet
reconstruction, evaluated for all candidates in one batched NNrec call
(read-only on z — the plant rollout stays scalar and deterministic).

The two flags that define this final controller (full investigation: 76/NOTES.md):

  use_wnext=True  Candidates are evaluated against the NEXT-step gust W(t+dt)
                  instead of W(t). The plant advances into W(t+dt), so the
                  W(t) cost is a half-step stale and over-commits flap during
                  the gust rise, landing on a bad closed-loop branch. With the
                  correct gust phase the controller reverses the flap earlier
                  and lands the good branch BY CONSTRUCTION: at W30/Tg0.4
                  (DAMULT=3) CLred goes from ~+17% (chaotic: 60-pt ensemble
                  spread) to +76.5% (robust: 0.4-pt spread). W(t+dt) is a
                  2 ms / 0.16 m preview — legal under the study's gust-oracle
                  premise, physically the mildest LIDAR-preview assumption.

  refine=True     Parabolic sub-cell interpolation of the argmin on the SAME
                  grid -> continuous delta instead of a 0.175 deg staircase
                  (same argmin, same branch, ~8x lower flap roughness).
                  A finer grid is NOT equivalent: it changes the near-tie
                  discretization and can jump to a worse branch.

History note: clean/propw.py's batched R-sweep reported +76.1% here for the
PLAIN W(t) controller — a batch-position FP artifact (that controller is a
knife-edge between the +17% and +76% branches; rows of one TF batch round
differently by position and the argmin amplifies it). This controller reaches
the same good branch deterministically, without batching luck.
"""
import numpy as np


class OptimalController:
    """
    Parameters
    ----------
    aero          : LDNetAero — batch_step for the candidate scan, predict for
                    the causal gate; the controller never advances the latent z
    U             : float   freestream velocity [m/s]
    dt            : float   simulation time step [s]
    R             : float   flap-effort weight (3e-4 = universal choice, 76/)
    C_L_trim      : float   trim lift coefficient
    delta_max     : float   flap deflection limit [deg]
    delta_dot_max : float   flap rate limit [deg/s]
    n_grid        : int     flap candidates on [-delta_max, delta_max]
    use_wnext     : bool    evaluate candidates at W(t+dt)  (final: True)
    refine        : bool    parabolic sub-cell argmin refine (final: True)
    """

    def __init__(self, aero, U=80.0, dt=0.002, R=3e-4, C_L_trim=0.0,
                 delta_max=14.0, delta_dot_max=300.0, n_grid=161,
                 use_wnext=True, refine=True):
        self.aero          = aero
        self.U             = float(U)
        self.dt            = float(dt)
        self.R             = float(R)
        self._C_L_trim     = float(C_L_trim)
        self.delta_max     = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)
        self.n_grid        = int(n_grid)
        self.use_wnext     = bool(use_wnext)
        self.refine        = bool(refine)
        self._dg           = np.linspace(-self.delta_max, self.delta_max, self.n_grid)
        self._delta_prev   = 0.0

    def compute(self, state, W, W_next=0.0):
        """
        Return optimal flap deflection [deg].

        Parameters
        ----------
        state  : array-like (h, h_dot, alpha, alpha_dot)
        W      : float   gust velocity at t [m/s]
        W_next : float   gust velocity at t+dt [m/s] (used when use_wnext)
        """
        aero = self.aero
        dg = self._dg; G = self.n_grid
        reach = self.delta_dot_max * self.dt
        Wc = float(W_next) if self.use_wnext else float(W)

        # frozen-z candidate C_L for all deltas (batched reconstruction only,
        # does not touch aero._z)
        z_b = np.tile(np.asarray(aero._z, float).reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        CLg = aero.batch_step(z_b, x_b, dg, Wc, self.U, self.dt)[0]

        # causal gate: restrict to the C_L-reducing half — C_L(delta) is
        # non-monotone off-manifold and the wrong-sign half has spurious minima
        cl0 = float(aero.predict(state, 0.0, Wc, self.U)[0])
        neg = dg <= 0.0
        causal = neg if cl0 >= self._C_L_trim else ~neg
        ratem = np.abs(dg - self._delta_prev) <= reach + 1e-9

        cost = (CLg - self._C_L_trim) ** 2 + self.R * dg ** 2
        cost = np.where(causal & ratem, cost, np.inf)
        j = int(np.argmin(cost))
        d = float(dg[j])

        if self.refine and 0 < j < G - 1 \
                and np.isfinite(cost[j - 1]) and np.isfinite(cost[j + 1]):
            c0, c1, c2 = cost[j - 1], cost[j], cost[j + 1]
            denom = c0 - 2.0 * c1 + c2
            if denom > 1e-30:
                d += 0.5 * (c0 - c2) / denom * (dg[1] - dg[0])

        d = float(np.clip(d, self._delta_prev - reach, self._delta_prev + reach))
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        self._delta_prev = d
        return d

    def reset(self):
        """Reset flap state — call before each new simulation run."""
        self._delta_prev = 0.0


# ---------------------------------------------------------------------------
# Batched Dormand-Prince RK45 structural step — for MPCPreviewController horizon
# (verbatim of controllers_ref.dp45_batch; kept here so optimal.py is self-contained)
# ---------------------------------------------------------------------------

def dp45_batch(x_b, Fy_b, Mz_b, dt):
    """
    Advance G structural states by one step using the DP45 5th-order coefficients.

    Parameters
    ----------
    x_b   : (G, 4) array  — [h, hd, alpha, ad] for each candidate
    Fy_b  : (G,) array    — aerodynamic heave force per candidate
    Mz_b  : (G,) array    — aerodynamic pitch moment per candidate
    dt    : float         — time step [s]

    Returns
    -------
    x_next : (G, 4) array
    """
    import structure as _S

    def _rhs(xx):
        hd = xx[:, 1]; ad = xx[:, 3]
        rh = -Fy_b - _S.D_H * hd - _S.K_H * xx[:, 0]
        ra = Mz_b - _S.D_ALPHA * ad - _S.K_ALPHA * xx[:, 2]
        hdd = (_S.M_AA * rh - _S.M_HA * ra) / _S.DET
        add = (_S.M_HH * ra - _S.M_HA * rh) / _S.DET
        return np.stack([hd, hdd, ad, add], axis=1)

    k1 = _rhs(x_b)
    k2 = _rhs(x_b + dt * (1.0 / 5) * k1)
    k3 = _rhs(x_b + dt * (3.0 / 40 * k1 + 9.0 / 40 * k2))
    k4 = _rhs(x_b + dt * (44.0 / 45 * k1 - 56.0 / 15 * k2 + 32.0 / 9 * k3))
    k5 = _rhs(x_b + dt * (19372.0 / 6561 * k1 - 25360.0 / 2187 * k2
                           + 64448.0 / 6561 * k3 - 212.0 / 729 * k4))
    k6 = _rhs(x_b + dt * (9017.0 / 3168 * k1 - 355.0 / 33 * k2
                           + 46732.0 / 5247 * k3 + 49.0 / 176 * k4
                           - 5103.0 / 18656 * k5))
    return x_b + dt * (35.0 / 384 * k1 + 500.0 / 1113 * k3
                       + 125.0 / 192 * k4 - 2187.0 / 6784 * k5 + 11.0 / 84 * k6)


# ---------------------------------------------------------------------------
# T1 DLR massed-measurement fusion sensor — port of e2_combo.FusedSensor
# Reference: light/noise/NOTES.md §E2-combo, §E2CC
# ---------------------------------------------------------------------------

class FusedPreviewSensor:
    """
    Rolling inverse-variance fusion database that delivers a fused N-node
    preview VECTOR at each time step.

    Mechanics (verbatim of e2_combo.FusedSensor, kept in sync by eye):
    - At step i, measure node m = i+j for j in 1..Jmax with sigma_fun(j).
    - Accumulate (num, den) running sums per spatial node (inverse-variance).
    - Pre-warm with Jmax-1 virtual steps before t=0 so the database is full
      at the first real step.
    - Optional Tikhonov smoothness penalty (lam) via second-difference operator.
    - Raw samples NEVER clamped; output (self.last) clamped to >= 0.
    - self.last[k] = fused estimate of W at node i+(k+1), k=0..N-1.
    - wc_fun() returns float(self.last[0]) — the scalar the harness logs as Wc.

    Parameters
    ----------
    rng       : numpy Generator — fresh per rollout (rng 100+seed)
    sigma_fun : callable j -> sigma [m/s] — noise of a j-step-ahead measurement
    Jmax      : int — lookahead window (50 for the E2 winner)
    N         : int — preview horizon length (8 for the E2 winner)
    lam       : float — Tikhonov regularisation (0=none, 1=DLR-realistic arm)
    """

    def __init__(self, rng, sigma_fun, Jmax, N, lam=0.0):
        self.rng = rng
        self.js = np.arange(1, int(Jmax) + 1)
        self.sigs = np.array([float(sigma_fun(j)) for j in self.js])
        self.inv2 = 1.0 / self.sigs ** 2
        self.Jmax = int(Jmax)
        self.N = int(N)
        self.lam = float(lam)
        self.num = None
        self.den = None
        self.last = None

    def wc_fun(self, i, Wt, Nsteps):
        """
        Harness-compatible wc_fun signature: called at step i, returns scalar.
        Side-effect: sets self.last to the N-node fused preview vector.
        """
        js, sigs, inv2 = self.js, self.sigs, self.inv2
        if self.num is None:
            self.num = np.zeros(Nsteps)
            self.den = np.zeros(Nsteps)
            for ii in range(-(self.Jmax - 1), 0):
                mm = ii + js
                keep = mm >= 0
                mk = np.minimum(mm[keep], Nsteps - 1)
                yk = Wt[mk] + self.rng.normal(0.0, sigs[keep])
                np.add.at(self.num, mk, yk * inv2[keep])
                np.add.at(self.den, mk, inv2[keep])

        m_idx = np.minimum(i + js, Nsteps - 1)
        y = Wt[m_idx] + self.rng.normal(0.0, sigs)
        np.add.at(self.num, m_idx, y * inv2)
        np.add.at(self.den, m_idx, inv2)

        lo = min(i + 1, Nsteps - 1)
        hi = min(i + self.Jmax, Nsteps - 1)
        w = self.den[lo:hi + 1].copy()
        n = w.size
        ybar = np.zeros(n)
        good = w > 0.0
        ybar[good] = self.num[lo:hi + 1][good] / w[good]

        if self.lam == 0.0 or n < 3:
            u = ybar
        else:
            lam_eff = self.lam * float(np.mean(w))
            D = (np.eye(n - 2, n, 0) - 2.0 * np.eye(n - 2, n, 1)
                 + np.eye(n - 2, n, 2))
            A = np.diag(w) + lam_eff * (D.T @ D)
            u = np.linalg.solve(A, w * ybar)

        idx = np.minimum(np.arange(self.N), n - 1)
        self.last = np.maximum(0.0, u[idx])
        return float(self.last[0])

    def reset(self):
        """Clear database state — call before reuse on a new gust (or create fresh)."""
        self.num = None
        self.den = None
        self.last = None


# ---------------------------------------------------------------------------
# T7 N-step constant-flap MPC with fused preview — port of e2_combo.MPCPrevRdu
# Reference: light/noise/NOTES.md §E2-combo verdict
#
# Recorded results (cluster rk4 tree, W30/Tg0.4, DAMULT=3, N=8, R=3e-4, R_du=0):
#   clean oracle       +80.5%
#   flat sigma=2%*W0   +80.5 [80.4,80.6] 0/6 flags
#   DLR raw 1-3 m/s    +81.1 [80.3,83.7] 0/6 flags  (lam=1)
#   W10/Tg0.7 clean    +93.5%,  W30/Tg0.7 clean  +91.5%
# dp45 anchors: TBD (fill after Task 0 regression run).
# ---------------------------------------------------------------------------

class MPCPreviewController:
    """
    N-step constant-flap MPC that consumes a fused N-node gust preview vector.

    Cost over horizon: J = R*delta^2 + R_du*(delta-delta_prev)^2
                           + sum_{k=0}^{N-1} (C_L_k - trim)^2
    wnext convention: horizon step k is evaluated at w_seq[k] = W(t+(k+1)*dt).
    No causal sign gate (unlike OptimalController).
    Rate limit enforced; argmin from a G-point grid (no refine needed with N>1).

    Parameters
    ----------
    aero          : LDNetAero — shared plant model
    U             : float  freestream velocity [m/s]  (default 80)
    dt            : float  time step [s]  (default 0.002)
    rho           : float  air density [kg/m^3]  (default 1.225)
    S             : float  reference area [m^2]  (default 0.05)
    C             : float  chord [m]  (default 1.0)
    C_L_trim      : float  trim C_L  (default 0.0; pass CLTRIM from run.py)
    N             : int    preview horizon length  (default 8)
    R             : float  flap-effort weight  (default 3e-4)
    R_du          : float  move-suppression weight  (default 0 — keep at 0)
    G             : int    flap grid points  (default 161)
    delta_max     : float  flap limit [deg]  (default 14)
    delta_dot_max : float  rate limit [deg/s]  (default 300)

    API
    ---
    compute(state, w_seq) -> delta [deg]
        state  : (h, hd, alpha, ad)
        w_seq  : array-like length >= N, w_seq[k] = W(t+(k+1)*dt)
    reset()  -> clears _delta_prev
    _delta_prev  : float — last commanded delta (read/written by run.py LPF chain)
    """

    def __init__(self, aero, U=80.0, dt=0.002, rho=1.225, S=0.05, C=1.0,
                 C_L_trim=0.0, N=8, R=3e-4, R_du=0.0, G=161,
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
        self.G = int(G)
        self.delta_max = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)
        self._dg = np.linspace(-self.delta_max, self.delta_max, self.G)
        self._delta_prev = 0.0

    def compute(self, state, w_seq):
        """
        Return optimal constant-flap deflection [deg].

        w_seq : array-like of length N — fused/oracle gust preview
                w_seq[k] = W(t + (k+1)*dt)  (wnext convention)
        """
        aero = self.aero
        dg = self._dg
        G = self.G
        reach = self.delta_dot_max * self.dt
        ratem = np.abs(dg - self._delta_prev) <= reach + 1e-9

        z_b = np.tile(np.asarray(aero._z, float).reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        J = self.R * dg ** 2 + self.R_du * (dg - self._delta_prev) ** 2

        for k in range(self.N):
            Wk = float(w_seq[k]) if k < len(w_seq) else 0.0
            CL, CM, z_new = aero.batch_step(z_b, x_b, dg, Wk, self.U, self.dt)
            z_b = z_new - self.lam * z_b
            Fy = self.q * CL
            Mz = self.q * CM * self.C
            x_b = dp45_batch(x_b, Fy, Mz, self.dt)
            J = J + (CL - self.C_L_trim) ** 2

        J = np.where(ratem, J, np.inf)
        d = float(dg[int(np.argmin(J))])
        d = float(np.clip(d, self._delta_prev - reach, self._delta_prev + reach))
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        self._delta_prev = d
        return d

    def reset(self):
        """Reset flap state — call before each new simulation run."""
        self._delta_prev = 0.0

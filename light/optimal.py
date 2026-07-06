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

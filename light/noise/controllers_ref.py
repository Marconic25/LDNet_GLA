"""
Reference / mitigation controllers for the noise study. All follow the harness
API  compute(state, W_true, Wc) -> delta  and document which of the two gust
arguments they are allowed to read. light/optimal.py stays untouched.

  OptimalRdu   final controller + R_du move-suppression term (mitigation E2).
               compute() is a verbatim copy of OptimalController.compute with
               ONE added cost term R_du*(delta-delta_prev)^2 — kept in sync by
               eye with light/optimal.py (32 lines, checked against the
               regression anchors at R_du=0).
  PropWRef     proportional + gust feedforward, the equal-info baseline of
               76/honest_home.py (gains g_CL=-60, g_W=-0.5 = its no-flag best
               at W30/Tg0.4). Reads W_true only through the C_L measurement
               (a plant output sensor); the W feedforward term uses Wc.
  MPCConstRef  N-step constant-flap MPC, port of 76/controllers.MPCConst
               (N=4, gate='none' = the known noise-tolerant reference). Uses
               NO preview: the whole horizon is driven by Wc = the (noisy)
               current gust estimate.
"""
import numpy as np
import harness_noise as H
from optimal import OptimalController


class OptimalRdu(OptimalController):
    """Final controller + R_du*(delta - delta_prev)^2 in the candidate cost."""

    def __init__(self, R_du=0.0, **kw):
        super().__init__(**kw)
        self.R_du = float(R_du)

    def compute(self, state, W, W_next=0.0):
        aero = self.aero
        dg = self._dg; G = self.n_grid
        reach = self.delta_dot_max * self.dt
        Wc = float(W_next) if self.use_wnext else float(W)

        z_b = np.tile(np.asarray(aero._z, float).reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        CLg = aero.batch_step(z_b, x_b, dg, Wc, self.U, self.dt)[0]

        cl0 = float(aero.predict(state, 0.0, Wc, self.U)[0])
        neg = dg <= 0.0
        causal = neg if cl0 >= self._C_L_trim else ~neg
        ratem = np.abs(dg - self._delta_prev) <= reach + 1e-9

        cost = ((CLg - self._C_L_trim) ** 2 + self.R * dg ** 2
                + self.R_du * (dg - self._delta_prev) ** 2)
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


def make_optimal_rdu(R=3e-4, R_du=0.0, refine=True):
    return OptimalRdu(R_du=R_du, aero=H.aero, U=H.U, dt=H.DT, R=R,
                      C_L_trim=H.CLTRIM, delta_max=H.DMAX,
                      delta_dot_max=H.DDOT_MAX, n_grid=161,
                      use_wnext=True, refine=refine)


class PropWRef:
    """delta = g_CL*(C_L_meas - trim) + g_W*Wc  (76/controllers.PropW)."""

    def __init__(self, gain_CL=-60.0, gain_W=-0.5,
                 delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX):
        self.gain_CL = float(gain_CL); self.gain_W = float(gain_W)
        self.delta_max = float(delta_max); self.delta_dot_max = float(delta_dot_max)
        self._prev = 0.0

    def reset(self):
        self._prev = 0.0

    def compute(self, state, W_true, Wc):
        # C_L measurement = plant output sensor -> true gust, applied flap
        clm = float(H.aero.predict(state, self._prev, W_true, H.U)[0])
        d = self.gain_CL * (clm - H.CLTRIM) + self.gain_W * float(Wc)
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        rate = self.delta_dot_max * H.DT
        d = float(np.clip(d, self._prev - rate, self._prev + rate))
        self._prev = d
        return d


def dp45_batch(x_b, Fy_b, Mz_b, dt):
    """Batched Dormand-Prince RK45 structural step (fixed-step 5th-order, 6 stages)."""
    import structure
    def rhs(xx):
        h = xx[:, 0]; hd = xx[:, 1]; a = xx[:, 2]; ad = xx[:, 3]
        rh = -Fy_b - structure.D_H * hd - structure.K_H * h
        ra =  Mz_b - structure.D_ALPHA * ad - structure.K_ALPHA * a
        hdd = (structure.M_AA * rh - structure.M_HA * ra) / structure.DET
        add = (structure.M_HH * ra - structure.M_HA * rh) / structure.DET
        return np.stack([hd, hdd, ad, add], axis=1)
    k1 = rhs(x_b)
    k2 = rhs(x_b + dt*(1.0/5)*k1)
    k3 = rhs(x_b + dt*(3.0/40*k1 + 9.0/40*k2))
    k4 = rhs(x_b + dt*(44.0/45*k1 - 56.0/15*k2 + 32.0/9*k3))
    k5 = rhs(x_b + dt*(19372.0/6561*k1 - 25360.0/2187*k2 + 64448.0/6561*k3 - 212.0/729*k4))
    k6 = rhs(x_b + dt*(9017.0/3168*k1 - 355.0/33*k2 + 46732.0/5247*k3 + 49.0/176*k4 - 5103.0/18656*k5))
    return x_b + dt*(35.0/384*k1 + 500.0/1113*k3 + 125.0/192*k4 - 2187.0/6784*k5 + 11.0/84*k6)


class MPCConstRef:
    """
    Constant-flap N-step MPC (port of 76/controllers.MPCConst, gate='none').
    J = sum_k (C_L_k - trim)^2 + R*delta^2, z leak-corrected each step.
    Reads only Wc (its estimate of the CURRENT gust; no preview) — held
    constant over the horizon exactly as in 76/mpc_noise2.py.
    """

    def __init__(self, N=4, R=3e-4, G=161,
                 delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX):
        self.N = int(N); self.R = float(R); self.G = int(G)
        self.delta_max = float(delta_max); self.delta_dot_max = float(delta_dot_max)
        self._dg = np.linspace(-self.delta_max, self.delta_max, self.G)
        self._prev = 0.0

    def reset(self):
        self._prev = 0.0

    def compute(self, state, W_true, Wc):
        aero = H.aero
        dg = self._dg; G = self.G
        reach = self.delta_dot_max * H.DT
        ratem = np.abs(dg - self._prev) <= reach + 1e-9

        z_b = np.tile(np.asarray(aero._z, float).reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        J = self.R * dg ** 2
        for _ in range(self.N):
            CL, CM, z_new = aero.batch_step(z_b, x_b, dg, float(Wc), H.U, H.DT)
            z_b = z_new - H.LAM * z_b
            Fy = H.q * CL; Mz = H.q * CM * H.C
            x_b = dp45_batch(x_b, Fy, Mz, H.DT)
            J = J + (CL - H.CLTRIM) ** 2
        J = np.where(ratem, J, np.inf)
        d = float(dg[int(np.argmin(J))])
        d = float(np.clip(d, self._prev - reach, self._prev + reach))
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        self._prev = d
        return d

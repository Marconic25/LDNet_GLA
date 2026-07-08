"""
Single-step controllers for the 76/ investigation.

OptGrid is one flexible one-step-optimal controller covering every single-step
hypothesis via flags:

  R          effort weight            J += R*delta^2
  R_du       move suppression         J += R_du*(delta-delta_prev)^2
  G          # flap candidates on [-DMAX,DMAX]
  gate       'hard'  : causal sign gate at cl0 vs CLTRIM (== oracle/light)
             'db'    : gate with a deadband db around trim -> inside the band the
                       full range is allowed (removes the hard switch)
             'none'  : no causal gate (rely on R_du to avoid the wrong-sign basin)
  db         deadband half-width in C_L units (gate=='db')
  use_wnext  evaluate candidates against W(t+dt) instead of W(t) (still one step,
             W is a legal oracle so W(t+dt) is known)
  refine     parabolic refine of the argmin within its grid cell

Candidate C_L is the frozen-z reconstruction (== aero.predict), evaluated for all
G deltas in one batched NNrec call for speed; the plant is then advanced by the
honest scalar map (aero.predict + aero.advance), exactly as light/run.py does.
So the ROLLOUT is B=1 (deterministic, batch-position independent); only the
candidate scan is vectorized, and it does not touch plant state.
"""
import numpy as np
import harness as H


class OptGrid:
    def __init__(self, R=3e-4, R_du=0.0, G=161, gate='hard', db=0.0,
                 use_wnext=False, refine=False, z_aware=False,
                 delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX):
        self.R = float(R); self.R_du = float(R_du); self.G = int(G)
        self.gate = gate; self.db = float(db)
        self.use_wnext = bool(use_wnext); self.refine = bool(refine)
        self.z_aware = bool(z_aware)
        self.delta_max = float(delta_max); self.delta_dot_max = float(delta_dot_max)
        self._dg = np.linspace(-self.delta_max, self.delta_max, self.G)
        self._prev = 0.0

    def reset(self):
        self._prev = 0.0

    def _causal_mask(self, cl0):
        dg = self._dg; e0 = cl0 - H.CLTRIM
        neg = dg <= 0.0
        if self.gate == 'none':
            return np.ones_like(dg, dtype=bool)
        if self.gate == 'db' and abs(e0) <= self.db:
            return np.ones_like(dg, dtype=bool)      # full range inside deadband
        # hard (or db outside band): sign gate
        return neg if e0 >= 0.0 else ~neg

    def compute(self, aero, state, prev, Wi, Wn):
        # keep rate-limit consistent with the plant's actually-applied prev
        self._prev = float(prev)
        W = float(Wn) if self.use_wnext else float(Wi)
        dg = self._dg; G = self.G
        reach = self.delta_dot_max * H.DT

        z = np.asarray(aero._z, float)
        z_b = np.tile(z.reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        # frozen-z candidate C_L for all deltas (batch_step reconstructs from current z_b)
        CLg, _, z_new = aero.batch_step(z_b, x_b, dg, W, H.U, H.DT)
        if self.z_aware:
            # z-aware next-step C_L (== predict_step): reconstruct from the
            # leak-corrected advanced latent z_next = (1-LAM)z + (dt/dt_ref)dz.
            z_next = z_new - H.LAM * z_b
            CLg = aero.batch_step(z_next, x_b, dg, W, H.U, H.DT)[0]

        cl0 = float(aero.predict(state, 0.0, W, H.U)[0])
        causal = self._causal_mask(cl0)
        ratem = np.abs(dg - self._prev) <= reach + 1e-9
        feas = causal & ratem
        cost = (CLg - H.CLTRIM)**2 + self.R * dg**2 + self.R_du * (dg - self._prev)**2
        cost = np.where(feas, cost, np.inf)
        j = int(np.argmin(cost))
        d = float(dg[j])

        if self.refine and 0 < j < G - 1 and np.isfinite(cost[j-1]) and np.isfinite(cost[j+1]):
            c0, c1, c2 = cost[j-1], cost[j], cost[j+1]
            denom = (c0 - 2*c1 + c2)
            if denom > 1e-30:
                shift = 0.5 * (c0 - c2) / denom            # in grid-cells, (-0.5,0.5)
                d = d + shift * (dg[1] - dg[0])

        d = float(np.clip(d, self._prev - reach, self._prev + reach))
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        self._prev = d
        return d


class MPCConst:
    """
    Short-horizon MPC: hold a constant flap delta over N steps, pick the delta
    minimising the accumulated cost

        J = sum_{k=1..N} [ (C_L_k - CLTRIM)^2 + Qad*(ad_k - ad_{k-1})^2 ]
            + R*delta^2 + R_du*(delta-delta_prev)^2

    Rolls z (leak-corrected) + structural state forward with LDNetAero.batch_step
    and the batched RK4 (== clean/controller._rollout_cost_batch, no DLPF).
    The horizon lets the cost SEE the C_L overshoot that a one-step cost misses.
    """
    def __init__(self, N=4, R=3e-4, R_du=0.0, Qad=0.0, G=161, gate='hard',
                 delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX):
        self.N = int(N); self.R = float(R); self.R_du = float(R_du)
        self.Qad = float(Qad); self.G = int(G); self.gate = gate
        self.delta_max = float(delta_max); self.delta_dot_max = float(delta_dot_max)
        self._dg = np.linspace(-self.delta_max, self.delta_max, self.G)
        self._prev = 0.0

    def reset(self):
        self._prev = 0.0

    def compute(self, aero, state, prev, Wi, Wn):
        self._prev = float(prev)
        dg = self._dg; G = self.G; reach = self.delta_dot_max * H.DT
        # causal gate from the frozen-z natural C_L (== oracle)
        cl0 = float(aero.predict(state, 0.0, float(Wi), H.U)[0]); e0 = cl0 - H.CLTRIM
        neg = dg <= 0.0
        if self.gate == 'none':
            causal = np.ones_like(dg, dtype=bool)
        else:
            causal = neg if e0 >= 0.0 else ~neg
        ratem = np.abs(dg - self._prev) <= reach + 1e-9
        feas = causal & ratem

        z_b = np.tile(np.asarray(aero._z, float).reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        J = self.R * dg**2 + self.R_du * (dg - self._prev)**2
        prev_ad = float(state[3])
        for _ in range(self.N):
            CL, CM, z_new = aero.batch_step(z_b, x_b, dg, float(Wi), H.U, H.DT)
            z_b = z_new - H.LAM * z_b
            Fy = H.q * CL; Mz = H.q * CM * H.C
            x_b = H.rk4_batch(x_b, Fy, Mz, H.DT)
            J = J + (CL - H.CLTRIM)**2 + self.Qad * (x_b[:, 3] - prev_ad)**2
            prev_ad = x_b[:, 3].copy()
        J = np.where(feas, J, np.inf)
        d = float(dg[int(np.argmin(J))])
        d = float(np.clip(d, self._prev - reach, self._prev + reach))
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        self._prev = d
        return d


class PropW:
    """
    Proportional + gust feedforward (equal-info baseline, == light/proportional.py).

    use_wnext=True feeds the feedforward term the next-step gust W(t+dt) instead of
    W(t) — the SAME one-step preview the wnext optimal uses, so the two laws see an
    identical information set {x(t), C_L measurement, W(t), W(t+dt)}. The C_L
    measurement stays at time t (it is a sensor reading, not a prediction).
    """
    def __init__(self, gain_CL=-40.0, gain_W=0.0, use_wnext=False,
                 delta_max=14.0, delta_dot_max=H.DDOT_MAX):
        self.gain_CL = float(gain_CL); self.gain_W = float(gain_W)
        self.use_wnext = bool(use_wnext)
        self.delta_max = float(delta_max); self.delta_dot_max = float(delta_dot_max)
        self._prev = 0.0

    def reset(self):
        self._prev = 0.0

    def compute(self, aero, state, prev, Wi, Wn):
        clm = float(aero.predict(state, prev, Wi, H.U)[0])   # measure with applied flap
        Wff = float(Wn) if self.use_wnext else float(Wi)
        d = self.gain_CL * (clm - H.CLTRIM) + self.gain_W * Wff
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        rate = self.delta_dot_max * H.DT
        d = float(np.clip(d, prev - rate, prev + rate))
        self._prev = d
        return d

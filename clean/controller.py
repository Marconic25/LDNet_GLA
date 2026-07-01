"""
One-step optimal controller for aeroelastic gust load alleviation.

At each time step the controller solves the scalar optimisation:

    min_{δ}  Q_h · h(k+1)²  +  Q_α · α(k+1)²  +  Q_CL · C_L(k+1)²  +  R · δ²

subject to   |δ| ≤ δ_max   and   |δ - δ_prev| / dt ≤ δ̇_max

The next state x(k+1) and lift coefficient C_L(k+1) are predicted by:
  1. calling the aerodynamic model with the candidate δ and the current
     gust estimate W_hat;
  2. advancing the structural state one RK4 step with the resulting forces.

Because the cost is smooth and the search is over a scalar bounded interval,
scipy.optimize.minimize_scalar with method='bounded' is fast and reliable.

This controller is model-agnostic: swapping `aero_predict` for an LDNet
wrapper gives the neural-network version with no other changes.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import structure as _st
from structure import rhs as structural_rhs

# Air and geometry constants (must match the values used in the simulation)
RHO   = 1.225   # air density [kg/m³]
S_REF = 0.05    # reference wing area [m²]
C_REF = 1.0     # reference chord [m]


class Controller:
    """
    One-step optimal controller.

    Parameters
    ----------
    aero_predict : callable(state, delta_deg, W, U) -> (C_L, C_M)
        Aerodynamic prediction function.  Pass `aero.predict` for the linear
        model or a wrapper around LDNet for the neural-network version.
    U            : float   freestream velocity [m/s]
    dt           : float   simulation time step [s]
    Q_h          : float   cost weight on heave displacement h [m]
    Q_alpha      : float   cost weight on pitch angle α [rad]
    Q_alpha_dot  : float   cost weight on pitch rate α̇ [rad/s] — damps pitch oscillations
    Q_CL         : float   cost weight on lift coefficient C_L
    R            : float   cost weight on control effort δ [deg]
    delta_max    : float   flap deflection limit [deg]
    delta_dot_max: float   flap rate limit [deg/s]
    """

    def __init__(self, aero_predict, U=80.0, dt=0.01,
                 Q_h=1e4, Q_alpha=1e4, Q_alpha_dot=1e4, Q_CL=1e3, R=1.0,
                 delta_max=20.0, delta_dot_max=100.0,
                 C_L_trim=0.0, Fy_trim=0.0, Mz_trim=0.0,
                 global_search=True, n_grid=25, causal_basin=False, R_du=0.0, target_lpf=0.0,
                 e_ref=0.0, R_sched_gain=1.0, lpf_max=0.0,
                 mpc_horizon=1, aero=None):
        self.aero_predict  = aero_predict
        self.U             = float(U)
        self.dt            = float(dt)
        self.Q_h           = float(Q_h)
        self.Q_alpha       = float(Q_alpha)
        self.Q_alpha_dot   = float(Q_alpha_dot)
        self.Q_CL          = float(Q_CL)
        self.R             = float(R)
        self.delta_max     = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)

        self._q_dyn    = 0.5 * RHO * U**2 * S_REF
        self._delta_prev = 0.0
        # Trim offsets: controller integrates perturbations from equilibrium,
        # and penalizes C_L deviation from trim (not absolute C_L).
        self._C_L_trim = float(C_L_trim)
        self._Fy_trim  = float(Fy_trim)
        self._Mz_trim  = float(Mz_trim)
        self.global_search = bool(global_search)
        self.n_grid = int(n_grid)
        self.causal_basin = bool(causal_basin)
        self.R_du = float(R_du)   # move-suppression: penalize (delta - delta_prev)^2
        self.target_lpf = float(target_lpf)  # low-pass the optimal target delta (0=off)
        self._d_filt = 0.0
        self.e_ref = float(e_ref)          # C_L-excursion ref for gust gain-scheduling (0=off)
        self.R_sched_gain = float(R_sched_gain)
        self._R_eff = float(self.R)
        self.lpf_max = float(lpf_max)   # schedule target_lpf up to lpf_max at high excursion
        self.mpc_horizon = int(mpc_horizon)  # >1: receding-horizon (constant-delta rollout)
        self._aero = aero                    # LDNetAero object (needed for the rollout)

    # ------------------------------------------------------------------
    def _predict_next(self, state, delta_deg, W_hat):
        """
        Predict the state and C_L one step ahead for a given δ.

        Uses the aerodynamic model to get forces, then integrates
        the structural equations with RK4.
        """
        C_L, C_M = self.aero_predict(state, delta_deg, W_hat, self.U)

        Fy = self._q_dyn * C_L - self._Fy_trim
        Mz = self._q_dyn * C_M * C_REF - self._Mz_trim

        x = np.asarray(state, dtype=float)
        dt = self.dt

        def f(s):
            return np.array(structural_rhs(s, Fy, Mz))

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return x_next, float(C_L)

    # ------------------------------------------------------------------
    def _rollout_cost(self, delta_deg, state, W_hat):
        """Accumulated cost of HOLDING delta_deg over mpc_horizon steps (constant-delta
        receding horizon). Rolls the latent z + structural state forward and sums the
        stage costs, so a delta that excites the pitch mode is penalized by the growing
        alpha_dot over the horizon -- which a one-step cost cannot see. Restores z."""
        aero = self._aero
        z_save = aero._z.copy()
        x = np.asarray(state, dtype=float)
        W = float(W_hat); dt = self.dt
        J = self.R * delta_deg**2 + self.R_du * (delta_deg - self._delta_prev)**2
        x_prev_ad = float(state[3])
        try:
            for _ in range(self.mpc_horizon):
                C_L, C_M = aero.predict(x, delta_deg, W, self.U)
                aero.advance(x, delta_deg, W, self.U, dt)
                Fy = self._q_dyn * float(C_L) - self._Fy_trim
                Mz = self._q_dyn * float(C_M) * C_REF - self._Mz_trim
                def f(ss): return np.array(structural_rhs(ss, Fy, Mz))
                k1=f(x); k2=f(x+0.5*dt*k1); k3=f(x+0.5*dt*k2); k4=f(x+dt*k3)
                x = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
                J += (self.Q_h*x[0]**2 + self.Q_alpha*x[2]**2
                      + self.Q_alpha_dot*(x[3]-x_prev_ad)**2
                      + self.Q_CL*(float(C_L)-self._C_L_trim)**2)
                x_prev_ad = float(x[3])
        finally:
            aero._z = z_save
        return J

    def _rk4_batch(self, x_b, Fy_b, Mz_b, dt):
        """Batched RK4 structural step (Fy_b, Mz_b held constant over the step)."""
        def rhs(xx):
            h=xx[:,0]; hd=xx[:,1]; a=xx[:,2]; ad=xx[:,3]
            rh = -Fy_b - _st.D_H*hd - _st.K_H*h
            ra =  Mz_b - _st.D_ALPHA*ad - _st.K_ALPHA*a
            hdd = (_st.M_AA*rh - _st.M_HA*ra)/_st.DET
            add = (_st.M_HH*ra - _st.M_HA*rh)/_st.DET
            return np.stack([hd, hdd, ad, add], axis=1)
        k1=rhs(x_b); k2=rhs(x_b+0.5*dt*k1); k3=rhs(x_b+0.5*dt*k2); k4=rhs(x_b+dt*k3)
        return x_b + (dt/6.0)*(k1+2*k2+2*k3+k4)

    def _rollout_cost_batch(self, deltas, state, W_hat):
        """Vectorized rollout cost for ALL candidate deltas at once (B = len(deltas)).
        One batched TF call per horizon step instead of B per step -> ~B x faster."""
        aero=self._aero; B=len(deltas)
        z_b=np.tile(np.asarray(aero._z).reshape(1,-1),(B,1))
        x_b=np.tile(np.asarray(state,dtype=float).reshape(1,-1),(B,1))
        d_b=np.asarray(deltas,dtype=float)
        J=self.R*d_b**2+self.R_du*(d_b-self._delta_prev)**2
        x_prev_ad=float(state[3])
        for _ in range(self.mpc_horizon):
            CL,CM,z_b=aero.batch_step(z_b,x_b,d_b,float(W_hat),self.U,self.dt)
            Fy=self._q_dyn*CL-self._Fy_trim
            Mz=self._q_dyn*CM*C_REF-self._Mz_trim
            x_b=self._rk4_batch(x_b,Fy,Mz,self.dt)
            J=J+(self.Q_h*x_b[:,0]**2+self.Q_alpha*x_b[:,2]**2
                 +self.Q_alpha_dot*(x_b[:,3]-x_prev_ad)**2
                 +self.Q_CL*(CL-self._C_L_trim)**2)
            x_prev_ad=x_b[:,3].copy()
        return J

    # ------------------------------------------------------------------
    def _cost(self, delta_deg, state, W_hat):
        """Objective function evaluated at a candidate δ."""
        x_next, C_L = self._predict_next(state, delta_deg, W_hat)
        return (  self.Q_h         * x_next[0]**2
                + self.Q_alpha     * x_next[2]**2
                + self.Q_alpha_dot * (x_next[3] - state[3])**2
                + self.Q_CL        * (C_L - self._C_L_trim)**2
                + self._R_eff      * delta_deg**2
                + self.R_du        * (delta_deg - self._delta_prev)**2)

    # ------------------------------------------------------------------
    def compute(self, state, W_hat=0.0):
        """
        Compute the optimal flap deflection for the current time step.

        Parameters
        ----------
        state : array-like (h, ḣ, α, α̇)
        W_hat : float   estimated gust velocity [m/s] (from observer)

        Returns
        -------
        delta : float   flap deflection command [degrees]
        """
        # Rate limit: constrain search interval to reachable δ values
        reach = self.delta_dot_max * self.dt
        lb = max(-self.delta_max, self._delta_prev - reach)
        ub = min( self.delta_max, self._delta_prev + reach)

        # Causal-basin: restrict to the flap-sign half that reduces the lift
        # excursion. In this model dC_L/ddelta>0 (cf. mpc_gust: a *negative* gain
        # reduces a positive C_L excursion), so to reduce C_L above trim the
        # lift-reducing flap deflects NEGATIVE, and vice-versa. Avoids the
        # non-causal "nulling" basin that destabilizes the non-monotone C_L(delta).
        if self.causal_basin or self.e_ref > 0.0:
            cl0 = float(self.aero_predict(state, 0.0, float(W_hat), self.U)[0])
        else:
            cl0 = None
        # Gust gain-scheduling: gentler control (larger R) at larger C_L excursion,
        # so the optimizer does not over-react and excite pitch at strong gusts.
        if self.e_ref > 0.0:
            e = abs(cl0 - self._C_L_trim)
            self._R_eff = self.R * max(1.0, e / self.e_ref) ** self.R_sched_gain
        else:
            self._R_eff = self.R
        if self.causal_basin:
            if cl0 >= self._C_L_trim:
                g_lo, g_hi = -self.delta_max, 0.0   # reduce C_L>trim with negative flap
            else:
                g_lo, g_hi = 0.0, self.delta_max
        else:
            g_lo, g_hi = -self.delta_max, self.delta_max
        lb = max(lb, g_lo); ub = min(ub, g_hi)

        if lb >= ub:
            delta = float(np.clip(self._delta_prev, g_lo, g_hi))
            self._delta_prev = delta
            return delta

        # Fast vectorized MPC: evaluate the whole delta grid in batched TF calls.
        if self.mpc_horizon > 1 and self._aero is not None and hasattr(self._aero, 'batch_step'):
            grid = np.linspace(g_lo, g_hi, self.n_grid)
            J = self._rollout_cost_batch(grid, state, float(W_hat))
            delta = float(np.clip(float(grid[int(np.argmin(J))]), lb, ub))
            self._delta_prev = delta
            return delta

        if self.global_search:
            # Non-convex one-step cost: confine to causal range, take the GLOBAL
            # optimum, then rate-limit the move toward it.
            obj = self._rollout_cost if (self.mpc_horizon > 1 and self._aero is not None) else self._cost
            grid = np.linspace(g_lo, g_hi, self.n_grid)
            costs = [obj(float(d), state, float(W_hat)) for d in grid]
            j = int(np.argmin(costs)); d_target = float(grid[j])
            step = grid[1] - grid[0]
            ref = minimize_scalar(
                obj,
                bounds=(max(g_lo, d_target - step), min(g_hi, d_target + step)),
                method='bounded', args=(state, float(W_hat)), options={'xatol': 0.01})
            if float(ref.fun) <= costs[j]:
                d_target = float(ref.x)
            lpf_eff = self.target_lpf
            if self.lpf_max > self.target_lpf and self.e_ref > 0.0:
                frac = min(1.0, max(0.0, (abs(cl0 - self._C_L_trim) / self.e_ref - 1.0) / 0.6))
                lpf_eff = self.target_lpf + (self.lpf_max - self.target_lpf) * frac
            if lpf_eff > 0.0:
                self._d_filt = lpf_eff * self._d_filt + (1.0 - lpf_eff) * d_target
                d_target = self._d_filt
            delta = float(np.clip(d_target, lb, ub))   # rate-limit the move
        else:
            result = minimize_scalar(
                self._cost, bounds=(lb, ub), method='bounded',
                args=(state, float(W_hat)), options={'xatol': 0.01})
            delta = float(result.x)
        self._delta_prev = delta
        return delta

    def reset(self):
        """Reset internal state (call before each new simulation run)."""
        self._delta_prev = 0.0
        self._d_filt = 0.0


class ProportionalController:
    """Pure C_L-feedback flap law for gust load alleviation.

        delta = clip( gain * (C_L_meas - C_L_trim) )   [rate- and magnitude-limited]

    Needs only the measured lift coefficient — no gust estimate and no state
    estimate. This is robust where the model-based optimizer fails: LDNet's
    C_L(W) is non-invertible (so a gust observer cannot recover W) and its
    one-step optimizer prediction destabilizes the loop. Reacting directly to
    measured C_L sidesteps both problems.
    """

    def __init__(self, C_L_trim=0.0, gain=10.0, dt=0.01,
                 delta_max=20.0, delta_dot_max=100.0):
        self.C_L_trim      = float(C_L_trim)
        self.gain          = float(gain)
        self.dt            = float(dt)
        self.delta_max     = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)
        self._delta_prev   = 0.0

    def compute(self, C_L_meas):
        d = self.gain * (float(C_L_meas) - self.C_L_trim)
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        rate = self.delta_dot_max * self.dt
        d = float(np.clip(d, self._delta_prev - rate, self._delta_prev + rate))
        self._delta_prev = d
        return d

    def reset(self):
        self._delta_prev = 0.0

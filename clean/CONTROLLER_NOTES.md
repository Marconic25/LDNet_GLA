# GLA controllers on LDNet — comparison & robustness

Two working controllers (no gust observer — both use measured C_L; the C_L->W
inversion is non-invertible with LDNet):

- **ProportionalController**: delta = G*(C_L_meas - C_L_trim), G=10.
- **Controller (model-based one-step optimal)**: global + causal-basin search,
  target_lpf=0.95, Q_alpha_dot=3e5, Q_CL=1e3, R=0.3.

## Nominal gust (W0=10, gust window)
| ctrl | C_L | h_ddot | alpha | alpha_dot | flap |
|------|-----|--------|-------|-----------|------|
| proportional | +21.6% | +31.4% | +51% | +3.8% | 3.3 deg |
| optimal      | +23.8% | +31.4% | +54% | +11.1% | 7.1 deg |

Optimal beats proportional on every metric at the design gust. Keys to the optimal
working at all: global search (non-convex 1-step cost), causal-basin (avoid the
non-causal nulling basin), target LPF (kill chatter), Q_alpha_dot (protect pitch).

## Gust-velocity robustness (C_L / h_ddot / alpha_dot reduction %)
| W0 | proportional | optimal |
|----|--------------|---------|
|  5 | +16.7/+27.0/ -8.8 | +19.1/+27.0/ -0.5 |
| 10 | +21.6/+31.4/ +3.8 | +23.8/+31.4/+11.1 |
| 15 | +25.3/+35.5/ +1.6 | +27.3/+35.5/ +9.1 |
| 20 | +22.3/+38.9/ -8.5 |  +6.5/ -6.2/-168  |
| 30 | +17.2/+44.0/-38.9 | -35.7/-59.3/-466  |

**Proportional is robust across W0=5..30** (always alleviates C_L & h_ddot).
**Optimal beats it for W0<=15 but DESTABILIZES at W0>=20** (h_ddot worse than open,
alpha_dot blows up). The optimal's fixed weights are tuned to the W0=10 signal scale;
larger gusts push the model into regimes where those gains over-react.

To make the optimal robust: gain-schedule / normalize the cost weights by the
measured C_L-excursion (gust) magnitude, or adapt R / Q_alpha_dot online.

## Robustness investigation — making the optimal work across gusts
The optimal destabilizes at W0>=20 via a **pitch resonance**: at large gusts its
sharp one-step delta moves (incl. causal-basin sign flips when C_L crosses trim)
drive alpha_dot into a growing oscillation (instrumented in diag20.py).

Levers tried:
- **R gain-scheduling** (scale control penalty with C_L excursion): NO effect
  (R is not the dominant cost term vs Q_CL/Q_alpha_dot).
- **Fixed heavy LPF**: DOES stabilize high gust but kills C_L:
  | W0 | lpf=0.95 (adot) | lpf=0.99 | lpf=0.995 |
  | 20 | -168%  | -107% | **+2.6%** |
  | 30 | -466%  | **+11.4%** | (over-damped) |
  So a LPF schedule (light at design gust, heavy at high gust) is the mechanism.
- **LPF schedule keyed on closed-loop C_L excursion**: FAILS to activate —
  when the controller is active the cl0 excursion stays near trim, so the
  scheduling signal does not see the true gust strength. Fundamental flaw:
  the scheduling variable is corrupted by the control action.

Conclusion: simple gain-scheduling does not cleanly restore high-gust robustness.
The instability is multi-step (pitch coupling) and the gust strength is not
reliably observable with this model (C_L(W) non-invertible). Robust options:
  1) short receding-horizon MPC (anticipates the pitch dynamics that cause the
     resonance) — the principled fix, beyond one-step;
  2) a reliable independent gust/disturbance estimate to key the schedule on;
  3) operational: proportional for robustness across all gusts + optimal for
     peak performance within its validated range (W0<=15).

Controller now exposes e_ref/R_sched_gain/lpf_max (scheduling infrastructure),
target_lpf, causal_basin, global_search, R_du for future work.

## CRITICAL: physical regime + latent free-running instability (the real caveat)
The closed-loop study above was run COLD (latent z=0). That is NOT physical:
- z=0 gives trim C_L=0.543 and gust C_L excursion ~0.81 (W0=10).
- The CFD data has trim F_y=164 N (C_L~0.84) and gust excursion ~0.27.
The PHYSICAL regime is the WARMED one (run.py --warmup-csv): trim C_L=0.839 (matches
CFD exactly), gust excursion ~0.18. So the cold study inflates the gust load ~4x.

In the physical (warmed) regime the controllers reduce the C_L EXCURSION
(prop -51%, mpc -62%) BUT h_ddot is unchanged (+0.0%) for all of them. Reason
(figE_latent_drift.png): the LDNet latent state has NO equilibrium -- ||z|| drifts
+~4.4/step unbounded. The warmed 'trim' is therefore not a true equilibrium, so the
structure gets a startup transient at t=0 that rings down and DOMINATES h_ddot. The
h_ddot trajectory is identical with and without the gust -> the gust barely couples
to the structure at the high-z operating point.

Bottom line: LDNet is excellent for TEACHER-FORCED replay (NRMSE 0.019) but its
FREE-RUNNING latent dynamics are unstable, so it is not a physically reliable forward
simulator for closed-loop GLA. Fix is at the MODEL level: retrain with a latent
stability/equilibrium regularization (penalize ||z|| drift, or an explicit damped
latent ODE) so the free-running model has a trim and physical gust->structure
coupling. Inference-time leak on z is a possible stopgap (test pending) but will
likely degrade the replay accuracy the model was trained for.

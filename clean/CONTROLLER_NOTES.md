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

## Task A — damping-lambda sweep for gust->structure coupling (2026-06-10)
Retrained the damped-ODE LDNet at smaller lambda (NADAM=400/NBFGS=150 each) to
recover the gust's indirect channel (W -> latent -> C_L -> structure), which the
lambda=0.01 model damps away (latent memory 1/lambda too short vs the 1s gust).
Variants in clean/models_damped_l005 (lambda=0.005) and _l003 (lambda=0.003).
Evaluated all with clean/val_damped_param.py (replay on CFD sim_A_025_test +
free-run ||z|| stability + gust vs no-gust h_ddot coupling):

| model    | lambda | replay NRMSE_F_y | free-run ||z|| | trim C_L | gust C_L exc | gust->h_ddot |
|----------|--------|------------------|----------------|----------|--------------|--------------|
| original | 0.000  | 0.019            | 6567 (UNBND)   | 1.329    | 0.625        | 9.93 COUPLED |
| damped   | 0.010  | 0.036            | 175  (bnd)     | 0.640    | 0.299        | 0.15 decoup  |
| l005     | 0.005  | 0.092            | 416  (bnd)     | 0.793    | 0.120        | 0.29 decoup  |
| l003     | 0.003  | 0.146            | 244  (bnd)     | 0.930    | 0.459        | 2.38 COUPLED |

Finding: lower lambda restores coupling (longer latent memory) but degrades replay
accuracy at fixed (short) BFGS budget. Only lambda=0.003 hits BOTH bounded ||z||
(244) AND real gust->h_ddot (2.38 m/s2 > 2.0 goal); its replay NRMSE (0.146) misses
the <0.04 goal. lambda=0.005 is the worst trade (decoupled AND less accurate than
0.01). The coupling is structural (memory length), not from undertraining, so it
should survive longer training -> CHOSEN lambda=0.003, proceed to Task B
(full-convergence) to drive replay NRMSE down while keeping coupling.

Infra added: optimization.py gained a no-op-default checkpoint hook
(checkpoint_callback / checkpoint_every); sensitivity_latent_damped_ckpt.py wires it
to save weights+config every 200 iters (BFGS saves only at end otherwise).

## Task B — full-convergence of l003 (NBFGS=4000) + the coupling collapse (2026-06-10)
Trained lambda=0.003 to convergence (4000 BFGS, ~7h, val loss plateaued ~1.27e-4) ->
clean/models_damped_l003_full. Validation (val_damped_param.py):
  replay NRMSE_F_y 0.011 (test-set NRMSE 0.0024, BETTER than original 0.019/0.014),
  free-run ||z|| 521 (bounded), trim C_L 0.227, gust->h_ddot 0.10 (DECOUPLED).
vs l003@150iter: NRMSE 0.146, ||z|| 244, gust->h_ddot 2.38 (COUPLED).

KEY FINDING: the gust->structure coupling (2.38) seen in the UNDERTRAINED l003 was a
TRAINING ARTIFACT, not structural. At convergence lambda=0.003 decouples (0.10) just
like lambda=0.01. => Task A premise (tune lambda for coupling) FAILS: replay accuracy
and free-running gust-coupling are in tension; convergence favors accuracy and kills
the free-running W->latent->C_L->structure channel. Open question (investigating):
is the decoupling real, or a test artifact of the startup transient (x0 trim computed
at a latent zsave that then drifts once the sim runs)? Need a proper JOINT (latent+
structure) equilibrium test before concluding on Tasks C/D.

## DECISIVE: CFD ground truth reframes the whole coupling question (2026-06-10)
Checked the CFD reference sim_A_025_test (gust W_peak=11.46) directly:
  C_L trim 0.833, C_L excursion 0.321
  h_ddot PEAK = 0.113 m/s2 (rms 0.033); h peak-peak 2.6mm, alpha 0.06deg
The real structure barely accelerates under the gust: a 1s gust (~1 Hz) is QUASI-STATIC
for the 5.8 Hz heave / 14.5 Hz pitch modes -> small h_ddot is CORRECT PHYSICS.

=> Task A's goal (gust->h_ddot > 2 m/s2) was anchored to the COLD UNSTABLE model's
inflated 9.93 (~90x the true 0.11) and is WRONG. The proper alleviation metric is the
C_L EXCURSION (the load), not h_ddot.

Coupling diagnostic (invest_coupling.py), joint latent+structure fixed point then gust:
  lambda=0.01 (models_damped): settles to PHYSICAL trim C_L=0.843 (=CFD 0.833), ||z||175,
    residual accel 0; DIRECT channel dC_L/dW real (W=10 -> dC_L=-0.125), latent +0.085,
    gust C_L excursion 0.20, h_ddot 0.044  => correct order of magnitude vs CFD. GOOD.
  lambda=0.003_full: coupled latent+structure free-run is UNSTABLE (diverges to C_L=1.34,
    a_ddot=72, h_ddot 283) even though latent-alone (struct clamped) is bounded. BAD.

CONCLUSION: stop lowering lambda. lambda=0.01 is the correct closed-loop model (stable
physical trim, real gust load). Task B redefined = full-convergence of LAMBDA=0.01
(current model only NADAM400/NBFGS150 -> NRMSE 0.036; target ~0.01 like l003_full) while
KEEPING stability. Tasks C/D evaluate controllers on the C_L-excursion metric.

## RESOLUTION: closed-loop rollout training -> usable model -> working GLA (2026-06-11)
Root cause of 'cannot use the trained LDNet': teacher-forced (1-step) training feeds the
TRUE structural states as inputs, so the net reads C_L off the states and (a) ignores
delta/W when undertrained (flap-blind, but free-run stable because lazy), or (b) becomes
free-run UNSTABLE when fully converged. Accuracy vs closed-loop stability were in tension.

FIX (src/sensitivity_latent_rollout.py): closed-loop ROLLOUT training. The 2-DOF
structural ODE is propagated IN the loss from the model's OWN predicted loads (delta,W
exogenous from data); loss matches rolled-out states+loads to data over ROLLOUT_LEN=800
steps. Warm-start from l003_full (flap-aware), pure BFGS (Adam degraded the warm-start),
best-VALIDATION early stopping (BFGS overfits: train->0.007 while valid->0.10; best valid
0.0114 at ~iter 20). TF structure step verified bit-identical to structure.py.

RESULT — the rollout model is closed-loop USABLE:
- Free-run joint equilibrium STABLE: settles to C_L=0.856 (=CFD 0.84), residual accel 0
  (l003_full diverged here; l01_full was chaotic).
- Captures flap (B ablation 0.111, Cc 0.031) AND gust.
- Closed-loop replay (z=0 start, structure from model loads, FULL loads) on held-out
  trajectories: sim_A_025(gust) C_L/h/a NRMSE 0.11/0.12/0.12; sim_Cc(gust+flap) 0.055/
  0.063/0.046; sim_B(flap) 0.096. Stable, final state matches data.
  NOTE: the rollout model's regime is z=0 + FULL loads (data trim state is the eq under
  full load). Do NOT pre-settle the latent or subtract trim (those gave the spurious
  'decoupled' / 8 m/s2-transient artifacts).

GLA CONTROL WORKS (clean/taskC_gla.py, proportional delta=-G*(C_L-trim), real gust):
  sim_A_025 (exc 0.241): gain 10 -> -39%, gain 20 -> -59%, gain 80 -> -78% C_L excursion,
  flap only 1.5-2.8 deg, h_pk ALSO reduced (-9..-18%), no pitch destabilization.
  sim_A_026 (exc 0.60): optimal gain ~20 (-37%); gain>=40 saturates flap at 14 deg ->
  gain-scheduling needed for strong gusts.
The earlier 'controllers fail / flap counterproductive' conclusion was an artifact of the
flap-blind undertrained l=0.01 model; with a usable model GLA alleviates cleanly.

## Rollout model — GLA robustness across held-out gusts (2026-06-11)
Closed-loop accuracy (clcheck.py, z=0 + full loads) generalizes on the held-out TEST set:
gust families A/Cc C_L/h/a NRMSE 0.04-0.12 (A_025 0.11, A_027 0.05, A_029 0.05, Cc_060/063/
067 0.06/0.04/0.08). Caveat: very strong gusts are UNDER-predicted in magnitude (A_027
data C_L exc 0.99 -> model 0.59; extrapolation beyond training range). Flap-only B family
has higher relative NRMSE (0.14-0.27) but tiny absolute signals.

GLA control generalization (clean_sweep.py, proportional delta=-G*(C_L-trim), +LPF option):
| gust (open exc) | best C_L reduction | flap | notes |
| A_025 (0.241 weak)  | -84% (G150) / -96% (LPF G300) | 3-5 deg  | no saturation |
| A_028 (0.486 med)   | -61% (G40) / -64% (LPF G40)   | 7 deg    | saturates >G40 |
| A_027 (0.584 strong)| -57% (G80)                    | 14 (sat) | LPF no help; high G hurts heave |
Pattern: optimal gain DECREASES with gust strength; strong gusts saturate the flap (14 deg)
and over-driving raises h_pk -> GAIN-SCHEDULING needed (key schedule on a gust estimate, not
closed-loop C_L which the controller flattens). dC_L/ddelta~0.014/deg sets the authority
limit: cancelling a 0.6 excursion needs ~14 deg (saturation). NEXT: MPC in the z=0 regime
(anticipates), gust-magnitude-scheduled gain, extend ROLLOUT_LEN, run.py z=0 driver path.

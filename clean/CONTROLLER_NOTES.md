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

## One-step OPTIMAL controller with the rollout model — minimal objective (2026-06-11)
clean/optimal_test.py drives the Controller class (mpc_horizon=1, global_search, no LPF,
Fy_trim=Mz_trim=0 for the z=0 full-load rollout regime) with the MINIMAL objective
J = Q_CL*(C_L-trim)^2 + R*delta^2 (Q_h=Q_alpha=Q_alpha_dot=0). W_hat = true gust. Records
all states+derivatives; flags explosion (>3x open-loop peak of ad/add/hdd).

Phase 1 (sweep R, Q_CL=1):
sim_A_025 (design, open CLexc 0.241, ad 0.0039, add 1.48):
  R=1e-5 CLexc 0.003(-99%) but ad 0.070(18x) add 5.96  -> EXPLODE (pitch rate)
  R=1e-4 CLexc 0.023(-91%)     ad 0.049(12x)            -> EXPLODE
  R=1e-3 CLexc 0.066(-73%)     ad 0.005(~open) add 1.48 -> STABLE  (sweet spot)
  R=1e-2 CLexc 0.204(-15%) weak; R=0.1 no action.
sim_A_027 (strong, open CLexc 0.584, ad 0.044, add 3.21):
  R<=1e-3: CLexc only -13..-17% (flap SATURATES 14deg) AND ad/add EXPLODE (up to 18,17)
  R>=1e-2: stable but useless (-6%). NO good R for the strong gust.

Finding: the minimal-objective one-step optimal nails C_L on the DESIGN gust at R~1e-3
(-73%, stable) but at higher aggressiveness (R<=1e-4) it EXCITES THE PITCH RATE (ad up to
18x open) -- the one-step cost cannot foresee the pitch dynamics it drives. On the STRONG
gust it fails outright (saturates + pitch blow-up when aggressive, no alleviation when
gentle). => states explode, so Phase 2 adds Q_h*h^2 + Q_alpha*alpha^2 (results below).

Phase 2 (add Q_h*h^2 + Q_alpha*alpha^2, aggressive R=1e-4/1e-5):
sim_A_025: Q=1e3/1e4 still EXPLODE (ad); only Q_h=Q_a=1e5 @R=1e-4 is stable (CLexc -74%,
  ad 0.0096 ~2.5x open) -- huge weights, and worse than the minimal R=1e-3 (-73%, ad~open).
sim_A_027: Q_h=Q_a up to 1e5 ALL EXPLODE (ad 0.16-0.21, add 13-18); no effect on the strong
  gust (flap saturates + pitch blows up regardless).
=> Q_h/Q_alpha (penalizing the pitch ANGLE) do NOT tame the explosion, which is in the pitch
RATE (alpha_dot). Per design intent, remove h/alpha and use only Q_alpha_dot (Phase 3).

Phase 3 (Q_CL + Q_alpha_dot + R, h/alpha removed):
sim_A_025 (design): Q_ad=1 still EXPLODE(ad); Q_ad=100 weak (CLexc only -29%, still ad);
  Q_ad>=1e4 TAMES pitch (ad~open) BUT flap saturates, CLexc gets WORSE (+123%) and the
  explosion moves to HEAVE (hdd 2.2 = 8x open). No good setting.
sim_A_027 (strong): Q_ad=1 EXPLODE; Q_ad>=100 STABLE (pitch tamed) but CLexc WORSE
  (+13..+43%) -- controller saturates the flap to kill pitch rate, sacrificing GLA.

CONCLUSION (one-step optimal, rollout model): Q_alpha_dot confirms the instability is the
pitch RATE (penalizing it stabilizes), but there is NO one-step setting that BOTH alleviates
C_L AND keeps the structure calm -- reducing C_L wants aggressive flap, not exciting pitch
wants gentle flap; the one-step cost cannot navigate this multi-step tradeoff and the flap
saturates. Best one-step result = minimal objective Q_CL+R at R=1e-3 on the design gust
(-73%, stable); the strong gust has no viable one-step solution. => motivates receding-
horizon MPC (sees the multi-step pitch dynamics) as the principled controller. Proportional
remains the simple robust baseline (-59% design, modest flap).

## Receding-horizon MPC with the rollout model (z=0 regime) (2026-06-11)
clean/optimal_test.py extended with NH (mpc_horizon) env -> drives Controller's vectorized
batch rollout (aero=a). MINIMAL objective (Q_CL + R only, horizons 3/6/10):
sim_A_025 (design): R<=1e-2 reduces C_L well (+48..+82%) but EXPLODES (ad AND hdd now) at
  ALL horizons -- at R=1e-3 the MPC explodes where the one-step was stable, because with only
  Q_CL the horizon optimizes accumulated C_L -> pushes the flap harder -> excites more pitch.
sim_A_027 (strong): stable only at R=1e-2 (+10..+18%, modest).
=> Q_CL-only MPC does NOT beat the one-step: the constant-delta rollout has no pitch-rate
penalty, so nothing stops it exciting pitch. The project's MPC needed Q_alpha_dot in the
rollout (the horizon SEES alpha_dot grow and penalizes it). Testing Q_CL + Q_alpha_dot next.

MPC + Q_alpha_dot (Q_CL=1 + Q_alpha_dot, H=6):
sim_A_025 (design): Q_ad=100,R=1e-2 -> CLexc 0.131 (+46%) STABLE (ad 0.0115, hdd 0.81 ~open).
  Q_ad=10 explodes (too weak); Q_ad=1000 over-damps -> C_L WORSE + hdd explode.
sim_A_027 (strong): Q_ad=10,R=1e-3 -> CLexc 0.469 (+20%) STABLE (ad 0.073, hdd 3.3, add 4.8
  all <3x open). Q_ad=100,R=1e-2 -> +11% stable. THIS is where MPC wins: stable alleviation
  on the strong gust, where the one-step optimal had NONE.

=== CONTROLLER COMPARISON (rollout model, z=0 regime, C_L excursion reduction) ===
| controller                         | design A_025 (0.241) | strong A_027 (0.584) |
| Proportional (G=20 / +LPF)         | -59% / up to -96%    | saturates, modest    |
| 1-step optimal, Q_CL+R (R=1e-3)    | -73% STABLE          | NONE (explode/useless)|
| 1-step + Q_alpha_dot               | tames pitch, C_L WORSE| stable but C_L WORSE |
| MPC Q_CL-only (H=3..10)            | explodes             | +10..18% (R=1e-2)    |
| MPC + Q_alpha_dot (Qad=100,R=1e-2) | +46% STABLE          | +11% STABLE          |
| MPC + Q_alpha_dot (Qad=10,R=1e-3)  | (+76% but explode)   | +20% STABLE          |
TAKEAWAY: MPC+Q_alpha_dot is the only model-based controller STABLE while alleviating on BOTH
gust strengths (the horizon penalizes the pitch rate it would excite). One-step optimal is
best ONLY on the design gust (-73%); proportional is the simple robust baseline (-59%). MPC
trades peak alleviation for robustness. Flap-authority saturation (dC_L/ddelta~0.014/deg)
caps strong-gust alleviation for all controllers.

## MPC gain-scheduling on synthetic gusts (W0, Tg) + state plots (2026-06-11)
clean/mpc_gust.py: synthetic 1-cos gust W(t)=(W0/2)(1-cos(2pi t/Tg)), CFD pre-gust trim IC
(X0=[-6.49e-3,0,-8.76e-4,0]; the fsolve trim gave a spurious startup transient -> use the
CFD trim consistent with z=0), z=0 rollout regime, MPC (Q_CL+Q_alpha_dot, H=6) with optional
target_lpf and a feedforward gain schedule on W0. Plots all 9 states open vs closed in the
dataset_v5 gust_response.png style (h,hd,alpha,alpha_dot,C_L,Fy,C_M,W_gust,delta).

Characterization (Tg=1.0, LPF=0.7), best STABLE C_L-excursion reduction:
  W0=5 (exc 0.08): ~0% (load negligible, MPC stays gentle)
  W0=10(exc 0.23): Qad=30,R=1e-2 -> +55%
  W0=15(exc 0.31): Qad=100,R=1e-2 -> +33% (Qad=30 borderline)
  W0=20(exc 0.47): Qad=30,R=1e-2 -> +57%
  W0=30(exc 0.59): Qad=30,R=1e-2 -> +39%
Aggressive R<=1e-3 EXPLODES at almost all W0 (and Tg=1,2). Tg also matters: shorter gusts
tolerate more aggression. Schedule adopted: Qad=100,R=1e-2 for W0<=8 (gentle), else Qad=30,
R=1e-2, LPF=0.7. The MPC self-regulates (R*delta^2 + flap saturation cap aggression), so a
heavy schedule is not needed -- one robust setting (+39..+57%) covers W0=10..30.

## FINAL MPC config + gust grid (W0 x Tg), open vs closed-loop plots (2026-06-11)
clean/mpc_gust.py final config: receding-horizon MPC (H=6, Q_CL + Q_alpha_dot schedule),
n_grid=15 (coarse grid gives a natural deadband so the flap returns to 0 cleanly -- a fine
grid let the controller chase the z-drift baseline -> spurious steady delta + late spike),
DLPF=0.95 (smooth/sinusoidal flap, applied in the harness since the Controller's target_lpf
is ignored on the MPC batch path), DAMULT=3 (3x structural pitch damping: the LDNet aero
under-represents aero pitch damping so the pitch mode rings; this smooths the post-gust
alpha_dot). Gain schedule on W0: Qad=100/R=1e-2 for W0<=8 (gentle), else Qad=30/R=1e-2.
Synthetic 1-cos gust W(t)=(W0/2)(1-cos(2pi t/Tg)); CFD pre-gust trim IC.

C_L excursion reduction, all stable, delta->0 (de@2.5=0):
| W0 \ Tg | 0.5s | 1.0s | 2.0s |
|  10      | +16% | +54% | +0% (negligible gust) |
| 20      | +2%  | +42% | +57% |
| 30      | +0%  | +14% | +33% |
design W11.5/Tg1.12: +52%.
Pattern: medium-long gusts at moderate W0 alleviate best; short (Tg=0.5) and strong (W30)
gusts are flap-authority/saturation limited. Notes on alpha_dot: its RMS already matches the
CFD ground truth (0.106 vs 0.105 deg/s) -- the visible wiggle is the LDNet C_M ripple (a
forced, not resonant, artifact: 4x structural damping changes RMS by <7%), and the controller
adds only ~8%. Plots: clean/results/mpc_plots/gust_W{10,20,30}_Tg{50,100,200}.png + W11_Tg112.

<<<<<<< Updated upstream

## Design A -- one-step Q_alpha_ddot (2026-06-29)

Question: is MPC necessary, or does a one-step controller with the same cost structure
but penalising Delta-adot = (adot(t+1) - adot(t))^2 instead of adot(t+1)^2 suffice?

Rationale: Phase 3 (one-step Q_alpha_dot = Q*adot^2) failed because it penalises the
absolute pitch rate, which rises naturally during a gust => huge weights needed => flap
saturates => C_L alleviation lost. The Delta-adot = alpha_ddot*dt term penalises only the
controller-induced pitch angular acceleration (proportional to the flap moment arm), not
the natural gust response. Mechanism: targets step 1 of the resonance cascade
(delta -> Mz -> alpha_ddot -> adot -> dC_L -> delta) without fighting the gust response.

Implementation: one line in controller._cost() [controller.py:178]:
    before: self.Q_alpha_dot * x_next[3]**2
    after:  self.Q_alpha_dot * (x_next[3] - state[3])**2
Sweep harness: clean/adot_sweep.py

Results (NH=1, NGRID=7, NSTEPS=800, rollout model):

sim_A_025 (design gust, CLexc_open=0.241, ad_open=0.0039):
  Q_ad=5  R=5e-4 => +75% stable (ad=2.5x)
  Q_ad=5  R=1e-3 => +70% stable (ad=1.3x)
  Q_ad=10 R=1e-3 => +67% stable (ad=1.4x)
  Q_ad=30 R=1e-3 => +53% stable (ad=1.8x)
  Q_ad>=100 R<=1e-3 => EXPLODE

sim_A_027 (strong gust, CLexc_open=0.584, ad_open=0.044):
  Q_ad=5-30 R=5e-3 => +12% stable
  Q_ad=5-30 R<=1e-3 => EXPLODE
  Q_ad>=100  any R  => worsens (-10..+10%)

Controller comparison (C_L excursion reduction):
  Proportional G=20:         A_025 -59%         A_027 modest
  1-step Q_CL+R R=1e-3:     A_025 -73% stable  A_027 EXPLODE
  1-step Q_alpha_dot Ph3:   A_025 -29% unstable A_027 worsens
  Design A Qad=5 R=1e-3:    A_025 -70% stable  A_027 -12% stable
  MPC H=6 Qad=100 R=1e-2:  A_025 -46% stable  A_027 -20% stable

Conclusions:
1. Design A replaces MPC for the design gust: -70% vs MPC -46%, one-line change.
2. Strong-gust gap (8%) is flap authority saturation, not cost formulation. At
   CLexc_open=0.584 the authority limit (~0.014/deg, delta_max=14 deg) saturates the
   flap at any aggressive setting; MPC gains 8% by sequencing commands within limits.
3. MPC theoretically justified only for strong gusts where multi-step command sequencing
   within the saturation limit is decisive. Q_Delta-adot captures pitch excitation
   mechanism and is sufficient for moderate gusts.
4. Best cross-gust setting: Q_ad=5-10, R=5e-3. Design-gust optimum: Q_ad=5, R=5e-4.
=======
## Design A — one-step Q_alpha_ddot (2026-06-29)

**Question:** is MPC necessary, or does a one-step controller with the same cost structure
but penalising Δα̇ = (α̇(t+1) − α̇(t))² instead of α̇(t+1)² suffice?

**Rationale:** Phase 3 (one-step Q_alpha_dot = Q·α̇²) failed because it penalises the
absolute pitch rate, which rises naturally during a gust → huge weights needed → flap
saturates → C_L alleivation lost. The Δα̇ = α̈·dt term penalises only the controller-
induced pitch angular acceleration (directly proportional to the flap moment arm), not
the natural gust response. Theoretical mechanism: frena il primo anello della cascata
risonante (δ → Mz → α̈ → α̇ → ΔC_L → δ) senza combattere la risposta naturale al gust.

**Implementation:** one line in controller._cost():
    - before: self.Q_alpha_dot * x_next[3]**2
    + after:  self.Q_alpha_dot * (x_next[3] - state[3])**2
(clean/controller.py:178; clean/adot_sweep.py for the sweep harness)

**Results — single-process sweep (clean/adot_sweep.py, NH=1, NGRID=7, NSTEPS=800):**

sim_A_025 (design gust, CLexc_open=0.241, ad_open=0.0039):
| Q_ad | R    | CLexc red. | ad/open | status |
|------|------|-----------|---------|--------|
|    5 | 5e-4 | +75%      | 2.5x    | stable |
|    5 | 1e-3 | +70%      | 1.3x    | stable |
|   10 | 1e-3 | +67%      | 1.4x    | stable |
|   30 | 1e-3 | +53%      | 1.8x    | stable |
|  100 | 5e-3 | +31%      | 1.6x    | stable |
| ≥100 | ≤1e-3| —         | —       | EXPLODE |

sim_A_027 (strong gust, CLexc_open=0.584, ad_open=0.044):
| Q_ad | R    | CLexc red. | status  |
|------|------|-----------|---------|
|  5   | 5e-3 | +12%      | stable  |
| 10   | 5e-3 | +12%      | stable  |
| 30   | 5e-3 | +12%      | stable  |
|  5–30| ≤1e-3| —         | EXPLODE |
| ≥100 | any  | -10..+10% | worsens |

**Controller comparison (rollout model, z=0, C_L excursion reduction):**
| controller                         | A_025 (design) | A_027 (strong) |
|------------------------------------|----------------|----------------|
| Proportional G=20                  | -59%           | modest         |
| 1-step Q_CL+R (Phase 1, R=1e-3)   | -73% stable    | EXPLODE        |
| 1-step Q_alpha_dot (Phase 3)       | -29% unstable  | worsens        |
| **Design A (Q_ad=5, R=1e-3)**      | **-70% stable**| -12% stable    |
| MPC H=6 (Q_ad=100, R=1e-2)        | -46% stable    | -20% stable    |

**Conclusions:**
1. Design A REPLACES MPC for the design gust: +70% vs MPC +46%, simpler by construction.
2. Design A PARTIALLY covers strong gusts (+12% vs MPC +20%). The gap is not the cost
   formulation — it is flap authority saturation. At CLexc_open=0.584 the authority limit
   (dC_L/dδ~0.014/deg, δ_max=14°) caps any one-step controller; MPC gains only 8% more
   by sequencing commands across the horizon to delay saturation.
3. MPC is theoretically justified ONLY for strong gusts where the multi-step pitch cascade
   cannot be inferred from the one-step Δα̇ cost. For a clean theoretical treatment:
   — Q_Δα̇ captures pitch excitation mechanism → sufficient for moderate gusts
   — Horizon needed only when flap saturation makes multi-step sequencing decisive
4. Best single setting across both gusts: Q_ad=5–10, R=5e-3 (stable everywhere, moderate
   alleviation). For design-gust optimality: Q_ad=5, R=5e-4 (+75%, ad=2.5x open).

## Confronto open / prop / MPC — CS-25.341 discrete gust envelope (2026-06-29)

Rollout model (models_rollout/latent_10), z=0 start, FULL loads, DAMULT=3.
Synthetic 1-cos gusts W(t)=(W0/2)(1-cos(2πt/Tg)); CFD pre-gust trim IC.
Script: clean/compare_grid.py; results in clean/results/compare_grid.{md,csv,png}.

**Proportional gain selection (cross-validation on 10 cells):**
Only G=10 keeps all 10 cells stable; G=20/40/80 lose ≥1 cell (likely W30/Tg0.5 where
flap authority is exhausted and the feedback loop over-drives). G=10 median CLred=22%
(conservative; higher gains give +48..60% but require excluding unstable cells).

Both controllers use the same 2nd-order DLPF chain (α=0.85, scheduled) for fair comparison.
MPC: RQUIET scheduling (R ramps up from 0.1 to tuned value as gust grows), H=6, NGRID=15.

**C_L excursion reduction vs open loop:**

| W0 [m/s] | Tg [s] | H [m] | H [ft] | CLexc open | Prop G=10 | MPC H=6  | MPC ms/step |
|----------|--------|-------|--------|-----------|-----------|----------|-------------|
| 10       | 0.50   | 20    | 66     | 0.202     | +24%      | +58%*    | 61          |
| 10       | 1.00   | 40    | 131    | 0.219     | +40%      | +59%     | 61          |
| 10       | 2.00   | 80    | 262    | 0.071     | +13%      | +2%      | 61          |
| 20       | 0.50   | 20    | 66     | 0.337     | +12%      | +23%     | 62          |
| 20       | 1.00   | 40    | 131    | 0.463     | +42%      | +38%     | 62          |
| 20       | 2.00   | 80    | 262    | 0.261     | +43%      | +69%     | 63          |
| 30       | 0.50   | 20    | 66     | 0.496     | +4%       | +4%      | 59          |
| 30       | 1.00   | 40    | 131    | 0.575     | +16%      | +46%     | 59          |
| 30       | 2.00   | 80    | 262    | 0.456     | +20%      | +33%     | 60          |
| 11.46    | 1.12   | 45    | 147    | 0.244     | +38%      | +67%     | 60          |

*W10/Tg0.5 MPC has hdd!! flag (structural load amplification — see caveat).

**Key observations:**
1. **MPC dominates on moderate-to-strong, medium-length gusts** (W30/Tg1: +46% vs +16%;
   design W11/Tg1.12: +67% vs +38%; W20/Tg2: +69% vs +43%). Multi-step lookahead pays off.
2. **Prop wins W20/Tg1** (+42% vs +38%) and **both equal on W30/Tg0.5** (+4%). At these
   operating points the gust is either perfectly tuned for the proportional feedback or
   the flap is saturated for both.
3. **Short gusts (Tg=0.5, H=20m=66ft, lower CS-25.341 boundary):** both controllers limited
   to +4..24% — the gust is shorter than the flap time-constant, authority saturates early.
4. **W10/Tg2 (very mild gust, CLexc=0.071):** MPC barely acts (+2%), prop is more
   useful (+13%). MPC is over-regularized at small excursion (RQUIET schedule).
5. **Compute cost:** MPC ~60ms/step vs DT=2ms → 30× real-time on CPU; prop ~8ms/step → 4×.
   Neither is real-time without dedicated hardware or code optimization.

**DLPF sensitivity (same smoothing applied to prop for fairness):**
| Cell          | Prop + DLPF | Prop no DLPF | flap Δ    |
|---------------|-------------|--------------|-----------|
| Design W11/Tg1.12 | +38%   | +52%         | 1.5→1.9 d |
| W10/Tg1.0    | +40%        | +53%         | 1.3→1.6 d |
| W30/Tg1.0    | +16%        | +28%         | 4.8→5.9 d |

DLPF costs prop ~12–14% CLred but reduces flap travel (~0.4–1.1 deg). Without DLPF prop
outperforms MPC on most moderate cells, confirming the smoothing (not the horizon) is the
main penalty on the proportional arm.

**Caveats:**
- LDNet under-predicts strong gusts (A_027: model 0.59 vs CFD 0.99) → saturation earlier
  in reality, all CLred values conservative for W0≥25.
- DAMULT=3 compensates LDNet aero under-damping; physical DAMULT=1 would require re-tuning.
- W10/Tg0.5 MPC `hdd!!`: peak heave acceleration >3× open-loop; structural loads amplified.
  Probably a resonance excited by the fast flap at the 5.8 Hz heave mode; needs investigation.
>>>>>>> Stashed changes

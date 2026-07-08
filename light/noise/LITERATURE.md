# Literature survey — handling NOISE in gust/wind preview for load alleviation

Compiled 2026-07-07 (web survey, primary sources fetched where marked).
Context: the light/noise/ study found the wnext one-step optimal controller
(+76..80% CLred clean) collapses at sigma ~ 1%*W0 white preview noise
(branch lottery), break-even vs prop-clean +32% at ~0.7%*W0; failure modes
F1 = causal-gate chattering at W~0 (post-gust flap wind-up), F2 = bias/latency
kill, F3 = needs phase not amplitude. This file records how the published
field handles the same problem, and maps each technique onto our architecture.

**Verification key:** [V] = read from fetched full text; [S] = search snippet /
abstract only; [I] = our inference from verified numbers.

## Big-picture finding

No published system feeds a raw pointwise wind-preview sample to a controller
the way our one-step controller consumes Wc(t+dt). In every mature toolchain
(DLR aircraft GLA, Stuttgart/NREL wind turbines) there are TWO mandatory layers
between sensor and control decision:

1. a **redundancy-exploiting estimator** that fuses hundreds-to-thousands of
   noisy raw measurements of the same advecting wind profile into a smooth
   estimate, and
2. a **frequency-domain trust boundary** (filter + command shaping) so the
   controller physically cannot act on spectral content the sensor chain
   cannot deliver.

Raw airborne lidar noise is ~1-3 m/s per line-of-sight measurement [V] —
3-10x worse than the 0.3 m/s that breaks us — yet these systems work, because
the controller never sees that noise. Our verdict "realistic Doppler noise >>
tolerance" compares RAW sensor noise against a controller designed for clean
input; the literature's comparison point is AFTER the reconstruction layer,
where residual error is ~0.05-0.3 m/s [V, Cavaliere Table 2; I for the
interpretation] — i.e. AT our measured tolerance boundary, not 3-10x beyond.

## Technique 1 — Massed-measurement Tikhonov-regularized wind reconstruction (DLR pipeline)

**Refs:** Cavaliere, Fezans, Kiehn, "Method to Account for Estimator-Induced
Previewed Information Losses...", CEAS EuroGNC 2022, CEAS-GNC-2022-063
(full text [V]) — https://eurognc.ceas.org/archive/EuroGNC2022/pdf/CEAS-GNC-2022-063.pdf ;
Fezans, Schwithal, Fischenberg, CEAS Aeronaut. J. 8(2), 2017 (paywalled; role
verified via citations in [V] sources) ; Kiehn, Fezans, Vrancken, Deiler,
IFASD-2022-105 — https://elib.dlr.de/187627/ ; Kiehn, Schultz, Fezans, Römer,
"Adaptive Wind Field Estimation Using an Empirical Bayesian Approach",
JGCD 47(11), 2024 [S] — https://elib.dlr.de/207352/

**Mechanism [V]:** lidar fires at PRF = 500 Hz with 9 range gates (60-180 m
ahead); each line-of-sight measurement is
`u_meas = V_TAS*cos(a_apert) + w_turb*sin(a_apert)*sin(phi_scan) + nu`.
All measurements falling in a spatial window (-78 m behind to +143 m ahead)
are kept in a database with per-measurement expected noise sigma_i, and a
vertical-wind profile on 33 spatial nodes is re-fit at 10 Hz by weighted least
squares (weights 1/sigma_i^2) with Tikhonov penalties on the first and second
SPATIAL derivatives of the profile (Gauss-Newton). Because the aircraft flies
into the frozen field, every part of the gust is re-measured hundreds of times
as it approaches, and estimates of a given spatial point improve as it nears.

**Noise level [V]:** zero-mean Gaussian per LOS shot, sigma_i prop. to
R_i/(sqrt(dR)*sqrt(PAP)); numerically sigma_i/R_i = 0.0278 1/s -> ~1.7 m/s at
60 m, ~5 m/s at 180 m. Bias assumed pre-calibrated out. AWIATOR flight
hardware: sigma_LOS ~ 1.0-1.5 m/s at 50 m range, 60 Hz update (Rabadan et al.,
J. Aircraft 47(2), 2010) [S].

**What it buys [V/I]:** ~4500 raw measurements/s compressed into a smooth
33-node profile; residual filter-vs-estimator mismatch 0.02-0.27 m/s (Table 2).
Without regularization the estimate is "ultimately very sensitive to noise".

**Portability: (a) pure preprocessing wrapper.** Give the simulated sensor N
noisy re-measurements per step of the advecting profile (not one sample of
W(t+dt)), keep a rolling buffer, fuse per spatial point (inverse-variance) +
optional smoothness penalty. Unlike our centred time-window averaging, this
averages estimates OF THE SAME SPATIAL POINT -> no phase lag [I].
**-> implemented in e2_sensor.py**

## Technique 2 — Model the estimator's information loss inside the controller design

**Refs:** Cavaliere, Fezans, Kiehn, EuroGNC 2022 (full text [V], same PDF);
applied in Wallace & Fezans et al., "Lidar-based Gust Load Alleviation —
Results Obtained on a Generic Long Range Aircraft Configuration", EUCASS 2023
(full text [V]) — https://elib.dlr.de/196046/1/C_Wallace_Paper_AEC_EUCASS_2023_Final_Orga.pdf

**Mechanism [V]:** the whole lidar + regularized-estimation chain is
approximated as a single SIMO LTI filter K_WFE(z) (closed form, no Monte
Carlo) and inserted into the discrete-time H-inf preview synthesis plant. The
controller is optimized against the smoothed preview it will actually receive,
not the true wind.

**What it buys [V]:** "a significant improvement in load alleviation
performance" vs designing on clean preview then discovering the mismatch;
drastically easier tuning.

**Portability: (b)->(c).** For us: re-tune R / gate thresholds WITH the chosen
preview filter in the loop, instead of tuning clean and bolting the filter on.
Our "LPF helps +20% then lag kills" is exactly the mismatch this removes.
**-> follow-up campaign once the best E2 filter is known.**

## Technique 3 — Band-pass (washout + low-pass) filtering of the COMMAND

**Ref:** Wallace & Fezans et al., EUCASS 2023 (full text [V]).

**Mechanism [V]:** the preview FF controller (FIR gain matrix over an 83-sample
wind vector) is in series with fixed band-pass filters on each control-surface
command: high-pass cutting below 0.05 Hz, low-pass cutting above 5-7 Hz. Quote:
cutting low frequencies "ensures that no constant or quasi-constant deflections
can be commanded"; cutting high frequencies "prevents from propagating
measurement noise to the control surfaces, especially at frequencies (i.e.
gust scales) that are not well measured by the lidar."

**What it buys [V]:** part of a certifiable-load-alleviation pipeline on an
Airbus-class model across the full gust-length envelope.

**Portability: (b) small change — targets our worst failure directly.** Our
dominant failure is POST-gust flap wind-up (a quasi-constant erroneous
deflection); a washout on the flap command decays any DC flap regardless of
what the noisy preview says, without penalizing fast transient action (unlike
R_du, which punished exactly the fast action we need).
**-> implemented in e2_gate.py (washout arms)**

## Technique 4 — Noise-weighted robust preview synthesis; optimal preview length

**Ref:** Fournier, Massioni, Pham, Bako, Vernay, Colombo, "Robust Gust Load
Alleviation of Flexible Aircraft Equipped with Lidar", JGCD 45(1), 2022 (full
text via HAL [V]) — https://hal.science/hal-03282401/file/Article_GLA_LIDAR_7.0.pdf

**Mechanism [V]:** preview modeled as a delay chain z^-i; each previewed
sample carries additive white noise with sigma_i = k_n*(i*dx) — noise grows
linearly with look-ahead distance, k_n = 0.02 1/s (~1.8 m/s at 91 m [I]).
These weighted noise inputs are exogenous channels in H-inf/mu synthesis, so
the optimizer automatically discounts far preview samples. Robustness via MIMO
disk margins (D-K iteration).

**What it buys [V]:** wing-root bending moment -57% with lidar vs -22% without
(robust design), -72% without robustness constraints. A finite optimal lidar
distance (~300 ft) emerges: beyond it, added preview is too noisy to help.
Long gusts degrade more slowly than short ones because many noisy samples
cover them and "the controller will more easily filter the noise out."

**Portability: (c)** — a synthesis philosophy. Transferable design rule:
derate trust in preview in proportion to its noise level [I].

## Technique 5 — Coherence-based optimal preview filter with explicit re-timing (Schlipf/Stuttgart-NREL)

**Refs:** Simley & Pao, ACC 2013 [S] — https://ieeexplore.ieee.org/document/6579906/ ;
Guo, Schlipf, Chen, WESC 2021, Zenodo [V summary] — https://zenodo.org/records/4985412 ;
Guo, Schlipf et al., Wind Energy Science 8, 1893-1907, 2023 (full text [V]) —
https://wes.copernicus.org/articles/8/1893/2023/ ; Schlipf et al., WES 8, 149,
2023 (full text [V]) — https://wes.copernicus.org/articles/8/149/2023/

**Mechanism [V]:** compute the magnitude-squared coherence gamma^2(f) between
the lidar wind estimate and the rotor-effective wind. The MMSE use of the
noisy preview is the ideal feedforward in series with a **Wiener prefilter**
whose gain tracks the coherence — trust only frequencies where the sensor
demonstrably correlates with what will hit the plant. In practice a 1st-order
LPF with f_c at the -3 dB point of the modeled lidar-to-rotor transfer
function. Crucially the filter's lag is NOT tolerated but REPAID from the
preview budget: `T_buffer = T_lead - T_scan/2 - T_filter - T_pitch >= 0` —
configurations where the filter delay eats more than the available preview
are rejected outright.

**Noise model [V]:** not additive white noise but coherence loss (probe-volume
averaging, wind evolution), modeled via coherence decay constants.

**What it buys [V]:** rotor-speed std -47.4%, tower-base DEL -4.3% (spinner
lidar, 15 MW turbine). Warning result [V]: with a SUBOPTIMAL filter under
strong wind evolution, feedforward makes tower loads WORSE than feedback-only.
A fixed f_c is adequate across atmospheric stability classes [V].

**Portability: (a) with one hard constraint** — filtering is only viable if
preview length exceeds filter delay. Our tau=10ms LPF on a 2 ms preview is
exactly the ruled-out combination (explains axis B/E). The literature answer:
lengthen the preview buffer and spend the surplus on filtering.
**-> the re-timing constraint shapes e2_mpc.py and the follow-up tuning.**

## Technique 6 — Disturbance-state estimation with an internal gust model (extended-state KF)

**Ref:** Forte, Nguyen, Xiong, "Gust Load Alleviation Control and Gust
Estimation for a High Aspect Ratio Wing Wind Tunnel Model", AIAA SciTech 2023,
NASA (full text [V]) — https://ntrs.nasa.gov/api/citations/20220018624/downloads/AIAA_SciTech_2023_CRM_GLA_final.pdf

**Mechanism [V]:** the gust is not consumed as a measured signal; it is a
STATE of an assumed disturbance model (sinusoidal oscillator there; for
stochastic gusts a Dryden/von Karman shaping filter) appended to the plant.
An extended-state Kalman filter estimates plant + gust states from onboard
sensors plus an upstream probe; LQG acts on the estimate. When the assumed
gust frequency is wrong, an RLS layer re-estimates it online (its input
pre-low-pass-filtered because the 2nd-derivative computation "is sensitive
to noise").

**Noise level [V]:** sensor noise covariance = (5% of max reading)^2; process
noise = (5% of max state response)^2.

**What it buys [V]:** wing-root strain -69.07% clean vs -68.45/-68.51% with
sensor noise — essentially FREE graceful degradation, because the estimator's
internal model prevents noise from commanding physically implausible gust
trajectories. A 2.5% frequency mismatch collapsed the non-adaptive controller
from ~69% to 39%; adaptive estimation restored 71%.

**Portability: (c), with a cheap (a)-grade variant:** model W as the output of
a 1-2 state shaping filter, run a scalar KF with the noisy preview as
measurement. Rejects dynamically-impossible W-jumps (the post-gust sign
flips), causal, phase-aware, bias-tracking — unlike a centred moving average [I].
**-> implemented in e2_kalman.py**

## Technique 7 — Preview-horizon integration (MPC) as intrinsic noise dilution

**Refs:** Mendez, Whidborne, Chen, "Wind Preview-Based Model Predictive
Control of Multi-Rotor UAVs Using LiDAR", Sensors 23(7):3711, 2023 (full text
[V]) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10098597/ ; Sinner, Petrovic,
Stockhouse, Langidis, Pusch, Kühn, Pao, "Insensitivity to propagation timing
in a preview-enabled wind turbine control experiment", Frontiers in Mech.
Eng. 9, 2023 (full text [V]) — https://frontiersin.org/articles/10.3389/fmech.2023.1145305/full ;
Mirzaei & Soltani, ACC 2013 [S].

**Mechanism [V]:** certainty-equivalent MPC consumes the preview over an
N-step horizon inside an integrated cost: uncorrelated per-sample noise is
averaged down by the horizon itself; timing error shifts the whole predicted
profile, which the receding horizon partially re-plans away. Sinner et al.
filter the preview with a moving average chosen for its FREQUENCY-INDEPENDENT
group delay (constant lag -> compensable by a single time shift; an IIR
low-pass's frequency-dependent lag is not).

**What it buys [V]:** Mendez: preview-MPC stays better than no-preview-MPC up
to ~120% magnitude error or ~1.75 s timing error. Sinner: FF benefit retained
across +-20% timing error (wind-tunnel experiment).

**Portability: (b/c).** Extending our one-step argmin to an N-step scan stays
within the surrogate's validated short-horizon regime (batch_step OK for short
MPC horizons). The contrast is stark: horizon-integrating controllers tolerate
order-100% preview errors; our single-sample argmin dies at 1%. The DECISION
STRUCTURE, not the model, sets the noise tolerance [I].
**-> implemented in e2_mpc.py**

## Technique 8 — Hysteresis, deadband, dwell time on switching logic

**Refs:** Hespanha, Liberzon, Morse, "Hysteresis-based switching algorithms
for supervisory control of uncertain systems", Automatica 39:263-272, 2003 —
https://web.ece.ucsb.edu/~hespanha/published/journal-hhs-final.pdf [S:
canonical result]; Schlipf et al., WES 8, 149, 2023 (full text [V]) for the
applied example.

**Mechanism:** a switching decision driven by a noisy signal near its
threshold chatters; standard cures: (i) hysteresis: switch on at threshold+eps,
off at threshold-eps; (ii) dwell time: once switched, hold for a minimum time;
(iii) deadband: inside a band around zero, command nothing. Hespanha et al.
prove hysteresis switching yields bounded switch counts under bounded noise
[S]. Applied instance [V]: Schlipf's feedforward is only ACTIVATED when
rotor-effective wind speed > 14 m/s — a state-conditioned enable keeping the
FF channel silent exactly where its action could be counterproductive.

**Portability: (a/b) — cheapest on the list.** Our causal sign gate is a
textbook chattering switch: near trim (post-gust, W~0) noise flips
sign(C_L,pred - trim) every step. Deadband (-> flap decays to zero),
hysteresis margin, and dwell are a few lines in the gate logic and none touch
the clean-preview optimum (the gates only bind near zero, where the optimal
flap is ~zero anyway) [I].
**-> implemented in e2_gate.py (gate arms)**

## Technique 9 — Adaptive feedforward (LMS) / fusion with onboard sensing

**Refs:** Wildschek et al., AIAA GNC 2009; MIMO adaptive FF, J. Sound &
Vibration 2014 [S] — https://www.sciencedirect.com/science/article/abs/pii/S0022460X14002867 ;
alpha-probe CLLMS adaptive FF, Aerospace 10(12):981, 2023 [S] —
https://www.mdpi.com/2226-4310/10/12/981 ; B-2/787 alpha-vane gust estimation
noted in Khalil & Fezans 2019 (full text [V]) — https://elib.dlr.de/128624/1/AIAA-2019-0822_wCitationInfo.pdf

**Mechanism [S]:** feedforward = FIR filter from the upstream disturbance
sensor to the surfaces, taps adapted online by (leaky) LMS to minimize the
MEASURED downstream response; static gain/bias/phase errors in the sensing
channel are automatically absorbed into the learned taps.

**What it buys:** flight-tested vibration alleviation on a large aircraft
(AWIATOR lineage) [S]; production systems (B-2 GLAS, 787) trust PROCESSED
alpha-vane + inertial estimates, never raw probes [V, per the DLR intro].

**Portability: (c).** Transferable half-idea at level (b): use the onboard
C_L residual to slowly estimate the preview channel's bias/scale online and
correct Wc — a two-parameter recursive regression targeting our fatal
+5%*W0-bias mode [I]. **-> follow-up candidate, not in E2.**

## Ranked shortlist for our failure modes

(F1 gate chattering at W~0; F2 bias/latency; F3 phase-not-amplitude)

1. **Hysteresis + deadband + dwell on the causal gate, decay-to-zero in the
   band** (T8) — hits F1 at its mechanism; ~10 lines; zero clean-case cost.
2. **Washout on the flap command** (T3) — makes the wound-up-flap state
   impossible by construction (F1 damage path, F2-bias consequence) while
   keeping fast authority (what R_du could not do).
3. **Scalar disturbance-model Kalman on Wc** (T6 cheap variant) — causal, no
   centred-window lag; F1+F2+F3; NASA proof of graceful degradation.
4. **N-step preview + integrated cost** (T7, filtered per T5 with delay repaid
   from the preview budget) — the only decision-structure change; the field's
   most consistent finding (~100% tolerance vs our 1%).
5. **Redundant re-measuring sensor + regularized profile re-estimation** (T1)
   + re-tune against the filtered preview (T2) — the honest framing fix: with
   DLR-style reconstruction, 1-3 m/s raw noise becomes ~0.1-0.3 m/s delivered,
   i.e. AT our tolerance boundary. What makes a "realistic Doppler lidar"
   claim defensible in either direction.

Cross-cutting caution [V]: a mis-designed preview filter can make preview
control WORSE than no preview at all (Guo/Schlipf 2021), and every viable
design couples the filter to an explicit timing budget — filtering without
re-timing is the one combination the field has already ruled out, and it is
the one our axis-B/E mitigation tried.

## Mapping to this study (axis E2)

| Script          | Technique | Arms |
|-----------------|-----------|------|
| e2_gate.py      | T8 + T3   | deadband / hysteresis / dwell gate variants; command washout tau_w sweep; gate+washout combo |
| e2_kalman.py    | T6        | scalar constant-velocity KF on the noisy preview, process-noise sweep; delivered-noise + lag reported |
| e2_mpc.py       | T7 (+T5)  | N-step preview MPC (constant flap, wnext convention), N in {2,4,8} |
| e2_sensor.py    | T1        | redundant re-measurement + inverse-variance fusion (+ Tikhonov), lookahead sweep, DLR-realistic raw-noise arm; delivered sigma reported |

Follow-ups deliberately NOT in E2: re-tuning R/gate against the winning filter
(T2), online bias/scale estimation from the C_L residual (T9), cross combos
(best-of-E2 stacking).

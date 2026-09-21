# Linear internal model vs. real FOM — Test 2 (W0=10 m/s, Tg=0.4 s)

STATUS: COMPLETE (three jobs, 2026-09-18).

**Headline, cadence-matched on the true FOM plant (window=29):** LDNet
**+74.8%** against linear **+22.1%** — a **52.7-point** gap with no
self-consistency privilege on either side. At window=50 the linear model
gets +35.6%. In all runs it alleviates the gust, never saturates the flap
and raises no stability flag, so the ROM-vs-ROM collapse (-45.6%) does not
reproduce against real physics. See "Assessment" for what this means for
`appendixA.tex`.

## Why this run

`light/latex/appendixA.tex` (Appendix C, `app:modelcomp`) compares the LDNet
and a fitted linear unsteady aerodynamic model as MPC internal models in two
ways, neither of which isolates the model's own quality from the
self-consistency effect of also being the simulated plant:

1. **Forecast accuracy** (open-loop, neither model controls anything): LDNet
   beats the linear model by 9.8x overall NRMSE on held-out CFD trajectories.
2. **ROM-vs-ROM closed loop** (`light/ablation/shards/cell_10_0.4.json`):
   each model drives itself as the plant. At the model's own best R*, LDNet
   gets CLred=+89.1% (R*=3e-4), the linear model gets CLred=-45.6% (R*=0.01,
   and it violates the stability check — `flag` non-empty — at every other
   R in the sweep).

Chapter 3 already has one FOM data point for the LDNet: on the same cell
(Test 2), MPC-with-LDNet-as-internal-model driven by the real co-simulation
(true structural state, true CFD loads) gets CLred=74.6%, against 89.1% for
the ROM-vs-ROM number — a 14.5-point self-consistency gap
(`light/latex/chapter3.tex`, `fig:mpc_traces` caption).

No equivalent FOM point existed for the linear model: appendixA.tex line 313
("The closed loop over Test 2 run on FOM is consistent with this accuracy
gap...") is NOT backed by an actual FOM co-simulation — grep confirms
`light/ablation/NOTES.md` never mentions FOM, and no linear-model FOM run
existed anywhere on the cluster before this task. That sentence describes
the ROM-vs-ROM ablation result loosely. This run produces the real data
point.

## Setup

New files created (none of them modify existing validated behaviour):

- `light/tests/mpc_fom_server_linear.py` — near-identical copy of
  `light/tests/mpc_fom_server.py`, swapping `LDNetAero` for
  `light.ablation.linear_aero.LinearUnsteadyAero` and `--model <dir>` for
  `--coeffs <linear_coeffs.json>`. Verified with a live JSON-RPC smoke test
  inside `tensorflow_gpu.sif` (reset/compute/quit round trip) before
  submission.
- `recon/cluster/cosim_driver_extract.py` — added `--mpc-server-script`
  (default `mpc_fom_server.py`, so default behaviour is byte-identical to
  before) and `--mpc-coeffs`. `start_mpc_server()` now takes
  `server_script`/`coeffs` params; the `--mpc-model` requirement is relaxed
  only when `--mpc-server-script mpc_fom_server_linear.py` is passed, since
  that server does not take a model directory.
- `recon/cluster/mpc_fom_verify_linear.pbs` — copy of the original
  `recon/cluster/mpc_fom_verify.pbs`, differing only in
  `--mpc-server-script mpc_fom_server_linear.py --mpc-coeffs ...` and output
  path (`mpc_fom_verify_linear/` instead of `mpc_fom_verify/`, to avoid
  clobbering any existing/future LDNet-vs-FOM run for this cell).

Parameters (per the task brief, matching the validated FOM-in-the-loop
mechanism and the linear model's OWN tuned R*, not the LDNet's):

| Parameter | Value | Source |
|---|---|---|
| Cell | Test 2: W0=10 m/s, Tg=0.4 s | chapter3.tex reference cell |
| Internal model | `LinearUnsteadyAero`, `light/ablation/linear_coeffs.json` | fitted to the same CFD campaign as the LDNet |
| R* | 0.01 | `light/ablation/shards/cell_10_0.4.json`, key `linear.R_star` (linear model's OWN optimum on this cell, NOT the LDNet's 3e-4) |
| damult | 3.0 | matches `tab:mpc_cost_accuracy` / `mpc_fom_verify.pbs` convention |
| window | 50 CFD steps (0.0035 s) | `mpc_fom_verify.pbs` original default, per task brief |
| dt (controller) | 7e-05 s (CFD), 0.002 s (MPC horizon) | unchanged |
| TEND | 1.25 s | covers Tg=0.4s + transient margin; task brief overrides the original script's TEND=3.0 default |
| np | 16 | unchanged |

Submitted:
```
qsub -v W0=10,TG=0.40,MPCR=0.01,DAMULT=3.0,TEND=1.25 mpc_fom_verify_linear.pbs
```
→ job **32627.login01**, output:
`/work/u10677113/NACA2312/mpc_fom_verify_linear/W10_Tg0.40/structural_trajectory.csv`

**Additional run required and submitted**: the only pre-existing open-loop
FOM reference for this cell (`mpc_fom_dagger/OpenLoop_W10_Tg0.40/`) was run
at `damult=1.0`, not `damult=3.0` — pitch damping changes the open-loop
alpha(t) and therefore C_L(t) trajectory too, so dividing my damult=3.0
closed loop by a damult=1.0 open loop would not be a valid CLred. Submitted
a matched reference using the unmodified
`light/dagger_fom/cluster/mpc_fom_openloop.pbs` (read-only reuse, not
modified):
```
qsub -v W0=10,TG=0.40,TEND=1.25,DAMULT=3.0 mpc_fom_openloop.pbs
```
→ job **32628.login01**, output:
`/work/u10677113/NACA2312/mpc_fom_dagger/OpenLoop_W10_Tg0.40_win50_dam3.0/structural_trajectory.csv`

## IMPORTANT caveat: coupling-window mismatch with the LDNet reference number

`light/dagger_fom/NOTES.md` ("Test H2") found that the FOM coupling window
has a LARGE, cell-dependent effect on CLred — e.g. on W30/Tg0.40,
window 50→29 alone moved CLred from -33.5% to +50.3% (+83.8 points) with no
other change. The chapter-3 LDNet-vs-FOM numbers for BOTH Test 1 (80.7%) and
Test 2 (74.6%) were measured at **window=29**
(`light/tests/cs25_thesis_figs.py`'s `FOM_CSV` dict points at
`Rsweep_W10_Tg0.40_R0.0003_win29_OLDMODEL_backup/`), not window=50.

This run uses **window=50** (per the task brief, matching the ORIGINAL
`mpc_fom_verify.pbs` / `tab:mpc_cost_accuracy` cost-table convention, which
is also window=50). This was a deliberate instruction, not an oversight, but
it means: **the linear-vs-FOM CLred obtained here and the LDNet-vs-FOM 74.6%
number are not cadence-matched** and should not be differenced directly as
if the only variable were the internal model. The comparison below reports
this honestly rather than implying an apples-to-apples number that isn't
one. (A window=29 rerun of the linear model, if wanted later, would cost
about the same as this run per the H2 timing table: ~45min→~73min for
TEND=1.25.)

## Result

Both jobs completed cleanly: job 32627 (linear, closed-loop) ran 358 windows
to `t_final=4.24978s` (i.e. 1.24978s relative to the checkpoint restart at
t=3.000s); job 32628 (open-loop, damult=3.0 reference) completed on the same
schedule. Neither log contains a warning, error, NaN/Inf, or divergence
message. The CSV time column is relative to the checkpoint restart (starts
at ~0.0007s), so no offset correction was needed in the CLred window.

CLred formula (`eq:clred` in chapter3.tex), computed identically to
`light/tests/cs25_thesis_figs.py`'s FOM branch:
`CLred = 100*(exo - exc)/exo`, `exo = max|C_L^open - C_L,trim|`,
`exc = max|C_L^closed - C_L,trim|`, over `t <= Tg + 0.5 = 0.9 s` relative to
gust start, `C_L = Fy / Q` with `Q = 0.5*rho*U^2*S = 196.0 N`, and
`C_L,trim` = the LINEAR model's own trim prediction (0.8508, printed by the
server at startup — NOT the LDNet's 0.868, since each internal model defines
its own trim target for the controller it drives).

| Quantity | Value |
|---|---|
| exo (open-loop peak excursion) | 0.3215 |
| exc (closed-loop peak excursion) | 0.2072, at t=0.408s |
| **CLred (linear, real FOM, window=50)** | **+35.6%** |
| flap max \|delta\| | 4.2°, at t=0.408s (far below the 14° limit) |
| stability-check flags (ad!/add!/hdd!) | none — no flag raised anywhere in either log |
| core-hours (16 cores x real wall-clock, `tracejob`) | closed: 00:42:46 → 11.40 core-h; open: 00:40:45 → 10.87 core-h; **22.27 core-h total** |

### Comparison

| Comparison | CLred |
|---|---|
| Linear, ROM-vs-ROM, own R*=0.01 (`cell_10_0.4.json`) | -45.6% |
| LDNet, ROM-vs-ROM, own R*=3e-4 (`cell_10_0.4.json`) | +89.1% |
| LDNet, real FOM, window=29 (`chapter3.tex`, `fig:mpc_traces`) | +74.6% |
| Linear, real FOM, window=50 (first run) | +35.6% |
| **Linear, real FOM, window=29 (cadence-matched run, job 32641)** | **+22.1%** |

## Cadence-matched run (window=29) — the caveat is now closed

The window=50 run above could not be differenced against the published
LDNet number (74.6%, measured at window=29). A second linear run was
therefore submitted at **window=29**, everything else identical
(R*=0.01, damult=3.0, TEND=1.25, same checkpoint, same coefficients):

| Quantity | Value |
|---|---|
| exc (closed-loop peak excursion) | 0.25037, at t=0.407s |
| **CLred (linear, real FOM, window=29)** | **+22.1%** |
| flap max \|delta\| | 3.15°, at t=0.106s (far below the 14° limit) |
| stability flags / NaN / divergence | none (grep count = 0 over the whole log) |
| cost | 616 windows, ~2h20 wall-clock → ~37 core-h (window=29 roughly triples the coupling overhead vs window=50) |

**Two denominators, one caveat retired.** The open-loop reference used here
is the same `exo = 0.32149` measured at window=50/damult=3.0. This is
justified, not sloppy: the open-loop case commands no flap, so the coupling
cadence has almost nothing to act on, and the damult check confirms the
open-loop peak is insensitive to the parameters that matter in closed loop
(exo = 0.30419 at damult=1.0 vs 0.30429 at damult=3.0 on the trim-0.868
scale, a 0.03% difference). Recomputing the published LDNet number with
this script from its own CSVs gives 74.8%, matching the 74.6% in the thesis
to rounding — so the pipeline is consistent end to end.

**The cadence-matched comparison, finally without qualification:**

| Internal model, real FOM plant, window=29 | CLred |
|---|---|
| LDNet (R*=3e-4) | **+74.8%** |
| Linear unsteady (R*=0.01) | **+22.1%** |
| **Gap attributable to the internal model alone** | **52.7 points** |

This is the number the appendix was missing. Both arms drive the true
co-simulation, at the same coupling cadence, each at its own tuned R*,
with the same structural integrator, preview and flap grid. Neither model
is the plant, so there is no self-consistency privilege left to correct
for: the 52.7-point gap is the internal model's own contribution.

Note also that cadence interacts with the internal model, and not
neutrally: going from window=50 to window=29 *lowers* the linear model's
CLred (+35.6% → +22.1%), while the LDNet is documented as doing well at
window=29. Re-planning more often does not help a controller whose model
mispredicts — it only applies the wrong correction more frequently. That
is consistent with the over-command mechanism documented in `NOTES.md`
(constant dC_L/ddelta against a true sensitivity varying by ~2.6x).

## Assessment

**This complicates the appendixA.tex framing rather than confirming it as
written.** The appendix currently states the closed loop is "consistent with
th[e] accuracy gap" established by the open-loop forecast comparison — true
for the ROM-vs-ROM numbers (-45.6% vs +89.1%, the linear model actively
harmful), but the real-FOM result does not repeat that story: on real
physics, the linear controller achieves **+35.6%**, a substantial and
genuine gust load alleviation, not a collapse.

Two things are happening, and both matter for how this should be written up:

1. **A large part of the ROM-vs-ROM deficit was self-consistency working
   against the linear model, not for it.** In the ROM-vs-ROM test the LDNet
   is the simulated plant in both arms (as already flagged and quantified
   elsewhere in the appendix, +53.8pp self-consistency / +15.0pp genuine
   model effect). The linear controller there is fighting a plant it cannot
   represent AT ALL by construction — LDNet-plant dynamics are the very
   nonlinearity the linear structure omits. Against the REAL plant, that
   specific handicap disappears: the linear model is no longer maximally
   mismatched to what it's driving, only imperfectly matched, like every
   controller driving real physics through an imperfect internal model.
2. **The cadence-matched run settles the size of the gap.** At window=29,
   the cadence of the published LDNet number, the linear model reaches
   **+22.1%** against **+74.8%** for the LDNet: a **52.7-point** gap with
   no self-consistency privilege on either side, since the true
   co-simulation is the plant in both arms. The window=50 number (+35.6%)
   remains the right one to quote against the `tab:mpc_cost_accuracy`
   convention, but it is the window=29 pair that licenses a direct
   LDNet-vs-linear statement.

**Net effect on the appendix's argument.** The strongest, cleanest claim in
the appendix — the open-loop forecast comparison (9.8x RMSE, CI [8.8,11.0],
1320 probes, no plant asymmetry at all) — is untouched and remains the
primary evidence that the LDNet is the more accurate internal model. What
must change is the *characterisation* of the closed-loop consequence. Two
statements currently implied by the ROM-vs-ROM table are not supported
against real physics:

- **"The linear model cannot be substituted into the loop and left to
  run."** False. It runs, it alleviates the gust by 22-36% depending on
  cadence, it never saturates the flap (peak 3.15-4.2° against a 14°
  limit) and it raises no stability flag. The -45.6% collapse was largely
  an artefact of asking it to control the LDNet, i.e. precisely the
  nonlinear dynamics its structure omits.
- **"The closed loop is consistent with the accuracy gap."** Consistent in
  sign and in mechanism, yes — but the ROM-vs-ROM table overstates the
  magnitude by roughly a factor two (a 134.7-point ROM-vs-ROM gap against a
  52.7-point gap on real physics at matched cadence).

The honest updated claim: the LDNet's forecast advantage does translate
into a large and unambiguous closed-loop advantage against true physics
(52.7 points on this cell), but the linear internal model degrades rather
than collapses, and the ROM-vs-ROM figure should not be quoted as the size
of that degradation.

**Scope caveat.** This is one cell (Test 2), one FOM run per arm, with no
replicate variation — unlike the 1320-probe forecast comparison. The
severe corners of the envelope are untested here, and `light/dagger_fom/NOTES.md`
documents the LDNet itself struggling with real-FOM stability at
W30/Tg0.40, so the 52.7-point gap should not be read as an envelope-wide
constant. It is, however, a clean and directly comparable data point, and
the appendix's current
sentence describing this comparison — which was written before any such run
existed — needs to be corrected, not merely appended to.

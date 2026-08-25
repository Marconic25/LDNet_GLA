# M-SPLIT study — mean-field split vs baseline (near-wall static bias)

Date started: 2026-07-17. Status: **COMPLETE 2026-07-18** (jobs 27155–27164, 10/10 done,
~0.9–1.4 h each). Verdict: **mean-split wins decisively on every metric.**

## RESULTS (test = sim_Cc_060 full grid; mean ± std over 5 seeds)
| metric | base | ms | factor |
|---|---|---|---|
| NRMSE_vx (global) | 1.095e-1 ± 8e-4 | 1.888e-2 ± 1.3e-3 | **5.8×** |
| NRMSE_vx near-airfoil | 1.398e-1 | 2.439e-2 | **5.7×** |
| vx near STATIC component | 1.38e-1 | 4.1e-3 | **33×** (static share 97%→3%) |
| vx near DYNAMIC (fluct-range) | 3.27e-2 | 3.15e-2 | ~1× (unchanged, not worse) |
| rho_vx | 0.700 | 0.992 | — |
| combined NRMSE | 1.39e-2 | 7.2e-3 | 1.9× |
| NRMSE_vy / NRMSE_p | 2.4e-2 / 2.4e-2 | 9.7e-3 / 1.25e-2 | 2.4× / 1.9× |

sim_A_025 (gust-only): even larger — vx all 1.52e-1 → 6.4e-3 (**23.7×**), near 1.94e-1
→ 8.2e-3 (23.8×); the baseline's 19% near-airfoil vx (= the documented 15–20% BL
symptom) drops to 0.8%.

Reading vs pre-registered expectations:
1. CONFIRMED and exceeded — the static near-wall bias is annihilated (33× on the
   static component), and because the fluct-normalization also re-weights the loss,
   the total beats the ~3× ceiling estimated from static-removal alone.
2. CONFIRMED — combined NRMSE moves less (1.9×) than per-field vx (5.8×): unit-mixing
   still hides the vx win (H-METRIC point stands).
3. CONFIRMED — dynamic component unchanged (3.27e-2 → 3.15e-2): the smaller
   normalization range did NOT hurt optimization.
4. Seed spread: base extremely stable (σ≈1e-3 relative ~1%); ms σ≈7% relative but
   absolute error 6× lower everywhere; s300 mildly worse (2.13e-2 vs ~1.8e-2).
5. CONFIRMED — BFGS stops on maxiter with grad_inf ~1e-4 in BOTH arms: same stall
   signature; optimizer lever (R4, Adam→L-BFGS) still untouched.

REMAINING ERROR after the split: surface region on Cc_060 improves only 1.4×
(2.8e-2, static share 3%) — now dominated by the DYNAMIC flap response at the wall.
That is exactly the target of study R2 (wall-distance + BL-mask decoder inputs).

Logging quirk: base_s100 run_info final_valid_loss=0.775 (stale row) while its test
metrics are in-family — ignore, metrics.json is the source of truth.

## Follow-up study (LAUNCHED 2026-07-18, jobs 27276–27295, 4 arms × 5 seeds)
Same protocol/data/seeds as round 1; all arms include --mean-split. Control = the
round-1 `ms` arm (and `base`). Lanes per seed: ms_t0 → ms_tik → ms_wall → ms_tik_d10.
- `ms_t0`      --mean-ref t0 (trim-referenced mean: sim-mean of the first snapshot)
- `ms_tik`     --alpha-reg 3e-4 (reference-LDNet Tikhonov ported verbatim from
               src/sensitivity_latent.py: per-layer mean of squared kernels averaged
               over layers, no biases, NNdyn+NNrec; validation WITHOUT the term)
- `ms_tik_d10` same + d_s=10 (does regularization flip the d_s verdict here too,
               as it did for the loads sweep on 2026-07-17?)
- `ms_wall`    --wall-feats: +2 decoder point-feature columns [wall distance d,
               BL mask (1-d/τ)²₊, τ=0.02c] computed from analysis_hmetric airfoil
               surface nodes (387), carried as extra `points` columns so src/ is
               untouched (space dim 2→4, NNrec +48 params). Static reference
               geometry (flap deflection ignored in d — documented caveat).
               TARGET: the residual surface-region dynamic error (2.8e-2, 97%
               dynamic) that the plain mean split does not touch.
Smoke (smoke_ms2.sh) PASSED on all four paths incl. wall-feats reconstruct.
Analysis: meansplit_compare.py now auto-discovers arms and prints vx near+surface
focus tables with vs-base factors.

### ROUND-2 RESULTS (2026-07-19, 30 runs total, all arms 5 seeds)
vx near NRMSE (Cc_060 / A_025, ×better vs base):
  ms         2.44e-2 (5.7×) / 8.15e-3 (23.8×)   ← still the reference
  ms_t0      2.65e-2 (5.3×) / 7.30e-3 (26.6×)
  ms_tik     2.52e-2 (5.5×) / 8.77e-3 (22.2×)
  ms_tik_d10 2.95e-2 (4.7×) / 1.05e-2 (18.5×)
  ms_wall    2.60e-2 (5.4×) / 7.80e-3 (24.9×)
Verdicts:
1. **Plain ms captures essentially all the gain** — no follow-up lever beats it on
   Cc_060 beyond seed noise (differences ≲1–2σ). Adopt `--mean-split` as default.
2. **d_s verdict does NOT flip** for the field recon: ms_tik_d10 is worse than d_s=1
   even with the loads-sweep α=3e-4 (unlike the loads model). d_s=1 confirmed.
   Side finding: regularized d10 has tiny seed spread (±9.5e-5 near) — Tikhonov
   does stabilize seeds, it just doesn't add accuracy here.
3. **ms_t0** best on the gust-only sim (A_025 26.6×, static share 39→22%) but
   slightly worse on the flap sim — toss-up, not a clear upgrade.
4. **ms_wall**: best surface DYNAMIC component on A_025 (2.6e-2 vs ms 3.8e-2
   fluct-range, and best surface total 9.9e-3 tied with t0) but NO gain on
   Cc_060 — consistent with the static-SDF caveat: where the flap actually
   deflects, a time-independent wall distance cannot help the dynamic response.
   Informative negative; time-dependent SDF would be next, but the payoff cap is
   now small (surface vx ~2.8e-2 total).
5. **The remaining floor is dynamics, not decoder inputs**: dyn(fluct-range) vx
   ≈3.2e-2 near-region in EVERY arm including base — untouched by all round-1/2
   levers. Next levers live on the optimizer/dynamics side (R4 Adam→L-BFGS,
   latent-ODE capacity/rollout), with diminishing returns for the thesis.

## Dynamic-residual research (workflow wf_05b91437, 2026-07-21, 36 agents, adversarial 3-vote)
Question: how to reduce the DYNAMIC (fluctuation) residual (~3.2e-2 fluct-range, flap-gap
localized, input-invariant). 4 angles searched; top-8 sources fetched + verified. VERDICT:

**Optimizer lever DOWNGRADED (was my lead hypothesis).** Rathore 2402.01868 survived 0/3
but the verified claim carries its own kill: the BFGS/L-BFGS ill-conditioning is
PDE-RESIDUAL-specific (differential operator), and Rathore shows the data-fit/boundary
terms are the WELL-conditioned ones — so the mechanism does NOT transfer to our pure
data-fit MSE. Our grad~1e-4 line-search stall is a matching SYMPTOM, not the same cause.
Plus loss→error is sublinear (100× loss drop buys ~10× error). ⇒ Adam→L-BFGS→NNCG is at
best a cheap sanity check, not the fix.

**Best-verified POSITIVE lever = the DECODER, not the inputs.** CORAL 2306.07266 (survived
1/3, data-fit): shift-MODULATED SIREN decoder (φ_i=V_i·z+c_i into sin(ω0(W·η+b+φ))) +
meta-learned latent codes → ~11× lower time-dependent (Out-t) reconstruction MSE than DINo
(1.02e-3 vs 1.11e-2), DINo = the closest latent-ODE+coordinate-INR ROM to ours. CAVEAT: the
dynamic number is fixed-geometry NS; no CORAL number is simultaneously dynamic AND
moving-geometry; no direct modulation-vs-concatenation ablation. Points at conditioning
MECHANISM (we CONCATENATE) + activation (we use tanh).

**Moving-geometry (MB-PINN 2306.13395, survived 0/3):** confirms the error localizes at the
moving high-gradient region and that concatenation conditioning blurs secondary structure —
but its result is PDE-residual-weight tuning, not a lever we have (no physics term).

**Killed (5, mostly refuted-on-TRANSFER not on facts):** Fourier features 2006.10739 (facts
solid, data-fit, but verifiers won't grant our residual IS spectral bias), multistage
2407.17213 (reaches 1e-16 data-fit but on static snapshots), Aero-Nef 2407.19916 (">3×" is
3.7× vs GNN / 2.85× vs plain MLP, steady not unsteady), Geometric-DeepONet 2512.04434
(s=1..16 input context doesn't move a ~5% floor — CONSISTENT with our input-invariance),
Gauss-Newton lazy-regime 2411.07979 (over-extended).

**Workflow coverage GAP (honest):** stable-sort-by-relevance filled all 8 fetch slots from
the optimizer + moving-geometry angles; the rollout (PDE-Refiner 2308.05732) and
conditioning-ablation (Attention-Beats-Concatenation 2209.10684) angles were searched but
NOT source-verified, despite the most on-target SNIPPETS ("teacher-forced 1-step MSE fits
only high-amplitude low-freq"; "plain concatenation is worst at fixed latent"). Next
verification target if we go the decoder/temporal route.

**Untried cheap levers this surfaced:** we have NEVER Fourier-feature-encoded (x,y) on the
recon decoder (wall-feats added [d,σ_bl] columns, not FF); and never tried a flap-δ-aware
time-dependent SDF. Both are CHEAP one-arm diagnostics.

## D-RES arm A LAUNCHED (Fourier features, 2026-07-21, jobs 27409-27417)
Implements the cheap spectral-bias diagnostic surfaced by wf_05b91437: mean-split +
Gaussian random-Fourier-feature encoding of the decoder (x,y) inputs (NEVER tried before;
wall-feats added [d,σ_bl] columns, not FF). train_fields.py `--fourier-scales S1,S2
--fourier-m M`: fixed feature basis B~N(0,σ²) shape (2,m·nscales) seeded 12345 (same across
run-seeds → fair test; only net init varies); FF applied inside make_ldnet.reconstruct on the
(x,y) columns, wall columns (if any) pass through; decoder spatial input widens (2·m·nscales).
fourier_B.npy saved in the model dir; reconstruct_fields.py reloads + FF-encodes. Smoke PASSED
(NNrec 2115→3603 params at m=16/2scales, recon in total units). Scale sweep guards a
single-σ false null (coords min-max ≈[0,1], useful σ not obvious): arms ff1_5, ff5_20,
ff10_40 × seeds {0,100,200} = 9 jobs. Control = the `ms` arm. Analysis: meansplit_compare.py
auto-discovers the ff* arms (dir ff{SC}_s{seed}, dumps ms_ff{SC}_s{seed}_rom_*). READOUT: does
the dynamic vx NRMSE (~3.2e-2 fluct-range, flap-gap) drop? If NO across all scales → spatial
spectral bias ruled OUT as the cause.

## D-RES arm B (time-dependent SDF) — RECLASSIFIED MODERATE, not launched
Geometry RESOLVED from analysis/geometry.txt: airfoil = loop1 MAIN (292 nodes, x∈[0,0.72]) +
loop2 FLAP (95 nodes, x∈[0.75,0.978]); the 387 airfoil_nodes = main+flap, cleanly split by
x>~0.735 (gap 0.72→0.75); hinge ≈ flap LE (min-x flap node), chord 0.9776. δ(t) is input
signal index 4. BUT: the decoder receives STATIC per-point coordinates; a δ(t)-rotated wall
distance is a per-(sample,time,point) feature, which the current pipeline (points broadcast
static over time, subsampled per-(n,t) by src/utils.process_dataset internals) cannot carry
without either touching src/ (off-limits) or reimplementing point-subsampling + a threaded
time-dependent feature inside recon/ (fixed-subsample path + new dataset field concatenated in
make_ldnet.reconstruct + normalization + reconstruct_fields mirror). ⇒ moderate change, real
bug surface, and the static wall-feats arm already came up null — a rushed tdep-SDF risks a
FALSE null. DECISION: wait for arm A; the research (CORAL) favors the decoder over inputs, so
if arm A is null the better moderate spend is the CORAL-style modulated/SIREN decoder, not
tdep-SDF. Revisit arm B only if a moving-geometry input is still the hypothesis.

## D-RES RESULTS (2026-07-22 → 07-25): FF null, CORAL ω0=10 WINS

Metric = `dyn(fluct)` NRMSE of vx in the **near** region (fluctuation-range normalised),
mean over seeds {0,100,200}, from `meansplit_compare.py`. This is the flap-gap dynamic
residual every prior arm left untouched (~2.7–3.7e-2 across base/ms/tik/wall).

**Arm A (Fourier features) — VERDICT: NULL, confirmed.** No FF scale beats plain `ms`;
higher scale is monotonically worse. Spatial spectral bias RULED OUT as the cause.

| arm        | A_025 (gust) | Cc_060 (flap) |
|------------|--------------|---------------|
| ms (plain) | 2.743e-2     | 3.151e-2      |
| ff1_5      | 2.795e-2     | 3.314e-2      |
| ff5_20     | 2.901e-2     | 3.575e-2      |
| ff10_40    | 3.335e-2     | 3.637e-2      |

**CORAL (shift-modulated SIREN decoder) — VERDICT: ω0=10 WINS.** First and only lever to
move the dynamic floor. ω0=30/60 OVERSHOOT (worse than plain-ms, like high FF scale).

| arm        | A_025 (gust)          | Cc_060 (flap)         |
|------------|-----------------------|-----------------------|
| ms (plain) | 2.743e-2              | 3.151e-2              |
| coral ω10  | **2.406e-2 (−12%)**   | **2.887e-2 (−8%)**    |
| coral ω30  | 3.781e-2              | 3.913e-2              |
| coral ω60  | 3.760e-2              | 3.928e-2              |

vx surface dyn(fluct): coral ω10 3.019e-2 / 3.294e-2 vs ms 3.841e-2 / 3.649e-2 (−21% / −10%).
coral ω10 also best total NRMSE (28.95× vs base; ms 23.83×). Implemented as `--decoder coral`
in train_fields.py: SIREN sine base over coords + (z,u)→per-layer additive shift via a small
MLP; NNdyn/latent-ODE untouched. `ModulatedSiren`, 5139 params at 4×24/mod-2L. Local test
11/11 + cluster smoke passed. Sweep lost ~3 days to cluster account re-expiry (Rerunable jobs
killed+requeued at midnight boundaries).

**MECHANISM (confirmed):** after mean-split removes the static bias, the residual is the
DYNAMIC fluctuation, and it is SMOOTH in space (low ω helps, high ω / high FF-scale both
inject noise and hurt). It is NOT a spatial-resolution / spectral-bias problem — it is a
decoder-**conditioning** problem: how (z,u) modulate the field. Shift-modulation at the right
(low) frequency is the lever; input features (FF, wall, tdep-SDF) are not. This closes the
input-feature route and validates the CORAL/decoder hypothesis from wf_05b91437.

## D-RES follow-ups A+B (LAUNCHED 2026-07-25, on the winning coral ω0=10)
Two cheap arms to push below the coral ω10 floor (~2.4–2.9e-2), same one-arm protocol:
- **Arm A — latent-state sweep with coral** (`coralds.pbs`, DS∈{2,3,5}×seeds; d_s=1 = coral_o10):
  does the fixed decoder unblock latent scaling? d_s≈1 was optimal only while the tanh decoder
  was the bottleneck ([[recon-intrinsic-latent-dim-1]]); if d_s>1 now helps, the residual is
  latent-limited (latent-ODE lever next). Dumps `ms_coral_ds{DS}_s{seed}_*`.
- **Arm B — FiLM modulation** (`coralfilm.pbs`, seeds): `--siren-mod-type film` = scale+shift
  (γ·(Wh+b)+β, γ=1+MLP so ~identity at init) vs shift-only. Tests whether a multiplicative
  per-channel conditioning knob buys more. 7539 params. Dumps `ms_coral_film_s{seed}_*`.
READOUT: either arm dropping vx-near dyn(fluct) below coral ω10 identifies the next bottleneck
(latent capacity vs modulation form). `submit_coral_ab.sh`; local test 9/9 + smoke.

**A+B RESULT (final, n=3, 2026-07-27):** neither beats coral ω10. vx near dyn(fluct):
coral_o10(d_s1) 2.406e-2(A)/2.887e-2(Cc); ds2 2.40/3.69; ds3 3.01/2.77; ds5 2.27/3.02;
film 2.31/3.05. On the FLAP case (Cc_060, the GLA scenario) coral_o10 is BEST — d_s>1 and
FiLM sometimes edge out on gust-only (all ~2.3-2.4, within seed noise) but WORSEN the flap.
d_s sweep noisy/non-monotone ⇒ NO systematic latent-capacity gain, residual NOT
latent-limited, d_s=1 stays optimal; FiLM a wash-to-worse. **Coral ω0=10 shift d_s=1 is the
final answer; the ~2.4-2.9e-2 floor is a genuine limit of this LDNet class.** D-RES CLOSED.
(Ops notes: the 4 ds5/film s100/s200 seeds first died on a /work disk-quota overflow — freed
~9G by moving stale my_gpu_env + superseded dumps to /scratch_cpu03/u10677113/; the re-runs
were then CPU-contended on cpu01 and had to be pinned to free nodes cpu02-05 to finish.)

## Thesis deliverables (2026-07-21)
- `analysis/fig_meansplit.py` (login node, numpy+mpl on the dumps) →
  `analysis_meansplit/figs/`:
  - `fig_meansplit_decomp.png` — static (solid) vs dynamic (hatched) vx error by
    region, base (red) vs ms (blue), both test sims. Headline: baseline = pure
    static near-wall bias; ms removes it, dynamic unchanged.
  - `fig_meansplit_fields.png` — near-airfoil vx at gust peak (Cc_060): FOM |
    |err| base | |err| ms, shared scale. Baseline error halo around the whole
    profile → ms leaves only a flap-gap residual (the dynamic response).
  - `meansplit_numbers.tex` — \newcommand macros (msCcBaseNear=0.140,
    msCcMsNear=0.0244, msCcFactor=5.7, msABaseNear=0.194, msAMsNear=0.0082,
    msAFactor=24) so the text never hardcodes a number.
- `recon/analysis/meansplit_thesis.tex` — drop-in \subsubsection for Sec. 4.1
  (replaces the commented "Network depth and field reconstruction" placeholder in
  chapter3.tex line ~120): setup → symptom (static bias) → mean-split fix + table
  + 2 figures → residual-is-dynamic ablation verdict. Table tab:meansplit = per-
  region base vs ms vx NRMSE, both sims. Copied to light/latex/ + Images/;
  bib entry Catalani2025 (MARIO, verified author list) added to bibliography.bib;
  POD-split analogy cites BennerGugercinWillcox2015. Only external ref =
  subsec:training_results (exists). Validated: braces balanced, macros/cites
  resolve. NOT auto-inserted into chapter3.tex — placement is the user's call
  (article ~30pp; field recon is a secondary line). To use: \input the snippet
  where the placeholder is, \input meansplit_numbers.tex in the preamble, sync
  the 2 PNGs to the Overleaf Chapter 3/Images.

## Question
~90% of the recon field-LDNet's BL/near-wake vx error is a STATIC time-mean bias
(analysis_hmetric/static_vs_dynamic.csv). The literature synthesis
(near_wall_rom_literature_report.md, 2026-07-17) ranks the **mean-field split** as the
cheapest, most-targeted fix: store the train-set mean field, let the decoder learn only
FLUCTUATIONS. The static component then costs the network nothing (it is stored, not
learned), and min-max normalization re-derived on fluctuations re-scales the MSE loss
toward the gust/flap dynamics instead of the ~80 m/s mean structure.
Literature anchors: POD-DeepONet stores mean+POD modes explicitly, cavity flow error
1.20%→0.33% vs end-to-end DeepONet (Lu et al. 2022, via arXiv:2403.18735 Table 1);
mean-subtraction is standard in hybrid DL-ROMs (Gupta & Jaiman arXiv:2009.04396:
"s_n' = s_n − s̄"); GLOBE (arXiv:2511.15856) predicts ΔU/|U∞|, not total velocity.
No published ablation isolates "mean split removes static near-wall bias" — this study
IS that ablation.

## Implementation (recon/train_fields.py `--mean-split`, recon/reconstruct_fields.py)
- mean_fields (P,3) = ensemble+time mean of TRAIN output_fields, computed on the full
  reference grid BEFORE normalization/subsampling; saved to <out>/mean_fields.npy and
  <out>/latent_<ds>/mean_fields.npy (self-contained model dir).
- train/valid/test fields all mean-subtracted (shared reference grid asserted);
  normalization ranges recomputed on the fluctuations.
- evaluate() and reconstruct_fields.py add the mean back → all reported metrics stay in
  TOTAL-field units with the same full-range denominators as every previous run
  (directly comparable NRMSE). config.json carries "mean_split": true.
- Baseline path byte-identical (flag off ⇒ no behavior change; smoke-verified).
- Smoke (login node, smoke_meansplit.sh, 2026-07-17): PASSED — baseline unchanged,
  mean saved, roundtrip exact (1.4e-17 local check), ROM dump back in physical totals
  (vx [-25.0, 114.5] matching FOM), trainer-eval and dump NRMSE identical (3.583e-05
  on the trivially-easy quiescent smoke set).

## Design
- Data: FIELDS_div_train.h5 (15 sims) / FIELDS_div_valid.h5 / FIELDS_Cc060.h5 (test),
  same as final_div and the depth study. Eval dumps on sim_A_025 (gust) + sim_Cc_060
  (gust+flap).
- Arms: `base` (no flag) vs `ms` (--mean-split). Everything else identical:
  d_s=1, output-nl linear, subsample 1024 uniform, restarts 2, adam 300, bfgs 2000
  (= final_div protocol; reference: train 8.28e-3, Cc-test combined NRMSE 1.39e-2).
- Seeds: --seed-base ∈ {0,100,200,300,400} (spacing ≥ restarts so Adam-restart seeds
  never collide across runs). 5 seeds × 2 arms = 10 runs ≈ 2 h each.
- Jobs: cluster/meansplit.pbs (qsub -v ARM,SEED), cluster/submit_meansplit.sh = one
  lane per seed, base → ms chained afterany, 5 lanes parallel (cpu queue had one
  running depthrun; 5×8 cores fits). Submitted 2026-07-17: 27155(base_s0 R),
  27157/59/61/63 (base s100–400 Q), 27156/58/60/62/64 (ms arms, H on afterany).
- Outputs: models/meansplit_study/{arm}_s{seed}/ (train.log, latent_1/*, recon.log)
  + results/ms_{arm}_s{seed}_rom_{a025,cc060}/ (full-grid ROM/FOM dumps).

## Analysis (after completion)
On the login node:
    python3 /work/u10677113/NACA2312/recon/analysis/meansplit_compare.py
→ analysis_meansplit/{meansplit_runs.csv, meansplit_static_dynamic.csv,
meansplit_arm_summary.csv} + printed vx-by-region focus table.
Decomposition conventions identical to hmetric_static_dynamic.py (regions
near/wake/far/surface from analysis_hmetric/region_labels.npy + airfoil_nodes.npy;
err_static = time-mean(rom−fom)).

## What to look for (pre-registered expectations)
1. PRIMARY: near/surface-region vx static NRMSE collapses in the ms arm
   (static_share ~90% → small). If total near-wall vx error is ~90% static, ceiling
   is ~3× NRMSE reduction (√10) in the BL band.
2. Combined/global NRMSE may move little (dominated by easy far field) — that is
   expected and is itself a thesis point.
3. Dynamic-component NRMSE (nrmse_dyn_fluctrange) should be ≤ baseline: the fluct
   normalization re-weights the loss toward dynamics. If it worsens, the smaller
   normalization range is hurting BFGS conditioning — report, don't hide.
4. Seed spread (5 seeds/arm): does the split also stabilize training? (Curriculum
   literature says variance reduction is plausible; not the primary claim.)
5. BFGS termination (bfgs_msg/grad_inf_norm in meansplit_runs.csv): unchanged stall
   signature expected — the split is not an optimizer fix (that is study R4).

## Caveats
- The stored mean is the ensemble+time mean over 15 div-train sims (mixes gust and
  flap responses), not the trim snapshot; if ms wins, a trim-referenced variant
  (t=0 field) is a cheap follow-up arm.
- delta-normalization guard (_rng) already handles constant channels; fluct fields
  are near-zero-mean by construction so min-max ranges are roughly symmetric.
- viz_fields.py compare works on the dumps unchanged (they are total-field npy).

## POST-CLOSURE ROUND (2026-08-18/19): depth, RAD-lite sampling, BFGS budget --
## every consolidated lever tried LOST to the champion. coral_o10 stands.

Champion = coral_o10_s0 (L6, mean-split+CORAL o10, d_s=1, bfgs=2000, uniform
sampling): vx 1.621e-2, combined NRMSE 6.141e-3 (Cc_060 test, full grid).

| run | vx NRMSE | combined | vs champion |
|---|---|---|---|
| coral_o10_L12_s0/s100/s200 (depth L12, dyn=4x7 rec=8x24) | 2.019/2.003/2.257e-2 | 7.058/6.208/6.859e-3 | worse, all 3 seeds |
| coral_o10_nw4_s0/s100/s200 (--sampling near-wall boost=4) | 2.558/3.670/3.538e-2 | 7.555/14.19/9.511e-3 | much worse, all 3 seeds |
| coral_o10_bfgs6000_s0 (bfgs 2000->6000) | 2.649e-2 | 7.072e-3 | worse |
| ms_L12mlp_s0 (L12 + mean-split, plain MLP decoder, NO coral) | 1.993e-2 | 6.232e-3 | worse (barely, on combined) |

Readings:
1. Depth+CORAL (L12) confirmed negative on all 3 seeds -- consistent, not noise.
2. Near-wall RAD-lite sampling (boost=4, subsample unchanged at 1024) is a clear
   loss on all 3 seeds. Diagnosis: a fixed total point budget means boosting
   near-wall draw probability 5x starves the far field, which carries most of
   the domain's dynamic range (vx to +-100 m/s far field vs a much smaller
   near-wall scale) -- global NRMSE degrades even if the near-wall region
   itself improved locally (not decomposed here). Do not reuse boost=4 with
   subsample=1024; a future attempt needs either a much gentler boost (~1-1.5)
   or a larger --subsample so far-field coverage isn't sacrificed.
3. R4 (more BFGS budget) made things WORSE -- the first empirical answer after
   the 2026-07-21 literature-only downgrade. BFGS at 2000 was not budget-
   starved; the extra iterations overfit (matches the val-loss-plateaus-while-
   train-loss-still-falls pattern already visible in the L12 logs). Do not
   raise the BFGS cap as a lever; if anything a LOWER cap (early stopping) is
   the more promising untested direction, cautiously.
4. Disambiguation isolates Tier 1's cause: it's CORAL-specific, not a general
   mean-split+depth incompatibility. L12+mean-split+plain-MLP (6.232e-3) beats
   L6+mean-split+plain-MLP-alone (7.2e-3, round-1 baseline) -- depth still
   helps the ORIGINAL decoder, consistent with the pre-CORAL H-DATA finding
   (N60+L12=0.0127, see recon/analysis/learning_curve_summary.csv). But it
   still doesn't beat L6+CORAL (6.141e-3). Verdict: the CORAL SIREN decoder
   specifically does not tolerate added depth (plausible cause: phase/
   frequency effects compounding across sine layers, unlike tanh layers) --
   depth and CORAL are each independently useful but must not be stacked
   without first addressing SIREN-specific depth conditioning (e.g. per-layer
   omega0 scaling) -- that is research territory, not a cheap flag combo.

NET CONCLUSION: coral_o10_s0 remains UNBEATEN after this full round. Every
cheap, consolidated lever available (depth, sampling reweight, optimizer
budget) has been tried and lost. Corroborates the original D-RES CLOSED
verdict from a wider angle. Remaining candidate levers (NNdyn/latent-ODE
architecture change, a retuned gentler RAD-lite, SIREN-specific depth
conditioning) are genuinely open research questions, not more flag sweeps --
scope and cost need to be discussed before spending more compute on them.

Also recovered from the cluster this round, not relaunched: the H-DATA
learning-curve results (models/learning_curve_summary.csv, written
2026-07-21, never previously synced) -- see recon/analysis/learning_curve_summary.csv
and recon-hdata-ladder-status in Claude's session memory for the full reading;
N100 rows in that file are unreliable (land in the known extraction-corruption
window) and should not be cited without a clean re-extraction.

## MULTIPLE SHOOTING (2026-08-19/20): global NRMSE improves, target metric LOSES

First lever aimed at NNdyn/the training procedure (all prior levers only ever
touched the decoder). --shooting-segments/--shooting-lambda added to
train_fields.py: splits each training trajectory into K segments, each with
its own free trainable initial latent state, plus a continuity penalty
(Turan & Jaeschke, arXiv:2109.06786). Training-only; validation/test/inference
unchanged (standard single-shot rollout). K=4 fixed, lambda in {0.1,1.0,10.0},
single seed, on top of the champion (coral_o10_s0).

GLOBAL numbers look like a clean win: combined NRMSE 5.76-5.85e-3 across all
3 lambdas vs champion's 6.141e-3. **This is misleading** -- per-region
static/dynamic decomposition (recon/analysis/decomp_shooting.py, same method
as fig_meansplit.py) shows the near-wall/surface DYNAMIC vx residual (the
actual quantity this whole D-RES thread targets, flap-driven) gets ~10-11%
WORSE at every lambda (near: 2.054e-2 champion -> 2.27-2.28e-2 shooting;
surface: 2.266e-2 -> 2.20-2.40e-2, mixed but not a real win at any lambda).
Only the far-field dynamic component improves (~10% at best). The combined-
NRMSE "win" is a H-METRIC-style illusion: far field dominates the global
number by point-count/mass, masking a real regression exactly where it
matters. Higher lambda (pulling harder toward single-shot consistency) does
NOT reverse the near-region trend -- looks structural (free boundary
variables let the optimizer "reset" locally instead of being forced to track
the fast near-wall dynamics continuously across the whole horizon), not a
lambda-tuning artifact.

VERDICT: multiple shooting (K=4, this form) LOSES on the metric that matters.
coral_o10_s0 (no shooting) still stands as champion. Open, untested
questions if revisited: much larger K with a weak/annealed penalty
(near-continuous shooting, softly enforcing full consistency only late in
training) vs this coarse fixed-K/fixed-lambda version -- not a cheap next
step, needs a real annealing schedule (checkpoint_callback hook already
exists in RecordingOptimizationProblem for this), discuss before spending
more compute.

LESSON reinforced (again): never trust combined/global NRMSE alone on this
project -- always pull the per-region static/dynamic decomposition before
declaring a result. This is the second time in the project's history (after
the original H-METRIC finding) that global NRMSE and the real per-region
picture disagreed in opposite directions.

## NEURAL-CDE (2026-08-20/21): loses cleanly, no metric-illusion this time

User explicitly asked for the most-promising lever regardless of implementation
cost after shooting lost. Chosen: --dyn-cond cde in train_fields.py -- NNdyn
depends on the latent state z ALONE (no concatenated exogenous input) and
outputs a (d_s x n_channels) matrix, contracted with the exogenous signal
path's per-step finite-difference derivative (dz = f_theta(z) @ dX), instead
of the standard concat-everything-into-one-MLP update (Kidger et al., Neural
CDEs, arXiv:2005.08926; skips their cubic-spline interpolation of X since its
stated purpose -- adjoint-backprop stability -- doesn't apply, this project
backprops through the unrolled loop directly). Concrete motivation: the flap
deflection rate and gust rate have NO path into NNdyn at all under the
standard concat scheme (only h-dot, alpha-dot are given), yet the flap drives
the worst residual; under CDE every channel's rate becomes structurally
load-bearing by construction.

Caught and fixed a real bug before spending cluster compute: f_theta(z)
evaluated at the physical z=0 initial condition, with Keras' default zero
bias, makes an all-tanh MLP output EXACTLY zero regardless of the kernels
(tanh cascades through zero at every layer) -- the state could never leave
the origin and every kernel got a bit-exact-zero gradient. Fixed with a small
random bias initializer specific to this path; verified locally (nonzero
kernel gradients, genuine state movement) before launching anything.

RESULT (n=3 seeds, coral_o10_cde_s{0,100,200}): loses CLEANLY this time, no
combined-NRMSE illusion to untangle -- every region, static AND dynamic, on
every seed, is worse than the champion:

| region/component | champion | cde s0 | cde s100 | cde s200 |
|---|---|---|---|---|
| near dynamic vx | 2.054e-2 | 2.415e-2 (+18%) | 2.283e-2 (+11%) | 2.411e-2 (+17%) |
| surface dynamic vx | 2.266e-2 | 2.960e-2 (+31%) | 2.433e-2 (+7%) | 3.119e-2 (+38%) |
| near static vx | 3.80e-3 | 4.12e-3 | 3.46e-3 | 5.01e-3 |
| combined NRMSE | 6.141e-3 | 6.222e-3 | 6.598e-3 | 6.847e-3 |

Reading: by removing the raw concatenated signal values in favor of
rate-only, matrix-contracted conditioning, the CDE formulation likely lost
access to information the standard concat approach had "for free" -- the
ABSOLUTE value of the exogenous signals (e.g. the actual flap angle, not just
how fast it's currently moving). For a nonlinear aerodynamic response, the
absolute forcing state plausibly matters as much as its rate; this
formulation subtracted information instead of adding it. A more conservative
middle ground (never tried): keep the standard concatenation AND add the
missing rate channels (W-dot, delta-dot) as EXTRA inputs, rather than
replacing value-conditioning with rate-only structural conditioning -- this
was flagged as the cheap precursor probe in LATENTODE_LITERATURE_NOTES.md
item 4 and was skipped in favor of the full restructure; it remains the one
NNdyn-conditioning idea from that document not yet empirically tested.

CUMULATIVE STATUS: five substantial, independently-motivated levers now
tried and lost against coral_o10_s0 (depth+CORAL, near-wall RAD-lite
sampling, tripled BFGS budget, multiple shooting, Neural-CDE). The champion
(mean-split + CORAL omega0=10, d_s=1, L6, uniform sampling, bfgs=2000) has
been attacked from the decoder side, the sampling side, the optimizer side,
the training-procedure side, and now the dynamics-conditioning side, and has
not been beaten once. This substantially strengthens the original "D-RES
CLOSED, genuine limit of this LDNet class" verdict -- it now holds up from
five independent angles, not just the original decoder-only sweep.

## STALL/SEPARATION HYPOTHESIS (2026-08-22) -- NEW, physically CONFIRMED

New framing, not previously established anywhere in this project: is the D-RES
residual actually localized flow SEPARATION (boundary-layer/shear-layer
reversal) at the flap, rather than just "a smooth region the model happens to
fit poorly"? A repo-wide grep found zero prior aerodynamic use of "stall" --
every existing hit was the unrelated BFGS line-search "stall" symptom. No
Cl(t)/AoA(t)/separation/recirculation diagnostic existed anywhere before this.

**Diagnostic** (`recon/analysis/decomp_stall.py`, local-only, no cluster
needed, run on the already-synced champion FOM/ROM dump for sim_Cc_060):
per near-airfoil node, take its own FOM velocity vector at t=0 (quiescent,
attached) as a fixed per-node reference direction; at every later time,
project the local velocity onto that reference. A negative projection means
the local flow has reversed relative to its own attached baseline -- this
sidesteps needing to reason about surface-tangent sign conventions around the
closed main+flap contour (upper/lower loop winding flips sign easily).

**VERDICT: CONFIRMED, cleanly, on both criteria that would have refuted it.**
- Temporally gated (not a static feature): reversed-flow fraction in the
  near-flap band is exactly 0 for t<0.5s and t>0.8s, spikes to a peak of
  23.3% at t=0.584s (near-main-element fraction peaks far lower, 7.7%, same
  time) -- a genuine transient event tied to the gust+flap excitation, not a
  constant wake artifact.
- Spatially localized at the flap (see `figs_stall/stall_diagnostic.png`):
  the reversed-flow mask at peak time sits right on and immediately behind
  the flap upper surface/trailing edge -- exactly the same flap-gap/wake zone
  every D-RES lever already identified as the untouched residual.
- The champion ROM MISSES it almost entirely: at the same peak time (t=0.584s)
  ROM's own reversed-flow fraction in that band is 2.1% vs FOM's 23.3% -- the
  model does not just have quantitatively larger error there, it fails to
  represent the separation event at all, predicting attached flow throughout.
- At the 15 worst-error near-flap points at peak time, FOM shows vx from -3
  to -16 m/s (reversed) while ROM predicts vx from +50 to +80 m/s (strongly
  attached-like) -- not a magnitude miss, a SIGN miss. Across all 1726
  near-flap points at peak time, ROM gets the attached/reversed sign wrong at
  21.7% of them.

**Reading**: this is not merely "the model is locally worse near the flap" --
it is a qualitative failure mode: a real, brief, localized shear-layer
reversal event that the smooth globally-coordinate-conditioned latent
dynamics structurally cannot represent, consistent with every levers-tried
result so far (decoder conditioning, sampling, optimizer, shooting, CDE all
touched EITHER the whole-field decoder OR the whole-trajectory dynamics --
none specifically targeted a localized, transient, sign-changing event).
This reframes D-RES from "a stubborn smooth residual" to "an unmodeled
localized separation event" -- a materially different, and more actionable,
diagnosis. See `recon/analysis/decomp_stall.py` to reproduce/extend (control
case on sim_A_025 gust-only, and a vorticity-based corroboration, are the two
open follow-ups noted but not yet run).

Next: literature review targeting regime-aware/local architectural levers
(deliberately excluding the 5 already-closed mechanisms), in parallel with
two cheap, code-grounded candidates already identified: (1) a loss-weighting
hook -- confirmed NEVER tried, the current loss (`train_fields.py` ~line
432-433) is a plain unweighted MSE with zero per-point/per-region weighting
mechanism, distinct from the already-failed *sampling* reweight since it
doesn't sacrifice far-field point budget; (2) an engineered regime-indicator
input channel (e.g. a gust-rate x flap-angle threshold trigger derived from
this diagnostic) concatenated into NNdyn/NNrec, informed by the CDE lesson
that removing information (there: absolute signal values) loses more than it
gains -- so add, don't replace.

## RESIDUAL-CURRICULUM LOSS WEIGHTING (2026-08-22/23): LOSES cleanly, worse
## than plain uniform MSE on EVERY axis -- not just a null result

STALL_LITERATURE_NOTES.md's #1-ranked candidate (section 5/9): reweight each
sampled point's squared training error by its own DYNAMIC residual magnitude
-- `--loss-weight-mode residual`, weight = stop_gradient(|pred-target|_2
across output fields)^power, mean-normalized, recomputed fresh every Adam
epoch / BFGS function evaluation (no persistent EMA state, deliberately: a
rejected BFGS line-search trial would corrupt any cross-step EMA). Distinct
from the STATIC geometric `--loss-weight-mode flap` lever tested in parallel
(fixed flap-proximity weight, precomputed once) and from the already-failed
near-wall *sampling* reweight (biases which points are seen, not how much
each residual counts). n=3 seeds x 2 strengths (power=1.0 moderate, power=2.0
strong) on top of the champion (mean-split+CORAL o10) = 6 runs, all completed.

**VERDICT: LOSES on every region, every component (static AND dynamic), every
seed, both power settings -- no combined-vs-regional illusion to untangle
this time, the global number and the per-region picture agree for once, they
just both say "worse":**

| arm | near static | near dynamic | combined NRMSE |
|---|---|---|---|
| **champion** | 3.80e-3 | 2.054e-2 | 6.141e-3 |
| residual p1.0 (3 seeds) | 9.1-11.2e-3 | 2.90-3.15e-2 | 9.90-10.81e-3 |
| residual p2.0 (3 seeds) | 14.6-17.7e-3 | 2.84-3.46e-2 | 11.54-13.47e-3 |

Every region (near/wake/far/surface) and both components (static/dynamic) are
worse at every seed, for both powers -- including the STATIC component and
the FAR FIELD, which no prior lever meaningfully touched (all 5 previously-
closed levers left static roughly flat or improved it slightly). Stronger
weighting (p2.0) is consistently and monotonically worse than moderate
(p1.0) across seeds -- a real dose-dependent harm, not noise.

**Sign-flip readout (the literature's own falsifiable prediction, most direct
test): did NOT improve -- if anything it got worse.** Champion ROM already
massively underestimates the FOM's 23.35% peak reversed-flow fraction at
2.14%, sign-flip rate 21.67%. Under residual weighting the ROM's reversed-
flow fraction at the same instant is even LOWER (p1.0: 0.00-0.35%; p2.0:
0.87-1.74%) -- the model represents the reversal event *less*, not more, and
sign-flip rate stays flat at ~21-23% throughout (unchanged, within noise of
champion). The literature's own pre-registered "null result" reading
("sign-flip rate unchanged... residual is a representational ceiling") does
not even fully apply -- this is not a null, it is a regression on the exact
metric the lever targeted.

**Reading:** the likely mechanism is that a *dynamic*, model's-own-error-
driven weight is a fundamentally noisier, less stable training signal than a
static geometric one -- early/mid-training residuals are large and volatile
everywhere (not yet informative about where the *true* hard region is), and
feeding that noise back into the loss with power>=1 amplifies whatever the
optimizer is currently worst at fitting, which on a d_s=1, 1024-point-
subsample budget model degrades general fit quality broadly rather than
concentrating gradient usefully on the flap-gap transient. This is a
qualitatively different failure mode from every previously-closed lever
(none of which touched the static component this severely) -- it is not just
"another way to not fix D-RES", it actively demonstrates that naive
error-driven curriculum weighting is actively harmful here, a useful negative
result for the thesis write-up. Full per-region numbers in
`recon/analysis/decomp_residual.py`'s output; champion (mean-split+CORAL o10)
remains unbeaten.

**CUMULATIVE STATUS UPDATE:** six independently-motivated levers now tried
and lost against coral_o10_s0 (depth+CORAL, near-wall RAD-lite sampling,
tripled BFGS budget, multiple shooting, Neural-CDE, and now residual-
curriculum loss weighting). The remaining open candidates from
STALL_LITERATURE_NOTES.md are the Goman-Khrabrov separation-lag state
(dynamics-side memory, `--dyn-sep-state`, in progress) and a local/gated
decoder near the flap (major, held back pending the cheaper items' verdicts).

## GOMAN-KHRABROV SEPARATION-LAG STATE (2026-08-22/23): implemented, launched

`--dyn-sep-state`: a dedicated scalar attachment-state X(t) (1=attached,
0=separated), lag-ODE form `tau1*Xdot + X = X0(gust_rate, flap_rate)`
(Goman-Khrabrov structure, STALL_LITERATURE_NOTES.md section 2/9 item 2),
discretized as its own forward-Euler update PARALLEL to NNdyn's (not folded
into num_latent_states -- the lag-ODE is a specific constrained recurrence,
structurally different from NNdyn's free vector field, and folding it in
would discard exactly the structural prior motivating the lever). X(0)=1
(fully attached IC, matching every sim's quiescent start); tau1 is a single
trainable softplus-parameterized scalar (init 5 steps, ~matching the
confirmed ~0.3s/~15-step separation-burst width). X is concatenated as an
extra channel into BOTH NNdyn's regular update and NNrec's decoder input
(`extra_cond` in `reconstruct_states`), matching the literature's
"concatenated alongside z" recommendation. `SepStateDynamics` class,
`train_fields.py`.

**Two real bugs caught before/at the cluster smoke-test stage** (both now
fixed, verified via a real cluster training run, not just locally):
1. `sepnet.count_params()` crashed ("layer isn't built") because
   `SepStateDynamics` has no `call()` (only `.x0()`/`.tau1`), so Keras never
   flips its own `built` flag even though its Dense sublayers built
   individually on first use. Fixed by summing `trainable_variables` shapes
   directly instead of relying on Keras' `count_params()`.
2. **More serious: a regression in the BASELINE (flag OFF) path.** The
   original implementation put `if dyn_sep_state:` conditionals on
   individual statements INSIDE the shared `for i in tf.range(nt-1):` loop
   body (assigning `x_history` conditionally). This works fine under eager
   execution (a local forward-pass check passed cleanly) but throws
   `ValueError: 'x_history' must be defined before the loop` under the REAL
   autograph-traced training path (`src/optimization.py`'s
   `@tf.function`-decorated gradient computation) -- autograph's `for_stmt`
   conversion does a static pass over the loop body's assigned names to set
   up `tf.while_loop`'s loop-carried variables, and a name only
   conditionally assigned inside the loop trips this even when the
   Python-level condition is a compile-time constant that's False. This
   would have broken every OTHER concurrently-running lever too (any run
   with `--dyn-sep-state` unset also hits `evolve_dynamics`), not just this
   one -- caught it before any queued job could pick up the broken file by
   checking `ps`/logs on the two other in-flight full launches immediately
   after the bad push. **Lesson reinforced**: a local eager forward-pass
   check does NOT exercise autograph tracing and cannot catch this whole
   class of bug -- only a real cluster training run (which goes through the
   actual `@tf.function` path) can. Fixed by fully duplicating the loop body
   into two separate top-level branches (`if dyn_sep_state: <loop A> else:
   <loop B>`) so autograph never sees the `x_history` name at all when
   tracing the off-path, instead of branching individual statements inside
   one shared loop.

Local forward-pass check (post-fix): X(0)=1.0000 exactly, X evolves over a
trajectory (not pinned), byte-identical-when-off confirmed exactly (loss ==
plain MSE). Cluster smoke test (`smoke_stall_sepstate.sh`, real TF 2.14
training, both the sep-state-ON and baseline-OFF paths): PASSED cleanly after
the fix. **Launched** n=3 seeds (0/100/200) on top of the champion
(mean-split+CORAL o10), jobs 29058-60 (`stall_sepstate.pbs`), queued behind
other in-flight work (shared `max_user_run=4` cap with a concurrent chat's
unrelated `mpcdagger` jobs and this investigation's own `stall_rate`/
`stall_lw` full launches) -- result pending.

## SIGNAL-RATES (--add-signal-rates) RESULT (2026-08-22/24): LOSES, n=3 confirmed

The cheap "add, don't replace" middle-ground lever (missing Wdot/deltad rate
channels appended to the standard concat, explicitly motivated by the
Neural-CDE post-mortem) -- full n=3 seeds (0/100/200) completed on top of the
champion (mean-split+CORAL o10). `recon/analysis/decomp_rates.py`.

**VERDICT: LOSES on the region that matters, consistently across all 3
seeds -- and shows the SAME combined-NRMSE illusion pattern as multiple
shooting.**

| seed | near static | near dynamic | surface dynamic | combined NRMSE |
|---|---|---|---|---|
| **champion** | 3.80e-3 | 2.054e-2 | 2.266e-2 | 6.141e-3 |
| rates s0 | 4.41e-3 | 2.155e-2 (+5%) | 2.488e-2 (+10%) | 5.715e-3 (looks better) |
| rates s100 | 4.04e-3 | 2.275e-2 (+11%) | 2.665e-2 (+18%) | 6.585e-3 (worse) |
| rates s200 | 4.22e-3 | 2.193e-2 (+7%) | 2.902e-2 (+28%) | 5.549e-3 (looks better) |

Near-region dynamic AND static are worse at every single seed (no
exceptions); surface dynamic is worse at every seed, by a growing margin
(+10% to +28%). The combined/global NRMSE is a mixed bag (2 of 3 seeds look
BETTER than champion) -- exactly the H-METRIC/multiple-shooting-style
illusion: far-field/point-count dominance in the global number masking a
real, consistent regression in the near-flap region that this whole
investigation targets.

**Sign-flip readout (the direct, targeted test): essentially flat, no real
improvement.** Champion 21.67% sign-flip rate (ROM reversed-flow frac 2.14%
vs FOM's 23.35%); rates gives 21.90% / 22.07% / 19.00% across the 3 seeds --
within seed-to-seed noise of the champion, no seed shows a materially better
ROM reversed-flow fraction (0.0156-0.0446, champion 0.0214 -- s200's 0.045 is
nominally higher but still misses over 80% of the event, not a meaningful
recovery).

**Reading:** adding the missing rate channels alongside the existing
concatenation (rather than replacing value-conditioning with rate-only
conditioning, as CDE did) avoided CDE's clean catastrophic loss, but did not
help either -- the extra channels apparently give the optimizer more
capacity to spend on the easy far-field/attached-flow majority of points
(where the combined-NRMSE improvement comes from) rather than the hard
near-flap separation event specifically. This closes the "add, don't
replace" hypothesis from the CDE post-mortem: extra rate INPUTS alone,
without any accompanying change to how the loss weights points or how the
dynamics represents regime transitions, are not sufficient.

## FLAP LOSS-WEIGHT (--loss-weight-mode flap) RESULT (2026-08-22/24): LOSES, n=3 confirmed -- and a lesson about premature single-seed reads

The static geometric flap-proximity loss reweight (distance-decay tau=0.3c,
boost=5.0) -- full n=3 seeds (0/100/200) completed on top of the champion.
`recon/analysis/decomp_lw.py`.

**A single-seed preview (s200 only, run while s0/s100 were still queued)
looked like the best result in the entire investigation**: sign-flip rate
dropped from the champion's 21.67% to 8.40%, and the ROM's reversed-flow
fraction at the gust peak jumped from 2.14% to 16.45% -- by far the largest
movement toward the FOM's 23.35% target seen from any lever, cheap or
literature-derived. This was reported to the user as a promising early
signal, explicitly caveated as single-seed and unconfirmed.

**It did not replicate. VERDICT: LOSES, n=3 confirmed -- s200 was an outlier,
not a real effect.**

| seed | near dynamic | surface dynamic | sign-flip% | ROM reversed-flow frac |
|---|---|---|---|---|
| **champion** | 2.054e-2 | 2.266e-2 | 21.67% | 2.14% |
| flap-lw s0 | 2.783e-2 (+35%) | 3.367e-2 (+49%) | 22.94% (worse) | 0.52% (worse) |
| flap-lw s100 | 2.831e-2 (+38%) | 3.852e-2 (+70%) | 22.65% (worse) | 0.70% (worse) |
| flap-lw s200 | 2.003e-2 (better) | 2.580e-2 (+14%) | 8.40% (much better) | 16.45% (much better) |

Two of three seeds (s0, s100) lose cleanly and consistently on EVERY axis --
static, dynamic, near, surface, combined NRMSE, AND the sign-flip target
metric itself gets slightly worse, not better. Only s200 shows the dramatic
improvement. Averaged across the 3 seeds, near/surface dynamic are clearly
worse than the champion (near avg 2.54e-2 vs 2.05e-2, +24%; surface avg
3.27e-2 vs 2.27e-2, +44%) and the sign-flip average (18.0%) is pulled toward
"better" almost entirely by the one outlier seed, not a systematic effect --
2 of 3 independent draws show no sign-flip improvement at all.

**Reading:** a static, precomputed geometric weight (fixed at the START of
training, based only on distance to the flap surface) is apparently a
high-variance, unstable training signal -- exactly the seed-sensitivity this
project's n=3 discipline exists to catch. Plausible mechanism: boosting the
loss weight of a small, hard, high-curvature spatial region by a fixed
multiplier throughout the ENTIRE training run (not annealed, not adaptive)
can push the optimizer into qualitatively different basins depending on
initialization -- one basin (hit by s200) that actually resolves the
near-flap structure, and two basins (s0, s100) that instead overfit the
up-weighted region in a way that hurts both static and dynamic accuracy
broadly. This is a DIFFERENT failure mode from the ALSO-tried residual-
curriculum lever (which lost cleanly and uniformly, no seed variance to
speak of) -- static-boost and residual-adaptive-boost are not interchangeable
approximations of the same idea.

**Process lesson, stated plainly for future sessions**: do not report a
single-seed result as "the most promising lever" even with a clear verbal
caveat -- the caveat does not fully undo the anchoring effect of a strong
early number. The correct sequence (used here once the s0 result came in)
is: flag single-seed reads as informative-but-provisional, and actively
update/retract the framing the moment contradicting data arrives, rather
than let the first favorable impression stand until "confirmed." No
compute or code was wasted by this -- the mistake was purely in how the
interim result was framed to the user -- but it is worth remembering for
future single-seed previews in this project.

## GOMAN-KHRABROV SEPARATION-LAG STATE RESULT (2026-08-24/25): no real effect, n=3 confirmed

The dedicated attachment-state lag-ODE lever (`--dyn-sep-state`) -- full n=3
seeds (0/100/200) completed on top of the champion. `recon/analysis/decomp_sepstate.py`.

**VERDICT: essentially a wash on aggregate metrics, and NO improvement
whatsoever on the target sign-flip metric in any seed -- this is the null
result the literature notes themselves pre-registered as the informative
outcome ("would argue the missing piece isn't a dynamics-side memory/
indicator but a decoder-side inability to spatially express the reversal").**

| seed | near dynamic | surface dynamic | combined NRMSE | sign-flip% | ROM reversed-flow frac |
|---|---|---|---|---|---|
| **champion** | 2.054e-2 | 2.266e-2 | 6.141e-3 | 21.67% | 2.14% |
| sep-state s0 | 1.845e-2 (better) | 1.936e-2 (better) | 6.020e-3 (better) | **21.67% (IDENTICAL)** | **2.14% (IDENTICAL)** |
| sep-state s100 | 2.032e-2 (~tied) | 2.176e-2 (better) | 5.945e-3 (better) | 20.80% (~tied) | 2.67% (~tied) |
| sep-state s200 | 2.480e-2 (+21% worse) | 2.677e-2 (+18% worse) | 6.581e-3 (worse) | 22.94% (worse) | 0.41% (worse) |
| **3-seed average** | 2.119e-2 (+3%) | 2.263e-2 (~tied) | 6.182e-3 (~tied) | 21.80% (~tied) | 1.74% (~tied/worse) |

Two of three seeds (s0, s100) show modest, consistent improvement on the
generic per-region NRMSE numbers (near/surface dynamic AND static, combined)
-- but seed s0's sign-flip readout is EXACTLY IDENTICAL to the champion down
to the same 4 significant figures (21.67%, ROM frac 0.0214), meaning the
sep-state mechanism had ZERO measurable effect on how the model represents
the actual reversal event for that seed, despite the generic metrics ticking
down slightly. s100 is marginally different, within noise. s200 cancels out
essentially all of the average gain by being clearly worse on every axis.
Averaged across all 3 seeds, the near-region dynamic residual (the specific
quantity this whole D-RES investigation targets) is not improved (+3%,
i.e. mildly worse), and the sign-flip rate -- the single most direct,
falsifiable test of whether this lever does what it was designed to do -- is
flat to worse in every individual seed, never better.

**Learned tau1 values** (relaxation timescale, init 5.0 steps): not
inspected in this pass -- if this thread is revisited, checking whether tau1
converged to something physically sensible (order ~15 steps, matching the
confirmed ~0.3s separation-burst width) vs. degenerated to a trivial extreme
(near-0 or very large) would help distinguish "the mechanism engaged but
didn't help" from "the mechanism never engaged at all" -- not done here
since the aggregate verdict (no sign-flip improvement in any seed) is
already conclusive enough not to warrant it.

**Reading:** the modest generic-NRMSE improvement in 2/3 seeds is most
plausibly just the effect of adding one extra conditioning channel/scalar
capacity to NNdyn and NNrec (more free parameters, slightly better generic
fit), NOT evidence that the Goman-Khrabrov lag-ODE structure is doing its
intended job of tracking an attachment-loss event. This is consistent with,
and now empirically confirms, the literature notes' own pre-registered null
reading: giving the dynamics an explicit memory/indicator channel does not
help if the actual bottleneck is the DECODER's inability to spatially
express a localized, sign-changing feature even when told (accurately or
not) that a regime transition is occurring. This directly motivates
STALL_LITERATURE_NOTES.md's item 3 (local/gated decoder near the flap) as
the next-in-line candidate if this line of investigation continues -- but
per that document's own reasoning, this null result narrows the diagnosis
usefully: the problem is very likely decoder-side locality, not
training-emphasis (residual-curriculum also failed) and not dynamics-side
memory (this lever also failed).

---

## CUMULATIVE SUMMARY -- STALL/SEPARATION INVESTIGATION (2026-08-22 to 2026-08-25)

**Starting point**: Phase 1 of this investigation physically CONFIRMED (not
assumed) that the long-standing D-RES residual is a real, localized,
transient flow-separation event at the flap -- not a diffuse smooth-fit
problem. At the gust+flap test case's gust peak, up to 23.35% of near-flap
nodes show FOM flow reversal (vs 0% at rest); the champion ROM represents
only 2.14% of this event and gets the local flow direction's SIGN wrong at
21.67% of near-flap points at the event's peak (`decomp_stall.py`,
`recon/analysis/MEANSPLIT_NOTES.md`'s STALL/SEPARATION HYPOTHESIS section).

**Four new-territory levers were designed, implemented, smoke-tested, and
run to full n=3-seed statistical completion, each targeting a different
axis never attacked by the five originally-closed D-RES levers:**

| lever | mechanism | axis attacked | verdict |
|---|---|---|---|
| Residual-curriculum weighting | loss reweighted by the model's own live detached residual | training emphasis (adaptive) | **LOSES cleanly** -- worse on every region/component/seed/power, sign-flip unimproved |
| Signal-rates (`--add-signal-rates`) | +2 input channels (Wdot, deltad) to the standard concat | missing-information (additive) | **LOSES** -- near/surface dynamic worse all 3 seeds; combined-NRMSE shows the same H-METRIC-style illusion as multiple shooting; sign-flip flat |
| Flap loss-weight (`--loss-weight-mode flap`) | loss reweighted by a fixed geometric flap-proximity mask | training emphasis (static) | **LOSES** -- 2/3 seeds lose on every axis including sign-flip; the 1/3 seed that looked dramatically promising in a premature single-seed preview did not replicate (documented process lesson on premature single-seed framing) |
| Sep-state (`--dyn-sep-state`) | Goman-Khrabrov lag-ODE attachment-state channel into NNdyn+NNrec | dynamics-side memory/regime-indicator | **No real effect** -- mild generic-NRMSE improvement in 2/3 seeds is not accompanied by ANY sign-flip improvement in any seed (one seed's sign-flip is bit-identical to the champion); net wash on the metric that matters |

**Combined with the five originally-closed D-RES levers** (decoder depth,
near-wall sampling, BFGS budget, multiple shooting, Neural-CDE rate-only
conditioning -- see the earlier sections of this file), **the champion
(`coral_o10_s0`: mean-split + CORAL shift-modulated SIREN decoder, omega0=10,
d_s=1, L6, uniform sampling, bfgs=2000) has now been attacked by NINE
independent, substantial levers spanning every axis this project could
identify** -- decoder architecture, decoder conditioning mechanism, point
sampling, optimizer choice/budget, training procedure (shooting), dynamics
conditioning mechanism (CDE rate-only), loss weighting (both static-
geometric and dynamic-adaptive), and dynamics-side regime memory -- **and has
not been beaten once**, on the metric that actually matters
(near/surface-region dynamic residual and, specifically for this second
round, the flap sign-flip/reversal-representation readout).

**Two consistent methodological lessons reinforced across this entire
campaign, worth restating plainly:**
1. Never trust combined/global NRMSE alone -- it has produced a misleading
   "win" reading at least three times now (multiple shooting, signal-rates,
   and marginally sep-state), always because the far-field/attached-flow
   majority of points dominates the point-count-weighted global metric while
   masking a flat-or-worse result in the near-flap region that is the actual
   target.
2. Never trust a single-seed preview as a verdict, even with a clear verbal
   caveat -- the flap loss-weight lever's s200 result looked like the best
   finding of the whole investigation and did not replicate at n=3.

**Implication for the thesis**: the evidence is now very strong -- from
nine structurally distinct angles -- that the residual is a genuine,
robust ceiling of this specific LDNet architectural class (global
coordinate-conditioned decoder + concatenation-or-simple-augmentation
dynamics conditioning), not an artifact of any one design choice tried so
far. The one remaining untried candidate, flagged consistently as the
most-informed next step by both the sign-flip-metric pattern here and the
original literature review, is a genuinely NEW architecture: a local/
spatially-gated decoder mechanism near the flap (STALL_LITERATURE_NOTES.md
section 9 item 3) -- ranked MAJOR cost, deliberately held back until the
cheaper items were exhausted, which they now are. This is a natural
stopping point to write up the D-RES/stall investigation as a well-
evidenced structural-limit finding for the thesis, with the local-decoder
idea noted as future work rather than pursued further in this campaign
without an explicit decision to invest in genuinely new architecture.

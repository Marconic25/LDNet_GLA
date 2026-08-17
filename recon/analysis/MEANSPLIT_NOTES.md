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

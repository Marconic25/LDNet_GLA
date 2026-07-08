# H-ARCH Depth Study (Thread D) — does adding layers help BFGS convergence / final error?

Status: IN PROGRESS (this header is updated as results land; see Results table below).
Date started: 2026-07-07.

## Question
Baseline field-LDNet (NNdyn 2×Dense(7,tanh)+head, NNrec 4×Dense(24,tanh)+head; 6 total
hidden layers) plateaus at train ~8.3e-3 (d_s=1) / ~1.02e-2 (d_s=10). H-OPT (4 restarts,
800 Adam, 4000 BFGS on d_s=10) proved "optimize harder" does NOT break the plateau
(train 1.021e-2, identical to the light run). Next candidate lever: CAPACITY VIA DEPTH.
Does escalating depth geometrically (×2 per rung, widths fixed 7/24, dyn:rec layer
ratio fixed 1:2) improve BFGS convergence and/or final test NRMSE, and where does the
ladder physically die (vanishing gradients / optimizer memory / walltime)?

## Optimizer facts that shape the study (from src/optimization.py — read, not modified)
- BFGS variant: **scipy `method='BFGS'` = FULL BFGS**, dense P×P float64 inverse-Hessian.
  - Memory: O(P²) → 8·P² bytes for H, ~3-5 P² temporaries at peak during the update.
  - Per-iteration cost: O(P²) linear algebra + 1-2 loss+grad evals (dominated by the
    latent-ODE forward/backward for our P range).
  - `gtol=1e-100, tol=1e-100` → BFGS never stops on tolerance; it stops on maxiter
    (budget) or line-search failure ("precision loss"). The recorded scipy message
    per run tells which — that IS the convergence-stall signal.
- Adam phase: full-batch, lr 1e-2, `restarts` seeds, best-val winner goes to BFGS.

### Predicted parameter counts and full-BFGS Hessian memory (d_s=10; d_s=1 slightly less)
P(dyn) = (d_s+7)·7+7 + (dyn_layers−1)·56 + 7·d_s+d_s ; P(rec) = (d_s+8)·24+24 + (rec_layers−1)·600 + 75

| L (total) | dyn×7 | rec×24 | P (ds=10) | 8·P² (H alone) | peak (~4×) |
|-----------|-------|--------|-----------|----------------|------------|
| 6         | 2     | 4      | 2,593     | 54 MB          | 0.2 GB     |
| 12        | 4     | 8      | 5,105     | 209 MB         | 0.8 GB     |
| 24        | 8     | 16     | 10,129    | 0.82 GB        | 3.3 GB     |
| 48        | 16    | 32     | 20,177    | 3.3 GB         | 13 GB      |
| 96        | 32    | 64     | 40,273    | 13.0 GB        | 52 GB      |
| 192       | 64    | 128    | 80,465    | 51.8 GB        | 207 GB     |
| 384       | 128   | 256    | 160,849   | 207 GB         | 828 GB     |

→ With the 64 GB job allocation the **full-BFGS memory wall is predicted at L=192**
(H alone ~52 GB; update temporaries push peak past 64 GB). L=96 is borderline
(~52 GB peak). Nodes have 528 GB / 1 TB physical, but the job requests mem=64gb.
An L-BFGS(m) variant would be O(mP) and dodge this wall entirely — documented here
because the study result "depth dies at L≈192 by optimizer memory" is a property of
FULL BFGS, not of the architecture.

### Trainability prediction (plain deep tanh, Glorot init, NO residuals/scaled init)
Plain tanh stacks lose signal/gradient norm multiplicatively with depth; beyond
~10–20 hidden layers, forward activations saturate/collapse and gradients vanish.
Expect degradation visible at L=24–48 and outright untrainability (loss stuck at
init-level, BFGS line-search failure within few iterations) at L≥96 — unless BFGS's
curvature scaling partially rescues it. This is exactly what the ladder measures.
Per the design brief, ONE clearly-labeled variant arm (e.g. scaled init) MAY be added
if rung 24/48 already collapses — only as an extra arm, never replacing the plain family.

## Experiment design
- Data (as-is, cluster): train `data/FIELDS_div_train.h5` (15 sims), valid
  `data/FIELDS_div_valid.h5`, test `data/FIELDS_Cc060.h5`. Subsample 1024 pts (default).
- Ladder (total hidden layers): 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144,
  12288, 24576, 49152, 98304 (cap <100,000). dyn = L/3, rec = 2L/3 (baseline ratio).
- Per rung: d_s = 1 and d_s = 10.
- Budget per run (matches baseline light run except longer BFGS, deliberate):
  `--restarts 2 --adam 300 --bfgs 4000 --output-nl linear --log-every 1`.
- Reference numbers (baseline arch, restarts=2/adam=300/bfgs=2000, linear):
  d_s=1/5/10 → train 8.28e-3 / 9.97e-3 / 1.02e-2; Cc-test NRMSE 1.39e-2/1.64e-2/1.59e-2.
  L=6 is RE-RUN inside the ladder with bfgs=4000 for identical logging.
- Stop rules: (a) both d_s ≥2× baseline train loss (or NaN) for 2 consecutive rungs;
  (b) crash (OOM/TF graph) — one retry, second crash = the wall; (c) single run >
  ~10 h wall; (d) NaN/divergence — NOTE: seeds fixed → training deterministic, so the
  "one retry" for NaN is provably futile; NaN counts as "bad" under (a) instead
  (written justification, per design-brief allowance).

## Implementation (all changes confined to recon/; src/ and clean/ untouched)
- `recon/train_fields.py`: new flags `--dyn-layers/--dyn-width/--rec-layers/--rec-width`
  (defaults 2/7/4/24 = EXACT original architecture, same layer order → same weight-init
  RNG draws for a given seed; other campaign runs unaffected) and `--log-every`
  (default 10 = original print/record cadence byte-identical; 1 = full history).
  New `RecordingOptimizationProblem(optimization.OptimizationProblem)` subclass records
  per-iteration train+val loss (train at BFGS iterations comes free from the cached
  last function evaluation), per-function-evaluation train loss (`kind=fev` rows),
  wall time per row, and the scipy BFGS result (nit/nfev/status/message/final grad
  inf-norm). The optimization trajectory itself is UNCHANGED (variables left at the
  last evaluated point, exactly like the base class).
  Per run artifacts (in `<model_dir>/latent_<ds>/`): `loss_history.csv`
  (phase,kind,step,train_loss,valid_loss,wall_s; phases adam_r0, adam_r1, bfgs),
  `run_info.json` (arch, param counts, phase wall-clock, BFGS termination, budgets,
  argv), `config.json` (now includes architecture), `metrics.json` (as before).
- `recon/depth_driver.py`: resumable ladder driver (runs inside one PBS job, exits
  before walltime, next chain link resumes; per-run status.json + attempts.txt;
  STOP file records the ladder-stop reason).
- `recon/depth_index.py`: idempotent rebuild of `models/depth_study/index.csv` from
  artifacts (one row per run: run_id, layers, widths, d_s, param counts, budgets,
  final train/val loss, per-field + combined test NRMSE, phase wall-clocks, BFGS
  nit/nfev/message/grad-norm, status, stop_reason, model_dir).
- `recon/cluster/depth_ladder.pbs`: cpu queue, select=1:ncpus=8:mem=64gb, walltime 24 h
  (mem=64gb sized for the L=96/ds=10 full-BFGS Hessian; nodes have 528 GB–1 TB).
- `recon/cluster/submit_depth_chain.sh`: N links chained 1-wide with afterany
  (queue politeness; 88-job extraction chain untouched).
- Model dirs: `recon/models/depth_study/L{total}_ds{K}/` (new tree; existing models/
  dirs untouched).

## Smoke test (login node, light)
- PENDING — recorded here when done: default-flag param counts must be 127 (NNdyn)
  / 2115 (NNrec) at d_s=1; deep-flag (4×7+8×24) counts 239 / 4515.

## Timing calibration
hopt_ds10 (L=6 arch, d_s=10, restarts=4/adam=800/bfgs=4000, 8 cores): 3 h 50 m total.
→ this study's budget (restarts=2/adam=300/bfgs=4000) ≈ 2.5–3 h at L=6/ds=10; cost
scales ~linearly with total layers (both NNdyn-in-ODE-loop and NNrec-over-points are
depth-dominated) → L=12 ≈ 2×, L=24 ≈ 4×, ... — UNLESS deep runs die early in the
line search, which makes them cheap. Measured walls land in the Results table.

## Results (filled as rungs complete — all numbers also in depth_study_index.csv)
| L | d_s | P | train | val | test NRMSE | BFGS nit/nfev | BFGS stop message | wall h | status |
|---|-----|---|-------|-----|-----------|----------------|-------------------|--------|--------|
| (pending) |

## BFGS convergence observations
(pending — plateau level vs depth, iterations-to-stall, gradient inf-norm at
termination, line-search failure modes; full per-iteration curves in
recon/analysis/depth_study/loss_history_L*_ds*.csv)

## Where the ladder stopped and why
(pending)

## Verdict
(pending)

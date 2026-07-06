# LDNet field-reconstruction — running log

Separate from the controller work in `clean/`. Goal: reconstruct unsteady CFD fields
(vx, vy, pressure) from a small latent state with LDNet and compare side-by-side against the
OpenFOAM ground truth (FOM) — paper-style images, a video, and the NRMSE-vs-latent-dim curve.

Reuses the field-capable LDNet machinery already in `src/` (`sensitivity_latent.py`,
`utils.process_dataset`, `aerodynamics/model.py`) — the controller work had collapsed it to a
single point + scalar forces. Here we point it at full fields instead.

## Folder layout
```
recon/
  RECON_NOTES.md          # this log
  extract_fields.py       # cluster-side: OpenFOAM -> gridded (vx,vy,p) -> FIELDS_*.h5
  train_fields.py         # field-LDNet training (adapts src/sensitivity_latent.py)
  viz_fields.py           # FOM vs LDNet images + video + NRMSE-vs-latent curve
  data/                   # FIELDS_{train,valid,test}.h5
  models/latent_{1,5,10}/ # trained weights + config.json
  results/                # png snapshots, mp4/gif, nrmse_vs_latent.png
```

## Phase 1 — cluster investigation (dataset_v5)
Access (per `server/guida_cluster_gpu.txt`): VPN `gp-dmat-saml.vpn.polimi.it`, then
`ssh u10677113@10.78.18.100`. Dataset at `/work/u10677113/NACA2312/dataset_v5/sim_*`.

Questions to answer before extraction:
- [ ] OpenFOAM time directories present (`0/`, `0.00x/` with `U`, `p`)? reconstructed vs `processor*/`?
- [ ] `postProcessing/`, `VTK/`, or `*.foam` present?
- [ ] Mesh: 2D? domain extent + resolution; same mesh across sims?
- [ ] Number of written timesteps + write interval vs CSV dt?
- [ ] Size per sim and for the target subset (download sizing).

### Findings (2026-06-29) — DECISIVE: fields must be regenerated, but it's feasible

Cluster reachable (key-based SSH, VPN up). `dataset_v5` has 146 `sim_*` dirs.

**No spatial field data exists anywhere:**
- Each `sim_*/` holds only `structural_trajectory.csv`, `postProcessing/forces/` (forces.dat
  only — no sampled planes/surfaces/volumes), `figures/gust_response.png`, `cosim_state.json`,
  `log.cosim_driver`, `sim_params.csv`, `job.pbs`. No time dirs, no VTK, no `*.foam`.
- `pca_results_v5/` is PCA on **(CL,CM)/(h,alpha) coefficients**, not on flow fields
  (`pca_weekend.py` docstring: prove the W,δ→aero→struct chain is 2D).
- The full CFD case ran on `/scratch_local/$USER/sim_*` which is **auto-deleted after 30 days**
  (`job.pbs`: "Full case on: $SCRATCH (auto-deleted after 30 days)"). Sims ran ~May 21; today
  Jun 29 → scratch gone. The rsync back to /work explicitly **excluded** `processor*` and field
  time dirs — only forces/CSV/figures were kept.
- No extracted field `.npy` were persisted: `NACA2312/data/GLA/` has only `timeseries/` CSVs +
  metadata; no `fields/`, no `mesh_*.npy`.

**Regeneration IS feasible — everything needed survives on /work:**
- `cosim_main/` = full OpenFOAM case: mesh `constant/polyMesh`, `system/` dicts,
  `constant/dynamicMeshDict` + `wingMotion.dat`/`flapMotion.dat` (prescribed-motion path),
  `0.orig/`, and `cosim_driver.py` (55 KB). OF7 container `/work/u10677113/of7.sif`.
- A checkpoint (decomposed `processor*/` at CFD t=3.0s) is the warm-start state used by every
  sim's `job.pbs`. Per-sim gust/flap params are in `metadata_v5.csv` (R, T_g, W_g0, flap law).
- **Existing field extractor `extract_data.py --fields --reconstruct`** already does the whole
  job: `reconstructPar`, pyvista `OpenFOAMReader`, slice z=0.125 (mid-plane), crop
  x∈[-1,4] y∈[-1,1], and saves `fields/sim_XXX.npy` shape **[150, N_pts, 3]** where channels are
  `[Ux, Uy, p]` (= vx, vy, pressure — exactly the target), plus `mesh_points.npy [N_pts,2]`,
  `mesh_triangles.npy`, `field_times.npy`. `extract_fields.pbs` is the batch wrapper (it looped
  over scratch right after each run — never got persisted).

**Conclusion / plan impact:** to get fields we must **re-run OpenFOAM** per target sim
(deterministic from checkpoint + metadata params), writing field time dirs, then run the
existing `extract_data.py --fields` ON THE SAME NODE before scratch is cleaned, and copy the
compact `.npy` to /work. Cost: each cosim re-run is t=3→6 s CFD at dt=7e-5 (~43k steps) on
16 cores — `job.pbs` budgets 6 h walltime. So regenerating N sims = N multi-hour batch jobs.
This makes the *number of sims to regenerate* a compute-budget decision (revisit with user).

Two regeneration paths (Phase 2 detail):
1. **Cosim replay** (validated): re-run `cosim_driver.py --from-checkpoint` with the sim's
   gust/flap params → reproduces the exact published trajectory + fields.
2. **Prescribed-motion CFD replay** (cheaper/simpler): drive the existing dynamic mesh from the
   recorded `structural_trajectory.csv` (h,α as wingMotion.dat, δ as flapMotion.dat) + gust
   inlet, run `pimpleFoam` writing fields. No structural solver. Fields are fully determined by
   prescribed motion + gust, so equally valid for reconstruction.

## Phase 2 — extraction (pilot: sim_A_025_test, cosim replay)

Decisions (user): pilot ONE sim first; cosim replay (validated path).

### Gotchas found before launching
- **`purgeWrite 2`** in `cosim_main/system/controlDict` discards all but the latest 2 time dirs
  → must set **`purgeWrite 0`** on scratch so per-window fields persist. `cosim_driver`'s
  `update_control_dict` only rewrites startFrom/startTime/endTime/writeInterval, not purgeWrite,
  so a one-time edit sticks.
- **`metadata_v5.csv` is STALE / inconsistent** with the actually-run sims: its `sim_A_025_test`
  row says R=0.3478/T_g=0.462/W_g0=27.8, but the in-dir `sim_params.csv` + `job.pbs` (what
  truly ran) say R=0.143/T_g=1.121/W_g0=11.46. → **authoritative re-run params come from each
  sim's own `dataset_v5/<sim>/job.pbs`**, NOT the global metadata. We also decouple extraction
  from metadata: LDNet inputs come from the saved `structural_trajectory.csv` (has W_gust,delta
  columns); the field extractor only needs the reconstructed OF case.
- `extract_data.py main()` skips fields if `extract_timeseries` throws → so we use a **standalone
  field-only extractor** (`recon/extract_fields.py`) to avoid that coupling and the metadata.
- Moving mesh morphs (same topology, points displaced; displacements tiny: h~mm, α~0.06°), so a
  crop mask + reference mesh_points computed once is valid across snapshots (same as the original
  extractor). Fields saved on the reference grid.
- Verified: checkpoint OK (16 processor dirs + state, t_cur=0, t_end=3.0); `cosim_env` has
  pyvista 0.47.1 / numpy 2.4.3 / scipy 1.17.1; `of7.sif` present (700 MB).

### sim_A_025_test authoritative cosim args (from its job.pbs)
```
cosim_driver.py --np 16 --window 50 --dt 7e-05 --t-end 3.0 --from-checkpoint \
  --gust-w0 11.4599 --gust-t-start 0.0 --gust-t-end 1.120581 \
  --delta-times 0.0 3.0 --delta-angles 0.0 0.0
```

### Approach
`recon/cluster/pilot_A025.pbs` (parametric via `T_END`): copy `cosim_main`+checkpoint to scratch,
set purgeWrite 0, run cosim, `reconstructPar`, run `recon/extract_fields.py` →
`fields_<sim>.npy [T,Npts,3]=(Ux,Uy,p)` + `mesh_points.npy` + `mesh_triangles.npy` +
`field_times.npy`, copy to `/work/u10677113/NACA2312/recon_fields/<sim>/`, then scp down to
`recon/data/`. Smoke first (`T_END=0.05`, ~30 min) to validate the full chain, then `T_END=3.0`.

### Results
- **Smoke (job 24213, T_END=0.05) SUCCEEDED** — full chain validated. Output
  `recon_fields/sim_A_025_test_T0p05/`: `fields_sim_A_025_test.npy (17, 11075, 3)` float32 =
  (Ux,Uy,p), `mesh_points (11075,2)`, `mesh_triangles (~18k)`, `field_times (17)`,
  `structural_trajectory.csv`. Crop OK (x∈[-0.98,3.98], y∈[-0.97,1.0]). Values physical:
  Ux mean ~64 (wake of 80 m/s), p within ±q∞≈3920 Pa, no NaNs. Window write spacing ≈0.0035 s.
- Confirmed: cosim's **per-window reconstruct already populates the case root**, so with
  `purgeWrite 0` the full trajectory persists in root and the extractor reads it directly
  (the explicit final `reconstructPar` is a redundant safety net).
- **Local rendering works** (WSL python: numpy 1.26 / matplotlib 3.6 / scipy 1.11). Repo files
  are written via WSL (`wsl.exe -d Ubuntu`) since Git Bash sees the WSL share read-only; the
  cluster is reachable from WSL with the same key. `recon/viz_fields.py snapshot` →
  `recon/results/fom_smoke.png`: clean NACA2312 (main+flap) with LE stagnation, suction peak,
  wake deficit, vy dipole. Renderer uses the native triangulation (airfoil hole respected).
- Cost: smoke 0.05 s took ~45 min, dominated by fixed setup (rsync case + 16 checkpoint
  processor dirs + container cold starts). Full 3 s = same setup + ~857 windows; production
  ran 3 s within 6 h on this setup, so feasible. **Full run job 24216 (T_END=3.0, 12 h
  walltime) RUNNING.** Per-sim cost ≈ a few hours → number of sims is a compute decision.

### Tooling built
- `recon/extract_fields.py` — cluster-side standalone field extractor (validated).
- `recon/cluster/pilot_A025.pbs` — cosim replay + extract, parametric `-v T_END`.
- `recon/viz_fields.py` — `snapshot` (FOM panels) + `video` (animated channel) DONE & tested on
  smoke data → `recon/results/fom_smoke.png`, `fom_smoke_vx.gif`. `compare`/`nrmse-curve` need
  the trained LDNet (Phase 4).
- `recon/build_fields_h5.py` — DONE & validated: npy + structural_trajectory → FIELDS h5 in the
  LDNet schema (points, times, input_signals, output_fields[vx,vy,p], output_signals, families),
  resampling each sim onto a common time grid so multiple sims stack. Local test → FIELDS_smoke.h5
  (1,16,11075,3), no NaNs.

### Phase 2b — FULL pilot run COMPLETE (job 24216)
- `sim_A_025_test` full 3 s: `fields (860, 11075, 3)` (Ux,Uy,p), times t_rel 0→3 s, 114 MB.
  Wall time ≈ 2 h on 16 cores (the poller's "time" column was CPU-time-used ≈16× wall, not
  elapsed — the job was never 18 h). Values physical (gust excursions larger than smoke).
- Downloaded → `recon/data/sim_A_025_test/`. Built `recon/data/FIELDS_A025.h5`
  (1,150,11075,3) via build_fields_h5 (resampled 860→150 times).
- Deliverables: `recon/results/fom_A025_gustpeak.png` (4-panel snapshot at gust peak),
  `recon/results/fom_A025_speed.gif` (full-trajectory |v| animation, stride 5 → 172 frames).
- **Per-sim cost ≈ 2 h wall.** cpu queue has 5 nodes mostly free → can fan out ~4-5 re-runs in
  parallel. Number of training sims is the remaining compute-budget decision (next).

### Local env note
WSL python (system) + `--user --break-system-packages h5py 3.16` covers numpy/scipy/matplotlib/
h5py for assembly+viz. TensorFlow training stays on the cluster GPU (Keras 2.14 container).

## Phase 3 — training (`recon/train_fields.py`)
Adapts `src/sensitivity_latent.py`: targets `output_fields` (vx,vy,p), NNrec output dim=3,
auto-computes normalization from data, latent sweep, NRMSE-vs-latent summary. Reuses
`src/utils.process_dataset` + `src/optimization` (uploaded to cluster `/work/.../NACA2312/src`).
Runs in the TF GPU container via `recon/cluster/train.pbs`.

- **BUG FIXED:** `delta` (flap) is identically 0 for gust-only family A → zero normalization
  range → divide-by-zero → NaN loss → NaN recon → pearsonr crash. `_rng()` now guards any
  zero-range channel (constant → maps to 0); `safe_rho()` guards degenerate correlation.
- **Validated end-to-end on GPU** (overfit pilot, latent 10, 60 Adam/150 BFGS): combined test
  NRMSE 0.052 (undertrained — captures gross field, smears near-airfoil detail). Code path OK:
  evolve+reconstruct over the real grid, weights+config+metrics saved.

## Phase 4 — visualization (`recon/viz_fields.py` + `recon/reconstruct_fields.py`)
- `reconstruct_fields.py` (TF container, `recon/cluster/recon.pbs`): trained model + FIELDS h5
  → `rom_<name>.npy` / `fom_<name>.npy` / points / times on the full grid.
- `viz_fields.py compare` / `compare-video`: FOM | LDNet | |error| panels (rows vx,vy,p) and
  side-by-side animation. VALIDATED → `recon/results/compare_pilot_gustpeak.png` (rough, from the
  undertrained pilot model; pipeline confirmed).

## Bulk re-run debugging (2026-06-30)
First 7-job batch (24251-57) FAILED (empty outputs): (a) all parallel jobs wrote the SAME
`$HOME/bin_of7` OF wrappers → race corruption; (b) `field_run.pbs` had dropped the explicit
`reconstructPar` that the validated pilot used; (c) PBS `-o` log invisible. FIXED: per-job
wrapper dir on scratch, restored `reconstructPar`, persistent `OUT_PERSIST/run.log`. Re-test on
`sim_A_000` (24275) started cleanly. Relaunched the set: 24275 (A_000), 24277-282
(A_002/005/010/012/017 train, A_020 val). ~2 h each, ~3-4 h wall for the batch.

## Status / next
Field set running. When complete → build FIELDS_{train,valid,test}.h5 (6 train + A_020 val +
A_025 test) → full training (latent 1/5/10, BFGS ~2000) → final FOM-vs-LDNet figs + NRMSE curve.

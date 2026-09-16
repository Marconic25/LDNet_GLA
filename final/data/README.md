# `final/data/` — dataset_v6 extraction and assembly

Three scripts, run in this order (see also `final/README.md` §"Ordine di esecuzione"):

```
1. (per sim, inside Agent A's co-simulation driver / window loop)
   reconstructPar -time T          # rebuild ONE time directory
   extract_fields_step.py --case C --time T --out O --name N     # slice + crop it
   rm -rf <processor time dirs for T>                            # purge before the next window
   ... repeat once per coupling window ...
   extract_fields_step.py --finalize --out O --name N            # stack into fields_<N>.npy

2. preprocess_GLA_v6.py   --matrix final/design/run_matrix.csv --root <dataset_v6> --out-dir final/data
   build_fields_h5_v6.py  --matrix final/design/run_matrix.csv --root <dataset_v6> --out-dir final/data

3. training reads final/data/GLA_{train,valid,test}.h5 and final/data/FIELDS_{train,valid,test}.h5
```

## 1. `extract_fields_step.py` — incremental field extractor

Why it exists: dataset_v6 uses a 29-timestep coupling window at `dt=3.16e-5 s`
(~3 274 windows per 3 s run). The old pattern — keep every processor time
directory, run one `reconstructPar` at the end, then extract — would need ~419 GB
of node-local scratch per run (see `final/README.md` / `ESTIMATE.md` §5). Instead,
the driver reconstructs, slices, and deletes ONE time directory per window, calling
this script once per window.

CLI contract (the driver depends on this exactly):

- `--case C --time T --out O --name N` — read the one reconstructed time directory
  `T` present in case `C`, slice+crop it exactly like `recon/extract_fields.py`
  (same `CROP_X/Y`, same `z=0.125` mid-span plane, same `(Ux,Uy,p)` channel order
  and `float32` dtype), and write `O/chunks/f_<key>.npy` — shape `(Npts,3)`. `<key>`
  is a canonical re-encoding of `T` (round-trips through `float()` exactly), not a
  literal copy of the `--time` string. The first call for a given `O` also writes
  `O/mesh_points.npy`, `O/mesh_triangles.npy`, and an internal `O/_crop_mask.npy`
  that later calls reuse so every snapshot selects the same physical points.
- `--finalize --out O --name N` — stack `O/chunks/f_*.npy` in ascending **numeric**
  time order into `O/fields_<N>.npy` `[T,Npts,3]` + `O/field_times.npy` `[T]`, then
  remove `O/chunks/`. Numeric order matters: OpenFOAM's `general` time format at
  `timePrecision 6` mixes fixed and scientific notation across a run (e.g.
  `3.16e-05` early, `0.0001264` later), and a plain string sort gets that backwards.
  Tolerant of a few missing/malformed chunks — logs how many and continues, so
  losing one snapshot out of ~860 never aborts a multi-hour run.
- `--selftest` — offline check of the time-parsing/round-trip/numeric-sort logic
  and of `--finalize`'s tolerance, on synthetic data. No pyvista, no OpenFOAM case
  needed. Run it any time with `python3 extract_fields_step.py --selftest`.

Requires `pyvista` only for the `--case/--time` extraction path (not for
`--finalize` or `--selftest`); on the cluster that means the `~/cosim_env` venv.

## 2/3. Ragged `t_end` — the decision

v5 ran every sim for a flat 3.0 s, so every `structural_trajectory.csv` had the
same row count at the same sample times, and the old scripts could reuse one sim's
raw time column as everyone's `times` dataset. v6's `run_matrix.csv` gives each row
its **own** `t_end` (`Tg`-dependent — `final/ESTIMATE.md` §3: family A ~1.65 s avg,
B ~1.80 s, Cc ~1.95 s, vs. a flat 3.0 s in v5), so raw per-sim sample counts differ
and cannot be stacked into one array as-is.

**Decision: resample per run onto one shared, absolute time grid, holding the last
recorded value past each run's own end (never truncate to the shortest run, never
extrapolate).**

- `T_MAX = max(t_end)` over the **whole** `run_matrix.csv` (every split, not just
  the file currently being written). `preprocess_GLA_v6.py` and
  `build_fields_h5_v6.py` both compute it this way, so `GLA_*.h5` and `FIELDS_*.h5`
  share one absolute time origin and span — required because a downstream joint
  loss could otherwise compare a loads sample at time `t` against a field frame
  that is actually at a different real time.
- Each run is linearly interpolated from its own recorded `(t, values)` onto
  `t_common = linspace(0, T_MAX, n_times)`. Query points past that run's own last
  recorded sample are clipped to that boundary *before* evaluating (`resample_hold`
  in both scripts), which holds the last value rather than extrapolating.
- Rejected: **truncating every run to the shortest run's `t_end`** — the shortest
  rows are the low-`Tg` A/B runs (~1.2 s); truncating everyone to that would discard
  a large fraction of the 57 Cc/MPC rows, the campaign's highest-value data
  (`final/README.md`: "le 57 run Cc-MPC *sono* la verifica MPC-sul-FOM").
- Rejected: **linear extrapolation** past a short run's last state — every row's
  `t_end` is deliberately `Tg` (or the flap-motion end) plus a settling buffer, so
  the state at `t_end` is already past the transient; extrapolating risks
  manufacturing drift in exactly the free-running regime already flagged unreliable
  for this model family (project memory: `ldnet-latent-instability`). Holding the
  last value adds no synthetic dynamics — it repeats a real, already-settled
  sample.

The two output files intentionally do **not** share `n_times`:
`preprocess_GLA_v6.py` defaults it to the largest raw row count among the runs
that loaded cleanly (keeps v5's native per-coupling-window resolution for the
loads-only signals — cheap, `N×T×6` floats), while `build_fields_h5_v6.py`
defaults to 150 as `recon/build_fields_h5.py` always has (fields are the memory-
heavy `N×T×Npts×3` array). They never needed to match in v5 either —
`output_signals` inside the FIELDS file is documented there as "compat; not the
field target". Only the time *axis* (origin, span, hold-padding convention) needs
to agree, and it does.

## Run/split selection

Both `preprocess_GLA_v6.py` and `build_fields_h5_v6.py` take `--matrix
final/design/run_matrix.csv` as the authoritative list of runs and splits (columns:
`sim_id,family,split,W0,Tg,r_g,controller,K_ff,t_d,delta_pk,rate_max,R_star,t_end`),
instead of parsing `sim_{family}_{idx}_{split}`-style directory names the way v5's
`data/preprocess_GLA.py` did. A run's directory is looked up as `<root>/<sim_id>`,
falling back to `<root>/sim_<sim_id>` (v5's naming) if the bare id isn't found —
**this fallback is an assumption**, made because `final/design/gen_matrix.py` did
not exist yet at the time these scripts were written and the actual v6 on-disk
naming convention could not be confirmed locally; verify it against the real
`dataset_v6` layout once available and adjust `find_sim_dir()` if it differs.

Every matrix row is validated (CSV/fields present, right shapes, no NaN/Inf,
monotonic time, `|delta|` in range, consistent mesh `Npts` across runs) and
skipped/corrupt runs are printed explicitly by `sim_id` and reason — never
silently dropped.

## Assumptions worth flagging

- **Mesh point ordering is stable across separate `extract_fields_step.py`
  invocations** (i.e. across process restarts, not just within one run of
  `recon/extract_fields.py`'s loop). This extends an assumption `recon/
  extract_fields.py` already makes within a single process/reader instance
  (constant mesh topology → one crop mask valid for the whole run) to also hold
  across the many short-lived processes the v6 driver launches. It could not be
  verified end-to-end without a live OpenFOAM case; if a run's field data comes out
  visibly scrambled, this is the first assumption to check.
- `find_sim_dir()`'s directory-naming fallback, above.
- `preprocess_GLA_v6.py`'s CSV validation (NaN/Inf/monotonic-time/`|delta|<=25`)
  mirrors `data/preprocess_GLA.py`'s `validate_csv`, extended to reject (not just
  flag) NaN/Inf rows, since those cannot be safely interpolated through.

## `valid_mask` — do not train on the padding

Both `preprocess_GLA_v6.py` and `build_fields_h5_v6.py` write a
`valid_mask (N, T)` boolean dataset alongside the signals. `valid_mask[i, k]`
is `True` where run `i` has real data at `times[k]`.

It exists because the shared time grid spans `T_MAX = 2.4 s` (the longest run in
the campaign) while the mean run is `1.83 s`. Everything past a run's own
`t_end` is held-last-value padding, and that padding is **fabricated**: the real
aeroelastic system is still ringing down there, not sitting flat. Measured on
`design/run_matrix.csv`:

| | mean padding | worst run |
|---|---:|---:|
| campaign-wide | 23.6% | — |
| family A | 31.4% | 48.7% |
| family B | 25.0% | 25.0% |
| family Cc | 19.4% | 37.5% |

Training unmasked on that teaches the model that the response goes constant —
a wrong dynamic, learned from roughly a quarter of the samples. Weight the loss
by `valid_mask` (or slice to it) in every training script that consumes these
files.

The two builders compute the mask consistently, and the fields builder is the
stricter of the two: a sample is valid only up to the earlier of the run's last
field snapshot and its last CSV row, so a run whose extraction was truncated
does not get credit for signal rows it has no fields for.

Both scripts print the padding fraction per split when they write, so the number
is visible at build time rather than buried here.

### Why not avoid the padding instead

Three alternatives were considered and rejected:

- **Truncate every run to the shortest.** Would cut the whole campaign to the
  1.23 s of the shortest family-A run and gut the 57 Cc/MPC rows, which are the
  highest-value data in the dataset.
- **Extrapolate past each run's end.** Manufactures drift in exactly the
  free-running regime this project has already found to be unreliable.
- **A per-run time grid.** Architecturally fine — the LDNet consumes `dt`
  explicitly, scaled by `dt_ref = 0.002 s` — but every existing training script
  assumes a fixed `T` across samples, so it would mean changing the training
  code as well as the data. Worth revisiting if the mask proves awkward.

Holding the last value and flagging it keeps the array shapes v5-compatible and
pushes the decision to the consumer, which is the only one of the four options
that does not either destroy data or silently invent it.

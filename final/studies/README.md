# final/studies/ — the v6 (dataset_v6) study chain

Everything the v5 thesis campaign in `light/` did against a model trained on
`dataset_v5`, re-run against a model trained on `dataset_v6` (new numerics:
`dt = 3.16e-5`, coupling window 29, Courant ~ 1) — without touching the v5
archive under `light/`. `light/latex/chapter3.tex` still reports the v5
numbers; nothing here changes that until a human decides to swap them in.

## Design: import, don't fork

Most of `light/`'s control/integration code is dataset-independent
(`structure.py`, `ldnet_aero.py`, `optimal.py`, `light/noise/harness_noise.py`,
`light/noise/controllers_ref.py`): it takes a model directory or an `aero`
object as a parameter and never hardcodes a path. Every script here imports
that code from `light/` — it is never copied — so the v5 and v6 campaigns
cannot silently diverge in their physics or control law.

`paths.py` is the one file where v6 differs from v5. It:
1. puts `light/` and `light/noise/` on `sys.path`, so `import structure`,
   `from ldnet_aero import LDNetAero`, `from optimal import ...`,
   `import harness_noise`, `from controllers_ref import ...` all resolve
   regardless of the caller's working directory;
2. exposes `V6_MODEL_DIR`, `V6_DATASET_ROOT`, `V6_TRAINING_DATA_ROOT`,
   `V6_RESULTS_ROOT`, `V6_IMG_DIR` — each overridable by environment
   variable (`LDNET_V6_MODEL_DIR`, etc.), each defaulting to a v6 sibling of
   the corresponding v5 location (e.g. `clean/models_rollout_v6/latent_10`,
   next to `clean/models_rollout/latent_10`);
3. provides `results_dir(name)` (-> `final/results_<name>/`) and `img_dir()`
   (-> `final/images/`);
4. provides `load_v6_harness()` — see below.

Two different retargeting mechanisms are needed, because the two v5 entry
points disagree on how their model path is set:

- **`light/run.py`** already reads `MD_OVERRIDE` from the environment
  (`os.environ.get('MD_OVERRIDE', <v5 default>)`). Retargeting it is a pure
  env-var change, so **`run_v6.py` does not fork it**: it sets
  `MD_OVERRIDE=paths.V6_MODEL_DIR` (via `setdefault`, so an explicit
  `MD_OVERRIDE` from the caller still wins) and then does
  `from run import simulate, metrics, gust, CLTRIM, ...`, before light/run.py's
  module-level code executes (env var must be set *before* the `import run`
  line, since light/run.py builds its `LDNetAero` at import time). Every
  downstream script that used to `import run as Rn` now does
  `import run_v6 as Rn` and needs no other change.
- **`light/noise/harness_noise.py`** computes its model path from its own
  `__file__` with no override hook at all. Forking it just to parameterize
  one path would duplicate ~170 lines of verified rollout/metrics logic —
  exactly what "import, don't fork" exists to avoid. Instead,
  `paths.load_v6_harness()` imports it unchanged and monkey-patches the
  already-imported module's `aero`, `CLTRIM`, `LAM`, `MD` globals. This
  works because `rollout()`/`metrics()` read those as module globals *at
  call time*, and because Python caches modules by name, so every other
  file that does `import harness_noise as H` in the same process
  (`controllers_ref.py`, `e2_combo.py`, ...) sees the same patched object.
  Call `paths.load_v6_harness()` before constructing any controller — some
  (`controllers_ref.MPCConstRef.__init__`) read `H.aero._num_z` at
  construction time and would otherwise bake in the v5 latent dimension.

Every forked file in this directory carries a header comment naming its
`light/` origin and listing exactly what changed (in every case: the
import/patch mechanism above, plus the output directory). No control law,
sigma model, sensor-fusion, or MPC-cost logic was touched.

## Files

| File | Forks | Backs |
|---|---|---|
| `paths.py` | n/a (v6-specific) | — |
| `run_v6.py` | `light/run.py` (env var only, not forked) | — |
| `cs25_combo_study.py` | `light/tests/cs25_combo_study.py` | `tab:cs25_grid` (raw data) |
| `cs25_combo_plots.py` | `light/tests/cs25_combo_plots.py` | `tab:cs25_grid` (summary.md table) |
| `cs25_thesis_figs.py` | `light/tests/cs25_thesis_figs.py` | `fig_ch3_trace_W30Tg04`, `fig_ch3_trace_W30Tg07`, `fig_ch3_trace_W10Tg07`, `fig_ch3_trace_W20Tg07`, `fig_ch3_envelope` |
| `noise_white_combo.py` + `plots_noise_white_combo.py` | `light/noise/{noise_white_combo,plots_noise_white_combo}.py` | `fig:noise_white` |
| `noise_calib_combo.py` | `light/noise/noise_calib_combo.py` | `fig:noise_bias_timing` (bias/gain half) |
| `noise_timing_combo.py` | `light/noise/noise_timing_combo.py` | `fig:noise_bias_timing` (shift/refit half) |
| `plots_thesis_bias_timing.py` | `light/noise/plots_thesis_bias_timing.py` | `fig:noise_bias_timing` |
| `e2_combo.py` | `light/noise/e2_combo.py` | multi-cell generality check (home cell W30/Tg0.4) |
| `e2_combo_cells.py` | `light/noise/e2_combo_cells.py` | multi-cell generality check (W10/Tg0.7, W30/Tg0.7) |
| `run_study.pbs` | n/a (v6-specific, PBS launcher) | — |
| `run_all.sh` | n/a (v6-specific, chain sequencer) | — |

`cs25_combo_plots.py` is not in the task's original file list but is
included because `cs25_combo_study.py` alone only writes raw
`traces_W*.npz`; the actual per-cell table that `tab:cs25_grid` is
transcribed from (`summary.md`) is produced by the plotter, exactly as
`noise_white_combo.py` needed "its plotter" listed alongside it.

## Re-run order

See `run_all.sh` for the full sequence with exact commands (preprocess ->
LDNet training -> accuracy studies -> depth ladder -> field reconstruction ->
CS-25 grid -> noise axes). `--dry-run` is the default; pass `--go` to
actually `qsub` the two stages that live in this repo (CS-25 grid, noise
axes). The other stages belong to `src/`, `recon/`, and the root
`train_*.sh` scripts and are only ever printed, never executed by this
script — confirm each precondition (previous stage's checkpoint exists,
dataset landed) before running them by hand.

**Compute discipline**: every launcher here submits through `run_study.pbs`
(`qsub`), never bare `nohup` on the login node. `light/noise/run_axis.sh`
uses that pattern and is explicitly not the one to copy —
`light/dagger_fom/NOTES.md` documents login01 hitting load average 107 with
45 users and the user's jobs being administratively removed as a result.
Plotting scripts (`cs25_combo_plots.py`, `cs25_thesis_figs.py`,
`plots_noise_white_combo.py`, `plots_thesis_bias_timing.py`) are cheap
(numpy + matplotlib over already-computed `.npz` files, no TensorFlow) and
are meant to run locally after scp, or on the login node — only the
TensorFlow rollout loops go through PBS.

## v5-vs-v6 comparison protocol

The v5 numbers are frozen in `light/results_cs25_combo/` (`traces_W*.npz`,
`summary.md`, the PNGs) and `light/noise/results/` (`W_combo.npz`,
`A2_calib.npz`, `B2_timing.npz`, `E2_combo_*.npz`) — never overwritten by
anything in this directory. The v6 equivalents land in
`final/results_cs25_combo/` and `final/results_noise/` (via
`paths.results_dir(...)`).

To compare a metric between campaigns:
1. Load both npz files (`np.load(..., allow_pickle=True)`) — the record
   schema (`kind='point'`/`'traj'`/`'open'`, field names) is identical
   between v5 and v6 since it comes from the same `harness_noise.py`
   (`point_record`/`traj_record`) and `cs25_combo_study.py` npz layout.
2. Diff the derived numbers, not the raw arrays: for the CS-25 grid, diff
   `summary.md` (v5: `light/results_cs25_combo/summary.md`; v6:
   `final/results_cs25_combo/summary.md`) cell by cell (`CLred`, `R*`,
   `|delta|_max`, pitch ratio). For noise axes, diff the `mean`/`lo`/`hi`/
   `nflag` fields of matching `(arm, value/frac)` records between
   `light/noise/results/<axis>.npz` and `final/results_noise/<axis>.npz`.
3. A regression is a v6 point whose `mean` CLred drops outside the v5
   point's `[lo, hi]` seed band (not just below the v5 mean — six-seed noise
   studies have real seed-to-seed spread; see the
   `honest-comparison-gust-oracle` / `preview-noise-sigma2-anchor` precedent
   in this project of not treating single-mean deltas as decisive), or a
   point that gains a stability flag (`nflag > 0`) where v5 had none.
4. Do not edit `light/latex/chapter3.tex` numbers from this comparison
   without an explicit decision to move the thesis onto dataset_v6 — until
   then v5 remains the numbers of record and this directory is a parallel,
   non-destructive re-run.

## Plot-style rules extracted from `light/latex/AGENTS.md`

("Linee guida per i plot", binding for any regenerated figure — see the file
for the Italian original.)

- One script per figure, archived in the repo; no hand-touched figures —
  every plot must be regenerable by a single script execution.
- `font.family: serif`, `mathtext.fontset: 'cm'`, ~9 pt at final print width
  (`figsize` sized to the LaTeX target: `\textwidth` ~ 6.3 in,
  `0.48\textwidth` ~ 3 in, so fonts are never rescaled by LaTeX); PNG at
  300 dpi.
- No in-figure titles (`suptitle`/`set_title`) — the description belongs in
  the LaTeX caption.
- Axis labels use the same notation as the text ($W_0$, $T_g$, $H$, $k$,
  $\delta$, CLred, ...) with units in square brackets.
- Only the MPC controller of the chapter appears in results figures — no
  legacy-controller series (one-step optimal, proportional).
- Legends use plain descriptive English, never internal dev codenames
  ("E2-combo", "A2/B2", "home cell").
- Multi-seed studies: mean line/bar plus a min-max band or whisker, seed
  count stated in the caption; points that raise the stability flag get a
  dedicated marker.
- Consistent, sober palette across all figures: same color for the same
  quantity everywhere (e.g. open-loop grey, closed-loop blue); light grid.
- Paired subfloats keep identical `figsize` and axis limits, exported
  without a variable crop (no `bbox_inches='tight'` when it would change the
  relative proportions between panels), so they compare at a glance; a
  shared legend appears in only one panel.

All forked plotting scripts here (`cs25_thesis_figs.py`,
`plots_noise_white_combo.py`, `plots_thesis_bias_timing.py`) already followed
these rules in `light/` and were forked verbatim on the plotting logic —
only their input/output directories changed. `cs25_combo_plots.py`'s
heatmap/line-plot PNGs are diagnostic (not AGENTS.md-styled thesis figures;
the actual thesis figures for the CS-25 grid are `cs25_thesis_figs.py`'s
output) and were left with their original titles, matching the v5 original.

# Field reconstruction: cost vs. accuracy (LDNet vs. full CFD)

## Table

| | Full CFD (OpenFOAM `pimpleFoam`, cosim replay) | LDNet field reconstruction (champion) |
|---|---|---|
| **Config.** | `cosim_main` case, dt=7e-5 s, ~43k steps for a 3 s window, 11075-node crop mesh | mean-split + CORAL (shift-modulated SIREN, omega0=10) decoder, d_s=1, L6 (2x7 dyn / 4x24 rec) |
| **Cores** | 16 (`ncpus=16:mpiprocs=16`) | 1 |
| **Wall-clock, 1 trajectory (3 s, 150 output frames, 11075 nodes)** | approx 2 h | 18.8-19.5 s |
| **Core-hours, 1 trajectory** | approx 32 core-h | approx 0.0053 core-h (19 s x 1 core) |
| **Speedup (wall-clock, single-run-to-single-run)** | 1x | approx 370-380x |
| **Speedup (core-hours)** | 1x | approx 6000x |
| **Model size** | -- (full 3-D discretized Navier-Stokes solve) | 5266 trainable parameters (127 NNdyn + 5139 NNrec) |
| **Per-field NRMSE (vx / vy / p), champion config (d_s=1, mean-split on)** | -- (ground truth) | see per-field table below |

Sources:
- CFD cost: `recon/RECON_NOTES.md:146-155` ("Phase 2b -- FULL pilot run COMPLETE (job 24216)") -- *"sim_A_025_test full 3 s: fields (860, 11075, 3) ... Wall time ~2 h on 16 cores (the poller's 'time' column was CPU-time-used ~16x wall, not elapsed -- the job was never 18 h)."* Core count from `recon/cluster/pilot_A025.pbs:4` (`select=1:ncpus=16:mpiprocs=16`). This is the same cosim/pimpleFoam pipeline (`cosim_driver.py --from-checkpoint`, prescribed gust+flap replay) used to generate every training/validation/test trajectory in the recon/ dataset -- not a separate benchmark run.
- LDNet cost: measured 2026-09-14, `recon/analysis/time_inference.py` (mirrors `reconstruct_fields.py`'s exact model-build/load/forward-pass code path), run on the cluster inside the same `tensorflow_gpu.sif` (Keras 2.14) container that trained the model, via `recon/cluster/timing.pbs` (PBS job 32226, `select=1:ncpus=1`, `OMP_NUM_THREADS=1`). Full forward pass = latent-ODE rollout (150 steps) + NNrec query over the full 11075-node grid, i.e. genuinely "reconstruct the whole trajectory," matching what the CFD number reconstructs. 10 repetitions after a warmup call (which pays a one-time TF tracing cost, excluded from the mean):
  - `sim_A_025` (gust-only, val/test): mean 18.832 s, std 1.006 s (`recon/analysis/timing_a025.json`)
  - `sim_Cc_060` (gust+flap, val/test): mean 19.497 s, std 0.827 s (`recon/analysis/timing_cc060.json`)
  - Build+load (one-time, amortizes over any number of trajectories): 0.72-0.76 s
  - Per (point, timestep): approx 11.3-11.7 microseconds
- Param count: `recon/models/meansplit_study/coral_o10_s0/latent_1/config.json` (identical file also present locally as `recon/models/coral_o10_s0_synced/latent_1/config.json`) -- `n_params_dyn=127`, `n_params_rec=5139` (`recon/models/.../run_info.json`).

Note on the local (Windows/WSL) tfvenv: it runs Keras 3.15, which cannot load these Keras-2.14-trained weights (nested-Sequential sublayer build-state incompatibility -- `reconstruct_fields.py` itself fails identically there, not a bug specific to the timing script). The timing measurement above is real cluster-container execution, not a local approximation.

## Per-field accuracy at the retained configuration (d_s=1, mean-split on, CORAL omega0=10)

This is the "champion" model (`coral_o10_s0`) that both `recon-intrinsic-latent-dim-1` (session memory) and `HMETRIC_VERDICT.md` independently identify as final for this architecture class: d_s=1 is simultaneously cheapest (fewest latent-ODE parameters) and most accurate (lowest NRMSE at every d_s tested: 1, 5, 10).

| quantity | value | source |
|---|---|---|
| Combined NRMSE (mixed-unit, pressure-dominated -- see caveat below), sim_Cc_060 | 6.141e-3 | `MEANSPLIT_NOTES.md`, "POST-CLOSURE ROUND" table |
| vx NRMSE (per-field), sim_Cc_060, full grid | 1.621e-2 | same |
| vx near-airfoil dynamic (fluctuation-normalized) NRMSE | 2.054e-2 | `MEANSPLIT_NOTES.md`, STALL/SEPARATION + D-RES campaign tables (champion row, repeated across many lever-comparison tables) |
| vx surface dynamic NRMSE | 2.266e-2 | same |

**Caveat carried from `HMETRIC_VERDICT.md`**: the "combined NRMSE" above mixes vx/vy/p in physical units, and pressure supplies 99.7-99.8% of the squared error by unit mixing alone -- it is effectively a pressure metric, not a balanced field metric. The per-field vx numbers (rows 2-4) are the ones that actually describe velocity-field accuracy. **The pre-mean-split/pre-CORAL per-field breakdown reported in `HMETRIC_VERDICT.md`** (base tanh-MLP decoder, no mean-split) is a *different, earlier* configuration and is not conflated with the champion numbers above:

| d_s (base decoder, no mean-split, no CORAL) | vx near NRMSE (A_025 / Cc_060) | surface-p NRMSE @ peak (A / Cc) |
|---|---|---|
| 1 (best) | 0.195 / 0.140 | 0.035 / 0.044 |
| 5 | 0.205 / 0.148 | 0.057 / 0.074 |
| 10 | 0.208 / 0.150 | 0.049 / 0.064 |

Mean-split alone then cut vx near-airfoil NRMSE 5.7-5.8x (Cc_060) / 23.7-24x (A_025) relative to this base-decoder table (`MEANSPLIT_NOTES.md` round-1 results), and CORAL's shift-modulated SIREN decoder cut the remaining *dynamic* residual a further ~8-12% on top of mean-split alone (`MEANSPLIT_NOTES.md`, "D-RES RESULTS ... CORAL omega0=10 WINS"). The two tables above are NOT directly comparable rows of the same architecture -- the top table is the final, retained configuration; the bottom table is the ablation baseline that motivated adopting mean-split+CORAL. Report only the top table as "the model's accuracy" in the thesis; use the bottom table only when the text is explicitly narrating the ablation history.

## Framing (for the thesis text)

The comparison above is deliberately not framed as a Pareto trade-off ("pay more compute, get more accuracy"), because the data does not support that story for this architecture. Two independent findings close off the trade-off framing:

1. **Accuracy is not on a size dial here.** The NRMSE-vs-latent-dimension study (session memory `recon-intrinsic-latent-dim-1`; reconfirmed by `HMETRIC_VERDICT.md`'s full field x region x window decomposition) found d_s=1 to have both the lowest *training* loss and the lowest *test* NRMSE among d_s in {1, 5, 10} -- not a case of an overfit small model beating a larger one on held-out data by chance, but a genuine indication that the intrinsic dimensionality of the unsteady aerodynamic response this LDNet is fit to is approx 1. Adding latent capacity does not trade cost for accuracy; it only adds cost. The same pattern reappears for decoder depth: L12+CORAL (2x the champion's decoder depth) was tried explicitly as part of the closed D-RES investigation and lost on every seed (`MEANSPLIT_NOTES.md`, "POST-CLOSURE ROUND", L12 rows worse than L6 on both vx and combined NRMSE). Note: `recon/analysis/DEPTH_STUDY_NOTES.md` (the dedicated H-ARCH depth-ladder study) was never filled in beyond its experiment design (all result sections read "pending") -- the L6-vs-L12 comparison actually used for this framing comes from the meansplit/CORAL post-closure round, not from that stub.
2. **The residual that remains is not closable by more model capacity.** The D-RES/stall investigation established across 11 independent architectural levers -- decoder conditioning, sampling, optimizer budget, training procedure, dynamics conditioning, loss weighting (2 forms), dynamics-side memory, and a local/gated decoder -- that the remaining near-flap dynamic error (a genuine, transient flow-separation event the champion under-represents) is a ceiling of this LDNet model class, not a capacity deficit that a bigger network would close. This closes off the "spend more, get more accuracy" reading of the accuracy axis entirely, independent of the cost axis.

Given both of these, the honest framing is: **LDNet is cheaper than the CFD solve by roughly three orders of magnitude in wall-clock (and closer to four in core-hours) at the same query resolution, AND it is already sitting at its own accuracy ceiling for this architecture** -- increasing latent width, decoder depth, or any of the 11 tried architectural levers does not move accuracy in the direction more compute would normally buy. There is no dial to turn between "cheap/less accurate" and "expensive/more accurate" within this model family on this problem; the only real trade-off in the data is LDNet-vs-CFD, and it is a one-sided one, not a curve.

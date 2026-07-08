# H-METRIC verdict — does d_s > 1 help anywhere that matters?

**Verdict: No. d_s=1 wins in every field, every region, and every time window that carries
meaningful signal — but the H-METRIC suspicion about the aggregate was half right: the reported
"combined NRMSE" is effectively a near-airfoil *pressure* metric (99.7–99.8 % of its squared
error comes from p), so it says nothing about the velocity fields. Decomposing them anyway
does not change the ranking: the flat/increasing NRMSE-vs-d_s curve is real, not a metric
artifact.**

## Methods (2–3 lines)
Loaded the 6 ROM/FOM pairs (`div_rom_{a,cc}_l{1,5,10}`, `final_div` models, T=150, 11075 nodes,
fields vx, vy, p in physical units) and decomposed the repo-convention error
`NRMSE = rms(rom−fom) / (max(fom)−min(fom))` per field, per region (near = dist-to-surface
< 0.5c, wake = TE→TE+3c × ±0.75c, far = rest; airfoil surface = 387 interior-boundary-loop
nodes, chord c = 0.978 m measured from the mesh hole), and per time window (excitation =
|W|≥10 % max or |δ|≥10 % max; peak = ≥50 %; decay = 1 s after; quiescent = rest), plus a
static (time-mean bias) vs dynamic (fluctuation) split and a linear-FE vorticity check.
Scripts: `hmetric_decompose.py`, `hmetric_static_dynamic.py` (run on the cluster login node),
`make_figs.py` (local).

## Reconciliation with the reported aggregates — EXACT
| sim | d_s=1 | d_s=5 | d_s=10 |
|---|---|---|---|
| sim_A_025 (reported 2.40/2.69/2.64e-2) | 2.3970e-2 | 2.6909e-2 | 2.6404e-2 |
| sim_Cc_060 (reported 1.39/1.64/1.59e-2) | 1.3871e-2 | 1.6431e-2 | 1.5849e-2 |

Recombining the field × region × window pieces (SSE-weighted by n_nodes × n_times) reproduces
these numbers to 4+ digits (see script stdout) — the decomposition is complete and consistent.

## (b) What the scalar aggregate is actually made of
- **Field shares of total squared error: p = 99.72–99.83 %, vx = 0.16–0.27 %, vy ≈ 0.01 %**
  (all 6 cases; `mse_shares.csv`, `combined_metric_composition.png`). Cause: the metric mixes
  units — the global range is set by p (10 606 Pa for A, 21 906 Pa for Cc vs vx range 149/210
  m/s), and p's absolute errors dominate the MSE. The combined scalar *is* a pressure NRMSE.
- **Region shares: near (r < 0.5c, 57 % of nodes) carries 89–99 % of the SSE.** The far-field
  node-count-dilution part of the hypothesis is wrong for this mesh: nodes concentrate near the
  airfoil, and the error genuinely lives there. The hiding mechanism is unit mixing, not node
  weighting.
- The vx field — the "hard" one at 14–21 % NRMSE near the airfoil — is numerically invisible
  in the aggregate. So the aggregate can't be read as a statement about velocity accuracy.

## (a) Where does d_s=5/10 beat d_s=1? Essentially nowhere.
Across 2 sims × 4 fields (vx, vy, p, surface-p) × 5 regions × 5 windows, **d_s=1 has the lowest
NRMSE in every single combination**. Representative numbers (window=all unless noted):

| metric | d_s=1 | d_s=5 | d_s=10 |
|---|---|---|---|
| vx near, A | **0.195** | 0.205 | 0.208 |
| vx near, Cc | **0.140** | 0.148 | 0.150 |
| vx far, A | **0.085** | 0.117 | 0.117 |
| surface-p @peak, A | **0.035** | 0.057 | 0.049 |
| surface-p @peak, Cc (loads proxy) | **0.044** | 0.074 | 0.064 |
| p near @peak, Cc | **0.043** | 0.057 | 0.054 |

Nominal exceptions, all in noise-dominated regimes:
1. **Wake/far vorticity**: d_s=10 "wins" the wake (−12…−39 %) and d_s=5 the far field
   (−33…−43 %) — but every model's vorticity error is ≥ 69 % of the signal RMS near the wall
   and ≥ 150–270 % in wake/far (error larger than the signal ⇒ predicting zero would be
   better). No model reconstructs vorticity; these are differences between failure modes
   (d_s=10's far-field vorticity error is 6–11× the signal — residual stripe/noise). Caveat:
   vorticity from linear-FE gradients on anisotropic near-wall triangles is noisy; treat
   qualitatively.
2. **Cc wake p @excitation/peak**: d_s=5 better by 0.6–1.7 % — negligible, and all three
   models sit at rel-to-fluctuation ≈ 1.0 there (none tracks wake-p dynamics).
3. **Cc far vx dynamic component**: d_s=10 −8.5 % — but rel-to-fluctuation ≥ 1.0 for all three.

## Bonus finding: the vx error is a *static* bias, and d_s=5/10 don't track dynamics at all
Error vs time is nearly flat (quiescent ≈ peak for vx), so we split err = static (time-mean)
+ dynamic (`static_vs_dynamic.csv`):
- The near-airfoil vx error is ~90 % a steady boundary-layer/wake mean-flow bias (static
  component = 0.84–0.90 of the fluctuation range for A). The gust-response part is small
  (dynamic NRMSE ≈ 3–4 % of fluctuation range).
- **Dynamic error / signal-fluctuation RMS in the near region (where the signal is):
  d_s=1 = 0.54–0.92 (tracks the dynamics); d_s=5 = 0.98–1.08 (no better than freezing the
  time-mean field); d_s=10 = 0.88–1.12 (marginal at best).** In the weak-signal wake/far vx
  pockets all three models sit at ≥ 1.0. Higher d_s doesn't just fail to add detail; it
  largely fails to track the unsteady response at all, consistent with the
  intrinsic-latent-dim≈1 finding.

## (c) Recommendation for the paper's NRMSE-vs-d_s figure
1. **Do not use the mixed-unit combined scalar** (or if kept for continuity, label it as
   pressure-dominated). Instead report **per-field NRMSE** (each field normalized by its own
   range), either as three curves or averaged after normalization.
2. Add the **airfoil-surface pressure NRMSE in the excitation window** as the physically
   meaningful loads-proxy curve (d_s=1: 0.035/0.044; it separates the models most:
   d_s=5 is +65/+69 % worse).
3. Optionally note (or plot) the **static/dynamic split**: the headline vx error is a steady
   near-wall bias common to all d_s; the dynamic-tracking metric is where d_s=1's advantage
   is cleanest.
4. Whatever variant is chosen, d_s=1 stays best (and d_s=10 ≲ d_s=5 on the aggregate) — the
   paper's d_s=1 conclusion is robust to the metric choice.

## Files
- `error_decomposition.csv` — tidy table (750 rows): sim, d_s, field, region, time_window,
  nrmse, n_nodes, n_times, rmse, denom, denom_type.
- `mse_shares.csv` — SSE shares by field/region (+ per-field within near region).
- `static_vs_dynamic.csv` — static vs dynamic error split with three normalizations.
- `timeseries_sim_{A_025,Cc_060}.csv` — per-time NRMSE curves + W(t), δ(t).
- `err_vs_time_sim_{A_025,Cc_060}.png`, `nrmse_by_field_region.png`,
  `combined_metric_composition.png`.
- `geometry.txt`, `region_labels.npy`, `airfoil_nodes.npy` — region definitions/masks.
- `hmetric_decompose.py`, `hmetric_static_dynamic.py` (cluster-side), `make_figs.py` (local).
  Cluster copies of the outputs: `/work/u10677113/NACA2312/recon/analysis_hmetric/`.

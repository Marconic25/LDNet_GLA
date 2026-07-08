# Design — preview-noise robustness study for the final wnext controller

Question: how much measurement error on the 1-step gust preview W(t+dt) does the
final one-step optimal controller (`light/optimal.py`, use_wnext+refine) tolerate
before the +76.6/+80.7% advantage at W30/Tg0.4 (DAMULT=3) collapses to the
prop-W equal-info baseline (+32%)? And which mitigations recover margin honestly?

## Ground rules (from the task spec + 76/ conventions)

- Plant ALWAYS advances with the true gust W(t); only the controller's preview
  argument is corrupted. Scalar B=1 rollout, loop body identical to
  `76/preview_study.py::rollout_customW` / `light/run.py`.
- `light/optimal.py` is imported, never modified. With `use_wnext=True` BOTH the
  candidate scan and the causal gate see the corrupted Wc — that is the honest
  sensor model (the controller has no clean W at all).
- DAMULT=3 read from env BEFORE importing structure (harness does the scaling
  exactly once; run.py is NOT imported to avoid double-scaling).
- Metrics + window t<=Tg+0.5 byte-identical to light/run.py; explosion flags
  (ad/add/hdd > 3x open) always reported.
- Noise conventions copied from 76 for comparability: white noise
  `default_rng(100+seed)`, band-limited `default_rng(300+seed)`, one draw per
  step, sensor clamp `Wc = max(0, .)`.
- R in {3e-4, 1e-4} everywhere except the R-mitigation sweep. >=6 seeds/point
  (seeds 0..5, same bases as 76), mean/min/max/std of CLred + flag count.
- Regression anchors (must reproduce before any study runs): open cex0=0.4600;
  sigma=0 -> +76.6% (R=3e-4), +80.7% (R=1e-4); sigma=2% with refine=False
  must give ~+5.2% mean (76 used OptGrid refine=False — the final controller has
  refine=True, so the sanity check instantiates OptimalController(refine=False)
  via constructor arg, defaults untouched).

## Architecture

```
light/noise/
  harness_noise.py    sys.path -> light/; DAMULT scaling; shared aero instance;
                      gust_array, rollout(ctrl, W0, Tg, wc_fun), metrics, seed_stats
  controllers_ref.py  reference laws for comparison/mitigation, all local to noise/:
                      OptimalRdu (subclass adding R_du move cost),
                      PropWRef (g_CL=-60, g_W=-0.5, honest_home best),
                      MPCConstRef (port of 76 MPCConst N4 gate-none, uses light aero)
  noise_white.py      axis A (+ prop-W noisy reference arm)
  noise_bandlim.py    axis B
  noise_struct.py     axis C
  noise_cells.py      axis D
  noise_mitigation.py axis E
  plots.py            all figures from results/*.npz
  results/            npz + png + logs
  NOTES.md            tables, regressions, verdict (written at the end)
```

`rollout(ctrl, W0, Tg, wc_fun)`: wc_fun(i, Wt, N) -> the Wc the controller
receives as W_next. Controller call: `ctrl.compute(x, Wt[i], Wc)` (light API;
W arg unused under use_wnext but passed true for interface fidelity).
Each axis script saves one npz: config arrays + per-(point,seed) CLred/flag/
flap_max/pitchpk + full trajectories for the money-plot cases.

## Study matrix (home cell W30/Tg0.4 unless stated)

A) White noise sigma/W0 in {0,.01,.02,.05,.10}, R in {3e-4,1e-4}, 6 seeds.
   + sanity arm refine=False @ sigma=2%, R=3e-4 (expect ~+5.2%).
   + prop-W reference under the SAME noisy W (sigma in {0,.02,.05}) so the
     break-even vs prop is honest (prop degrades too — its W-feedforward is
     smooth in W, expected mild).
B) Band-limited (5% white + 1st-order LPF), tau in {2,5,10,20,40} ms, both R,
   6 seeds (rng 300+seed). 76 found sweet spot tau~5-10ms for OptGrid
   refine=False; reproduce for the final refine=True law.
C) Structured errors, one at a time, deterministic (no seeds) except jitter:
   bias +-5/10% W0; scale x{0.8,0.9,1.1,1.2}; latency (Wc=W(t), i.e. k=0);
   jitter: per-step k in {0,1,2} uniform (rng 100+seed, 6 seeds). Both R.
D) Cell dependence: sigma in {0,.02,.05} at W10/Tg0.7 and W30/Tg0.7 (own
   open-loop refs). Is the noise knife-edge specific to the sharp-gust cell?
E) Mitigations, sigma in {.02,.05}, 6 seeds each, same pick rule (max mean
   CLred s.t. zero flags across seeds):
   1. R increase: R in {3e-4,1e-3,3e-3,1e-2}
   2. R_du move-suppression (OptimalRdu): R_du in {1e-4,1e-3,1e-2} at R=3e-4
   3. Multi-sample preview averaging (the physically-new one): a LIDAR measures
      the whole approaching profile (plus remembered past scans), so each step
      the controller has K independent noisy samples W(t+j*dt)+eps_j on a
      window CENTRED on the needed j=1 (K=3: j in {0..2}; K=5: {-1..3};
      K=9: {-3..5}); Wc = their mean. Centred, NOT forward (j=1..K), because
      a forward window shifts the effective preview to (K+1)/2 steps and k>=5
      is known to ring (76/ preview-horizon study). Trades noise 1/sqrt(K)
      against gust-shape smoothing over (K-1)*dt <= 16 ms, phase-neutral.
   4. Combo of the best two above
   5. MPC N4 gate-none reference (no preview, W(t) noisy) — the known
      noise-tolerant alternative, same seeds
   Money plots: worst sigma=2% seed unmitigated vs best mitigated (C_L, delta,
   W true vs W seen, alpha_dot).

## Execution

Cluster only (TF): sync light/noise/ to /work/u10677113/LDNet_GLA/light/noise/,
verify remote light/ + model dir exist first. Stage 0 = regression script
(fast, ~5 rollouts); gate everything on it. Then 5 independent nohup jobs
(A/B/C/D/E) in the apptainer image with PYTHONNOUSERSITE=1, DAMULT=3, polled
via ssh -n. Cost @ ~1 min/rollout: A~63, B~60, C~30, D~54, E~160 rollout-min
-> wall ~3h bounded by E (MPC arm ~4x/step). plots.py runs on the cluster
(matplotlib in container), results + figures scp'd back.

## Verdict deliverable (NOTES.md)

- Full tables per axis; regression section; explicit break-even: the sigma at
  which mean CLred crosses prop-W's SAME-noise mean (and the noise-free +32%
  line), for both R and for the best mitigation.
- Honest framing: what stays a knife-edge, what mitigations genuinely buy,
  whether MPC N4 remains the practical choice under noise.

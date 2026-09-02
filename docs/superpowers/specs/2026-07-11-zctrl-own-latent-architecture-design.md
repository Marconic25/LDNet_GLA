# Design: replace z_plant with z_ctrl across `light/`

Date: 2026-07-11
Status: approved (design), pending implementation plan

## Problem

The model-based GLA controllers in `light/` seed their MPC prediction horizon
from the **plant's** latent state, `z_b = tile(aero._z)`, where `aero` is the
*same* `LDNetAero` instance used as the plant. That is an idealization: on a
real aircraft the plant is not the LDNet, and the controller would carry its
own LDNet copy whose latent `z_ctrl` is propagated in closed loop and never
re-synchronised with the true latent.

The E2-zctrl experiment (`light/noise/e2_zctrl.py`, results
`results/E2_zctrl_{home,cells}.npz`) validated that a controller running on its
**own** latent holds performance: sanity arm B (copy fed the true gust)
reproduces the shared-z baseline bit-for-bit; the realistic arm C (copy fed the
fused current-gust estimate) matches the shared-z baseline within ≤2.8 pt with
0/6 stability flags across the lidar cells, degrading only past white noise
≥10 %·W0 (single-seed). The latent drift `‖z_ctrl − z_plant‖₂` stays small and
bounded (≤2.5e-2 at σ=20 %) because the copy is fed the true structural state
`x(t)` and the applied flap δ* each step — it is not a free run.

This design promotes that validated behavior into the actual `light/`
architecture: **the controllers run on their own `z_ctrl`, and the read of the
plant latent `aero._z` is removed.**

## Decisions (locked with user)

1. **Scope:** the canonical controller (`optimal.py::MPCPreviewController`,
   used by `run.py` and the `noise_*_combo.py` studies) **and** the noise-study
   controller copies (`e2_combo.py`, `e2_combo_cells.py`, `e2_mpc.py`,
   `controllers_ref.py`).
2. **No shared-z fallback:** the read of `aero._z` is removed entirely. The
   horizon always starts from `self._z_ctrl`. There is no `own_latent` flag.
   Consequence: re-running these scripts reproduces the own-latent numbers
   (≈ e2_zctrl arm C), not the archived idealized numbers, which survive only
   in already-saved npz. Accepted.
3. **Implementation = Approach B:** the controller stores only its own latent
   vector `self._z_ctrl`; the same trained weights advance it through a new
   *pure* method on `LDNetAero`. No second TF model is instantiated. (Approach
   A — a dedicated `LDNetAero` copy per controller, as in `e2_zctrl.py` — is the
   validated reference but heavier; B is numerically identical and lighter.)

## Mechanism

### New primitive: `LDNetAero.advance_z` (pure)

```python
def advance_z(self, z, state, delta_deg, W, U, dt):
    """Advance an EXTERNAL latent z one step; return z_new. Never touches self._z.
    Mirrors advance() exactly (forward-Euler sub-steps with leak)."""
    h, hd, a, ad = state
    sigs_n = self._normalize_signals(h, hd, a, ad, delta_deg, W)
    U_n = self._normalize_U(U)
    return self._advance_z(np.asarray(z, float), sigs_n, U_n, dt)
```

Because `advance_z` reuses the same `_advance_z` (leak included) as the plant's
`advance()`, feeding it identical `(state, δ, W, U, dt)` yields a latent
bit-identical to the plant's — the clean/oracle pipeline is unchanged.

### Controller changes (uniform pattern)

Each MPC controller:
- `reset()` sets `self._z_ctrl = np.zeros(num_z)` (num_z from the model:
  `self.aero._num_z` in optimal.py, `H.aero._num_z` in the noise copies).
- `compute()` seeds the horizon from the controller latent:
  `z_b = np.tile(np.asarray(self._z_ctrl, float).reshape(1, -1), (G, 1))`
  (replacing `tile(aero._z)`).
- After choosing δ*, advances its own latent with its **current-gust estimate**
  `w_now`:
  `self._z_ctrl = aero.advance_z(self._z_ctrl, state, d, w_now, U, dt)`.

The horizon's batched forward step (`aero.batch_step`, leak-corrected
`z_b = z_new - lam*z_b`) is unchanged — it already uses the shared weights; only
the *seed* latent changes.

### `w_now` — the controller's own current-gust estimate

Never the true gust unless that is genuinely all the controller has.

| Controller | Caller / sensor | `w_now` |
|---|---|---|
| `MPCPreviewController` | `run.py` (oracle preview) | true current gust `Wi` |
| `MPCPreviewController` | `noise_*_combo.py` wrappers | `sensor.cur` (fused current node) |
| `MPCPrevRdu` (e2_combo, cells) | `FusedSensor` | `sensor.cur` |
| `MPCConstRef` | harness | `Wc` (noisy current gust it already receives) |
| `MPCPrevRef` (e2_mpc) | `PreviewSensor` | `Wc` (= `last[0]`, nearest preview as proxy) |

### New: `FusedPreviewSensor.cur` / `FusedSensor.cur`

The fused current-gust estimate at node `m = i`: `num[i]/den[i]`, clamped ≥ 0.
Node `i` is untouched by step `i`'s update (which only writes nodes
`i+1..i+Jmax`), so it aggregates every prior measurement of the current gust —
the natural current-gust estimate. Computed at the end of `wc_fun`, cached in
`self.cur`. Subclasses `JitterSensor`, `RefitSensor` inherit it.

## API changes

`MPCPreviewController.compute(state, w_seq)` →
`MPCPreviewController.compute(state, w_seq, w_now)`.

Call sites updated to pass `w_now`:
- `run.py:84` → `ctrl.compute(x, w_seq, Wi)`
- `noise_white_combo.py:57`, `noise_calib_combo.py:70`,
  `noise_mismatch_combo.py:99`, `noise_jitter_combo.py:137`,
  `noise_timing_combo.py:151` → `self._mpc.compute(state, self._sensor.last, self._sensor.cur)`

The `e2_*.py` copies keep the harness signature `compute(state, W_true, Wc)`
and advance `self._z_ctrl` internally; `harness_noise.py` is **not** changed.

## Files touched

| File | Change |
|---|---|
| `light/ldnet_aero.py` | + pure `advance_z` |
| `light/optimal.py` | `MPCPreviewController` z_ctrl + `compute(...,w_now)`; `FusedPreviewSensor.cur` |
| `light/run.py` | pass `Wi` as `w_now` |
| `light/noise/controllers_ref.py` | `MPCConstRef` z_ctrl (advance with `Wc`) |
| `light/noise/e2_combo.py` | `MPCPrevRdu` z_ctrl (advance with `sensor.cur`); `FusedSensor.cur` |
| `light/noise/e2_combo_cells.py` | same as e2_combo |
| `light/noise/e2_mpc.py` | `MPCPrevRef` z_ctrl (advance with `Wc`) |
| `light/noise/noise_white_combo.py` | pass `sensor.cur` |
| `light/noise/noise_calib_combo.py` | pass `sensor.cur` |
| `light/noise/noise_mismatch_combo.py` | pass `sensor.cur` |
| `light/noise/noise_jitter_combo.py` | pass `sensor.cur` |
| `light/noise/noise_timing_combo.py` | pass `sensor.cur` |

Not touched: `harness_noise.py`, `structure.py`, `e2_zctrl.py` (the experiment
already carries both arms), `light/latex/`.

## Validation

1. **Clean bit-identity.** Capture `run.py` combo `de`/`CL` arrays before and
   after the change (W30/Tg0.4, DAMULT=3, oracle). Must be identical to
   roundoff: z_ctrl advanced with the true `Wi` equals z_plant.
2. **Noise = arm C.** Run the `e2_combo.py --smoke` (and a short flat-σ point)
   and confirm the numbers match the corresponding `e2_zctrl.py` arm-C values
   already in `results/E2_zctrl_home.npz` (e.g. white-2 % +80.6, dlr +81.1;
   w10t07 +89.2, w30t07 +91.4 from the cells file).
3. **No crashes on the converted noise axes.** Smoke each `noise_*_combo.py`
   (they now feed `sensor.cur`); results shift from the archived shared-z
   numbers to own-latent — expected, not a regression.
4. Runs on the cluster (TF container); local `tfvenv` for smokes.

## Risks

- **Reproducibility of archived shared-z E2 numbers is dropped by design.** They
  remain only in saved npz. Mitigation: the E2-zctrl experiment already recorded
  both arms (A shared vs C own), so the comparison is preserved there.
- **Sensor `.cur` sign/ordering.** Must be computed before the compute() read
  and match the fused-node convention; covered by validation step 2.
- **`e2_mpc` proxy.** `MPCPrevRef` lacks a fused current node; using `Wc`
  (`= last[0]`, a 1-step preview) as `w_now` is a documented proxy, comparable to
  e2_zctrl arm D (which tracked arm C closely).

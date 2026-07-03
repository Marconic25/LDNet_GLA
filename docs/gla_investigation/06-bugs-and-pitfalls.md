# 06 — Bugs and pitfalls found along the way

[← Back to index](README.md) · [← 05 Envelope edge & open questions](05-envelope-edge-and-open-questions.md)

Documenting these matters as much as the results: several apparent "findings" were
actually artifacts, and a couple of dormant bugs could have silently corrupted
long-horizon rollouts. A future reader should know exactly which knives were in the
drawer.

## 1. `batch_step` omits the latent leak `(1 - lambda)`

**Where:** `clean/ldnet_aero.py`, `LDNetAero.batch_step` (line ~225):

```python
z_new = z_b + (float(dt) / self._dt_ref) * dz          # in batch_step
```

versus the correct single-step update in `_step_z` (line ~127):

```python
return (1.0 - self._z_leak) * z + (sub_dt / self._dt_ref) * dz.numpy().flatten()
```

**Problem:** `batch_step` propagates the latent as `z_new = z + (dt/dt_ref)*dz`,
**dropping the `(1 - lambda)` leak term** that the true latent ODE has (the
damped-ODE leak from the rollout model's config). Over the **short MPC horizon**
(a few steps) the missing leak is negligible — fine. But over a **full-trajectory
rollout** (hundreds of steps in the batched analysis sweeps) the error **compounds**
and the latent diverges from the true model.

**Fix / workaround:** the batched analysis scripts correct for it *externally* after
each call, e.g. in `clean/propw.py`, `clean/honest_grid.py`, `clean/smoothprop.py`,
`clean/ksweep2.py`:

```python
LAM = float(M.a._z_leak)
...
cl, cm, z_new = a.batch_step(z_b, x_b, d, Wi, U, DT)
z_b = z_new - LAM * z_b          # re-apply the leak that batch_step omitted
```

i.e. `z_corrected = z_new - lambda*z_old`, reconstructing the intended
`(1 - lambda)*z + (dt/dt_ref)*dz`. **Lesson:** the fast batched path and the
authoritative single-step path had **divergent latent update laws**; only the
single-step one (`advance`) is exact. Any new batched rollout must re-apply the leak
or it will slowly lie.

## 2. The 2nd-order flap smoothing (`DLPF=0.95`) CRIPPLES the MPC on fast gusts

**Where:** the flap post-filter in `clean/mpc_gust.py::simulate`
(`de_f`, `de_f2` cascade) and the scheduled default `DLPF=0.95`.

**Problem:** a heavy 2nd-order low-pass (`DLPF=0.95`) **delays and attenuates** the
commanded flap. On a **fast / high-`k`** gust the whole gust is over before the
filtered command reaches its target — the alleviation lands *past the gust peak*,
where it is useless. This made an early reduced-frequency sweep (the original
`ksweep`) **invalid**: it was measuring the *filter's* lag, not the *controller's*
capability, and wrongly made the MPC look bad at high `k`.

**Diagnosis / fix:** stripping the extra smoothing (`DLPF=0`, no gust-gating,
keeping only the physical 300 deg/s rate limit) **recovered** the MPC — e.g. from
crippled back to ~73% reduction at `Tg=0.50`. The corrected sweep `clean/ksweep2.py`
runs **both** controllers on the same physically-limited post-processing (`DLPF=0` +
rate limit only), so the `k`-comparison reflects the control law, not a filter. See
the module docstring of `ksweep2.py` for the explicit "the previous ksweep was
invalid" note. **Lesson:** a shared post-filter that is fine at low `k` can be fatal
at high `k`; a filter artifact masqueraded as a model limitation.

## 3. `causal_basin` had the flap SIGN inverted (dormant bug)

**Where:** `clean/controller.py::compute`, the `causal_basin` branch (lines ~208–231).

**Problem:** `causal_basin` restricts the search to the flap-sign half that *reduces*
the lift excursion. In this plant `dC_L/ddelta > 0`, so to reduce a **positive**
`C_L` excursion (`C_L > trim`) the lift-reducing flap must go **negative**, and
vice-versa. The original code had this **backwards**, which would have steered the
optimizer into the **non-causal "nulling" basin** — the wrong side of the
non-monotone `C_L(delta)` — and destabilized the loop.

**Why it was dormant:** the production MPC path always ran with `causal_basin=False`
(see `clean/mpc_gust.py`, where `simulate('mpc', ...)` constructs the `Controller`
with `causal_basin=False`), so the buggy branch was never exercised in the headline
runs. It *was* used by `clean/onestep.py` (`causal_basin=True`), which is why the
sign had to be right there.

**Fix:** corrected so `C_L >= trim → [g_lo, g_hi] = [-DMAX, 0]` (negative flap) and
`C_L < trim → [0, DMAX]`. The current code and its comment reflect the correct sign.
**Lesson:** a dormant sign bug in a rarely-taken branch is exactly the kind of thing
that surfaces only when you finally enable the flag — check sign conventions against
the *measured* `dC_L/ddelta`, not intuition.

## 4. The "margin grows with `k`" pitch-pick artifact

**Where:** the win-selection logic when comparing arms under a pitch constraint.

**Problem:** to compare "at iso-pitch," each arm's reported score is the **best `C_L`
reduction among candidates whose pitch ratio stays under a threshold**. If the
selection **falls back** to a different candidate when the primary one violates the
pitch cap, the *chosen operating point* changes between cells in a way that has
nothing to do with `k`. An early pass read the resulting numbers as **"the optimal's
margin over the proportional grows with `k`"** — apparently confirming the
reduced-frequency prediction. On inspection this was an **artifact of the
pitch-constraint pick**, not a physical trend, and the reading was **walked back**.

**Mitigation:** `clean/propw.py::pick` was made to take the **unconstrained max `C_L`
reduction** (reporting pitch alongside for inspection) precisely to avoid the
"pitch-fallback artifact" — its docstring says so. `clean/ksweep2.py` instead
reports reduction at **several fixed pitch thresholds** (`@115`, `@105`) so the
constraint is applied *consistently* across `k` and a real trend (if any) can be
seen without a pick artifact. **Lesson:** when you constrain-then-maximize, the
*argmax location* can move for reasons unrelated to your independent variable;
always check whether a "trend" is really a change in which operating point got
picked.

## Meta-lesson

Three of the four items above are **artifacts that briefly looked like findings**
(the crippled-MPC `k`-sweep, the inverted causal basin's would-be instability, the
pitch-pick "`k`-trend"), and one is a **silent correctness bug** (the batched latent
leak). In a study whose whole point is a *fair* comparison, the failure mode is
subtle: a broken filter, a mis-picked operating point, or a divergent rollout can
each manufacture (or erase) an advantage. Every headline number here was re-derived
after these were caught.

---

Next: [07 — Literature →](07-literature.md)

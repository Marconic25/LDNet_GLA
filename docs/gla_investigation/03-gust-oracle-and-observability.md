# 03 — Gust oracle, observability, and the honest result

[← Back to index](README.md) · [← 02 One-step = proportional](02-single-step-equals-proportional.md)

This is the pivot of the whole investigation: the model's apparent win was bought
with information it cannot actually have.

## Step 1 — With a gust oracle, the optimal wins on sharp/strong gusts

First comparison (`clean/onestep.py`): give the model-based one-step optimal the
**true gust** `W(t)` (a "gust oracle" — it calls `ctrl.compute(x, W_hat=Wt[i])`
with the exact gust), and compare against a fixed-gain proportional.

| Cell | `k` | Prop (`C_L` red) | 1-step optimal, oracle (`C_L` red) |
|------|-----|-----------------|-------------------------------------|
| W10/Tg1.0 | 0.039 | tie | tie |
| W20/Tg1.0 | 0.039 | tie | tie |
| **W30/Tg0.5** | **0.079** | **~4%** | **~36%** |

On the strong, sharp gust (`W30/Tg0.5`) the oracle optimal reduced the peak lift
excursion by ~36% vs only ~4% for the proportional — a large, clean win.

> **Caveat #1 (amplitude):** the proportional here is **unsmoothed** (raw high-gain,
> chattering). See [04](04-equal-information-and-chatter.md) — smoothing it changes
> the picture.
>
> **Caveat #2 (information):** the optimal *knows the true gust* `W(t)`; the
> proportional only sees `C_L`. This is not yet a fair fight.

## Step 2 — Reduced-frequency physics: why a model *could* help

LDNet is **unsteady** (latent `z` = aerodynamic memory). The reduced frequency is
`k = pi / (80 * Tg)`. At low `k`, Theodorsen `C(k) -> 1` and the aero is
quasi-steady, so (per [02](02-single-step-equals-proportional.md)) a static gain is
near-optimal and the model adds nothing. At higher `k` the unsteady **lift deficit**
`1 - C(k)` grows, and a model that captures the memory could, in principle, do
better.

Corroborating evidence: the **linear-fit residual** of the optimal command
(~8%) matched the unsteady lift deficit `1 - C(k)` — i.e. the part of the optimal's
behaviour that a static proportional *cannot* reproduce is exactly the unsteady
part. This motivated the prediction that **the model advantage should grow with
`k`**. (This prediction was later only *partially* borne out and one over-eager
reading of it had to be walked back — see
[05](05-envelope-edge-and-open-questions.md) and
[06](06-bugs-and-pitfalls.md).)

## Step 3 — The honest question: can `W` be recovered from `C_L`?

The gust oracle is only legitimate if `W` is **observable** from the sensor
(`C_L`). So: sweep `W` at realistic operating points and check whether `C_L(W)` is
invertible. Script: `clean/clw_sweep.py`.

Method: run an open-loop gust (`W0=30, Tg=0.5`), capture the real operating point
`(state x, latent z)` at three instants (rise `t=0.12`, peak `t=0.25`, decay
`t=0.38`), then **freeze** `(x, z)` and sweep `W in [0, 40]` computing
`C_L(W) = predict(x, 0, W, U)`.

Finding:

> `C_L(W)` at frozen latent is **NON-MONOTONE** at the gust peak, and even
> **non-physical** (locally *decreasing* in `W`).

**Why.** In training, `W` and the latent `z` **co-vary** — the model never saw `W`
increased while `z` was held fixed. Probing `W` at a frozen `z` is therefore
**off-manifold**: the network extrapolates into a region it never learned, and the
`C_L(W)` curve turns over and even reverses slope. So the map from measured `C_L`
back to `W` is **not invertible**.

**Verdict:** `W` is **not reliably recoverable** from `C_L`. The existing gust
observer (`clean/observer.py`) had already run into this — it had to switch from a
**bisection** root-find (which assumes monotonicity) to a **bounded minimization**
precisely because `C_L(W)` is non-invertible. So **the gust oracle is not
realizable** on this plant. Any controller that needs `W` must either be given it by
an *independent* sensor (LIDAR / alpha-probe preview — see
[07](07-literature.md)) or do without.

## Step 4 — The honest optimal (no gust oracle): it ties or loses

If the optimal cannot know `W`, how should it use the model fairly? The honest
formulation (`clean/honest_grid.py`): **anchor the lift level to the measured
`C_L`** (the same sensor the proportional uses) and take only the model's
**delta-shape at `W = 0`** — the unknown-gust offset cancels in the difference:

```
Chat(delta) = C_L_meas + [ model(x, delta, W=0) - model(x, delta_prev, W=0) ]
delta*      = argmin_delta (Chat(delta) - C_L_trim)^2 + R*delta^2      (causal half, rate-limited)
```

The excursion sign (which flap direction reduces lift) is taken from the **measured**
excursion `C_L_meas - C_L_trim`, not from the (unknown) gust.

Result: the honest optimal **ties or loses** to the proportional.

| Cell | Honest optimal (`C_L` red) | Proportional (`C_L` red) | Winner |
|------|----------------------------|--------------------------|--------|
| W30/Tg0.5 | ~2.5% | ~3.7% | **proportional** |

**Conclusion.** The optimal's earlier win (~36% at `W30/Tg0.5`) came almost
entirely from the **gust oracle** — the unobservable `W`. Strip the oracle, and the
model's structural edge over a proportional collapses to a tie-or-loss. The model
was not "smarter"; it was *better informed*, with information that is not physically
available.

This directly sets up the equal-information test: *if we now give the proportional
the same `W`, does it match the optimal?* → [04](04-equal-information-and-chatter.md).

---

Next: [04 — Equal information and the chatter reversal →](04-equal-information-and-chatter.md)

# 02 — The single-step optimum IS a proportional law

[← Back to index](README.md) · [← 01 Setup](01-question-and-setup.md)

## The derivation

Take the one-step cost with only the lift term and control effort (the essential
GLA trade-off):

```
J(delta) = Q_CL * (C_L(delta) - C_L_trim)^2 + R * delta^2
```

On a **locally-linear** plant, linearize the lift response about the current
operating point with sensitivity `s = dC_L/ddelta`:

```
C_L(delta) ≈ C_L0 + s * delta
```

where `C_L0` is the lift at `delta = 0` (i.e. the lift the controller *sees*).
Substitute and minimize:

```
J(delta) = Q_CL * (C_L0 - C_L_trim + s*delta)^2 + R * delta^2

dJ/ddelta = 2*Q_CL*s*(C_L0 - C_L_trim + s*delta) + 2*R*delta = 0
```

Solving for the optimum:

```
delta* = - [ Q_CL * s / (Q_CL * s^2 + R) ] * (C_L0 - C_L_trim)
       = - K * (C_L0 - C_L_trim)
```

with the **model-computed gain**

```
K = Q_CL * s / (Q_CL * s^2 + R).
```

## The implication

**This is exactly a proportional law.** `delta* = -K * (C_L0 - C_L_trim)` is
`ProportionalController` with `gain = -K`. So:

> In the **quasi-steady** regime (`s` roughly constant), the best fixed-gain
> proportional controller **already realizes the one-step optimum**. There is
> nothing for the model to add.

The model-based one-step controller can therefore only **beat** the proportional
where `s = dC_L/ddelta` **varies** — i.e. where the plant is genuinely nonlinear or
unsteady. Two concrete mechanisms:

1. **Self-adapting gain.** `K` is recomputed from the model each step, so as `s`
   changes (with state, latent `z`, gust), the effective proportional gain adapts.
   A fixed-gain proportional cannot do this.
2. **Pitch-reversal safety.** Near a sensitivity sign flip / pitch reversal
   (`s -> 0`), `K -> 0`, so the optimal **stops commanding flap** exactly where a
   fixed-gain law would push in the *wrong* direction. This is the clearest place
   the model can help.

## Where this shows up in the code

- The equivalence is stated in the docstring of `clean/onestep.py` (lines 1–15) and
  is the reason that script logs the **effective gain**
  `K(t) = delta / (C_L0 - C_L_trim)` for the "money plot": it lets you *see* the
  optimal's gain self-adapt and collapse to zero at the reversal.
- `clean/onestep.py` runs the single-step optimal (`Controller`, `mpc_horizon=1`,
  `causal_basin=True`, **no heavy smoothing**) against the best fixed-gain
  `ProportionalController`, over cells chosen to be **linear** (expect tie) and
  **nonlinear** (expect win):

  | Cell | `(W0, Tg)` | Expectation |
  |------|-----------|-------------|
  | W10/T1.0 | (10, 1.0) | ~linear → **tie** |
  | W20/T1.0 | (20, 1.0) | ~linear → **tie** |
  | W10/T2.0 | (10, 2.0) | pitch reversal → **win** |
  | W30/T1.0 | (30, 1.0) | high amplitude → **win** |
  | W30/T0.5 | (30, 0.5) | strong/sharp → **win** |

- The `Controller` implements exactly this cost in `_cost()` (lines 179–187) — the
  `Q_CL*(C_L - trim)^2 + R*delta^2` terms are the derivation's `J`; the extra
  `Q_h`, `Q_alpha`, `Q_alpha_dot`, `R_du` terms are refinements on top of the base
  proportional-equivalent law.

## Why this framing matters for the whole study

Because the one-step optimal **is** an adaptive proportional, the honest question is
no longer "optimal vs proportional" in the abstract — it is:

> *Does the adaptivity of `K` (from the model's knowledge of `s`) buy enough, over a
> well-tuned fixed gain, to justify the model? And is the information the model uses
> to compute `K` even available to it fairly?*

The rest of the investigation answers **no** and **not for free**, respectively:

- The measured wins came from a **gust oracle**, not from the gain adaptivity — and
  the oracle is unobservable ([03](03-gust-oracle-and-observability.md)).
- Giving the proportional the same information (feedforward) closes the gap
  ([04](04-equal-information-and-chatter.md)).

---

Next: [03 — Gust oracle and observability →](03-gust-oracle-and-observability.md)

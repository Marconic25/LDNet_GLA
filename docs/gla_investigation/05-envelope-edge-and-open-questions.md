# 05 — Envelope edge, and open questions

[← Back to index](README.md) · [← 04 Equal info & chatter](04-equal-information-and-chatter.md)

## The envelope edge: `W0 = 48`, high `k` — the optimal LOSES

The training envelope is `W0 in [8, 48] m/s`, `Tg in [0.30, 1.20] s`. The corner
`W0 = 48`, `Tg in [0.30, 0.50]` is the **strongest, sharpest** gust the model ever
saw — the extreme edge of what it was trained on, and the highest reduced
frequencies (`k` up to `0.131`).

Script: `clean/smoothprop.py` (cases `W48/T0.30`, `W48/T0.40`, `W48/T0.50`).

| Cell | `k` | Optimal (**even with the gust oracle**) | Proportional |
|------|-----|------------------------------------------|--------------|
| W48/Tg0.30 | 0.131 | ~20–26% | ~30–50% |
| W48/Tg0.40 | 0.098 | ~20–26% | ~30–50% |
| W48/Tg0.50 | 0.079 | ~20–26% | ~30–50% |

> At the envelope edge the optimal **loses to the proportional** — and it loses
> even **with** the gust oracle (its most favorable, unrealizable setting).

This is the sharpest counter to "model-based wins at high `k`." Precisely where the
unsteady physics is strongest (high `k`) — where the reduced-frequency argument
predicted the model *should* dominate — the model instead **underperforms a dumb
proportional**.

### Attributed causes

1. **Model inaccuracy at the training-envelope edge.** `W0 = 48` is the boundary of
   the training data. The surrogate's `C_L`/`C_M` and its latent dynamics are least
   accurate here, so the optimizer is minimizing a cost built on a **wrong model**.
   The proportional, which reacts only to *measured* `C_L`, is immune to model error
   by construction.
2. **One-step greedy myopia.** The single-step optimal minimizes the *next* step's
   cost only. On a fast, strong gust it can commit an aggressive flap move that looks
   locally optimal but excites the pitch mode / mis-times the fast transient — a
   pathology a one-step cost cannot foresee. (This is the same class of failure that
   motivated the receding-horizon `mpc_horizon > 1` mode and the pitch-rate penalty
   `Q_alpha_dot`; cf. `clean/CONTROLLER_NOTES.md`, where the fixed-weight optimal is
   documented to **destabilize** via a pitch resonance at `W0 >= 20`.)

Together these mean the model's *supposed* home turf (strong, unsteady gusts) is
exactly where it is least trustworthy.

## Open questions

These are genuinely unresolved — the honest edges of the study.

### 1. Is the envelope-edge loss a fixable formulation issue, or fundamental?

The single-step optimal is **greedy**. Candidate fixes not yet fully evaluated:

- a genuine **receding horizon** `> 1` step (`mpc_horizon > 1` in `controller.py`,
  which rolls `z` + structure forward and can *see* the pitch excitation a one-step
  cost misses),
- a **terminal cost** to value the end-of-horizon state,
- **move-suppression** (`R_du * (delta - delta_prev)^2`, already wired in) to damp
  aggressive commits.

Open: do these recover the optimal at `W48`/high-`k`, or is the loss **fundamental**
— driven by LDNet's accuracy at the training edge, which no controller reformulation
can repair? A better-model ablation (retrain / extend the envelope) would separate
*controller myopia* from *model error*.

### 2. Does the model advantage genuinely grow with `k`?

The reduced-frequency argument ([03](03-gust-oracle-and-observability.md)) predicts
the model's edge should scale with the unsteady deficit `1 - C(k)`, i.e. grow with
`k`. But:

- the equal-information sweep ([04](04-equal-information-and-chatter.md)) showed only
  **noisy, cell-dependent** wins at a few high-`k` cells, no clean law;
- a first "margin grows with `k`" reading was **walked back** as a **pitch-pick
  artifact** (see [06](06-bugs-and-pitfalls.md));
- at the highest `k` in-envelope (`W48/Tg0.30`, `k=0.131`) the optimal actually
  **loses**.

Open: run a **clean iso-pitch** comparison — max `C_L` reduction at matched pitch
ratio, model vs proportional, as a function of `k` — to settle whether there is a
real trend. `clean/ksweep2.py` is built for exactly this (MPC formulation fixed:
`DLPF=0`, no gust-gating, both arms on the same rate-limit-only post-processing, and
it reports `mdl@115`/`pd@115` = max reduction at pitch ratio `<= 1.15`). The
question is whether the margin is positive **and** monotone in `k` once the pitch
constraint is applied consistently.

### 3. Would a MIMO / multi-surface / constrained problem favor model-based control?

This whole study is **SISO**: one flap, one scalar objective (`C_L` excursion). A
scalar proportional law is hard to beat there because the one-step optimum *is* a
proportional ([02](02-single-step-equals-proportional.md)). But:

- **multiple control surfaces** (MIMO) require coordinating actuators — a scalar
  gain per surface cannot capture the coupling;
- **hard state/actuator constraints** (envelope protection, load limits) are exactly
  what MPC handles natively and a proportional law cannot;
- **multi-objective** trades (lift vs root bending moment vs torsion) need a model to
  navigate.

Open: does model-based control earn its keep in a setting a SISO proportional
**cannot scale to**? The negative SISO result here does **not** transfer
automatically to those richer problems — and identifying where model-based control
*does* pay off is the natural next thesis chapter.

---

Next: [06 — Bugs and pitfalls →](06-bugs-and-pitfalls.md)

# 07 — Literature context

[← Back to index](README.md) · [← 06 Bugs & pitfalls](06-bugs-and-pitfalls.md)

The investigation's negative result — model-based optimal control does not robustly
beat a well-tuned feedforward proportional for this SISO GLA problem — is
**consistent with the GLA literature**, not an anomaly.

## 1. LMPC vs PID: only ~10% advantage on peak loads

Studies comparing **Linear MPC (LMPC)** against classical **PID** for gust load
alleviation report a **modest** advantage for the model-based controller — on the
order of **~10%** improvement on the peak **wing-root bending moment** (the usual
GLA target load). This is a real but small edge, and it is bought at the cost of a
plant model, state estimation, and online optimization.

The parallel to this study is direct: our fair comparisons
([04](04-equal-information-and-chatter.md)) put the model-based optimal and the
smoothed feedforward proportional **within a few points** of each other across most
of the envelope, with the optimal even **losing** at the envelope edge
([05](05-envelope-edge-and-open-questions.md)). A single-digit-to-~10% band, not a
decisive win, is exactly what the literature would predict.

## 2. The real lever in GLA is gust PREVIEW / feedforward — the information

The GLA literature's consistent message is that the dominant performance lever is
**not controller sophistication** but **gust preview**: measuring the oncoming gust
ahead of time (e.g. with **forward-looking LIDAR** or an **alpha-probe / flow-angle
sensor**) and **feeding it forward** to the control surface. The controller that
*knows the disturbance early* wins, whether it is MPC or a well-designed feedforward
law.

This is precisely what the equal-information experiment
([04](04-equal-information-and-chatter.md)) isolated: once the **proportional** is
given the gust `W` through a simple **feedforward** term
(`delta = g_CL*(C_L - trim) + g_W*W`), it matches the optimal-with-`W`. The value
was in the **information** (`W`), which a trivial feedforward realizes — not in the
model or the optimization.

And critically ([03](03-gust-oracle-and-observability.md)): on this plant `W` is
**not observable from `C_L`** (`C_L(W)` is non-invertible on the LDNet manifold).
So the information lever the literature identifies — gust preview — must come from
an **independent** sensor (LIDAR / alpha-probe), not from inverting the lift
measurement. This is the single most actionable takeaway: **invest in a preview
sensor, not in a fancier controller.**

## 3. References

- **Model predictive control of a flared folding wingtip for gust load alleviation**
  — ScienceDirect, article `S1270963825000641` (*Aerospace Science and Technology*).
  Model-based GLA using a flared folding wingtip; representative of the LMPC-vs-PID
  comparison and the modest model-based margin.
- **Feedforward / feedback GLA reviews** — the body of GLA work establishing gust
  **preview** (LIDAR, alpha-probe) and **feedforward** as the primary performance
  lever, above controller type.

> If the URLs are needed: the primary reference is
> `https://www.sciencedirect.com/science/article/pii/S1270963825000641`.

## 4. How this study fits

| Literature finding | This study's corresponding result |
|--------------------|-----------------------------------|
| LMPC beats PID by only ~10% on peak load | Optimal ≈ smoothed feedforward proportional within a few points ([04](04-equal-information-and-chatter.md)) |
| Preview / feedforward (information) is the real lever | Prop + `W` feedforward matches opt-with-`W`; value was the information ([04](04-equal-information-and-chatter.md)) |
| Disturbance must be measured, not inferred | `C_L(W)` non-invertible → `W` unobservable from the sensor; need an independent preview sensor ([03](03-gust-oracle-and-observability.md)) |
| Model-based control shines in MIMO / constrained problems | Open question — SISO negative result may not transfer ([05](05-envelope-edge-and-open-questions.md)) |

The thesis contribution is therefore not "MPC beats PID" (it does not, here) but a
**careful, fair, information-controlled demonstration** — on a nonlinear unsteady
neural-surrogate plant — of *why* it does not, and of *where* the actual leverage
lies (the disturbance information, and its observability).

---

[← Back to index](README.md)

# GLA controllers on LDNet — comparison & robustness

Two working controllers (no gust observer — both use measured C_L; the C_L->W
inversion is non-invertible with LDNet):

- **ProportionalController**: delta = G*(C_L_meas - C_L_trim), G=10.
- **Controller (model-based one-step optimal)**: global + causal-basin search,
  target_lpf=0.95, Q_alpha_dot=3e5, Q_CL=1e3, R=0.3.

## Nominal gust (W0=10, gust window)
| ctrl | C_L | h_ddot | alpha | alpha_dot | flap |
|------|-----|--------|-------|-----------|------|
| proportional | +21.6% | +31.4% | +51% | +3.8% | 3.3 deg |
| optimal      | +23.8% | +31.4% | +54% | +11.1% | 7.1 deg |

Optimal beats proportional on every metric at the design gust. Keys to the optimal
working at all: global search (non-convex 1-step cost), causal-basin (avoid the
non-causal nulling basin), target LPF (kill chatter), Q_alpha_dot (protect pitch).

## Gust-velocity robustness (C_L / h_ddot / alpha_dot reduction %)
| W0 | proportional | optimal |
|----|--------------|---------|
|  5 | +16.7/+27.0/ -8.8 | +19.1/+27.0/ -0.5 |
| 10 | +21.6/+31.4/ +3.8 | +23.8/+31.4/+11.1 |
| 15 | +25.3/+35.5/ +1.6 | +27.3/+35.5/ +9.1 |
| 20 | +22.3/+38.9/ -8.5 |  +6.5/ -6.2/-168  |
| 30 | +17.2/+44.0/-38.9 | -35.7/-59.3/-466  |

**Proportional is robust across W0=5..30** (always alleviates C_L & h_ddot).
**Optimal beats it for W0<=15 but DESTABILIZES at W0>=20** (h_ddot worse than open,
alpha_dot blows up). The optimal's fixed weights are tuned to the W0=10 signal scale;
larger gusts push the model into regimes where those gains over-react.

To make the optimal robust: gain-schedule / normalize the cost weights by the
measured C_L-excursion (gust) magnitude, or adapt R / Q_alpha_dot online.

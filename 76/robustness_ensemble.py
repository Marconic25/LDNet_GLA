"""
Phase 3 — basin/robustness ensemble for the candidate winner
  SS wnext R3e-4 : one-step optimal, causal gate, evaluated against W(t+dt).
Reached +76.5% deterministically (B=1) at W30/Tg0.4. Here we perturb the initial
condition (X0 +/- 1e-6, ~1e10x the rounding scale) and the gust amplitude
(W0 +/- 0.1%) and require ALL members to land within a few CLred points -- i.e.
the good branch is a real attractor, not a rounding knife-edge.

For contrast we run the SAME ensemble on the frozen-z W(t) controller (the one
that sits on the bad branch) to show it is the fragile one.
"""
import numpy as np
import harness as H
from controllers import OptGrid

W0, Tg = 30.0, 0.4
OL = H.scalar_rollout(None, W0, Tg)
cex0 = H.cex_of(OL['CL'], OL['_t'], Tg)
print(f"# W30/Tg0.4 DAMULT=3 | open cex0={cex0:.4f}", flush=True)

rng = np.random.default_rng(0)
members = [('base', None, 1.0)]
# pure X0 perturbations (each component +/-1e-6)
for k in range(4):
    for s in (+1.0, -1.0):
        dx = np.zeros(4); dx[k] = s * 1e-6
        members.append((f'X0[{k}]{s:+.0e}', dx, 1.0))
# pure gust-amplitude perturbations
for sc in (0.999, 0.9995, 1.0005, 1.001):
    members.append((f'W0x{sc}', None, sc))
# combined random perturbations
for j in range(4):
    dx = rng.uniform(-1e-6, 1e-6, 4)
    sc = 1.0 + rng.uniform(-1e-3, 1e-3)
    members.append((f'rand{j}', dx, sc))


def run_ensemble(make_ctrl, label):
    print(f"\n=== ensemble: {label} ===", flush=True)
    clreds = []
    for name, dx, sc in members:
        x0 = None if dx is None else (H.X0 + dx)
        r = H.scalar_rollout(make_ctrl(), W0, Tg, X0_override=x0, W0_scale=sc)
        # metric vs the MATCHING open loop (same perturbation) for fairness
        rO = H.scalar_rollout(None, W0, Tg, X0_override=x0, W0_scale=sc)
        m = H.metrics(r, rO, Tg)
        clreds.append(m['clred'])
        print(f"  {name:14s} CLred={m['clred']:+6.1f}%  flap={m['flap_max']:5.1f} "
              f"pitch={m['pitchpk']*180/np.pi:6.3f} {m['flag']}", flush=True)
    clreds = np.array(clreds)
    print(f"  --> min={clreds.min():+.1f}%  max={clreds.max():+.1f}%  "
          f"spread={clreds.max()-clreds.min():.1f}pts  std={clreds.std():.2f}", flush=True)
    return clreds


win = run_ensemble(lambda: OptGrid(R=3e-4, G=161, gate='hard', use_wnext=True),
                   'SS wnext R3e-4 (candidate)')
bad = run_ensemble(lambda: OptGrid(R=3e-4, G=161, gate='hard', use_wnext=False),
                   'SS W(t) R3e-4 (bad-branch control)')

print("\n# VERDICT:", flush=True)
print(f"#   wnext ensemble spread = {win.max()-win.min():.1f} pts (robust if small)", flush=True)
print(f"#   W(t)  ensemble spread = {bad.max()-bad.min():.1f} pts", flush=True)
print("# DONE", flush=True)

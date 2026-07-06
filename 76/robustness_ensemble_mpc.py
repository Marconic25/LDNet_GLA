"""
Perturbation ensemble for the multi-step winner MPC N4 gate=none R3e-4 (+79.2% at
W30/Tg0.4). Same protocol as robustness_ensemble.py: X0 +/-1e-6, W0 +/-0.1%.
A real attractor keeps every member within a few CLred points.
"""
import numpy as np
import harness as H
from controllers import MPCConst

W0, Tg = 30.0, 0.4
print(f"# MPC N4 none R3e-4 ensemble  W30/Tg0.4 DAMULT=3", flush=True)
rng = np.random.default_rng(1)
members = [('base', None, 1.0)]
for k in (0, 2):
    for s in (+1.0, -1.0):
        dx = np.zeros(4); dx[k] = s * 1e-6
        members.append((f'X0[{k}]{s:+.0e}', dx, 1.0))
for sc in (0.999, 1.001):
    members.append((f'W0x{sc}', None, sc))
for j in range(2):
    dx = rng.uniform(-1e-6, 1e-6, 4)
    sc = 1.0 + rng.uniform(-1e-3, 1e-3)
    members.append((f'rand{j}', dx, sc))

clreds = []
for name, dx, sc in members:
    x0 = None if dx is None else (H.X0 + dx)
    r = H.scalar_rollout(MPCConst(N=4, R=3e-4, G=161, gate='none'), W0, Tg,
                         X0_override=x0, W0_scale=sc)
    rO = H.scalar_rollout(None, W0, Tg, X0_override=x0, W0_scale=sc)
    m = H.metrics(r, rO, Tg)
    clreds.append(m['clred'])
    print(f"  {name:14s} CLred={m['clred']:+6.1f}%  flap={m['flap_max']:5.1f} "
          f"pitch={m['pitchpk']*180/np.pi:6.3f} {m['flag']}", flush=True)
clreds = np.array(clreds)
print(f"  --> min={clreds.min():+.1f}% max={clreds.max():+.1f}% "
      f"spread={clreds.max()-clreds.min():.1f}pts std={clreds.std():.2f}", flush=True)
print("# DONE", flush=True)

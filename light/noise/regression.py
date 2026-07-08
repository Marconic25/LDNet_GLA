"""
Stage-0 regression gate — MUST pass before any study axis runs.

Anchors (W30/Tg0.4, DAMULT=3):
  1. open-loop cex0 = 0.4600
  2. sigma=0, R=3e-4, refine=True  -> CLred = +76.6%
  3. sigma=0, R=1e-4, refine=True  -> CLred = +80.7%
  4. OptimalRdu(R_du=0) == OptimalController exactly (validates the copied
     compute in controllers_ref.py)
  5. sigma=2% white on W(t+dt), refine=False, R=3e-4, seeds 100..105
     -> mean CLred = -0.5% [-26.1, +18.5]  == 76/preview.log (cluster) section
     C, which this harness reproduces BIT-EXACTLY (verified 2026-07-07).
     NOTE: 76/NOTES.md quotes "+5.2% (-25..+29)" for this cell — that number
     is from 76/mpc_noise.py, which uses rng base 200+seed (not 100+seed):
     a different noise realization of the same experiment. With per-seed
     std ~19, a 6-seed mean has SE ~8 pts; -0.5 and +5.2 are the same
     statistical result. The regime is chaotic at seed level — that is the
     finding, not an error.

Exit code 1 on any failure.
"""
import numpy as np
import harness_noise as H
from controllers_ref import make_optimal_rdu

W0, Tg = 30.0, 0.4
FAIL = []

def check(name, val, ref, tol):
    ok = abs(val - ref) <= tol
    print(f"  {'OK ' if ok else 'FAIL'} {name}: {val:+.4g} (ref {ref:+.4g}, tol {tol:g})",
          flush=True)
    if not ok:
        FAIL.append(name)

print("# stage-0 regression gate | W30/Tg0.4 DAMULT=3", flush=True)

OL = H.rollout(None, W0, Tg)
cex0 = float(np.max(np.abs(OL['CL'][H._gust_window(OL['_t'], Tg)] - H.CLTRIM)))
check("open cex0", cex0, 0.4600, 5e-4)

r3 = H.rollout(H.make_optimal(R=3e-4), W0, Tg)
m3 = H.metrics(r3, OL, Tg)
check("sigma=0 R=3e-4 CLred", m3['clred'], 76.6, 0.5)
print(f"       flap={m3['flap_max']:.1f} flag='{m3['flag']}'", flush=True)

r1 = H.rollout(H.make_optimal(R=1e-4), W0, Tg)
m1 = H.metrics(r1, OL, Tg)
check("sigma=0 R=1e-4 CLred", m1['clred'], 80.7, 0.5)
print(f"       flap={m1['flap_max']:.1f} flag='{m1['flag']}'", flush=True)

rdu = H.rollout(make_optimal_rdu(R=3e-4, R_du=0.0), W0, Tg)
dmax = float(np.max(np.abs(rdu['de'] - r3['de'])))
check("OptimalRdu(R_du=0) == Optimal (max |d_de|)", dmax, 0.0, 1e-12)

crs = []
for seed in range(6):
    rng = np.random.default_rng(100 + seed)
    wf = lambda i, Wt, N: max(0.0, Wt[min(i + 1, N - 1)] + rng.normal(0.0, 0.02 * W0))
    r = H.rollout(H.make_optimal(R=3e-4, refine=False), W0, Tg, wc_fun=wf)
    m = H.metrics(r, OL, Tg)
    crs.append(m['clred'])
    print(f"  seed {100+seed}: CLred {m['clred']:+6.1f}%  flag='{m['flag']}'", flush=True)
crs = np.array(crs)
print(f"  sigma=2% refine=False: mean {crs.mean():+.1f}% "
      f"[{crs.min():+.1f},{crs.max():+.1f}] "
      f"(76/preview.log ref: -0.5 [-26.1,+18.5]; NOTES' +5.2 = rng 200+seed)",
      flush=True)
check("sigma=2% refine=False mean CLred", float(crs.mean()), -0.5, 1.0)
check("sigma=2% refine=False min CLred", float(crs.min()), -26.1, 1.0)
check("sigma=2% refine=False max CLred", float(crs.max()), 18.5, 1.0)

if FAIL:
    print(f"# REGRESSION FAILED: {FAIL}", flush=True)
    raise SystemExit(1)
print("# REGRESSION PASSED", flush=True)

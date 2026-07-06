"""
Phase 4 — generalization of the SS wnext controller to neighbour cells with an
R sweep only (NO per-cell hand tuning). Cells: W30/T0.4 (home), W30/T0.3,
W30/T0.5, W20/T0.4. Baselines per cell: open loop + best prop-W (small gain grid).
"""
import numpy as np
import harness as H
from controllers import OptGrid, PropW, MPCConst

CELLS = [(30.0, 0.4), (30.0, 0.3), (30.0, 0.5), (20.0, 0.4)]
R_SWEEP = (3e-4, 1e-3, 3e-3)

print(f"{'cell':10s} {'arm':16s} {'CLred':>7s} {'flap':>5s} {'pitch':>6s} {'adrms':>6s} {'flag':>6s}", flush=True)
print('-'*62, flush=True)
for (W0, Tg) in CELLS:
    OL = H.scalar_rollout(None, W0, Tg)
    cex0 = H.cex_of(OL['CL'], OL['_t'], Tg)
    cell = f'W{int(W0)}/T{Tg:.1f}'
    print(f"{cell:10s} {'open cex0':16s} {cex0:7.4f}", flush=True)
    # prop-W best
    bpw = None
    for gCL in (-60., -120.):
        for gW in (0.0, -0.3):
            r = H.scalar_rollout(PropW(gain_CL=gCL, gain_W=gW), W0, Tg)
            m = H.metrics(r, OL, Tg)
            if bpw is None or m['clred'] > bpw[0]['clred']:
                bpw = (m, gCL, gW)
    m, gCL, gW = bpw
    print(f"{'':10s} {'prop-W best':16s} {m['clred']:+6.1f}% {m['flap_max']:5.1f} "
          f"{m['pitchpk']*180/np.pi:6.3f} {m['adrms']:6.2f} {m['flag']:>6s}  (g={gCL:g},{gW:g})", flush=True)
    # wnext R sweep (single-step winner)
    for R in R_SWEEP:
        r = H.scalar_rollout(OptGrid(R=R, G=161, gate='hard', use_wnext=True), W0, Tg)
        m = H.metrics(r, OL, Tg)
        print(f"{'':10s} {'wnext R='+f'{R:g}':16s} {m['clred']:+6.1f}% {m['flap_max']:5.1f} "
              f"{m['pitchpk']*180/np.pi:6.3f} {m['adrms']:6.2f} {m['flag']:>6s}", flush=True)
    # MPC N4 gate=none R sweep (multi-step alternative)
    for R in (3e-4, 1e-3):
        r = H.scalar_rollout(MPCConst(N=4, R=R, G=161, gate='none'), W0, Tg)
        m = H.metrics(r, OL, Tg)
        print(f"{'':10s} {'MPC4none R='+f'{R:g}':16s} {m['clred']:+6.1f}% {m['flap_max']:5.1f} "
              f"{m['pitchpk']*180/np.pi:6.3f} {m['adrms']:6.2f} {m['flag']:>6s}", flush=True)
    print('', flush=True)
print("# DONE", flush=True)

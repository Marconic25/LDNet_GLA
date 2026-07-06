"""
Honest scalar prop-W baseline (equal-info: knows W) at W30/Tg0.4, DAMULT=3.
Wide gain grid; prop-W is a smooth law (no argmin near-ties) so B=1 scalar is
the robust number. Reference clean/propw.py reports +39.1% here.
"""
import numpy as np
import harness as H
from controllers import PropW

W0, Tg = 30.0, 0.4
OL = H.scalar_rollout(None, W0, Tg)
cex0 = H.cex_of(OL['CL'], OL['_t'], Tg)
print(f"# W30/Tg0.4 DAMULT=3 | open cex0={cex0:.4f}", flush=True)
print(f"{'gCL':>6s} {'gW':>6s} {'CLred':>7s} {'flap':>5s} {'pitch':>6s}", flush=True)
best = None
for gCL in (-40., -60., -80., -120., -160.):
    for gW in (0.0, -0.1, -0.2, -0.3, -0.5, -0.8):
        r = H.scalar_rollout(PropW(gain_CL=gCL, gain_W=gW), W0, Tg)
        m = H.metrics(r, OL, Tg)
        print(f"{gCL:6.0f} {gW:6.2f} {m['clred']:+6.1f}% {m['flap_max']:5.1f} "
              f"{m['pitchpk']*180/np.pi:6.3f}", flush=True)
        if best is None or m['clred'] > best[0]:
            best = (m['clred'], gCL, gW, r['CL'], r['de'], r['al'], r['ad'], m['flap_max'], m['pitchpk'])
print(f"\n# prop-W BEST: CLred={best[0]:+.1f}% at gCL={best[1]:g} gW={best[2]:g} "
      f"flap_max={best[7]:.1f} pitchpk={best[8]*180/np.pi:.3f}deg", flush=True)
np.savez_compressed('results_propw.npz', ts=OL['_t'], Wt=OL['_Wt'], cex0=cex0,
    CLTRIM=H.CLTRIM, OL_CL=OL['CL'], OL_al=OL['al'], OL_ad=OL['ad'],
    PW_clred=best[0], PW_gCL=best[1], PW_gW=best[2],
    PW_CL=best[3], PW_de=best[4], PW_al=best[5], PW_ad=best[6])
print("# DONE", flush=True)

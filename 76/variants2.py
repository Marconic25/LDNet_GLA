"""
Phase 2b — remaining single-step ideas (z-aware predict_step, W(t+dt)) and the
MPC N=2..6 fallback at W30/Tg0.4, DAMULT=3. Honest B=1 rollouts.
"""
import numpy as np
import harness as H
from controllers import OptGrid, MPCConst

W0, Tg = 30.0, 0.4
OL = H.scalar_rollout(None, W0, Tg)
cex0 = H.cex_of(OL['CL'], OL['_t'], Tg)
print(f"# W30/Tg0.4 DAMULT=3 | open cex0={cex0:.4f} CLTRIM={H.CLTRIM:.4f}", flush=True)

CONFIGS = [
    # remaining single-step ideas
    ('SS wnext R3e-4',          OptGrid(R=3e-4, G=161, gate='hard', use_wnext=True)),
    ('SS wnext R1e-3',          OptGrid(R=1e-3, G=161, gate='hard', use_wnext=True)),
    ('SS zaware R3e-4',         OptGrid(R=3e-4, G=161, gate='hard', z_aware=True)),
    ('SS zaware R1e-3',         OptGrid(R=1e-3, G=161, gate='hard', z_aware=True)),
    ('SS zaware R3e-4 Rdu1e-3', OptGrid(R=3e-4, G=161, gate='hard', z_aware=True, R_du=1e-3)),
    ('SS zaware wnext R3e-4',   OptGrid(R=3e-4, G=161, gate='hard', z_aware=True, use_wnext=True)),
    # MPC constant-delta sweep (no smoothing)
    ('MPC N2 R3e-4 hard',       MPCConst(N=2, R=3e-4, G=161, gate='hard')),
    ('MPC N3 R3e-4 hard',       MPCConst(N=3, R=3e-4, G=161, gate='hard')),
    ('MPC N4 R3e-4 hard',       MPCConst(N=4, R=3e-4, G=161, gate='hard')),
    ('MPC N6 R3e-4 hard',       MPCConst(N=6, R=3e-4, G=161, gate='hard')),
    ('MPC N4 R1e-3 hard',       MPCConst(N=4, R=1e-3, G=161, gate='hard')),
    ('MPC N4 R3e-4 none',       MPCConst(N=4, R=3e-4, G=161, gate='none')),
    ('MPC N4 R3e-4 Qad50',      MPCConst(N=4, R=3e-4, G=161, gate='hard', Qad=50.0)),
]

print(f"\n{'config':26s} {'CLred':>7s} {'flap':>5s} {'pitch':>6s} {'adrms':>6s} {'flag':>6s}", flush=True)
print('-'*64, flush=True)
save = dict(ts=OL['_t'], Wt=OL['_Wt'], cex0=cex0, CLTRIM=H.CLTRIM,
            OL_CL=OL['CL'], OL_al=OL['al'], OL_ad=OL['ad'], OL_de=OL['de'])
for name, ctrl in CONFIGS:
    r = H.scalar_rollout(ctrl, W0, Tg)
    m = H.metrics(r, OL, Tg)
    print(f"{name:26s} {m['clred']:+6.1f}% {m['flap_max']:5.1f} "
          f"{m['pitchpk']*180/np.pi:6.3f} {m['adrms']:6.2f} {m['flag']:>6s}", flush=True)
    key = name.replace(' ', '_').replace('.', 'p')
    save[f'{key}__CL'] = r['CL']; save[f'{key}__de'] = r['de']
    save[f'{key}__al'] = r['al']; save[f'{key}__ad'] = r['ad']
    save[f'{key}__clred'] = m['clred']
np.savez_compressed('results_variants2.npz', **save)
print("\nSaved results_variants2.npz", flush=True)
print("# DONE", flush=True)

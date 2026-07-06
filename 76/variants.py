"""
Phase 2 — single-step controller variants at W30/Tg0.4, DAMULT=3.
Each config is an honest B=1 scalar rollout (deterministic, batch-position
independent). Baselines: open loop and prop-W (best of a small gain grid).

Reference: open cex0=0.4600; good branch +76.1%; prop-W +39.1%; honest scalar
one-step best +28.9% (R=1e-3, coarse grid). Goal: robustly beat +39.1%, target +76%.
"""
import numpy as np
import harness as H
from controllers import OptGrid, PropW

W0, Tg = 30.0, 0.4

OL = H.scalar_rollout(None, W0, Tg)
cex0 = H.cex_of(OL['CL'], OL['_t'], Tg)
print(f"# W30/Tg0.4 DAMULT=3 | open cex0={cex0:.4f} CLTRIM={H.CLTRIM:.4f}", flush=True)

# prop-W: small gain grid, keep the best (equal-info baseline)
best_pw = None
for gCL in (-40., -80., -120.):
    for gW in (0.0, -0.2):
        r = H.scalar_rollout(PropW(gain_CL=gCL, gain_W=gW), W0, Tg)
        m = H.metrics(r, OL, Tg)
        if best_pw is None or m['clred'] > best_pw[1]['clred']:
            best_pw = ((gCL, gW), m, r)
(gCL, gW), mpw, rpw = best_pw
print(f"# prop-W best: gain_CL={gCL:g} gain_W={gW:g} -> CLred={mpw['clred']:+.1f}% "
      f"flap_max={mpw['flap_max']:.1f} pitchpk={mpw['pitchpk']*180/np.pi:.3f}deg", flush=True)

CONFIGS = [
    # sanity / grid effect
    ('hard G161 R3e-4',            OptGrid(R=3e-4, G=161, gate='hard')),
    ('hard G161 R1e-3',            OptGrid(R=1e-3, G=161, gate='hard')),
    ('hard G15  R1e-3 refine',     OptGrid(R=1e-3, G=15,  gate='hard', refine=True)),
    # R_du move suppression (gate hard, G161, R=3e-4)
    ('hard R3e-4 Rdu1e-4',         OptGrid(R=3e-4, G=161, gate='hard', R_du=1e-4)),
    ('hard R3e-4 Rdu3e-4',         OptGrid(R=3e-4, G=161, gate='hard', R_du=3e-4)),
    ('hard R3e-4 Rdu1e-3',         OptGrid(R=3e-4, G=161, gate='hard', R_du=1e-3)),
    ('hard R3e-4 Rdu3e-3',         OptGrid(R=3e-4, G=161, gate='hard', R_du=3e-3)),
    ('hard R3e-4 Rdu1e-2',         OptGrid(R=3e-4, G=161, gate='hard', R_du=1e-2)),
    # gate deadband (G161, R=3e-4, Rdu=0)
    ('db2e-3 R3e-4',               OptGrid(R=3e-4, G=161, gate='db', db=2e-3)),
    ('db5e-3 R3e-4',               OptGrid(R=3e-4, G=161, gate='db', db=5e-3)),
    ('db1e-2 R3e-4',               OptGrid(R=3e-4, G=161, gate='db', db=1e-2)),
    ('db2e-2 R3e-4',               OptGrid(R=3e-4, G=161, gate='db', db=2e-2)),
    # combos
    ('db1e-2 R3e-4 Rdu1e-3',       OptGrid(R=3e-4, G=161, gate='db', db=1e-2, R_du=1e-3)),
    ('none R3e-4 Rdu1e-3',         OptGrid(R=3e-4, G=161, gate='none', R_du=1e-3)),
    ('none R3e-4 Rdu3e-3',         OptGrid(R=3e-4, G=161, gate='none', R_du=3e-3)),
]

print(f"\n{'config':26s} {'CLred':>7s} {'flap':>5s} {'pitch':>6s} {'adrms':>6s} {'flag':>6s}", flush=True)
print('-'*64, flush=True)
save = dict(ts=OL['_t'], Wt=OL['_Wt'], cex0=cex0, CLTRIM=H.CLTRIM,
            OL_CL=OL['CL'], OL_al=OL['al'], OL_ad=OL['ad'], OL_de=OL['de'],
            PW_CL=rpw['CL'], PW_al=rpw['al'], PW_ad=rpw['ad'], PW_de=rpw['de'],
            PW_clred=mpw['clred'])
names = []
for name, ctrl in CONFIGS:
    r = H.scalar_rollout(ctrl, W0, Tg)
    m = H.metrics(r, OL, Tg)
    print(f"{name:26s} {m['clred']:+6.1f}% {m['flap_max']:5.1f} "
          f"{m['pitchpk']*180/np.pi:6.3f} {m['adrms']:6.2f} {m['flag']:>6s}", flush=True)
    key = name.replace(' ', '_').replace('.', 'p')
    save[f'{key}__CL'] = r['CL']; save[f'{key}__de'] = r['de']
    save[f'{key}__al'] = r['al']; save[f'{key}__ad'] = r['ad']
    save[f'{key}__clred'] = m['clred']
    names.append((name, key))
save['names'] = np.array([f'{n}||{k}' for n, k in names])
np.savez_compressed('results_variants.npz', **save)
print("\nSaved results_variants.npz", flush=True)
print("# DONE", flush=True)

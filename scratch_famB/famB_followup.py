#!/usr/bin/env python3
"""Follow-up: absolute errors, family shares of test MSE, delta ramp timing."""
import numpy as np, h5py

ROOT = '/home/marco/LDNet_OF'
NORM = {'Fy': (-18.65, 547.29), 'Mz': (-44.99, 42.24)}
def nf(v, o):
    lo, hi = NORM[o]; return (2*v - lo - hi)/(hi - lo)

for tag, p in [('TF in6/d10', f'{ROOT}/results/sensitivity/in6/latent_10/traces.npz'),
               ('ROLLOUT', f'{ROOT}/results/sensitivity/rollout_eval/in6/latent_10/traces.npz'),
               ('TF in6/d1', f'{ROOT}/results/sensitivity/in6/latent_1/traces.npz')]:
    d = np.load(p)
    print(f'--- {tag} ---')
    tot = 0.0; shares = {}
    for fam in ['A', 'B', 'Cc']:
        s = 0.0
        for o in ['Fy', 'Mz']:
            f, r = d[f'{fam}_fom_{o}'], d[f'{fam}_rom_{o}']
            rms = np.sqrt(np.mean((r-f)**2))
            fam_rng = f.max()-f.min()
            glob_nrmse = rms/(NORM[o][1]-NORM[o][0])
            e2 = np.mean((nf(r,o)-nf(f,o))**2)
            s += e2
            print(f'  {fam:2s} {o}: abs RMS={rms:7.3f}  fam.range={fam_rng:7.2f}  '
                  f'NRMSE(global)={glob_nrmse*100:5.2f}%  norm.MSE={e2:.2e}')
        shares[fam] = s; tot += s
    for fam in shares:
        print(f'  share of shown-trace normalized MSE: {fam}: {shares[fam]/tot*100:.1f}%')

# delta ramp timing in test B
with h5py.File(f'{ROOT}/data/GLA_test.h5','r') as f:
    fams = f['sim_families'][:].astype(str)
    tin = f['input_signals'][:]; tt = f['times'][:]; tout = f['output_signals'][:]
iB = int(np.where(fams=='B')[0][0])
dlt = tin[iB,:,4]
ipk = np.argmax(np.abs(dlt))
print(f'\nB delta: value t=0 {dlt[0]:.3f} deg -> extreme {dlt[ipk]:.3f} deg at t={tt[ipk]:.3f}s')
i10 = np.argmax(np.abs(dlt) > 0.1*np.abs(dlt[ipk])); i90 = np.argmax(np.abs(dlt) > 0.9*np.abs(dlt[ipk]))
print(f'  10-90% rise: {tt[i10]:.4f}s -> {tt[i90]:.4f}s')
print(f'  delta at end: {dlt[-1]:.3f} deg')
# B Fy trajectory summary
fy = tout[iB,:,0,0]
print(f'B Fy: t=0 {fy[0]:.1f}, min {fy.min():.1f} at t={tt[np.argmin(fy)]:.3f}s, final {fy[-1]:.1f}')
# all-B train delta extremes
with h5py.File(f'{ROOT}/data/GLA_train.h5','r') as f:
    trf = f['sim_families'][:].astype(str)
    dtr = f['input_signals'][:,:,4]
mB = trf=='B'
print(f'train B delta extremes per sim: min {dtr[mB].min():.2f}, max {dtr[mB].max():.2f} deg; '
      f'global delta range across all train: [{dtr.min():.2f},{dtr.max():.2f}]')

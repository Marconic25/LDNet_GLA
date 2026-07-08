"""Is C_L(W) invertible? Sweep W at realistic operating points (state x + latent z
captured during a gust) and check monotonicity -- decides whether a gust observer can
recover W_hat from measured C_L."""
import os, numpy as np
import mpc_gust as M

DT = M.DT; U = M.U; CLTRIM = M.CLTRIM; q = M.q; C = M.C
W0, Tg = 30.0, 0.5
TEND = 1.5
a = M.a
N = int(round(TEND / DT)) + 1
t = np.arange(N) * DT
Wt = np.array([M.gust(tt, W0, Tg) for tt in t])

# open-loop rollout, capture (x, z) at a few instants
a.reset(dt=DT); x = M.X0.copy()
snaps = {}
idx = {'rise(t=0.12)': int(0.12 / DT), 'peak(t=0.25)': int(0.25 / DT), 'decay(t=0.38)': int(0.38 / DT)}
for i in range(N):
    for lab, ii in idx.items():
        if i == ii:
            snaps[lab] = (x.copy(), np.array(a._z, float).copy(), float(Wt[i]))
    cl, cm = a.predict(x, 0.0, Wt[i], U)
    a.advance(x, 0.0, Wt[i], U, DT); x = M.structure.step_dp45(x, q * cl, q * cm * C, DT)

Wg = np.linspace(0.0, 40.0, 81)
print(f'# C_L(W) sweep at captured operating points (delta=0)  DAMULT={os.environ.get("DAMULT","1")}', flush=True)
out = {'Wg': Wg}
for lab, (xs, zs, Wtrue) in snaps.items():
    a._z = zs.copy()
    CLs = np.array([float(a.predict(xs, 0.0, float(w), U)[0]) for w in Wg])
    dC = np.diff(CLs)
    mono = 'MONOTONE-up' if np.all(dC >= -1e-6) else ('MONOTONE-dn' if np.all(dC <= 1e-6) else 'NON-MONOTONE')
    nsign = int(np.sum(np.diff(np.sign(dC)) != 0))
    print(f'{lab:16s} Wtrue={Wtrue:5.1f}  CL range [{CLs.min():.3f},{CLs.max():.3f}]  '
          f'{mono}  sign-changes(dCL/dW)={nsign}', flush=True)
    out[lab] = CLs
import os as _o; _o.makedirs('results', exist_ok=True)
np.savez('results/clw_sweep.npz', **out)
print('# saved results/clw_sweep.npz', flush=True)
print('# DONE', flush=True)

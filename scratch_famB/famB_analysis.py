#!/usr/bin/env python3
"""Family-B diagnosis: per-family metrics, error characterization, H1-H4 evidence."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

ROOT = '/home/marco/LDNet_OF'
OUT = os.path.join(ROOT, 'scratch_famB')
FAMS = ['A', 'B', 'Cc']
OUTS = ['Fy', 'Mz']

# Normalization (from sensitivity_latent.py)
NORM = {'Fy': (-18.65, 547.29), 'Mz': (-44.99, 42.24)}
ALPHA = 0.05

def norm_fw(v, name):
    lo, hi = NORM[name]
    return (2.0*v - lo - hi) / (hi - lo)

def g(u):      return (u**3 + ALPHA*u) / (1 + ALPHA)
def gprime(u): return (3*u**2 + ALPHA) / (1 + ALPHA)

def g_inv(y):
    """invert g elementwise (monotone cubic)."""
    y = np.atleast_1d(np.asarray(y, float))
    out = np.empty_like(y)
    for i, yi in enumerate(y.ravel()):
        r = np.roots([1.0, 0.0, ALPHA, -yi*(1+ALPHA)])
        r = r[np.abs(r.imag) < 1e-9].real
        out.ravel()[i] = r[np.argmin(np.abs(r))] if len(r) else np.nan
    return out.reshape(y.shape)

# ------------------------------------------------------------------
# TASK 1: per-family NRMSE / 1-rho for every traces.npz
# ------------------------------------------------------------------
runs = []
for p in sorted(glob.glob(f'{ROOT}/results/sensitivity/in*/latent_*/traces.npz')):
    parts = p.split('/')
    runs.append((parts[-3], int(parts[-2].split('_')[1]), 'TF', p))
runs.append(('in6', 10, 'ROLLOUT', f'{ROOT}/results/sensitivity/rollout_eval/in6/latent_10/traces.npz'))
runs.sort(key=lambda r: (r[0], r[1], r[2]))

table = []
for inset, ds, mode, p in runs:
    d = np.load(p)
    t = d['time']
    row = {'inset': inset, 'ds': ds, 'mode': mode}
    for fam in FAMS:
        for o in OUTS:
            f = d[f'{fam}_fom_{o}']; r = d[f'{fam}_rom_{o}']
            rng = f.max() - f.min()
            nrmse = np.sqrt(np.mean((r-f)**2)) / rng
            rho = np.corrcoef(f, r)[0, 1]
            row[f'{fam}_{o}_nrmse'] = float(nrmse)
            row[f'{fam}_{o}_1mrho'] = float(1-rho)
    table.append(row)

with open(f'{OUT}/per_family_metrics.json', 'w') as fp:
    json.dump(table, fp, indent=2)

# text table
lines = []
hdr = f"{'run':>16} | " + ' | '.join(f'{fam}.{o}' for fam in FAMS for o in OUTS)
lines.append('NRMSE (per-family range-normalized)')
lines.append(hdr)
for row in table:
    tag = f"{row['inset']}/d{row['ds']}/{row['mode']}"
    lines.append(f"{tag:>16} | " + ' | '.join(f"{row[f'{fam}_{o}_nrmse']*100:5.1f}%" for fam in FAMS for o in OUTS))
lines.append('')
lines.append('1 - rho')
lines.append(hdr)
for row in table:
    tag = f"{row['inset']}/d{row['ds']}/{row['mode']}"
    lines.append(f"{tag:>16} | " + ' | '.join(f"{row[f'{fam}_{o}_1mrho']:7.1e}" for fam in FAMS for o in OUTS))
txt = '\n'.join(lines)
print(txt)
with open(f'{OUT}/per_family_metrics.txt', 'w') as fp:
    fp.write(txt + '\n')

# ------------------------------------------------------------------
# TASK 2: characterize B error (best TF model in6/d10 + rollout)
# ------------------------------------------------------------------
print('\n' + '='*70)
print('TASK 2: B error split, transient vs ramp')
for tag, p in [('TF in6/d10', f'{ROOT}/results/sensitivity/in6/latent_10/traces.npz'),
               ('ROLLOUT in6/d10', f'{ROOT}/results/sensitivity/rollout_eval/in6/latent_10/traces.npz')]:
    d = np.load(p); t = d['time']
    early = t < 0.5; late = ~early
    for o in OUTS:
        f = d[f'B_fom_{o}']; r = d[f'B_rom_{o}']
        e = r - f
        E_early = np.mean(e[early]**2); E_late = np.mean(e[late]**2)
        frac_early = np.sum(e[early]**2)/np.sum(e**2)
        print(f'{tag} B {o}: RMS early={np.sqrt(E_early):.3f}, RMS late={np.sqrt(E_late):.3f}, '
              f'mean offset late={np.mean(e[late]):+.3f}, energy frac t<0.5s={frac_early:.2f}')

# FFT: B residual (early, detrended) vs A FOM ring-down
d = np.load(f'{ROOT}/results/sensitivity/in6/latent_10/traces.npz')
t = d['time']; dt = np.median(np.diff(t))
def spec(x):
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 2), np.arange(len(x)))
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), dt)
    return fr, X
early = t < 0.5
frB, XB = spec((d['B_rom_Fy']-d['B_fom_Fy'])[early])
frBm, XBm = spec((d['B_rom_Mz']-d['B_fom_Mz'])[early])
# A ring-down: use segment after the gust peak; take second half of A trace
frA, XA = spec(d['A_fom_Fy'][len(t)//2:])
frA2, XA2 = spec(d['A_fom_Mz'][len(t)//2:])
def peaks(fr, X, n=3):
    i = np.argsort(X)[::-1]
    sel = []
    for j in i:
        if fr[j] < 0.5: continue
        if all(abs(fr[j]-s) > 1.0 for s in sel): sel.append(fr[j])
        if len(sel) >= n: break
    return sel
print(f'\nB Fy residual peaks (t<0.5s): {[f"{x:.1f}" for x in peaks(frB,XB)]} Hz')
print(f'B Mz residual peaks (t<0.5s): {[f"{x:.1f}" for x in peaks(frBm,XBm)]} Hz')
print(f'A Fy FOM ring-down peaks   : {[f"{x:.1f}" for x in peaks(frA,XA)]} Hz')
print(f'A Mz FOM ring-down peaks   : {[f"{x:.1f}" for x in peaks(frA2,XA2)]} Hz')

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].semilogy(frB, XB/XB.max(), label='B $F_y$ residual, $t<0.5$ s')
axs[0].semilogy(frA, XA/XA.max(), '--', label='A $F_y$ FOM ring-down')
axs[0].set_xlim(0, 60); axs[0].set_xlabel('f [Hz]'); axs[0].legend(); axs[0].set_title('$F_y$')
axs[1].semilogy(frBm, XBm/XBm.max(), label='B $M_z$ residual, $t<0.5$ s')
axs[1].semilogy(frA2, XA2/XA2.max(), '--', label='A $M_z$ FOM ring-down')
axs[1].set_xlim(0, 60); axs[1].set_xlabel('f [Hz]'); axs[1].legend(); axs[1].set_title('$M_z$')
fig.suptitle('B ROM-error spectrum vs structural ring-down (in6/d10, teacher-forced)')
fig.tight_layout(); fig.savefig(f'{OUT}/B_error_spectrum.png', dpi=150); plt.close(fig)

# Trace overlay with windows
fig, axs = plt.subplots(2, 2, figsize=(12, 7))
dr = np.load(f'{ROOT}/results/sensitivity/rollout_eval/in6/latent_10/traces.npz')
for j, (dd, tag) in enumerate([(d, 'teacher-forced in6/d10'), (dr, 'rollout in6/d10')]):
    for i, o in enumerate(OUTS):
        ax = axs[i, j]
        ax.plot(t, dd[f'B_fom_{o}'], 'b-', lw=1, label='FOM')
        ax.plot(t, dd[f'B_rom_{o}'], 'r--', lw=1, label='ROM')
        ax.axvspan(0, 0.5, color='0.85')
        ax.set_ylabel(f'B {o}')
        if i == 0: ax.set_title(tag)
        if i == 1: ax.set_xlabel('t [s]')
        ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/B_traces_windows.png', dpi=150); plt.close(fig)

# ------------------------------------------------------------------
# TASK 3: H1 / H2 / H3 / H4 with the h5 data
# ------------------------------------------------------------------
print('\n' + '='*70)
print('H1: family composition')
for split in ['train', 'valid', 'test']:
    with h5py.File(f'{ROOT}/data/GLA_{split}.h5', 'r') as f:
        fams = f['sim_families'][:].astype(str)
        u, c = np.unique(fams, return_counts=True)
        print(f'  {split}: {dict(zip(u, c.tolist()))} (total {len(fams)})')

print('\nH2: normalization squeeze — per-family share of the train-MSE signal')
with h5py.File(f'{ROOT}/data/GLA_train.h5', 'r') as f:
    fams = f['sim_families'][:].astype(str)
    ysig = f['output_signals'][:]          # (100, T, 1, 2)
    insig = f['input_signals'][:]          # (100, T, 6)
    times = f['times'][:]
yn = np.stack([norm_fw(ysig[..., 0, 0], 'Fy'), norm_fw(ysig[..., 0, 1], 'Mz')], axis=-1)  # (100,T,2)
h2 = {}
for o, oi in [('Fy', 0), ('Mz', 1)]:
    print(f'  --- {o} (normalized units, full range = 2.0) ---')
    for fam in FAMS:
        m = fams == fam
        y = yn[m, :, oi]
        rng = y.max() - y.min()
        # variance about the per-sim final/trim value: use variance about per-sim mean
        var = np.mean((y - y.mean(axis=1, keepdims=True))**2)
        h2[f'{fam}_{o}'] = dict(rng=float(rng), var=float(var), n=int(m.sum()))
        print(f'   {fam:2s}: n={m.sum():3d}  norm.range={rng:.4f} ({rng/2*100:.1f}% of global)  '
              f'temporal var={var:.2e}')
    # fraction of total dynamic MSE signal that is family B
    tot = sum(h2[f'{fam}_{o}']['var'] * h2[f'{fam}_{o}']['n'] for fam in FAMS)
    for fam in FAMS:
        sh = h2[f'{fam}_{o}']['var'] * h2[f'{fam}_{o}']['n'] / tot
        print(f'   share of summed temporal variance ({o}) — {fam}: {sh*100:.1f}%')
# equal-relative-error loss contribution: err = eps*range_f  -> MSE ratio
rB = h2['B_Fy']['rng']; rA = h2['A_Fy']['rng']; rC = h2['Cc_Fy']['rng']
print(f'  Fy: same RELATIVE error costs (range^2 * n): '
      f'B/(A+B+Cc) = {rB**2*30/(rA**2*20+rB**2*30+rC**2*50)*100:.2f}%')

print('\nH3: flap-step transient + cold start')
with h5py.File(f'{ROOT}/data/GLA_test.h5', 'r') as f:
    tfams = f['sim_families'][:].astype(str)
    tin = f['input_signals'][:]
    tout = f['output_signals'][:]
    ttimes = f['times'][:]
iB = int(np.where(tfams == 'B')[0][0])
iA = int(np.where(tfams == 'A')[0][0])
names = ['h', 'hd', 'a', 'ad', 'delta', 'W_gust']
print(f'  test B sim index {iB}; inputs at first 5 samples:')
for k, nm in enumerate(names):
    v = tin[iB, :5, k]
    print(f'   {nm:7s}: {np.array2string(v, precision=4)}   (t=0 value {v[0]:+.4f}, max|.| {np.abs(tin[iB,:,k]).max():.4f})')
print(f'  B Fy FOM at t=0..5 samples: {np.array2string(tout[iB,:5,0,0], precision=2)}')
print(f'  B Fy FOM min over first 0.2s: {tout[iB, ttimes<0.2, 0, 0].min():.2f}, trim(t=0)={tout[iB,0,0,0]:.2f}, final={tout[iB,-1,0,0]:.2f}')
# ROM at t=0 (z=0 cold start) from traces
print(f'  traces t[0]={t[0]:.5f}s: B ROM Fy(0)={d["B_rom_Fy"][0]:.2f} vs FOM {d["B_fom_Fy"][0]:.2f}')
print(f'  A ROM Fy(0)={d["A_rom_Fy"][0]:.2f} vs FOM {d["A_fom_Fy"][0]:.2f}')
print(f'  Cc ROM Fy(0)={d["Cc_rom_Fy"][0]:.2f} vs FOM {d["Cc_fom_Fy"][0]:.2f}')

# plot B inputs first 0.6 s
fig, axs = plt.subplots(2, 3, figsize=(13, 6))
for k, nm in enumerate(names):
    ax = axs.ravel()[k]
    m = ttimes < 0.6
    ax.plot(ttimes[m], tin[iB, m, k])
    ax.set_title(f'B input: {nm}'); ax.set_xlabel('t [s]')
fig.tight_layout(); fig.savefig(f'{OUT}/B_inputs_early.png', dpi=150); plt.close(fig)

print('\nH4: cubic output layer g(u)=(u^3+0.05u)/1.05 — operating point per family')
h4rows = []
for o, oi in [('Fy', 0), ('Mz', 1)]:
    for fam in FAMS:
        m = fams == fam
        y = yn[m, :, oi]
        lo, hi = np.percentile(y, [1, 99])
        u_lo, u_hi = g_inv(lo)[0], g_inv(hi)[0]
        gp = gprime(g_inv(np.percentile(y, [1, 25, 50, 75, 99])))
        # effective sensitivity for the family's dynamic swing:
        du = u_hi - u_lo; dy = hi - lo
        print(f'  {fam:2s} {o}: y_norm in [{lo:+.3f},{hi:+.3f}]  u in [{u_lo:+.3f},{u_hi:+.3f}]  '
              f"g' @p50={gp[2]:.3f}  g' min={gp.min():.3f}  eff dy/du={dy/du if du>0 else float('nan'):.3f}")
        h4rows.append((fam, o, lo, hi, u_lo, u_hi, float(gp[2])))

# plot g' with family bands (in u space mapped from y)
uu = np.linspace(-1.2, 1.2, 400)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, o in zip(axs, OUTS):
    ax.plot(g(uu), gprime(uu), 'k-')
    cols = {'A': 'tab:blue', 'B': 'tab:red', 'Cc': 'tab:green'}
    for fam, oo, lo, hi, ulo, uhi, gp50 in h4rows:
        if oo != o: continue
        ax.axvspan(lo, hi, color=cols[fam], alpha=0.25, label=f'{fam} range')
    ax.set_xlabel('normalized output $y=g(u)$'); ax.set_ylabel("$g'(u)$")
    ax.set_title(o); ax.legend(fontsize=8); ax.grid(ls=':')
fig.suptitle("Cubic output-layer gain $g'$ vs family operating range")
fig.tight_layout(); fig.savefig(f'{OUT}/H4_cubic_gain.png', dpi=150); plt.close(fig)

# per-family NRMSE vs latent dim figure (Fy)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, o in zip(axs, OUTS):
    for fam in FAMS:
        for inset, mk in [('in2', ':'), ('in4', '--'), ('in6', '-')]:
            xs, ys = [], []
            for row in table:
                if row['inset'] == inset and row['mode'] == 'TF':
                    xs.append(row['ds']); ys.append(row[f'{fam}_{o}_nrmse'])
            ax.semilogy(xs, ys, mk, marker='o', color={'A':'tab:blue','B':'tab:red','Cc':'tab:green'}[fam],
                        label=f'{fam} {inset}' if inset=='in6' else None, alpha=0.7 if inset=='in6' else 0.35)
    rr = [row for row in table if row['mode']=='ROLLOUT'][0]
    for fam in FAMS:
        ax.plot(10, rr[f'{fam}_{o}_nrmse'], '*', ms=14, color={'A':'tab:blue','B':'tab:red','Cc':'tab:green'}[fam])
    ax.set_xlabel('$d_s$'); ax.set_ylabel(f'NRMSE {o} (family range)'); ax.set_title(o)
    ax.legend(fontsize=8); ax.grid(ls=':', which='both')
fig.suptitle('Per-family NRMSE (solid=in6, dashed=in4, dotted=in2, star=rollout)')
fig.tight_layout(); fig.savefig(f'{OUT}/per_family_nrmse.png', dpi=150); plt.close(fig)

print('\nDone. Outputs in', OUT)

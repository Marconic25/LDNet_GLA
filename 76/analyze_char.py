"""Offline analysis of results_char.npz (no TF). Shows what the good branch does
differently during the gust: samples de(t), CL(t), gate(t) for the good and bad
rows and reports where they diverge substantially."""
import numpy as np
d = np.load('results_char.npz')   # self-generated, trusted; no object arrays
ts = d['ts']; CLTRIM = float(d['CLTRIM']); cex0 = float(d['cex0'])

# good = run A row 0 (cex 0.11); bad = run A row 4 (cex 0.38)
gde, gCL, ggate = d['A_de'][0], d['A_CL'][0], d['A_gate'][0]
bde, bCL, bgate = d['A_de'][4], d['A_CL'][4], d['A_gate'][4]
gmarg, bmarg = d['A_margin'][0], d['A_margin'][4]

print(f"cex0={cex0:.4f} CLTRIM={CLTRIM:.4f}", flush=True)
print("\n t(s) | de_good de_bad  | CL_good CL_bad   | gate g/b | mrg_good mrg_bad", flush=True)
for k in range(0, len(ts), 10):     # every 0.02 s
    if ts[k] > 1.0: break
    print(f"{ts[k]:.3f} | {gde[k]:+7.3f} {bde[k]:+7.3f} | "
          f"{gCL[k]:+.4f} {bCL[k]:+.4f} | {ggate[k]}/{bgate[k]} | "
          f"{gmarg[k]:.2e} {bmarg[k]:.2e}", flush=True)

# first step where |de| differ by > 0.5 deg
dd = np.abs(gde - bde)
big = np.where(dd > 0.5)[0]
if len(big):
    i = int(big[0])
    print(f"\nFirst |de_good-de_bad|>0.5deg at i={i} t={ts[i]:.4f}s: "
          f"de {gde[i]:+.3f} vs {bde[i]:+.3f}", flush=True)
# gate flips (causal switch) timing
gflip = np.where(np.diff(ggate) != 0)[0]
bflip = np.where(np.diff(bgate) != 0)[0]
print(f"\ngood gate flips at t=", [f"{ts[i+1]:.3f}" for i in gflip[:12]], flush=True)
print(f"bad  gate flips at t=", [f"{ts[i+1]:.3f}" for i in bflip[:12]], flush=True)

# peak flap timing and CL excursion timing
mw = ts <= 0.9
print(f"\ngood: flap_max={np.max(np.abs(gde[mw])):.2f} at t={ts[np.argmax(np.abs(gde*mw))]:.3f}; "
      f"CLexc={np.max(np.abs(gCL[mw]-CLTRIM)):.4f} at t={ts[np.argmax(np.abs((gCL-CLTRIM)*mw))]:.3f}", flush=True)
print(f"bad : flap_max={np.max(np.abs(bde[mw])):.2f} at t={ts[np.argmax(np.abs(bde*mw))]:.3f}; "
      f"CLexc={np.max(np.abs(bCL[mw]-CLTRIM)):.4f} at t={ts[np.argmax(np.abs((bCL-CLTRIM)*mw))]:.3f}", flush=True)
print("# DONE", flush=True)

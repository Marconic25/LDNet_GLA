#!/usr/bin/env python3
"""Two of my scripts disagree on how many near-flap nodes the N99 model
reverses at the gust peak: 403 (signflip_prf.py) vs 173 (the figure).
Recompute both ways from one place and find which is wrong."""
import numpy as np
import decomp_rates as base

region = np.load("region_labels.npy")
air = np.load("airfoil_nodes.npy")
near = region == 0

D = "ms_coral_o10_N100_s0_rom_cc060"
fom, rom, pts = base.load(D)
print("fom", fom.shape, "rom", rom.shape)

d2 = ((pts[:, None, :] - pts[air][None, :, :]) ** 2).sum(-1)
flap = near & (d2.argmin(1) >= 292)

vref = fom[0, :, :2]
vn = vref / (np.linalg.norm(vref, axis=1, keepdims=True) + 1e-12)

# way A: full-array, index at the end (signflip_prf.py / probe)
prA = (rom[:, :, :2] * vn[None]).sum(-1)
revA = prA < 0
fA = ((fom[:, :, :2] * vn[None]).sum(-1)) < 0
tpk = int(fA[:, flap].mean(1).argmax())
print(f"\nway A: tpk={tpk}  FOM count={int(fA[tpk,flap].sum())}  "
      f"ROM count={int(revA[tpk,flap].sum())}")

# way B: slice the snapshot first, then project (the figure)
rpk = rom[tpk]
prB = (rpk[:, :2] * vn).sum(-1)
revB = prB < 0
print(f"way B: tpk={tpk}  ROM count={int((revB & flap).sum())}")

print(f"\nidentical arrays? {np.array_equal(revA[tpk], revB)}")
diff = np.where(revA[tpk] != revB)[0]
print(f"nodes differing: {len(diff)}")
if len(diff):
    i = diff[0]
    print(f"  example node {i}: wayA proj={prA[tpk,i]:.6e}  wayB proj={prB[i]:.6e}")
    print(f"  rom[tpk,{i},:2]={rom[tpk,i,:2]}  rpk[{i},:2]={rpk[i,:2]}")
    print(f"  vn[{i}]={vn[i]}")

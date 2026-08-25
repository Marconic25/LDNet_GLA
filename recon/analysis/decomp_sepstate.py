#!/usr/bin/env python3
"""Per-region/static-dynamic decomposition + sign-flip-rate readout for the
--dyn-sep-state lever (Goman-Khrabrov attachment-state), n=3 seeds.
Same methodology as decomp_lw.py/decomp_rates.py."""
import numpy as np
import decomp_rates as base

arms = {
    "champion (s0)": "ms_coral_o10_s0_rom_cc060",
    "sep-state s0": "ms_coral_o10_sep_s0_rom_cc060",
    "sep-state s100": "ms_coral_o10_sep_s100_rom_cc060",
    "sep-state s200": "ms_coral_o10_sep_s200_rom_cc060",
}

print(f"{'arm':16s} {'region':8s} {'static':>10s} {'dynamic':>10s}")
sf_rows = []
for name, dirname in arms.items():
    fom, rom, pts = base.load(dirname)
    for rname, mask in base.REGIONS:
        ns, nd = base.decomp(fom, rom, mask)
        print(f"{name:16s} {rname:8s} {ns:10.4e} {nd:10.4e}")
    ff, rf, sf, tpk = base.signflip_readout(fom, rom, pts)
    sf_rows.append((name, ff, rf, sf))
    combined = np.sqrt(np.mean((rom - fom) ** 2)) / (fom.max() - fom.min())
    print(f"{name:16s} combined NRMSE = {combined:.4e}\n")

print("=== SIGN-FLIP READOUT (flap-region, gust-peak time) ===")
print(f"{'arm':16s} {'FOM frac':>10s} {'ROM frac':>10s} {'sign-flip%':>11s}")
for name, ff, rf, sf in sf_rows:
    print(f"{name:16s} {ff:10.4f} {rf:10.4f} {100*sf:10.2f}%")

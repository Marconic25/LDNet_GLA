#!/usr/bin/env python3
"""PRELIMINARY (single-seed, NOT the final n=3 verdict) decomposition +
sign-flip readout for whatever arms of the stall investigation are available
so far. Same methodology as decomp_residual.py/decomp_stall.py -- just run
against however many seeds have landed to date, clearly labeled as partial."""
import numpy as np
from pathlib import Path

AN = Path(__file__).resolve().parent
RES = AN.parent / "results"

region_label = np.load(AN / "region_labels.npy")
air = np.load(AN / "airfoil_nodes.npy")
N = len(region_label)
REGIONS = [("near", region_label == 0), ("wake", region_label == 1),
           ("far", region_label == 2), ("surface", np.isin(np.arange(N), air))]


def decomp(fom, rom, mask, k=0):
    err = rom - fom
    e_s = err.mean(0)
    e_d = err - e_s[None]
    rng = fom[:, :, k].max() - fom[:, :, k].min()
    ns = np.sqrt((e_s[mask, k] ** 2).mean()) / rng
    nd = np.sqrt((e_d[:, mask, k] ** 2).mean()) / rng
    return ns, nd


def signflip_readout(fom, rom, pts):
    near = region_label == 0
    d2 = ((pts[:, None, :] - pts[air][None, :, :]) ** 2).sum(-1)
    nn = d2.argmin(1)
    is_flap_near = near & (nn >= 292)
    vref = fom[0, :, :2]
    vref_norm = np.linalg.norm(vref, axis=1) + 1e-6
    proj_fom = (fom[:, :, 0] * vref[None, :, 0] + fom[:, :, 1] * vref[None, :, 1]) / vref_norm[None, :]
    proj_rom = (rom[:, :, 0] * vref[None, :, 0] + rom[:, :, 1] * vref[None, :, 1]) / vref_norm[None, :]
    frac_flap_fom = (proj_fom < 0)[:, is_flap_near].mean(1)
    tpk = int(frac_flap_fom.argmax())
    frac_flap_rom_pk = (proj_rom[tpk] < 0)[is_flap_near].mean()
    flap_idx = np.where(is_flap_near)[0]
    signflip = float(((proj_fom[tpk, flap_idx] < 0) != (proj_rom[tpk, flap_idx] < 0)).mean())
    return frac_flap_fom[tpk], frac_flap_rom_pk, signflip


arms = {
    "champion (s0)": "ms_coral_o10_s0_rom_cc060",
    "rates s0": "ms_coral_o10_rates_s0_rom_cc060",
    "rates s200": "ms_coral_o10_rates_s200_rom_cc060",
    "flap-lw s200": "ms_coral_o10_lw_s200_rom_cc060",
    "sep-state s200": "ms_coral_o10_sep_s200_rom_cc060",
}

print("*** PRELIMINARY / SINGLE-SEED -- NOT the final n=3 verdict ***\n")
print(f"{'arm':18s} {'region':8s} {'static':>10s} {'dynamic':>10s}")
sf_rows = []
for name, dirname in arms.items():
    d = RES / dirname
    if not d.exists():
        print(f"{name:18s}  MISSING ({dirname})")
        continue
    fom = np.load(d / "fom_sim_Cc_060.npy").astype(np.float64)
    rom = np.load(d / "rom_sim_Cc_060.npy").astype(np.float64)
    pts = np.load(d / "rom_points.npy").astype(np.float64)
    for rname, mask in REGIONS:
        ns, nd = decomp(fom, rom, mask)
        print(f"{name:18s} {rname:8s} {ns:10.4e} {nd:10.4e}")
    ff, rf, sf = signflip_readout(fom, rom, pts)
    sf_rows.append((name, ff, rf, sf))
    combined = np.sqrt(np.mean((rom - fom) ** 2)) / (fom.max() - fom.min())
    print(f"{name:18s} combined NRMSE = {combined:.4e}\n")

print("=== SIGN-FLIP READOUT (flap-region, gust-peak time) ===")
print(f"{'arm':18s} {'FOM frac':>10s} {'ROM frac':>10s} {'sign-flip%':>11s}")
for name, ff, rf, sf in sf_rows:
    print(f"{name:18s} {ff:10.4f} {rf:10.4f} {100*sf:10.2f}%")

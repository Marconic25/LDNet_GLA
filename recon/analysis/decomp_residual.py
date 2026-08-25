#!/usr/bin/env python3
"""Per-region/static-dynamic decomposition + sign-flip-rate readout for the
residual-curriculum loss-weighting lever (--loss-weight-mode residual),
mirroring decomp_cde.py/decomp_shooting.py's methodology exactly (never trust
combined/global NRMSE alone -- this project has been fooled by that twice).

Also re-runs the flap sign-flip diagnostic from decomp_stall.py on each arm's
FOM/ROM dump: does the reversed-flow fraction gap (champion ROM 2.1% vs FOM
23.3% at the gust peak) actually close under residual-curriculum weighting?
"""
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


def load(dirname):
    d = RES / dirname
    fom = np.load(d / "fom_sim_Cc_060.npy").astype(np.float64)
    rom = np.load(d / "rom_sim_Cc_060.npy").astype(np.float64)
    pts = np.load(d / "rom_points.npy").astype(np.float64)
    return fom, rom, pts


def signflip_readout(fom, rom, pts):
    """Same convention as decomp_stall.py: reference = each node's own FOM
    velocity at t=0 (attached baseline); reversed = negative projection onto
    that reference. Returns (peak FOM frac, ROM frac at same time, sign-flip
    rate among near-flap points at that time, peak time index)."""
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
    return frac_flap_fom[tpk], frac_flap_rom_pk, signflip, tpk


arms = {
    "champion (coral_o10_s0)": "ms_coral_o10_s0_rom_cc060",
    "residual p1.0 s0": "ms_coral_o10_res_p1_s0_rom_cc060",
    "residual p1.0 s100": "ms_coral_o10_res_p1_s100_rom_cc060",
    "residual p1.0 s200": "ms_coral_o10_res_p1_s200_rom_cc060",
    "residual p2.0 s0": "ms_coral_o10_res_p2_s0_rom_cc060",
    "residual p2.0 s100": "ms_coral_o10_res_p2_s100_rom_cc060",
    "residual p2.0 s200": "ms_coral_o10_res_p2_s200_rom_cc060",
}

print(f"{'arm':26s} {'region':8s} {'static':>10s} {'dynamic':>10s}")
signflip_rows = []
for name, dirname in arms.items():
    d = RES / dirname
    if not d.exists():
        print(f"{name:26s}  MISSING ({dirname})")
        continue
    fom, rom, pts = load(dirname)
    for rname, mask in REGIONS:
        ns, nd = decomp(fom, rom, mask)
        print(f"{name:26s} {rname:8s} {ns:10.4e} {nd:10.4e}")
    fom_frac, rom_frac, signflip, tpk = signflip_readout(fom, rom, pts)
    signflip_rows.append((name, fom_frac, rom_frac, signflip, tpk))
    combined = np.sqrt(np.mean((rom - fom) ** 2)) / (fom.max() - fom.min())
    print(f"{name:26s} combined NRMSE = {combined:.4e}")
    print()

print("\n=== SIGN-FLIP READOUT (flap-region, gust-peak time) ===")
print(f"{'arm':26s} {'FOM frac':>10s} {'ROM frac':>10s} {'sign-flip%':>11s} {'tpk':>5s}")
for name, ff, rf, sf, tpk in signflip_rows:
    print(f"{name:26s} {ff:10.4f} {rf:10.4f} {100*sf:10.2f}% {tpk:5d}")

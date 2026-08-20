import numpy as np
from pathlib import Path

BASE = Path("/work/u10677113/NACA2312/recon")
RES = BASE / "results"
HM = BASE / "analysis_hmetric"
region_label = np.load(HM / "region_labels.npy")
air = np.load(HM / "airfoil_nodes.npy")
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
    return fom, rom

arms = {
    "champion (coral_o10_s0)": "ms_coral_o10_s0_rom_cc060",
    "shoot L0.1": "ms_coral_o10_shootK4L0.1_s0_rom_cc060",
    "shoot L1.0": "ms_coral_o10_shootK4L1.0_s0_rom_cc060",
    "shoot L10.0": "ms_coral_o10_shootK4L10.0_s0_rom_cc060",
}

print(f"{'arm':28s} {'region':8s} {'static':>10s} {'dynamic':>10s}")
for name, dirname in arms.items():
    fom, rom = load(dirname)
    for rname, mask in REGIONS:
        ns, nd = decomp(fom, rom, mask)
        print(f"{name:28s} {rname:8s} {ns:10.4e} {nd:10.4e}")
    print()

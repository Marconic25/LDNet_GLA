#!/usr/bin/env python3
"""Sign-flip / reversed-flow-fraction readout for the RESIDUAL-CURRICULUM lever,
adapted directly from decomp_stall.py (identical separation indicator and
near-flap/near-main masks) but generalized to loop over multiple arms/seeds
and print a compact comparison table against the already-established champion
numbers (ROM 2.1% vs FOM 23.3% reversed-flow fraction at t=0.584s, 21.7%
sign-flip rate among near-flap points).

This is THE most direct, falsifiable readout for this lever's literature
prediction (STALL_LITERATURE_NOTES.md section 5): does upweighting points by
their own residual magnitude measurably shrink the sign-flip rate at the
gust-peak transient, with far-field/attached NRMSE flat or only mildly worse?

Runs LOCALLY on FOM/ROM dumps scp'd back into recon/results/<dirname>/ (same
local-copy convention as decomp_stall.py/decomp_residual.py).
"""
import numpy as np
from pathlib import Path

AN = Path(__file__).resolve().parent
RES = AN.parent / "results"

region = np.load(AN / "region_labels.npy")     # 0 near, 1 wake, 2 far
air = np.load(AN / "airfoil_nodes.npy")        # main[0:292) + flap[292:387) perimeter-ordered
near = region == 0


def signflip_readout(dirname, simname="sim_Cc_060", tpk_from=None):
    """Returns dict with peak-time reversed-flow fractions (FOM/ROM, near-flap
    and near-main) and the sign-flip rate at that peak time. tpk_from: if
    given, reuse this fixed time index (e.g. the champion's own peak index)
    instead of re-finding each arm's own FOM peak -- lets every arm be read
    out at the EXACT SAME instant for a fair comparison; if None, uses this
    arm's own FOM peak (its FOM data barely differs run to run since FOM is
    ground truth, but the test-set FOM should be identical across arms since
    it's the same CFD sim -- this is mostly a safety fallback)."""
    d = RES / dirname
    fom = np.load(d / f"fom_{simname}.npy").astype(np.float64)
    rom = np.load(d / f"rom_{simname}.npy").astype(np.float64)
    pts = np.load(d / "rom_points.npy").astype(np.float64)
    times = np.load(d / "rom_times.npy").astype(np.float64)
    t_rel = times - times[0]

    d2 = ((pts[:, None, :] - pts[air][None, :, :]) ** 2).sum(-1)
    nn = d2.argmin(1)
    is_flap_near = near & (nn >= 292)
    is_main_near = near & (nn < 292)

    vref = fom[0, :, :2]
    vref_norm = np.linalg.norm(vref, axis=1) + 1e-6
    proj_fom = (fom[:, :, 0] * vref[None, :, 0] + fom[:, :, 1] * vref[None, :, 1]) / vref_norm[None, :]
    proj_rom = (rom[:, :, 0] * vref[None, :, 0] + rom[:, :, 1] * vref[None, :, 1]) / vref_norm[None, :]
    reversed_fom = proj_fom < 0
    reversed_rom = proj_rom < 0

    frac_flap_fom = reversed_fom[:, is_flap_near].mean(1)
    frac_main_fom = reversed_fom[:, is_main_near].mean(1)
    frac_flap_rom = reversed_rom[:, is_flap_near].mean(1)

    tpk = tpk_from if tpk_from is not None else int(frac_flap_fom.argmax())

    flap_idx = np.where(is_flap_near)[0]
    n_sign_flip = int(((proj_fom[tpk, flap_idx] < 0) != (proj_rom[tpk, flap_idx] < 0)).sum())
    signflip_pct = 100.0 * n_sign_flip / len(flap_idx)

    return {
        "tpk_idx": tpk, "t_rel": float(t_rel[tpk]),
        "fom_flap_frac": float(frac_flap_fom[tpk]),
        "rom_flap_frac": float(frac_flap_rom[tpk]),
        "fom_main_frac": float(frac_main_fom[tpk]),
        "n_flap_near": len(flap_idx), "n_sign_flip": n_sign_flip,
        "signflip_pct": signflip_pct,
    }


if __name__ == "__main__":
    CHAMPION_DIR = "ms_coral_o10_s0_rom_cc060"
    arms = {
        "champion (coral_o10_s0)": CHAMPION_DIR,
        "residual p1 s0":   "ms_coral_o10_res_p1_s0_rom_cc060",
        "residual p1 s100": "ms_coral_o10_res_p1_s100_rom_cc060",
        "residual p1 s200": "ms_coral_o10_res_p1_s200_rom_cc060",
        "residual p2 s0":   "ms_coral_o10_res_p2_s0_rom_cc060",
        "residual p2 s100": "ms_coral_o10_res_p2_s100_rom_cc060",
        "residual p2 s200": "ms_coral_o10_res_p2_s200_rom_cc060",
    }

    # find the champion's own peak index first (each arm is then ALSO read at
    # that same fixed index for a fair apples-to-apples comparison, in
    # addition to its own peak, since gust+flap forcing is identical across
    # arms/seeds and the peak time should not move much).
    champ_path = RES / CHAMPION_DIR
    tpk_fixed = None
    if champ_path.exists():
        r0 = signflip_readout(CHAMPION_DIR)
        tpk_fixed = r0["tpk_idx"]
        print(f"champion peak index tpk={tpk_fixed} (t={r0['t_rel']:.3f}s) "
              f"-- all arms below also read out at this SAME index\n")

    print(f"{'arm':22s} {'t_pk[s]':>8s} {'FOM flap%':>10s} {'ROM flap%':>10s} "
          f"{'sign-flip%':>11s} {'n_near-flap':>12s}")
    for name, dirname in arms.items():
        d = RES / dirname
        if not d.exists():
            print(f"{name:22s} -- MISSING ({d}) --")
            continue
        r_own = signflip_readout(dirname)
        line = (f"{name:22s} {r_own['t_rel']:8.3f} "
                f"{100*r_own['fom_flap_frac']:10.1f} {100*r_own['rom_flap_frac']:10.1f} "
                f"{r_own['signflip_pct']:11.1f} {r_own['n_flap_near']:12d}  (own peak)")
        print(line)
        if tpk_fixed is not None and dirname != CHAMPION_DIR:
            r_fix = signflip_readout(dirname, tpk_from=tpk_fixed)
            print(f"{'':22s} {r_fix['t_rel']:8.3f} "
                  f"{100*r_fix['fom_flap_frac']:10.1f} {100*r_fix['rom_flap_frac']:10.1f} "
                  f"{r_fix['signflip_pct']:11.1f} {r_fix['n_flap_near']:12d}  (champion's peak idx)")

#!/usr/bin/env python3
"""Depth vs LOADS bridge (option 1): surface-pressure reconstruction accuracy of the
existing depth-ladder field models. For each model, reconstruct (vx,vy,p) on the
gust+flap test sim Cc_060, then compute the NRMSE of the pressure on the airfoil
SURFACE nodes only (the physical source of the aerodynamic loads) — overall and in
the gust-peak time window. No retraining; pure post-processing of trained models.

Run inside the TF container on the login node:
  apptainer exec --bind /work/u10677113:/work/u10677113 tensorflow_gpu.sif \
      python3 recon/cluster/depth_loads.py
"""
import subprocess, sys, pathlib, csv
import numpy as np

RECON = pathlib.Path("/work/u10677113/NACA2312/recon")
DATA = RECON / "data" / "FIELDS_Cc060.h5"
NODES_F = RECON / "analysis_hmetric" / "airfoil_nodes.npy"
OUTROOT = RECON / "results" / "depth_loads"
OUTROOT.mkdir(parents=True, exist_ok=True)

# (tag, model dir) — depth ladder ds1 spine + ds10 contrast + L24 seed-100 replication
DS = RECON / "models" / "depth_study"
RP = RECON / "models" / "depth_replication"
MODELS = [
    ("L6_ds1",        DS / "L6_ds1"  / "latent_1"),
    ("L6_ds10",       DS / "L6_ds10" / "latent_10"),
    ("L12_ds1",       DS / "L12_ds1" / "latent_1"),
    ("L12_ds10",      DS / "L12_ds10"/ "latent_10"),
    ("L24_ds1_s0",    DS / "L24_ds1" / "latent_1"),
    ("L24_ds10_s0",   DS / "L24_ds10"/ "latent_10"),
    ("L24_ds20_s0",   DS / "L24_ds20"/ "latent_20"),
    ("L24_ds1_s100",  RP / "L24_ds1_s100" / "latent_1"),
    ("L24_ds10_s100", RP / "L24_ds10_s100"/ "latent_10"),
]

# --- surface nodes ---
nodes = np.load(NODES_F)
if nodes.dtype == bool:
    surf = np.where(nodes)[0]
else:
    surf = nodes.astype(int).ravel()
print(f"surface nodes: {len(surf)}")

# --- reconstruct each model (skip if already done) ---
for tag, mdir in MODELS:
    outd = OUTROOT / tag
    if (outd / "rom_sim_Cc_060.npy").exists():
        print(f"[skip recon] {tag}")
        continue
    if not (mdir / "config.json").exists():
        print(f"[MISSING MODEL] {tag}: {mdir}"); continue
    print(f"[recon] {tag} ...")
    r = subprocess.run([sys.executable, str(RECON / "reconstruct_fields.py"),
                        "--model", str(mdir), "--data", str(DATA),
                        "--index", "0", "--name", "sim_Cc_060", "--out", str(outd)],
                       cwd=str(RECON), capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  RECON FAILED {tag}:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")

# --- surface-pressure NRMSE ---
rows = []
for tag, _ in MODELS:
    outd = OUTROOT / tag
    rp = outd / "rom_sim_Cc_060.npy"
    fp = outd / "fom_sim_Cc_060.npy"
    if not rp.exists():
        print(f"[no recon] {tag}"); continue
    rom = np.load(rp); fom = np.load(fp)           # (T, Npts, 3) = vx,vy,p
    p_rom = rom[:, surf, 2]; p_fom = fom[:, surf, 2]   # (T, Nsurf)
    rng = float(p_fom.max() - p_fom.min())
    nrmse_all = float(np.sqrt(np.mean((p_rom - p_fom) ** 2)) / rng)
    # gust-peak window: time index of max spatial-RMS deviation of surface p from its time-mean
    sig = np.sqrt(((p_fom - p_fom.mean(0)) ** 2).mean(1))   # (T,)
    tk = int(sig.argmax())
    nrmse_peak = float(np.sqrt(np.mean((p_rom[tk] - p_fom[tk]) ** 2)) / rng)
    rows.append((tag, len(surf), rng, nrmse_all, nrmse_peak, tk))

print(f"\n{'model':16s} {'Nsurf':>6s} {'p_range[Pa]':>12s} {'surfP_NRMSE':>12s} {'peak_NRMSE':>11s} {'tk':>4s}")
for tag, ns, rng, na, npk, tk in rows:
    print(f"{tag:16s} {ns:6d} {rng:12.1f} {na:12.4e} {npk:11.4e} {tk:4d}")

out_csv = RECON / "analysis" / "depth_loads_surfP.csv"
out_csv.parent.mkdir(exist_ok=True)
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "n_surf_nodes", "p_range_Pa", "surfP_NRMSE", "peak_surfP_NRMSE", "peak_time_idx"])
    w.writerows(rows)
print(f"\nwrote {out_csv}")

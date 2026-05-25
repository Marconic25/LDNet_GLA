import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------
# Configuration
DATASET_DIR   = Path("/work/u10677113/NACA2312/dataset_v5")
OUT_DIR       = Path(__file__).parent
KNOWN_FAMILIES = {"A", "B", "Cc"}
U_INF         = 80.0

# Column indices in structural_trajectory.csv
# t, h, hd, alpha, ad, Fy, Mz, W_gust, delta
COL_T     = 0
COL_H     = 1
COL_HD    = 2
COL_A     = 3
COL_AD    = 4
COL_FY    = 5
COL_MZ    = 6
COL_WGUST = 7
COL_DELTA = 8
EXPECTED_COLS = 9

# ---------------------------------------------------------------
# Step 1: Scan dataset directory

def scan_dir(dataset_dir):
    """Scan for simulation subdirectories matching sim_{family}_{index}_{split}."""
    grouped = defaultdict(list)
    anomalies = []

    dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    print(f"Cartelle trovate: {len(dirs)}")

    for d in dirs:
        parts = d.name.split("_")
        if len(parts) != 4:
            anomalies.append(f"Nome non valido (non 4 parti): {d.name}")
            continue

        prefix, family, idx, split = parts

        if prefix != "sim":
            anomalies.append(f"Prefisso inatteso: {d.name}")
            continue

        if family not in KNOWN_FAMILIES:
            anomalies.append(f"Famiglia sconosciuta: {d.name} ('{family}')")
            continue

        if split not in ("train", "val", "test"):
            anomalies.append(f"Split sconosciuto: {d.name} ('{split}')")
            continue

        csv_path = d / "structural_trajectory.csv"
        if not csv_path.exists():
            anomalies.append(f"structural_trajectory.csv mancante in {d.name}")
            continue

        grouped[(family, split)].append(csv_path)

    return grouped, anomalies


grouped, anomalies = scan_dir(DATASET_DIR)

for key, files in sorted(grouped.items()):
    print(f"  {key}: {len(files)} simulazioni")

if anomalies:
    print("\nAnomalie:")
    for a in anomalies:
        print(f"  - {a}")

# ---------------------------------------------------------------
# Step 2: Validate and load CSVs

def validate_csv(csv_path):
    """Load and validate a structural_trajectory.csv. Returns (problems, data)."""
    problems = []

    try:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    except Exception as e:
        return [f"Errore di lettura: {e}"], None

    if data.ndim != 2 or data.shape[1] != EXPECTED_COLS:
        problems.append(f"Colonne inattese: {data.shape} (atteso Tx{EXPECTED_COLS})")
        return problems, None

    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))
    if nan_count > 0:
        problems.append(f"Valori NaN: {nan_count}")
    if inf_count > 0:
        problems.append(f"Valori Inf: {inf_count}")

    dt_array = np.diff(data[:, COL_T])
    if not np.all(dt_array > 0):
        problems.append("Tempo non monotono crescente")

    if np.max(np.abs(data[:, COL_DELTA])) > 25:
        problems.append(f"delta fuori range: max |delta| = {np.max(np.abs(data[:, COL_DELTA])):.2f} deg")

    return problems, data


def load_all(grouped):
    all_data = {}
    all_problems = {}
    for (family, split), files in sorted(grouped.items()):
        for csv_path in sorted(files):
            sim_name = csv_path.parent.name
            problems, data = validate_csv(csv_path)
            if problems:
                all_problems[sim_name] = problems
            if data is not None:
                idx = sim_name.split("_")[2]
                all_data[(family, split, idx)] = data

    if all_problems:
        print(f"\nProblemi trovati in {len(all_problems)} file:")
        for name, probs in sorted(all_problems.items()):
            print(f"  {name}:")
            for p in probs:
                print(f"    - {p}")
    else:
        print("Tutti i CSV validati correttamente.")

    return all_data


all_data = load_all(grouped)

# ---------------------------------------------------------------
# Step 3: Normalization stats on training set

def compute_normalization(all_data):
    sig_cols = {
        "h": COL_H, "hd": COL_HD, "alpha": COL_A, "ad": COL_AD,
        "delta": COL_DELTA, "W_gust": COL_WGUST, "Fy": COL_FY, "Mz": COL_MZ,
    }
    train_data = [v for k, v in all_data.items() if k[1] == "train"]
    if not train_data:
        print("Nessun dato di training trovato.")
        return {}
    minmax = {}
    for name, col in sig_cols.items():
        vmin = min(np.min(d[:, col]) for d in train_data)
        vmax = max(np.max(d[:, col]) for d in train_data)
        minmax[name] = (vmin, vmax)
    print("\nMin/max su training set:")
    for name, (vmin, vmax) in minmax.items():
        print(f"  {name}: [{vmin:.4f}, {vmax:.4f}]")
    return minmax

compute_normalization(all_data)

# ---------------------------------------------------------------
# Step 4: Write HDF5 files

def split_by_filename(all_data):
    datasets = {"train": {}, "valid": {}, "test": {}}
    split_map = {"train": "train", "val": "valid", "test": "test"}
    for (fam, split, idx), data in all_data.items():
        dest = split_map.get(split)
        if dest is not None:
            datasets[dest][(fam, idx)] = data
    return datasets


def write_h5(data_dict, filename):
    """Write simulations to HDF5.

    input_signals:  (N, T, 6)    — [h, hd, alpha, ad, delta, W_gust]
    output_signals: (N, T, 1, 2) — [Fy, Mz]
    """
    if not data_dict:
        print(f"Nessuna simulazione per {filename}, file non creato.")
        return

    simulations = list(data_dict.values())
    n_times = simulations[0].shape[0]

    times            = simulations[0][:, COL_T]
    input_signals    = np.stack([s[:, [COL_H, COL_HD, COL_A, COL_AD, COL_DELTA, COL_WGUST]] for s in simulations])
    output_signals   = np.expand_dims(np.stack([s[:, [COL_FY, COL_MZ]] for s in simulations]), axis=2)
    input_parameters = np.full((len(simulations), 1), U_INF)
    output_fields    = np.zeros((len(simulations), n_times, 1, 2))

    families_arr = np.array([k[0].encode() for k in data_dict.keys()])

    with h5py.File(filename, 'w') as f:
        f.create_dataset("points",           data=np.array([[0.0, 0.0]]))
        f.create_dataset("times",            data=times)
        f.create_dataset("input_parameters", data=input_parameters)
        f.create_dataset("input_signals",    data=input_signals)
        f.create_dataset("output_signals",   data=output_signals)
        f.create_dataset("output_fields",    data=output_fields)
        f.create_dataset("sim_families",     data=families_arr)

    print(f"Dataset salvato: {filename}  "
          f"[{len(simulations)} sims, input_signals: {input_signals.shape}, "
          f"output_signals: {output_signals.shape}]")


datasets = split_by_filename(all_data)

write_h5(datasets["train"], OUT_DIR / "GLA_train.h5")
write_h5(datasets["valid"], OUT_DIR / "GLA_valid.h5")
write_h5(datasets["test"],  OUT_DIR / "GLA_test.h5")

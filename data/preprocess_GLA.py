import csv
from operator import index
from operator import index
from os import name
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

#---------------------------------------------------------------
#Step 1: Configuration - scanning the directory and defining expected counts
DATA_DIR = Path("/home/marco/LDNet_OF/data/GLA_data/timeseries")

#Check the content of the directory
EXPECTED_COUNTS = {
    "A":  {"train": 6,  "test": 2},
    "B1": {"train": 16, "test": 4},
    "B2": {"train": 12, "test": 3},
    "B3": {"train": 10, "test": 3},
    "C":  {"train": 6,  "test": 2},
}

#Scanning and parsing the directory
def scan_dir(DATA_DIR):
    """Scans the directory and counts files according to the expected naming convention."""

    grouped = defaultdict(list) #dictionary to create empty lists for new keys
    anomalies = [] #list to store anomalies

    #find all csv files in the directory
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    print(f"File CSV trovati: {len(csv_files)}")

    for file in csv_files:
        name_parts = file.stem.split("_") #split the filename by "_"
        
        if len(name_parts) != 4:
            anomalies.append((file.name, "Filename does not have 4 parts"))
            continue
        
        prefix, family, index, split = name_parts

        if prefix != "sim":
            anomalies.append(f"Prefisso inatteso: {name} ('{prefix}' invece di 'sim')")
            continue
        
        
        if family not in EXPECTED_COUNTS:
            anomalies.append(f"Famiglia sconosciuta: {name} ('{family}')")
            continue
        
        # Verifica che lo split sia train o test
        if split not in ("train", "test"):
            anomalies.append(f"Split sconosciuto: {name} ('{split}')")
            continue
        
        # Tutto ok: aggiungi al gruppo corrispondente
        grouped[(family, split)].append(file)
    
    return grouped, anomalies

#Verify that the counts match the expected ones
grouped, anomalies = scan_dir(DATA_DIR)

for key, files in sorted(grouped.items()):
    print(f"{key}: {len(files)} file")

if anomalies:
    print("\nAnomalie:")
    for a in anomalies:
        print(f"  - {a}")

#----------------------------------------------------------------
#Step 2: Data loading and inspection
EXPECTED_ROWS = 1500
EXPECTED_COLS = 11
INP_COLS = 9
DT_SAVE = 0.002

COLUMNS = ["t", "h", "h_dot", "a", "ad", "delta", "W_gust", "C_L", "C_M"]

def validate_csv(file_path):
    """Validates the CSV file format and content."""
   
    problems = []

    try: #loads the file - skip first row
        data = np.loadtxt(file_path, delimiter=",", skiprows=1) #skip the header row

    except Exception as e:
        problems.append(f"Errore di lettura: {e}")
        return problems, None

    # Check dimensions
    if data.shape != (EXPECTED_ROWS, EXPECTED_COLS):
        problems.append(f"Dimensioni inattese: {data.shape} (atteso {EXPECTED_ROWS}x{EXPECTED_COLS})")
        return problems, None

    # Check NaN values
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))
    if nan_count > 0:
        problems.append(f"Valori NaN trovati: {nan_count}")
    if inf_count > 0:
        problems.append(f"Valori Inf trovati: {inf_count}")

        # --- Controllo colonna tempo ---
    time_col = data[:, 0]
    
    # Il tempo deve essere monotono crescente
    dt_array = np.diff(time_col)
    if not np.all(dt_array > 0):
        problems.append("Tempo non monotono crescente")
    
    # Il timestep deve essere costante (~0.002)
    dt_mean = np.mean(dt_array)
    dt_std = np.std(dt_array)
    if abs(dt_mean - DT_SAVE) > 1e-6:
        problems.append(f"Timestep medio inatteso: {dt_mean:.6f} (atteso: {DT_SAVE})")
    if dt_std > 1e-8:
        problems.append(f"Timestep non costante: std = {dt_std:.2e}")
    
    # --- Controllo range fisici base ---
    # delta (col 5) deve essere in gradi, range ragionevole
    delta = data[:, 5]
    if np.max(np.abs(delta)) > 25:
        problems.append(f"delta fuori range: max |delta| = {np.max(np.abs(delta)):.2f} deg")
    
    # W_g (col 6) deve essere >= 0 per un gust 1-cosine
    w_gust = data[:, 6]
    if np.min(w_gust) < -0.01:
        problems.append(f"W_g negativo: min = {np.min(w_gust):.4f}")

    return problems, data

def validate_all(grouped):
    """Validates all files in the grouped dictionary."""
    all_data = {}
    all_problems = {}
    checked = 0
    for (family, split), files in sorted(grouped.items()):
        for filepath in sorted(files):
            checked += 1
            name = filepath.stem    

            problems, data = validate_csv(filepath)

            if problems:
                all_problems[name] = problems

            if data is not None:
                index = filepath.stem.split("_")[2]
                all_data[(family, split, index)] = data
    return all_problems, all_data

# Validazione contenuto
#print("Validazione contenuto CSV...")
all_problems, all_data = validate_all(grouped)

#if all_problems:
 #   print(f"\nProblemi trovati in {len(all_problems)} file:")
  #  for name, probs in sorted(all_problems.items()):
   #     print(f"\n  {name}:")
    #    for p in probs:
     #       print(f"    - {p}")
#else:
 #   print("Tutti i CSV validati correttamente.")

 #----------------------------------------------------------------
#Step 3: Min e max calculation to find value for normalization(in TestCase_OF)

#csv contains 11 columns but we want 9 (no Fy and no Mz)
all_data = {k: v[:, :INP_COLS] for k, v in all_data.items()}

def compute_normalization(all_data):

    train = []
    minmax_values = {}
    for key in all_data:
        if key[1] == "train":
            train.append(all_data[key])

    for i in range(0, INP_COLS):
        min_val = min([np.min(d[:, i]) for d in train])
        max_val = max([np.max(d[:, i]) for d in train])
        minmax_values[COLUMNS[i]] = (min_val, max_val) #store also the family for reference
    return minmax_values

# Compute and print normalization parameters (based on train data)
minmax_values = compute_normalization(all_data)
print(minmax_values)


#----------------------------------------------------------------
#Step4: Packaging the data in h5 format for LDNet training
#train
#valid
#test



def split_train_valid_test(all_data):
    """Splits the data into train, valid, and test sets based on the naming convention. Note that
    the 'valid' set is not explicitly defined in the files, so we will create it by taking a portion of the 'train' data."""
    datasets = {"train": {}, "valid": {}, "test": {}}
    families = sorted(list(set(k[0] for k in all_data.keys()))) #get unique families
    val_ratio = 0.2 #20% of train data will be used for validation

    # Add test splits immediately
    for key, data in all_data.items():
        if key[1] == "test":
            datasets["test"][(key[0], key[2])] = data

    # Split training data into train/valid per family
    for fam in families:
        train_keys = sorted([k for k in all_data.keys() if k[0] == fam and k[1] == "train"], key=lambda x: x[2])
        if not train_keys:
            continue

        indices = [k[2] for k in train_keys]
        np.random.seed(42)
        np.random.shuffle(indices)

        n_val = max(1, int(len(indices) * val_ratio))
        val_idx = set(indices[:n_val])
        train_idx = set(indices[n_val:])

        for idx in val_idx:
            datasets["valid"][(fam, idx)] = all_data[(fam, "train", idx)]
        for idx in train_idx:
            datasets["train"][(fam, idx)] = all_data[(fam, "train", idx)]

    return datasets


#initialize lists to store the data for each set
times = []
input_parameters = [] #N_sim, 1 (80 m/s)
input_signals = [] #N_sim, N_time, 6 (h, h_dot, a, ad, delta, W_gust)
output_signals = [] #N_sim, N_time, 1, 2 (C_L, C_M)
output_fields = [] 

import h5py

def write_h5(data_dict, filename):
    # 1. Trasformiamo il dizionario in una lista di array NumPy
    simulations = list(data_dict.values())
    num_sims = len(simulations)

    # 2. Prepariamo i pezzi (assumendo 1500 righe e le colonne definite)
    times = simulations[0][:, 0]
    
    # 3. Aggreghiamo gli input signals (N, 1500, 6)
    Hint: input_signals = np.stack([s[:, 1:7] for s in simulations])
    
    # 4. Aggreghiamo gli output fields (N, 1500, 1, 2)
    output_fields = np.expand_dims(np.stack([s[:, 7:9] for s in simulations]), axis=2)
    input_parameters = np.full((num_sims, 1), 80.0)
    
    # 5. Creiamo il file H5
    with h5py.File(filename, 'w') as f:
        f.create_dataset("times", data=times)
        f.create_dataset("input_signals", data=input_signals)
        f.create_dataset("output_fields", data=output_fields)
        f.create_dataset("input_parameters", data=input_parameters)
        # Aggiungiamo anche il punto fittizio
        f.create_dataset("points", data=np.array([[0.0, 0.0]]))

    print(f"Dataset salvato correttamente in: {filename}")

    # Esempio di chiamata finale
datasets = split_train_valid_test(all_data)

write_h5(datasets["train"], "GLA_train.h5")
write_h5(datasets["valid"], "GLA_valid.h5")
write_h5(datasets["test"], "GLA_test.h5")
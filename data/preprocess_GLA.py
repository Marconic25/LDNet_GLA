import numpy as np
from pathlib import Path
from collections import defaultdict

#Step 1: Configuration
DATA_DIR = Path('/GLA_data/timeseries')

#Check the content of the directory
EXPECTED_COUNTS = {
    "A":  {"train": 6,  "test": 2},
    "B1": {"train": 16, "test": 4},
    "B2": {"train": 12, "test": 3},
    "B3": {"train": 10, "test": 3},
    "C":  {"train": 6,  "test": 2},
}

print(DATA_DIR.exists())
print(list(DATA_DIR.glob("*.csv"))[:3])
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "/root/workspace/workspace/03-Code/Projects/Legacy/PolyMas/services/ml-engine-python")
from polymas_ml.sequence.train import train_smoke

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

data_dir = Path("/root/workspace/workspace/03-Code/Projects/Legacy/PolyMas/results/sequence/smoke_kmer")
output_dir = Path("/root/workspace/workspace/03-Code/Projects/Legacy/PolyMas/results/sequence/smoke_kmer_out")

print("Loading tokens...")
tokens = np.load(data_dir / "tokens.npy")
print("Loading labels...")
labels_df = pd.read_csv(data_dir / "labels.csv")

# Train on just 1-2 diseases first
diseases = ["RA", "SLE"]
assert list(labels_df[diseases].columns) == diseases, f"Label columns mismatch"
y = labels_df[diseases].values

print("Running train_smoke...")
report = train_smoke(
    tokens=tokens,
    y=y,
    diseases=diseases,
    output_dir=output_dir,
    n_epochs=20,
    device_str="cuda"
)

print("\n--- Smoke Test Report ---")
print(json.dumps(report, indent=2))

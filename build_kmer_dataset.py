import json
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sys.path.insert(0, "/root/workspace/workspace/03-Code/Projects/Legacy/PolyMas/services/ml-engine-python")
from polymas_ml.sequence.dataset import build_dataset

results_dir = Path("/root/workspace/workspace/03-Code/Projects/Legacy/PolyMas/results")
ensembl_dir = results_dir / "raw" / "ensembl"
output_dir = results_dir / "sequence" / "smoke_kmer"

windows = json.loads((ensembl_dir / "reference_windows.json").read_text())
variant_info = json.loads((ensembl_dir / "variant_info.json").read_text())

print("Building k-mer dataset...")
build_dataset(
    results_dir=results_dir,
    output_dir=output_dir,
    windows=windows,
    variant_info=variant_info
)
print("Done.")

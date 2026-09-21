"""Build the k-mer sequence dataset (System B) from System A's features.

Run with: python build_kmer_dataset.py [--n-patients 50]
Reads results/features/prs_features.csv written by the System A pipeline and
results/raw/ensembl/{reference_windows,variant_info}.json.
"""
import argparse
import json
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "services" / "ml-engine-python"))
from polymas_ml.sequence.dataset import build_dataset  # noqa: E402

results_dir = PROJECT_ROOT / "results"

parser = argparse.ArgumentParser(description="Build System B k-mer dataset")
parser.add_argument("--n-patients", type=int, default=None,
                    help="Optional: subsample the first N patients (default: all)")
parser.add_argument("--out", type=str, default="smoke_kmer",
                    help="Output dir name under results/sequence/ (default: smoke_kmer)")
parser.add_argument("--max-context", type=int, default=None,
                    help="Optional: stride-subsample reference context to at most N k-mers per locus "
                         "(genotype tokens always kept). E.g. 64 -> 8x65=520 tokens/patient.")
args = parser.parse_args()

ensembl_dir = results_dir / "raw" / "ensembl"
output_dir = results_dir / "sequence" / args.out

windows = json.loads((ensembl_dir / "reference_windows.json").read_text())
variant_info = json.loads((ensembl_dir / "variant_info.json").read_text())

if args.n_patients:
    # Subsample to the first N patients by trimming the label/PRS inputs.
    prs_path = results_dir / "features" / "prs_features.csv"
    labels_path = results_dir / "features" / "labels.csv"
    import pandas as pd

    labels = pd.read_csv(labels_path)
    keep = labels["patient_id"].head(args.n_patients)
    labels[labels["patient_id"].isin(keep)].to_csv(labels_path.with_name("labels_full_backup.csv"), index=False)
    prs = pd.read_csv(prs_path)
    prs[prs["patient_id"].isin(keep)].to_csv(prs_path.with_name("prs_features_full_backup.csv"), index=False)

    # Temporarily narrow the CSVs in place so build_dataset reads only N patients.
    labels[labels["patient_id"].isin(keep)].to_csv(labels_path, index=False)
    prs[prs["patient_id"].isin(keep)].to_csv(prs_path, index=False)
    print(f"Subsampled to first {args.n_patients} patients.")

print("Building k-mer dataset...")
build_dataset(
    results_dir=results_dir,
    output_dir=output_dir,
    windows=windows,
    variant_info=variant_info,
    max_context_per_locus=args.max_context,
)
print("Done. Output:", output_dir)

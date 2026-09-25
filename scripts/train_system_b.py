"""Train the Mamba sequence model (System B) on the k-mer smoke dataset.

Run with: python run_smoke.py [--epochs 20] [--batch-size 2] [--diseases RA SLE]
Reads stash/results/sequence/smoke_kmer/{tokens.npy,labels.csv} written by
build_kmer_dataset.py and writes the training report to
stash/results/sequence/smoke_kmer_out/.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "services" / "ml-engine-python"))
from polymas_ml.sequence.train import train_smoke  # noqa: E402

parser = argparse.ArgumentParser(description="System B Mamba smoke training")
parser.add_argument("--data-dir", type=str, default="smoke_kmer",
                    help="Dataset dir name under stash/results/sequence/ (default: smoke_kmer)")
parser.add_argument("--out-dir", type=str, default="smoke_kmer_out",
                    help="Output dir name under stash/results/sequence/ (default: smoke_kmer_out)")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--eval-batch-size", type=int, default=16,
                    help="Batch size for train/val evaluation passes (default: 16)")
parser.add_argument("--diseases", type=str, nargs="+", default=["RA", "SLE"])
parser.add_argument("--resume", action="store_true",
                    help="Resume from an existing checkpoint.pt in --out-dir (default: fresh start)")
parser.add_argument("--require-gpu", action="store_true",
                    help="Fail fast if CUDA is unavailable instead of silently training on CPU")
args = parser.parse_args()

# Results root for this run; override with POLYMAS_RESULTS_DIR to write a
# fresh run folder without touching previous runs.
results_dir = Path(os.environ.get("POLYMAS_RESULTS_DIR", PROJECT_ROOT / "stash" / "results"))
data_dir = results_dir / "sequence" / args.data_dir
output_dir = results_dir / "sequence" / args.out_dir

print("Loading tokens...")
tokens = np.load(data_dir / "tokens.npy")
print("Loading labels...")
labels_df = pd.read_csv(data_dir / "labels.csv")

diseases = args.diseases
missing = [d for d in diseases if d not in labels_df.columns]
if missing:
    raise SystemExit(f"Missing disease columns in labels.csv: {missing}")
y = labels_df[diseases].values.astype(np.float32)

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
if args.require_gpu and device != "cuda":
    raise SystemExit("--require-gpu: CUDA unavailable (host GPU driver likely down); refusing CPU fallback")
print(f"Training on {diseases}: {tokens.shape[0]} patients x {tokens.shape[1]} tokens, {args.epochs} epochs, device={device}")
report = train_smoke(
    tokens=tokens,
    y=y,
    diseases=diseases,
    output_dir=output_dir,
    n_epochs=args.epochs,
    batch_size=args.batch_size,
    eval_batch_size=args.eval_batch_size,
    device_str=device,
    resume=args.resume,
)

print("\n--- Smoke Test Report ---")
print(json.dumps(report, indent=2))

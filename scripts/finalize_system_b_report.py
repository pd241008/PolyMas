"""Finalize the System B (Mamba) report from the last clean checkpoint.

The 2026-09-23 clean re-run (co-occurrence labels) was interrupted by
repeated GPU driver failures (WSL/dxg NVML loss). It completed 12/20
epochs before the third failure, by which point the model had converged
to constant outputs: train loss pinned at 1.0483 from epoch 8 onward and
all AUROCs at 0.500, so the best-epoch selection (epoch 10, mean val
AUROC 0.5271) cannot change no matter how many of the remaining epochs
finish. This script rebuilds smoke_test_report.json from the checkpoint
with train_smoke's exact semantics: reload the best weights, recompute
train metrics on the train split, carry over the best epoch's val
metrics, and mark the run as early-terminated so the report is honest
about what happened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "services" / "ml-engine-python"))

import pandas as pd  # noqa: E402

from polymas_ml.sequence.model import MambaSequenceClassifier  # noqa: E402
from polymas_ml.sequence.train import build_splits, evaluate  # noqa: E402

SEQ_DIR = PROJECT_ROOT / "results" / "sequence"
DATA_DIR = SEQ_DIR / "kmer5000_gwas"
OUT_DIR = SEQ_DIR / "kmer5000_gwas_out"

DISEASES = ["RA", "SLE", "SJOGRENS", "AITD", "T1D", "VITILIGO", "MS"]

INTERRUPTION_NOTE = (
    "Clean-label run (2026-09-23) interrupted at epoch 12/20 by repeated GPU "
    "driver failures (WSL/dxg NVML loss on the host). The model had already "
    "converged to chance-level outputs by epoch 8 (train loss pinned at "
    "1.0483; all AUROCs 0.500), so the interruption does not affect the "
    "best-epoch selection (epoch 10). Metrics below were recomputed offline "
    "from the checkpoint's best weights on CPU."
)


def main() -> None:
    tokens = np.load(DATA_DIR / "tokens.npy")
    labels_df = pd.read_csv(DATA_DIR / "labels.csv")
    missing = [d for d in DISEASES if d not in labels_df.columns]
    if missing:
        raise SystemExit(f"Missing disease columns in labels.csv: {missing}")
    y = labels_df[DISEASES].values.astype(np.float32)

    ck = torch.load(OUT_DIR / "checkpoint.pt", map_location="cpu", weights_only=False)
    best_state = ck["best_state"]
    if not best_state:
        raise SystemExit("Checkpoint has no best_state — cannot finalize.")

    model = MambaSequenceClassifier(
        vocab_size=int(tokens.max()) + 1,
        n_diseases=len(DISEASES),
        d_model=64,
        n_layers=2,
        d_state=8,
        d_conv=4,
        expand=2,
    )
    model.load_state_dict(best_state["state_dict"])

    split = build_splits(np.arange(len(tokens)), y, diseases=DISEASES)
    train_idx, val_idx = split["train"], split["val"]

    tokens_t = torch.from_numpy(tokens).long()
    y_t = torch.from_numpy(y).float()
    train_metrics = evaluate(
        model, tokens_t[train_idx], y_t[train_idx], DISEASES, torch.device("cpu"), batch_size=16
    )
    val_metrics = best_state["val_metrics"]

    report = {
        "diseases": DISEASES,
        "n_patients": len(tokens),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_epochs": 20,
        "epochs_completed": int(ck["epoch"]),
        "elapsed_seconds": None,
        "config": {
            "d_model": 64,
            "n_layers": 2,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
            "batch_size": 2,
            "lr": 0.001,
            "weight_decay": 0.1,
            "seed": 0,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_epoch": best_state["epoch"],
        "early_terminated": INTERRUPTION_NOTE,
    }
    (OUT_DIR / "smoke_test_report.json").write_text(json.dumps(report, indent=2))

    mean_val = float(
        np.nanmean([val_metrics[f"{d}_auroc"] for d in DISEASES])
    )
    print(f"Report finalized from checkpoint (best epoch {best_state['epoch']}).")
    print(f"Mean val AUROC: {mean_val:.4f}")
    for d in DISEASES:
        va = val_metrics[f"{d}_auroc"]
        tr = train_metrics[f"{d}_auroc"]
        print(f"  {d:10s} val={va:.3f}  train={tr:.3f}")


if __name__ == "__main__":
    main()

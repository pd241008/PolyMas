"""F-18: label-noise robustness curves (ROADMAP Phase 2).

Pre-registered claim: performance degrades gracefully and MONOTONICALLY
under 5/10/20% label flipping; the curve shape is the deliverable (R3).

Protocol (pre-registered before the sweep):
  - Data: the canonical Phase-2 run's patients, features, and F-12 split
    (loaded from <results_root>); NOTHING is rebuilt, so the only changing
    variable is label noise.
  - Noise: per disease, each label is flipped 1<->0 independently with
    probability r, r in {0.00, 0.05, 0.10, 0.20}, flip mask drawn once per
    rate with seed 42 (deterministic). Applied to ALL splits (the label
    matrix itself is noisy, train and test alike).
  - Model: the same System A ensemble (xgboost/catboost/lightgbm + Platt)
    retrained per rate on the canonical train split.
  - Metric: held-out AUROC per disease against the FLIPPED test labels
    (that is what robustness to label noise means here).
  - Curve check: per disease, AUROC(r) must be non-increasing within
    tolerance 0.01 per 0.05 step (bootstrap noise). Violations reported.

System B curves run through the guarded GPU wrapper (same flip protocol
over the k-mer dataset builder) and are merged into the same CSV.

Evidence: <results_root>/f18_noise_sweep/
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "services" / "ml-engine-python"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("f18_noise")

RATES = [0.0, 0.05, 0.10, 0.20]
FLIP_SEED = 42
MONOTONE_TOL = 0.01


def main() -> int:
    results = Path(os.environ.get(
        "POLYMAS_RESULTS_DIR", ROOT / "stash/results_final_20260926"))
    out_dir = results / "f18_noise_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    import run_real_pipeline as pipe

    prs_df = pd.read_csv(results / "features" / "prs_features.csv")
    clinical_df = pd.read_csv(results / "features" / "clinical_features.csv")
    labels = pd.read_csv(results / "features" / "labels.csv").set_index("patient_id")
    split = pd.read_csv(results / "models" / "split_indices.csv")

    X = pipe.prepare_feature_matrix(prs_df, clinical_df)
    X.index = clinical_df["patient_id"].values
    y = labels.loc[X.index, pipe.DISEASE_LABELS].copy()
    idx_train = split.loc[split["split"] == "train", "patient_id"].map(X.index.get_loc).to_numpy()
    idx_val = split.loc[split["split"] == "val", "patient_id"].map(X.index.get_loc).to_numpy()
    idx_test = split.loc[split["split"] == "test", "patient_id"].map(X.index.get_loc).to_numpy()

    rows = []
    for rate in RATES:
        rng = np.random.default_rng(FLIP_SEED)
        flip = rng.random(y.shape) < rate
        y_noisy = pd.DataFrame(
            np.where(flip, 1 - y.to_numpy(), y.to_numpy()),
            index=y.index, columns=y.columns)
        logger.info("rate=%.2f: training System A...", rate)
        ensemble = pipe.train_ensemble(X, y_noisy, train_indices=idx_train)
        from sklearn.metrics import roc_auc_score
        preds = ensemble.predict_proba(X.iloc[idx_test])
        for disease in pipe.DISEASE_LABELS:
            y_true = y_noisy.iloc[idx_test][disease].to_numpy()
            if len(np.unique(y_true)) < 2:
                continue
            auroc = float(roc_auc_score(y_true, preds[disease].to_numpy()))
            rows.append({"rate": rate, "disease": disease, "auroc": round(auroc, 4),
                         "system": "A"})
            logger.info("  %s AUROC=%.4f", disease, auroc)

    df = pd.DataFrame(rows)
    if not (out_dir / "noise_curves.csv").exists():
        df.to_csv(out_dir / "noise_curves.csv", index=False)
    else:
        existing = pd.read_csv(out_dir / "noise_curves.csv")
        pd.concat([existing, df]).drop_duplicates(
            subset=["system", "disease", "rate"], keep="last"
        ).to_csv(out_dir / "noise_curves.csv", index=False)
        df = pd.read_csv(out_dir / "noise_curves.csv")

    # Monotonicity check per disease (System A only here).
    checks = {}
    a = df[df["system"] == "A"]
    for disease, g in a.groupby("disease"):
        g = g.sort_values("rate")
        aucs = g["auroc"].to_numpy()
        drops = np.diff(aucs)
        checks[disease] = {
            "curve": aucs.tolist(),
            "monotone_within_tol": bool(all(d <= MONOTONE_TOL for d in drops)),
            "total_drop": round(float(aucs[0] - aucs[-1]), 4),
        }
    summary = {
        "protocol": "labels flipped 1<->0 with prob r on all splits; seed 42; "
                    "System A ensemble retrained per rate on the canonical train split",
        "rates": RATES,
        "monotone_tolerance": MONOTONE_TOL,
        "checks": checks,
        "system_b_pending": True,
    }
    (out_dir / "noise_sweep_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("F-18 summary:\n%s", json.dumps(checks, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

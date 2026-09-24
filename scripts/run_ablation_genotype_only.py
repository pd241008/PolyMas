"""Genotype-only ensemble ablation (apples-to-apples vs System B).

Re-runs the ensemble training step restricted to the 16 PRS features
(__score / __zscore columns) — dropping age, bmi, family_history, sex and
the three ethnicity dummies — keeping everything else identical to the
full-feature run: same patients, same composite-stratified 80/20 split,
same Platt calibration, same three base learners. Outputs are suffixed
with _genotype_only so the full-feature tables are never overwritten,
giving the clean three-way comparison: full ensemble vs genotype-only
ensemble vs Mamba (sequence-only).

Run with: PYTHONPATH=services/ml-engine-python services/ml-engine-python/.venv/bin/python \
          scripts/run_ablation_genotype_only.py --n-patients 5000
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path
import sys

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "services" / "ml-engine-python"))

from polymas_ml.data.patients import DISEASE_LABELS  # noqa: E402
from polymas_ml.models.ensemble import MultiLabelEnsemble  # noqa: E402


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location(
        "run_real_pipeline", PROJECT_ROOT / "scripts" / "run_real_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Genotype-only ensemble ablation")
    parser.add_argument("--n-patients", type=int, default=5000)
    args = parser.parse_args()

    rrp = _load_pipeline_module()

    logger.info("=== Genotype-only ablation (n=%d) ===", args.n_patients)
    prs_df, clinical_df, labels_df, _ = rrp.build_real_dataset(args.n_patients)
    X_full = rrp.prepare_feature_matrix(prs_df, clinical_df)
    X_full.index = clinical_df["patient_id"].values
    y = labels_df.set_index("patient_id").loc[X_full.index, DISEASE_LABELS].copy()

    geno_cols = [c for c in X_full.columns if c.endswith("__score") or c.endswith("__zscore")]
    X_geno = X_full[geno_cols].copy()
    logger.info("Genotype-only feature matrix: %d patients x %d features", len(X_geno), len(geno_cols))

    idx_train, idx_test = rrp.make_train_test_split(y)
    logger.info("Split: %d train / %d test (same composite-stratified split as the full run)", len(idx_train), len(idx_test))

    ensemble = MultiLabelEnsemble(learner_names=["xgboost", "catboost", "lightgbm"], platt_scaling=True)
    ensemble.fit(X_geno, y, train_indices=idx_train)

    logger.info("Scoring held-out test split (genotype-only features)...")
    metrics = rrp.run_metrics(ensemble, X_geno, y, idx_test, tag="genotype_only")
    logger.info("Saved %s", rrp.MODELS_DIR / "per_disease_metrics_genotype_only.csv")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

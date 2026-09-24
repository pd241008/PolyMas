"""Ancestry-stratified held-out evaluation.

Computes AUROC/AUPRC per disease within each ancestry group (EUR/AFR/EAS
from the REAL ImmPort demographic.race mapping), plus pooled stratified
bootstrap CIs, so the paper can show results split by ancestry instead of
only flagging ancestry as a limitation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)

# Minimum positives per disease within a stratum to attempt AUROC.
MIN_POSITIVES = 5


def ancestry_stratified_metrics(
    y_test: pd.DataFrame,
    preds: pd.DataFrame,
    ancestry: pd.Series,
    diseases: list[str],
) -> pd.DataFrame:
    """Per-disease discrimination metrics within each ancestry group.

    Rows: disease x ancestry with n, positives, AUROC, AUPRC. Cells with
    too few positives (or a single class) are NaN with a note, not dropped,
    so the table honestly shows where signal cannot be estimated.
    """
    ancestry = ancestry.reset_index(drop=True)
    rows = []
    for group in sorted(ancestry.unique()):
        idx = np.flatnonzero(ancestry == group)
        for disease in diseases:
            if disease not in y_test.columns or disease not in preds.columns:
                continue
            y_true = y_test[disease].to_numpy()[idx]
            p = preds[disease].to_numpy()[idx]
            n_pos = int(y_true.sum())
            if n_pos < MIN_POSITIVES or len(np.unique(y_true)) < 2:
                auroc, auprc, note = float("nan"), float("nan"), "insufficient positives"
            else:
                auroc = float(roc_auc_score(y_true, p))
                auprc = float(average_precision_score(y_true, p))
                note = ""
            rows.append({
                "disease": disease,
                "ancestry": group,
                "n": int(len(idx)),
                "n_positives": n_pos,
                "auroc": round(auroc, 4) if not np.isnan(auroc) else np.nan,
                "auprc": round(auprc, 4) if not np.isnan(auprc) else np.nan,
                "note": note,
            })
    out = pd.DataFrame(rows)
    logger.info("Ancestry-stratified metrics: %d groups", out["ancestry"].nunique())
    return out


def ancestry_bootstrap_intervals(
    y_test: pd.DataFrame,
    preds: pd.DataFrame,
    ancestry: pd.Series,
    diseases: list[str],
    metric: str = "auroc",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 123,
) -> pd.DataFrame:
    """Per-disease stratified bootstrap CIs on the pooled test split."""
    from polymas_ml.evaluation.stats import stratified_bootstrap_intervals

    rows = []
    ancestry = ancestry.reset_index(drop=True)
    for i, disease in enumerate(diseases):
        if disease not in y_test.columns or disease not in preds.columns:
            continue
        res = stratified_bootstrap_intervals(
            y_test[disease].to_numpy(dtype=float),
            preds[disease].to_numpy(dtype=float),
            ancestry.to_numpy(),
            metric=metric,
            n_boot=n_boot,
            ci=ci,
            seed=seed + 1000 + i,
        )
        rows.append({
            "disease": disease,
            "metric": metric,
            "point": round(res["point"], 4),
            "ci_lo": round(res["lo"], 4),
            "ci_hi": round(res["hi"], 4),
            "boot_se": round(res["boot_se"], 4),
            "n_boot": n_boot,
            "ci_level": ci,
            "stratified_by": "ancestry",
        })
    return pd.DataFrame(rows)

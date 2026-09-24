"""Statistical rigor layer: bootstrap CIs and permutation tests.

Computable from the existing pipeline outputs without new data:

- Bootstrap confidence intervals (percentile, patient-level resampling)
  for held-out AUROC/AUPRC per disease.
- Permutation test for the clustering silhouette score: repeatedly cluster
  within-disease-shuffled prediction vectors to build the null distribution
  of "no real cluster structure" while preserving per-disease marginal
  distributions and the correlation between predictions and labels.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.metrics import silhouette_score as sk_silhouette


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals for held-out discrimination metrics
# --------------------------------------------------------------------------- #
def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "auroc",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 123,
) -> dict[str, float]:
    """Patient-level percentile bootstrap CI for AUROC or AUPRC.

    Resamples test patients with replacement n_boot times; the point
    estimate is computed on the original sample. Returns the interval
    [lo, hi], the point estimate, and the bootstrap standard error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    fn: Callable[[np.ndarray, np.ndarray], float]
    if metric == "auroc":
        fn = roc_auc_score
    elif metric == "auprc":
        fn = average_precision_score
    else:
        raise ValueError(f"Unsupported metric {metric!r} (expected 'auroc' or 'auprc')")

    n = len(y_true)
    point = float(fn(y_true, y_score))

    # Single-class samples cannot produce an interval.
    if len(np.unique(y_true)) < 2:
        return {
            "point": point,
            "lo": float("nan"),
            "hi": float("nan"),
            "boot_se": float("nan"),
            "n_boot_used": 0,
            "n_test": int(n),
        }

    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y_true[idx], y_score[idx]
        if len(np.unique(yb)) < 2:
            continue  # degenerate resample: skip
        boots.append(float(fn(yb, sb)))

    if not boots:
        return {
            "point": point,
            "lo": float("nan"),
            "hi": float("nan"),
            "boot_se": float("nan"),
            "n_boot_used": 0,
            "n_test": int(n),
        }

    boots_arr = np.asarray(boots)
    alpha = (1.0 - ci) / 2.0
    return {
        "point": point,
        "lo": float(np.quantile(boots_arr, alpha)),
        "hi": float(np.quantile(boots_arr, 1.0 - alpha)),
        "boot_se": float(boots_arr.std(ddof=1)),
        "n_boot_used": int(len(boots)),
        "n_test": int(n),
    }


def per_disease_bootstrap_table(
    y_test: pd.DataFrame,
    preds: pd.DataFrame,
    diseases: list[str],
    metric: str = "auroc",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 123,
) -> pd.DataFrame:
    """Bootstrap CI table per disease for the held-out test split.

    Rows: disease, n_test, n_positives, point estimate, CI lo/hi, boot SE.
    Diseases with fewer than 2 classes in y_test yield NaN intervals.
    """
    rows = []
    for i, disease in enumerate(diseases):
        if disease not in y_test.columns or disease not in preds.columns:
            continue
        y_true = y_test[disease].to_numpy(dtype=float)
        y_score = preds[disease].to_numpy(dtype=float)
        res = bootstrap_metric_intervals(
            y_true, y_score, metric=metric, n_boot=n_boot, ci=ci, seed=seed + i
        )
        rows.append({
            "disease": disease,
            "metric": metric,
            "n_test": res["n_test"],
            "n_positives": int(y_true.sum()),
            "point": round(res["point"], 4),
            "ci_lo": round(res["lo"], 4),
            "ci_hi": round(res["hi"], 4),
            "boot_se": round(res["boot_se"], 4),
            "n_boot": res["n_boot_used"],
            "ci_level": ci,
        })
    return pd.DataFrame(rows)


def stratified_bootstrap_intervals(
    y_true: np.ndarray,
    y_score: np.ndarray,
    strata: np.ndarray,
    metric: str = "auroc",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 123,
) -> dict[str, float]:
    """Ancestry-stratified bootstrap CI: resample within each stratum.

    Preserves the ancestry composition of the test split in every resample
    (marginal-matched resampling), so the CI reflects within-group
    variability of a mixed split rather than group-composition jitter.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    strata = np.asarray(strata)
    fn = roc_auc_score if metric == "auroc" else average_precision_score

    point = float(fn(y_true, y_score)) if len(np.unique(y_true)) >= 2 else float("nan")
    stratum_indices = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        parts_y, parts_s = [], []
        for idx in stratum_indices:
            take = idx[rng.integers(0, len(idx), size=len(idx))]
            parts_y.append(y_true[take])
            parts_s.append(y_score[take])
        yb = np.concatenate(parts_y)
        sb = np.concatenate(parts_s)
        if len(np.unique(yb)) < 2:
            continue
        boots.append(float(fn(yb, sb)))

    if not boots:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "boot_se": float("nan")}
    arr = np.asarray(boots)
    alpha = (1.0 - ci) / 2.0
    return {
        "point": point,
        "lo": float(np.quantile(arr, alpha)),
        "hi": float(np.quantile(arr, 1.0 - alpha)),
        "boot_se": float(arr.std(ddof=1)),
    }


# --------------------------------------------------------------------------- #
# Permutation test for the silhouette score
# --------------------------------------------------------------------------- #
def silhouette_permutation_test(
    risk_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    n_permutations: int = 500,
    seed: int = 123,
    metric: str = "euclidean",
    cluster_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
) -> dict[str, float]:
    """Permutation test for hierarchical-cluster structure.

    H0: patient risk vectors carry no real cluster structure beyond their
    per-disease marginals. Null: each permutation independently shuffles the
    values WITHIN each disease column, destroying the patient-level joint
    (co-occurrence structure across diseases) while preserving each
    disease's marginal distribution exactly. The observed silhouette from
    the actual clustering is compared against the silhouette distribution
    of Ward re-clusterings (same k) of the shuffled matrices.

    Returns the observed silhouette, the null mean/sd, the one-sided
    p-value, and the null z-score.
    """
    X = np.asarray(risk_matrix, dtype=float)
    observed = float(sk_silhouette(X, cluster_labels, metric=metric))
    n_clusters = len(np.unique(cluster_labels))
    rng = np.random.default_rng(seed)

    def default_cluster(mat: np.ndarray, gen: np.random.Generator) -> np.ndarray:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        Z = linkage(pdist(mat, metric=metric), method="ward")
        return fcluster(Z, t=n_clusters, criterion="maxclust")

    cluster_null = cluster_fn or default_cluster

    null_scores: list[float] = []
    for _ in range(n_permutations):
        X_perm = X.copy()
        for col in range(X_perm.shape[1]):
            X_perm[:, col] = X_perm[rng.permutation(X_perm.shape[0]), col]
        labels_perm = cluster_null(X_perm, rng)
        if len(np.unique(labels_perm)) < 2:
            continue
        null_scores.append(float(sk_silhouette(X_perm, labels_perm, metric=metric)))

    if not null_scores:
        return {
            "observed": observed,
            "null_mean": float("nan"),
            "null_sd": float("nan"),
            "p_value": float("nan"),
            "z_score": float("nan"),
            "n_permutations_used": 0,
        }

    null_arr = np.asarray(null_scores)
    # One-sided: proportion of null silhouettes >= observed (+1 correction).
    p_value = float((np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1))
    null_sd = float(null_arr.std(ddof=1))
    z = float((observed - null_arr.mean()) / null_sd) if null_sd > 0 else float("nan")
    return {
        "observed": round(observed, 6),
        "null_mean": round(float(null_arr.mean()), 6),
        "null_sd": round(float(null_arr.std(ddof=1)), 6),
        "p_value": round(p_value, 6),
        "z_score": round(z, 4),
        "n_permutations_used": int(len(null_arr)),
        "null_scores": [round(float(v), 6) for v in null_arr],
    }


# --------------------------------------------------------------------------- #
# Co-occurrence statistics (phi correlations with p-values)
# --------------------------------------------------------------------------- #
def pairwise_phi_pvalues(labels: pd.DataFrame, diseases: list[str]) -> pd.DataFrame:
    """Pairwise phi (2x2 correlation) between disease labels with p-values.

    Pearson chi-squared (no Yates correction) on each 2x2 contingency
    table; phi is the signed sqrt(chi2/n) with the sign of the 2x2
    cross-product, so negative (incoercible/exclusion) pairs show up as
    negative phi.
    """
    rows = []
    n = len(labels)
    for i, a in enumerate(diseases):
        for b in diseases[i + 1:]:
            if a not in labels.columns or b not in labels.columns:
                continue
            xa = labels[a].to_numpy(dtype=float)
            xb = labels[b].to_numpy(dtype=float)
            va, vb = xa.std(), xb.std()
            phi = (
                float(((xa - xa.mean()) * (xb - xb.mean())).mean() / (va * vb))
                if va > 0 and vb > 0
                else 0.0
            )
            n11 = int(((xa == 1) & (xb == 1)).sum())
            n10 = int(((xa == 1) & (xb == 0)).sum())
            n01 = int(((xa == 0) & (xb == 1)).sum())
            n00 = int(((xa == 0) & (xb == 0)).sum())
            table = np.array([[n11, n10], [n01, n00]])
            if table.min() == 0:
                pval = float("nan")
            else:
                pval = float(sps.chi2_contingency(table, correction=False).pvalue)
            rows.append({
                "disease_a": a,
                "disease_b": b,
                "phi": round(phi, 4),
                "p_value": pval,
                "n11_both_positive": n11,
                "n_patients": n,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bonferroni"] = (out["p_value"] * len(out)).clip(upper=1.0).round(6)
    return out.sort_values("phi", ascending=False).reset_index(drop=True)

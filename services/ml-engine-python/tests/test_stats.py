"""Tests for the statistical rigor layer (bootstrap CIs + permutation tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from polymas_ml.evaluation.ancestry import (
    ancestry_bootstrap_intervals,
    ancestry_stratified_metrics,
)
from polymas_ml.evaluation.stats import (
    bootstrap_metric_intervals,
    pairwise_phi_pvalues,
    per_disease_bootstrap_table,
    silhouette_permutation_test,
)


def _make_scores(n: int, seed: int, auroc_strength: float = 2.0):
    """Score construction with a controllable signal strength."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    signal = y * auroc_strength + rng.normal(0, 1, n)
    return y.astype(float), signal


def test_bootstrap_ci_covers_point_and_has_width():
    y, s = _make_scores(400, seed=0)
    res = bootstrap_metric_intervals(y, s, metric="auroc", n_boot=300, seed=1)
    assert res["lo"] <= res["point"] <= res["hi"]
    assert res["hi"] - res["lo"] > 0.0
    assert res["n_boot_used"] == 300
    # Decent separation should be well above chance.
    assert res["point"] > 0.6


def test_bootstrap_ci_single_class_returns_nan():
    y = np.zeros(100)
    s = np.random.default_rng(0).normal(size=100)
    res = bootstrap_metric_intervals(y, s, metric="auroc", n_boot=50)
    assert np.isnan(res["lo"]) and np.isnan(res["hi"])


def test_per_disease_bootstrap_table_shape():
    n = 500
    rng = np.random.default_rng(2)
    y_test = pd.DataFrame({
        "RA": (rng.random(n) < 0.4).astype(int),
        "SLE": (rng.random(n) < 0.2).astype(int),
    })
    preds = pd.DataFrame({
        "RA": rng.normal(size=n),
        "SLE": rng.normal(size=n),
    })
    table = per_disease_bootstrap_table(y_test, preds, ["RA", "SLE"], n_boot=100, seed=3)
    assert set(table["disease"]) == {"RA", "SLE"}
    assert {"point", "ci_lo", "ci_hi", "boot_se"} <= set(table.columns)
    assert (table["ci_lo"] <= table["point"]).all()
    assert (table["point"] <= table["ci_hi"]).all()


def test_silhouette_permutation_detects_real_structure():
    """Well-separated blobs should beat the within-column-shuffle null."""
    rng = np.random.default_rng(4)
    blob_a = rng.normal(0.0, 0.05, size=(60, 7))
    blob_b = rng.normal(1.0, 0.05, size=(60, 7))
    X = np.vstack([blob_a, blob_b])
    Z = linkage(pdist(X), method="ward")
    labels = fcluster(Z, t=2, criterion="maxclust")
    res = silhouette_permutation_test(X, labels, n_permutations=30, seed=5)
    assert res["p_value"] <= 0.1
    assert res["z_score"] > 2
    assert res["observed"] > res["null_mean"]


def test_silhouette_permutation_null_for_structureless_matrix():
    """Independent columns (no patient-level joint) should NOT beat null."""
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, size=(120, 7))
    Z = linkage(pdist(X), method="ward")
    labels = fcluster(Z, t=3, criterion="maxclust")
    res = silhouette_permutation_test(X, labels, n_permutations=30, seed=7)
    assert res["p_value"] > 0.05 or res["z_score"] < 2


def test_pairwise_phi_pvalues_flags_and_orders():
    rng = np.random.default_rng(8)
    n = 2000
    latent = rng.random(n) < 0.3
    a = latent | (rng.random(n) < 0.1)
    b = latent | (rng.random(n) < 0.1)
    c = ((~latent) & (rng.random(n) < 0.2)).astype(int)  # antagonistic
    labels = pd.DataFrame({"A": a.astype(int), "B": b.astype(int), "C": c})
    table = pairwise_phi_pvalues(labels, ["A", "B", "C"])
    row_ab = table[(table.disease_a == "A") & (table.disease_b == "B")].iloc[0]
    row_ac = table[((table.disease_a == "A") & (table.disease_b == "C"))].iloc[0]
    assert row_ab["phi"] > 0.2 and row_ab["p_bonferroni"] < 0.05
    assert row_ac["phi"] < 0


def test_ancestry_stratified_metrics_computes_by_group():
    n = 600
    rng = np.random.default_rng(9)
    ancestry = pd.Series(rng.choice(["EUR", "AFR", "EAS"], size=n, p=[0.6, 0.2, 0.2]))
    y = pd.DataFrame({"RA": (rng.random(n) < 0.35).astype(int)})
    preds = pd.DataFrame({"RA": rng.normal(size=n) + y["RA"]})
    out = ancestry_stratified_metrics(y, preds, ancestry, ["RA"])
    assert set(out["ancestry"]) == {"EUR", "AFR", "EAS"}
    assert out["auroc"].notna().all()
    assert (out["n"] > 0).all()


def test_ancestry_stratified_metrics_flags_insufficient():
    ancestry = pd.Series(["EUR"] * 90 + ["EAS"] * 10)
    y = pd.DataFrame({"RA": [1] * 8 + [0] * 82 + [0] * 10})
    preds = pd.DataFrame({"RA": np.linspace(0, 1, 100)})
    out = ancestry_stratified_metrics(y, preds, ancestry, ["RA"])
    eas_row = out[out["ancestry"] == "EAS"].iloc[0]
    assert np.isnan(eas_row["auroc"])
    assert "insufficient" in eas_row["note"]


def test_ancestry_bootstrap_intervals_run():
    n = 400
    rng = np.random.default_rng(10)
    ancestry = pd.Series(rng.choice(["EUR", "AFR"], size=n))
    y = pd.DataFrame({"RA": (rng.random(n) < 0.4).astype(int)})
    preds = pd.DataFrame({"RA": rng.normal(size=n) + 1.5 * y["RA"]})
    ci = ancestry_bootstrap_intervals(y, preds, ancestry, ["RA"], n_boot=100, seed=11)
    assert len(ci) == 1
    row = ci.iloc[0]
    assert row["ci_lo"] <= row["point"] <= row["ci_hi"]

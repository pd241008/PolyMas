"""Tests for hierarchical clustering."""

import numpy as np
import pandas as pd
import pytest

from polymas_ml.clustering.hierarchical import DiseaseRiskClusterer


@pytest.fixture
def risk_matrix():
    np.random.seed(42)
    return pd.DataFrame(
        np.random.rand(50, 5),
        columns=["T1D", "T2D", "LADA", "GESTATIONAL_DM", "MONOGENIC_DIABETES"],
    )


def test_fit_predict(risk_matrix):
    clusterer = DiseaseRiskClusterer(min_clusters=3, max_clusters=5)
    labels = clusterer.fit_predict(risk_matrix, n_clusters=3)
    assert len(labels) == 50
    assert len(set(labels)) == 3


def test_dendrogram_json(risk_matrix):
    clusterer = DiseaseRiskClusterer()
    clusterer.fit_predict(risk_matrix, n_clusters=3)
    patient_ids = [f"P{i:04d}" for i in range(50)]
    dendro = clusterer.dendrogram_json(patient_ids)
    assert isinstance(dendro, str)
    assert "distance" in dendro


def test_silhouette(risk_matrix):
    clusterer = DiseaseRiskClusterer()
    clusterer.fit_predict(risk_matrix, n_clusters=3)
    score = clusterer.silhouette_score(risk_matrix)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0

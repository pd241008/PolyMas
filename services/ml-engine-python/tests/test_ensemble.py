"""Tests for the ML ensemble pipeline."""

import numpy as np
import pandas as pd
import pytest

from polymas_ml.models.ensemble import MultiLabelEnsemble
from polymas_ml.models.shared_features import SHARED_LOCI


@pytest.fixture
def synthetic_data():
    """Generate semi-synthetic patient data for testing."""
    np.random.seed(42)
    n_patients = 200
    n_features = 15

    X = pd.DataFrame(
        np.random.randn(n_patients, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    y = pd.DataFrame({
        "RA": np.random.binomial(1, 0.20, n_patients),
        "SLE": np.random.binomial(1, 0.10, n_patients),
        "SJOGRENS": np.random.binomial(1, 0.08, n_patients),
        "AITD": np.random.binomial(1, 0.15, n_patients),
        "T1D": np.random.binomial(1, 0.08, n_patients),
        "VITILIGO": np.random.binomial(1, 0.06, n_patients),
        "MS": np.random.binomial(1, 0.12, n_patients),
    })

    return X, y


def test_ensemble_fit_predict(synthetic_data):
    X, y = synthetic_data
    ensemble = MultiLabelEnsemble(learner_names=["xgboost"], platt_scaling=False)
    importances = ensemble.fit(X, y)
    assert "RA" in importances

    predictions = ensemble.predict_proba(X)
    assert predictions.shape == (200, 7)
    assert (predictions >= 0).all().all()
    assert (predictions <= 1).all().all()


def test_platt_scaling(synthetic_data):
    X, y = synthetic_data
    ensemble = MultiLabelEnsemble(learner_names=["xgboost"], platt_scaling=True)
    ensemble.fit(X, y)
    predictions = ensemble.predict_proba(X)
    assert predictions.shape == (200, 7)


def test_shared_loci_coverage():
    assert "rs2187668" in SHARED_LOCI
    assert "rs2476601" in SHARED_LOCI
    assert "rs3087243" in SHARED_LOCI
    assert "rs11209026" in SHARED_LOCI
    assert all("gene" in v for v in SHARED_LOCI.values())
    assert all("diseases" in v for v in SHARED_LOCI.values())

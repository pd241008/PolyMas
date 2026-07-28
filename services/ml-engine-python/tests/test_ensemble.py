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
        "T1D": np.random.binomial(1, 0.15, n_patients),
        "T2D": np.random.binomial(1, 0.25, n_patients),
        "LADA": np.random.binomial(1, 0.08, n_patients),
        "GESTATIONAL_DM": np.random.binomial(1, 0.12, n_patients),
        "MONOGENIC_DIABETES": np.random.binomial(1, 0.05, n_patients),
    })

    return X, y


def test_ensemble_fit_predict(synthetic_data):
    X, y = synthetic_data
    ensemble = MultiLabelEnsemble(learner_names=["xgboost"], platt_scaling=False)
    importances = ensemble.fit(X, y)
    assert "T1D" in importances

    predictions = ensemble.predict_proba(X)
    assert predictions.shape == (200, 5)
    assert (predictions >= 0).all().all()
    assert (predictions <= 1).all().all()


def test_platt_scaling(synthetic_data):
    X, y = synthetic_data
    ensemble = MultiLabelEnsemble(learner_names=["xgboost"], platt_scaling=True)
    ensemble.fit(X, y)
    predictions = ensemble.predict_proba(X)
    assert predictions.shape == (200, 5)


def test_shared_loci_coverage():
    assert "HLA_DR3" in SHARED_LOCI
    assert "CTLA4" in SHARED_LOCI
    assert "PTPN22" in SHARED_LOCI
    assert "TCF7L2" in SHARED_LOCI
    assert all("gene" in v for v in SHARED_LOCI.values())
    assert all("diseases" in v for v in SHARED_LOCI.values())

"""Tests for F-15 uncertainty metrics (NLL / ECE / Brier)."""

import numpy as np
import pytest

from polymas_ml.evaluation.uncertainty import all_metrics, brier, ece, nll


def test_perfect_predictor_scores_best() -> None:
    rng = np.random.default_rng(0)
    p_true = rng.beta(2, 5, 5000)
    y = (rng.uniform(size=5000) < p_true).astype(float)
    perfect = y  # p = y exactly
    assert nll(y, perfect) < 0.01
    assert brier(y, perfect) < 0.01
    assert ece(y, perfect) < 0.01


def test_constant_predictor_nll_matches_formula() -> None:
    y = np.array([1, 0, 1, 1, 0, 0])
    p = np.full(6, 0.5)
    assert nll(y, p) == pytest.approx(np.log(2), rel=1e-6)


def test_confidently_wrong_is_punished() -> None:
    y = np.array([1, 1, 1, 0, 0, 0])
    p_wrong = np.array([0.01, 0.01, 0.01, 0.99, 0.99, 0.99])
    p_mild = np.array([0.4, 0.4, 0.4, 0.6, 0.6, 0.6])
    assert nll(y, p_wrong) > nll(y, p_mild) > nll(y, y)


def test_ece_zero_for_perfectly_calibrated() -> None:
    rng = np.random.default_rng(1)
    p = np.repeat([0.2, 0.5, 0.8], 1000)
    y = (rng.uniform(size=3000) < p).astype(float)
    assert ece(y, p, n_bins=15) < 0.03


def test_ece_binning_property() -> None:
    # All predictions in one bin: ECE = |mean p - mean y| exactly.
    p = np.full(100, 0.7)
    y = np.array([1] * 70 + [0] * 30)  # mean y = 0.7 = mean p
    assert ece(y, p, n_bins=15) == pytest.approx(0.0, abs=1e-9)
    # Same p against a mismatched y: ECE = the bin gap exactly.
    y2 = np.array([1] * 60 + [0] * 40)  # mean y = 0.6
    assert ece(y2, p, n_bins=15) == pytest.approx(0.1, abs=1e-9)


def test_all_metrics_keys() -> None:
    y = np.array([0, 1])
    m = all_metrics(y, np.array([0.3, 0.6]))
    assert set(m) == {"nll", "ece", "brier"}


def test_ensemble_beats_worse_member_in_expectation() -> None:
    """Averaging two members' probs should not be worse than the worse
    member on Brier (regression-to-the-mean property)."""
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=3000) < 0.3).astype(float)
    good = np.clip(0.3 + rng.normal(0, 0.05, 3000), 0, 1)
    bad = np.clip(0.7 + rng.normal(0, 0.05, 3000), 0, 1)
    ens = (good + bad) / 2
    assert brier(y, ens) < brier(y, bad)

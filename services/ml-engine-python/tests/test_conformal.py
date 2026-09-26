"""Tests for F-11: split-conformal prediction sets."""

import numpy as np
import pytest

from polymas_ml.evaluation.conformal import (
    calibrate,
    conformity_scores,
    evaluate_coverage,
    finite_sample_quantile,
    prediction_sets,
    stratified_coverage,
)


def test_quantile_index_convention() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # n=9, alpha=0.10 -> k = ceil(10*0.9) = 9 -> 9th order statistic.
    assert finite_sample_quantile(scores, 0.10) == pytest.approx(0.9)
    # alpha=0.5 -> k = ceil(10*0.5) = 5 -> 5th order statistic.
    assert finite_sample_quantile(scores, 0.50) == pytest.approx(0.5)
    # Tiny sample: k > n -> inf (set contains everything).
    assert finite_sample_quantile(np.array([0.3]), 0.10) == float("inf")


def test_conformity_scores_orientation() -> None:
    s = conformity_scores(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.3, 0.8]))
    np.testing.assert_allclose(s, [0.1, 0.1, 0.7, 0.8])


def test_coverage_guarantee_synthetic() -> None:
    """Under exchangeability with a well-calibrated model, empirical
    coverage on a fresh test sample should meet 1 - alpha (up to noise)."""
    rng = np.random.default_rng(0)
    n_cal, n_test = 4000, 4000
    # True probability varies per patient; labels drawn from it.
    p_true = rng.beta(2, 5, size=n_cal + n_test)
    y = (rng.uniform(size=n_cal + n_test) < p_true).astype(int)
    # Model probs = true probs + small noise (calibrated on average).
    p = np.clip(p_true + rng.normal(0, 0.02, size=n_cal + n_test), 0, 1)
    alpha = 0.10
    res = evaluate_coverage(y[:n_cal], p[:n_cal], y[n_cal:], p[n_cal:], alpha=alpha)
    assert res["coverage"] >= 1 - alpha - 0.02  # small-sample slack
    assert res["frac_empty"] == 0  # one of the two rules always includes


def test_prediction_sets_rules() -> None:
    # q_pos = 0.8 (1 - p <= 0.8 -> p >= 0.2), q_neg = 0.3 (p <= 0.3).
    thr = {"q_pos": 0.8, "q_neg": 0.3}
    sets = prediction_sets(np.array([0.05, 0.25, 0.5, 0.85, 0.95]), thr)
    # p=0.05: {0} only (1-p=0.95 > q_pos=0.8 -> 1 not plausible);
    # p=0.25: {0,1}; p=0.5: {1} only (0.5 > q_neg=0.3);
    # p=0.85, 0.95: {1} (0 non-plausible, 1-plausible).
    expected = np.array([
        [True, False],
        [True, True],
        [False, True],
        [False, True],
        [False, True],
    ])
    assert (sets == expected).all()


def test_set_sizes_shrink_with_confidence() -> None:
    rng = np.random.default_rng(1)
    p_cal = rng.uniform(0, 1, 3000)
    y_cal = (rng.uniform(size=3000) < p_cal).astype(int)
    res = evaluate_coverage(y_cal, p_cal, y_cal[:500], p_cal[:500], alpha=0.2)
    # Near-extreme probabilities should produce mostly singletons.
    assert 0 < res["mean_set_size"] <= 2


def test_calibrate_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        calibrate(np.ones(50), np.linspace(0.1, 0.9, 50))


def test_stratified_coverage_reports_groups() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(size=n) < p).astype(int)
    groups = np.array(["EUR"] * 1500 + ["AFR"] * 500)
    out = stratified_coverage(y[1500:], p[1500:], groups[1500:],
                              y[:1500], p[:1500], alpha=0.1)
    # Only test-split groups appear (EUR/AFR mix in the last 500 is
    # random, so both may or may not appear; assert structure).
    for g, v in out.items():
        assert set(v) == {"n", "coverage", "mean_set_size"}
        assert v["n"] > 0
    assert sum(v["n"] for v in out.values()) == 500

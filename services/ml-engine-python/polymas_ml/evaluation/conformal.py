"""F-11: split-conformal prediction sets (ROADMAP Phase 4).

Pre-registered claim: split-conformal sets achieve >= 88% empirical
coverage at the nominal 90% level on held-out data, per disease
(tolerance: coverage >= 0.90 - 0.02 = 0.88). Miscoverage reported per
disease AND per ancestry.

Method (standard split conformal for binary classification, marginal
coverage, Vovk et al. / Papadopoulos et al.):
  - Calibration: honest held-out calibration probabilities p_cal for
    patients NOT used to fit the model, with their true labels y_cal.
  - Conformity score per calibration patient:
        s_i = |y_i - p_i|   (= 1 - p_i if y_i = 1, else p_i)
    A prediction is "conforming" for a patient when the true label would
    sit inside the set; smaller s = more conforming.
  - Finite-sample quantile:
        q_hat = ceil((n_cal + 1) * (1 - alpha)) / n_cal -quantile of {s_i}
    (the +1/ceil convention guarantees >= 1 - alpha coverage under
    exchangeability, not just asymptotically.)
  - Test sets: label d is in patient j's set iff
        |1 - p_jd| <= q_hat_d   (y=1 plausible)
        |0 - p_jd| <= q_hat_d   (y=0 plausible), i.e. p_jd <= q_hat_d
    Note q_hat_d from scores {1 - p_cal | y=1} gives the positive rule
    and {p_cal | y=0} gives the negative rule; we track two thresholds
    per disease, q_pos_d and q_neg_d, which is exactly the standard
    construction.

Coverage guarantee is MARGINAL (over the population), not conditional;
per-ancestry miscoverage is REPORTED, expected to deviate, and is not a
gate — pre-registered as descriptive.
"""
from __future__ import annotations

import numpy as np


def conformity_scores(y_true: np.ndarray, p: np.ndarray) -> np.ndarray:
    """|y - p| per patient."""
    return np.abs(np.asarray(y_true, dtype=float) - np.asarray(p, dtype=float))


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conformal quantile: the ceil((n+1)(1-alpha))/n empirical quantile.

    Returns 1.0 (everything in the set) when the index exceeds n — the
    correct behavior for tiny calibration samples."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def calibrate(
    y_cal: np.ndarray, p_cal: np.ndarray, alpha: float = 0.10,
) -> dict[str, float]:
    """Per-disease conformal thresholds.

    Returns {"q_pos": threshold for including label 1,
             "q_neg": threshold for including label 0}."""
    y = np.asarray(y_cal)
    p = np.asarray(p_cal, dtype=float)
    pos_mask = y == 1
    if pos_mask.sum() == 0 or (~pos_mask).sum() == 0:
        raise ValueError("calibration split needs both classes")
    return {
        "q_pos": finite_sample_quantile(1.0 - p[pos_mask], alpha),
        "q_neg": finite_sample_quantile(p[~pos_mask], alpha),
    }


def prediction_sets(
    p_test: np.ndarray, thresholds: dict[str, float],
) -> np.ndarray:
    """Boolean set matrix (n, 2)? No — (n, ) per disease as a 2-column
    (in_set[0], in_set[1]) layout for stacking across diseases."""
    p = np.asarray(p_test, dtype=float)
    in0 = p <= thresholds["q_neg"]
    in1 = (1.0 - p) <= thresholds["q_pos"]
    return np.column_stack([in0, in1])


def evaluate_coverage(
    y_test: np.ndarray, p_test: np.ndarray, y_cal: np.ndarray,
    p_cal: np.ndarray, alpha: float = 0.10,
) -> dict[str, float | int]:
    """Calibrate on (y_cal, p_cal), evaluate on (y_test, p_test).

    Returns coverage (fraction of test patients whose true label is in
    the set), mean set size, and counts."""
    thr = calibrate(y_cal, p_cal, alpha=alpha)
    sets = prediction_sets(p_test, thr)
    y = np.asarray(y_test)
    covered = np.where(y == 1, sets[:, 1], sets[:, 0])
    return {
        "n_test": int(len(y)),
        "coverage": round(float(covered.mean()), 4),
        "mean_set_size": round(float(sets.sum(axis=1).mean()), 4),
        "frac_singletons": round(float((sets.sum(axis=1) == 1).mean()), 4),
        "frac_empty": round(float((sets.sum(axis=1) == 0).mean()), 4),
        "q_pos": round(float(thr["q_pos"]), 4),
        "q_neg": round(float(thr["q_neg"]), 4),
    }


def stratified_coverage(
    y_test: np.ndarray, p_test: np.ndarray, groups: np.ndarray,
    y_cal: np.ndarray, p_cal: np.ndarray, alpha: float = 0.10,
) -> dict[str, dict]:
    """Descriptive per-group coverage (pre-registered: NOT a gate)."""
    out: dict[str, dict] = {}
    thr = calibrate(y_cal, p_cal, alpha=alpha)
    sets = prediction_sets(p_test, thr)
    y = np.asarray(y_test)
    covered = np.where(y == 1, sets[:, 1], sets[:, 0])
    for g in np.unique(groups):
        m = groups == g
        out[str(g)] = {
            "n": int(m.sum()),
            "coverage": round(float(covered[m].mean()), 4),
            "mean_set_size": round(float(sets[m].sum(axis=1).mean()), 4),
        }
    return out

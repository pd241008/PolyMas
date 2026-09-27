"""F-15: uncertainty / calibration metrics (NLL, ECE, Brier).

All functions take (y_true, p_pred) binary vectors per disease and return
plain floats; the F-15 runner aggregates across diseases and seeds.
"""
from __future__ import annotations

import numpy as np


def nll(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Mean binary negative log-likelihood (natural log)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Mean Brier score: mean (p - y)^2."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    return float(np.mean((p - y) ** 2))


def ece(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: sum over bins of (n_bin / n) *
    |mean(p_bin) - mean(y_bin)|. Confidence bins over p in [0, 1]."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def all_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    return {
        "nll": round(nll(y_true, p_pred), 4),
        "ece": round(ece(y_true, p_pred), 4),
        "brier": round(brier(y_true, p_pred), 4),
    }

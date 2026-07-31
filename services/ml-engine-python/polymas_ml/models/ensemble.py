"""Multi-label ensemble with binary relevance strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base_learners import BaseLearner, get_learner


class MultiLabelEnsemble:
    """
    Multi-label classifier using binary relevance with an ensemble of
    base learners (XGBoost, CatBoost, LightGBM) per disease label.

    Supports:
    - Weighted soft voting across base learners
    - Platt scaling for score normalization
    - Per-label model selection
    """

    DISEASE_LABELS = [
        "T1D",
        "T2D",
        "LADA",
        "GESTATIONAL_DM",
        "MONOGENIC_DIABETES",
    ]

    def __init__(
        self,
        learner_names: list[str] | None = None,
        learner_weights: list[float] | None = None,
        platt_scaling: bool = True,
    ) -> None:
        self._learner_names = learner_names or ["xgboost", "catboost", "lightgbm"]
        self._learner_weights = (
            learner_weights
            or [1.0 / len(self._learner_names)] * len(self._learner_names)
        )
        self._platt_scaling = platt_scaling
        self._models: dict[str, list[BaseLearner]] = {}  # label -> list of base learners
        self._platt_params: dict[str, tuple[float, float]] = {}  # label -> (a, b)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> dict[str, dict[str, float]]:
        """
        Fit one set of base learners per disease label.

        Args:
            X: Feature matrix (n_patients x n_features).
            y: Binary label matrix (n_patients x n_labels), columns = DISEASE_LABELS.

        Returns:
            Dictionary of {label: {learner_name: feature_importance_sum}}.
        """
        importances: dict[str, dict[str, float]] = {}

        for label in self.DISEASE_LABELS:
            if label not in y.columns:
                continue

            self._models[label] = []
            label_importances = {}

            y_binary = y[label].values

            for learner_name in self._learner_names:
                learner = get_learner(learner_name)
                learner.fit(X, pd.Series(y_binary))
                self._models[label].append(learner)
                label_importances[learner_name] = sum(learner.get_feature_importances().values())

            if self._platt_scaling:
                raw_scores = self._score_label(X, label)
                self._platt_params[label] = self._fit_platt(raw_scores, y_binary)

            importances[label] = label_importances

        return importances

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict soft probabilities for all disease labels.

        Returns:
            DataFrame (n_patients x n_labels) with calibrated probabilities.
        """
        results = {}
        for label in self.DISEASE_LABELS:
            if label not in self._models:
                results[label] = np.zeros(X.shape[0])
                continue

            scores = self._score_label(X, label)

            if self._platt_scaling and label in self._platt_params:
                a, b = self._platt_params[label]
                scores = 1.0 / (1.0 + np.exp(-(a * scores + b)))

            results[label] = scores

        return pd.DataFrame(results)

    def _score_label(self, X: pd.DataFrame, label: str) -> np.ndarray:
        """Weighted soft vote of base learners for a single label."""
        scores = np.zeros(X.shape[0])
        for learner, weight in zip(
            self._models[label], self._learner_weights, strict=True
        ):
            scores += weight * learner.predict_proba(X)
        return scores

    @staticmethod
    def _fit_platt(
        scores: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 100
    ) -> tuple[float, float]:
        """Fit Platt scaling parameters (a, b) via gradient descent."""
        a, b = 0.0, 0.0
        for _ in range(epochs):
            z = a * scores + b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_a = np.mean((p - y) * scores)
            grad_b = np.mean(p - y)
            a -= lr * grad_a
            b -= lr * grad_b
        return a, b

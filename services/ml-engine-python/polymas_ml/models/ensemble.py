"""Multi-label ensemble with binary relevance strategy."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .base_learners import BaseLearner, get_learner

logger = logging.getLogger(__name__)


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
        "RA",
        "SLE",
        "SJOGRENS",
        "AITD",
        "T1D",
        "VITILIGO",
        "MS",
    ]

    def __init__(
        self,
        learner_names: list[str] | None = None,
        learner_weights: list[float] | None = None,
        platt_scaling: bool = True,
        calibration_size: float = 0.2,
    ) -> None:
        self._learner_names = learner_names or ["xgboost", "catboost", "lightgbm"]
        self._learner_weights = (
            learner_weights
            or [1.0 / len(self._learner_names)] * len(self._learner_names)
        )
        self._platt_scaling = platt_scaling
        self._calibration_size = calibration_size
        self._models: dict[str, list[BaseLearner]] = {}  # label -> list of base learners
        self._platt_params: dict[str, LogisticRegression] = {}  # label -> fitted model
        self._calibration_indices: dict[str, np.ndarray] = {}  # label -> indices used for calibration

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> dict[str, dict[str, float]]:
        """
        Fit one set of base learners per disease label.

        Uses a held-out calibration split to fit Platt scaling, avoiding
        calibration leakage from training-set raw scores.

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

            if self._calibration_size > 0 and self._platt_scaling:
                X_train, X_cal, y_train, y_cal = train_test_split(
                    X, y_binary, test_size=self._calibration_size, stratify=y_binary, random_state=42
                )
                self._calibration_indices[label] = X_cal.index.values
            else:
                X_train, y_train = X, y_binary
                self._calibration_indices[label] = np.array([], dtype=int)

            for learner_name in self._learner_names:
                learner = get_learner(learner_name)
                learner.fit(X_train, pd.Series(y_train))
                self._models[label].append(learner)
                label_importances[learner_name] = sum(learner.get_feature_importances().values())

            if self._platt_scaling and self._calibration_size > 0:
                raw_scores = self._score_label(X_cal, label)
                self._platt_params[label] = self._fit_platt(raw_scores, y_cal)
                lr_model = self._platt_params[label]
                logger.debug(
                    "Platt scaling fitted for %s on %d held-out samples: A=%.4f, B=%.4f",
                    label,
                    len(y_cal),
                    lr_model.coef_[0][0],
                    lr_model.intercept_[0],
                )
            elif self._platt_scaling:
                raw_scores = self._score_label(X, label)
                self._platt_params[label] = self._fit_platt(raw_scores, y_binary)
                lr_model = self._platt_params[label]
                logger.debug(
                    "Platt scaling fitted for %s on training data (no split): A=%.4f, B=%.4f",
                    label,
                    lr_model.coef_[0][0],
                    lr_model.intercept_[0],
                )

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
                lr_model = self._platt_params[label]
                scores = lr_model.predict_proba(scores.reshape(-1, 1))[:, 1]

            results[label] = scores

        return pd.DataFrame(results)

    def predict_proba_with_diagnostics(self, X: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
        """
        Predict probabilities and return per-learner raw scores and final calibrated scores
        for diagnostic logging.
        """
        diagnostics: dict[str, dict[str, np.ndarray]] = {}
        for label in self.DISEASE_LABELS:
            if label not in self._models:
                diagnostics[label] = {
                    "raw": np.zeros(X.shape[0]),
                    "calibrated": np.zeros(X.shape[0]),
                    "learners": {name: np.zeros(X.shape[0]) for name in self._learner_names},
                }
                continue

            raw_scores = self._score_label(X, label)
            learner_scores = {
                name: learner.predict_proba(X)
                for name, learner in zip(self._learner_names, self._models[label])
            }

            if self._platt_scaling and label in self._platt_params:
                lr_model = self._platt_params[label]
                calibrated_scores = lr_model.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
            else:
                calibrated_scores = raw_scores

            diagnostics[label] = {
                "raw": raw_scores,
                "calibrated": calibrated_scores,
                "learners": learner_scores,
            }

        return diagnostics

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
    ) -> LogisticRegression:
        """Fit Platt scaling via scikit-learn LogisticRegression (no regularization)."""
        model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        model.fit(scores.reshape(-1, 1), y)
        return model

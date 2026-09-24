"""Multi-label ensemble with binary relevance strategy."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .base_learners import BaseLearner, _preflight_learner, get_learner

logger = logging.getLogger(__name__)


class _AffinePlatt:
    """Affine Platt fallback for tiny calibration sets.

    Standardizes raw scores to z-scores and maps them through a unit sigmoid:
    p = sigmoid((s - mu) / std). Preserves the raw-score dynamic range (no
    slope collapse) while keeping outputs in (0, 1). Exposes the same
    predict_proba / coef_ / intercept_ surface as sklearn LogisticRegression
    so the ensemble code needs no special-casing.
    """

    def __init__(self, mu: float, std: float) -> None:
        self._mu = mu
        self._std = std
        self.coef_ = np.array([[1.0 / std]])
        self.intercept_ = np.array([-mu / std])

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if s.ndim == 2:
            s = s[:, 0]
        z = (s - self._mu) / self._std
        p = 1.0 / (1.0 + np.exp(-z))
        return np.stack([1.0 - p, p], axis=1)


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
        # Pre-flight each learner once in a subprocess: learners whose native
        # backends cannot run here (segfault-prone xgboost in GPU-less
        # sandboxes) are dropped up-front instead of crashing the whole run.
        usable = [n for n in self._learner_names if _preflight_learner(n)]
        dropped = [n for n in self._learner_names if n not in usable]
        if dropped:
            logger.warning("Pre-flight dropped learners (crashed in probe): %s", dropped)
        if not usable:
            raise RuntimeError("No usable base learners — all failed pre-flight")
        self._learner_names = usable
        self._learner_weights = (
            learner_weights
            or [1.0 / len(self._learner_names)] * len(self._learner_names)
        )
        self._platt_scaling = platt_scaling
        self._calibration_size = calibration_size
        self._models: dict[str, list[BaseLearner]] = {}  # label -> list of base learners
        self._platt_params: dict[str, LogisticRegression | _AffinePlatt] = {}  # label -> fitted model
        self._calibration_indices: dict[str, np.ndarray] = {}  # label -> indices used for calibration

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        train_indices: np.ndarray | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Fit one set of base learners per disease label.

        Uses a held-out calibration split to fit Platt scaling, avoiding
        calibration leakage from training-set raw scores.

        Args:
            X: Feature matrix (n_patients x n_features).
            y: Binary label matrix (n_patients x n_labels), columns = DISEASE_LABELS.
            train_indices: Optional positional indices of the patients to fit
                on (an 80% train split). When provided, base learners and
                calibration are fit ONLY on these patients, leaving the rest
                untouched for held-out evaluation.

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
            if train_indices is not None:
                X_fit = X.iloc[train_indices]
                y_fit = y_binary[train_indices]
            else:
                X_fit, y_fit = X, y_binary

            if self._calibration_size > 0 and self._platt_scaling:
                X_train, X_cal, y_train, y_cal = train_test_split(
                    X_fit, y_fit, test_size=self._calibration_size, stratify=y_fit, random_state=42
                )
                self._calibration_indices[label] = X_cal.index.values
            else:
                X_train, y_train = X_fit, y_fit
                self._calibration_indices[label] = np.array([], dtype=int)

            for learner_name in self._learner_names:
                learner = get_learner(learner_name)
                try:
                    learner.fit(X_train, pd.Series(y_train))
                except Exception as e:
                    logger.warning("Learner %s failed to fit for %s (%s) — skipping", learner_name, label, e)
                    continue
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
            # Diagnostics use the learners actually fitted for this label —
            # self._learner_names may include learners dropped by pre-flight.
            learner_scores = {
                type(learner).__name__.replace("Learner", "").lower(): learner.predict_proba(X)
                for learner in self._models[label]
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
    ) -> LogisticRegression | _AffinePlatt:
        """Fit Platt scaling via scikit-learn LogisticRegression.

        With tiny calibration sets (< ~30 samples), an unregularized fit
        (C=1e10) can find a near-perfect separating line and become a step
        function, saturating predictions at exactly 0/1. Regularization
        strength is scaled to the sample size: C=1.0 for small n_cal, C=1e4
        (effectively unregularized) when the calibration set is large enough.

        A negative fitted slope means the calibration split ranks the classes
        inverted (possible on small or weak-signal splits); an inverted
        calibrator is worse than none, so the raw ensemble score is kept.
        """
        c_value = 10.0 if len(y) < 30 else 1e4
        model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=1000)
        model.fit(scores.reshape(-1, 1), y)
        slope = float(model.coef_[0][0])
        if slope < 0:
            logger.info(
                "Platt slope negative (A=%.4f) — inverted ranking on calibration split; keeping raw scores",
                slope,
            )
            return _AffinePlatt(mu=float(np.mean(scores)), std=max(float(np.std(scores)), 1e-9))
        # Small calibration sets cannot support a steep slope; if the fitted
        # sigmoid is flatter than the raw-score spread (regularization pulled
        # A toward 0), fall back to an affine standardization of the raw
        # scores through a unit sigmoid so the calibrated output keeps a
        # usable dynamic range instead of collapsing to the base rate.
        raw_std = float(np.std(scores))
        cal_std = float(np.std(model.predict_proba(scores.reshape(-1, 1))[:, 1]))
        if len(y) < 30 and raw_std > 1e-9 and cal_std < 0.05 * raw_std:
            mu = float(np.mean(scores))
            logger.info(
                "Platt slope collapsed (cal_std=%.4g << raw_std=%.4g) — using affine standardized fallback",
                cal_std,
                raw_std,
            )
            return _AffinePlatt(mu=mu, std=raw_std)
        return model

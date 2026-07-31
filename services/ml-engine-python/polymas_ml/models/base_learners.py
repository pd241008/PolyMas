"""Base learner registry and training pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseLearner(ABC):
    """Abstract base class for ensemble base learners."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def get_feature_importances(self) -> dict[str, float]: ...


class XGBoostLearner(BaseLearner):
    """XGBoost binary relevance base learner."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = params or {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05}
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        import xgboost as xgb

        self._model = xgb.XGBClassifier(**self._params)
        self._model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "Model not fitted"
        return self._model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> dict[str, float]:
        assert self._model is not None, "Model not fitted"
        return dict(zip(self._model.feature_names_in_, self._model.feature_importances_))


class CatBoostLearner(BaseLearner):
    """CatBoost binary relevance base learner with native categorical support."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = params or {"iterations": 500, "depth": 6, "learning_rate": 0.05, "verbose": 0}
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        from catboost import CatBoostClassifier

        self._model = CatBoostClassifier(**self._params)
        self._model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "Model not fitted"
        return self._model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> dict[str, float]:
        assert self._model is not None, "Model not fitted"
        return dict(zip(self._model.feature_names_, self._model.feature_importances_))


class LightGBMLearner(BaseLearner):
    """LightGBM binary relevance base learner."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = params or {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "verbose": -1}
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        import lightgbm as lgb

        self._model = lgb.LGBMClassifier(**self._params)
        self._model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "Model not fitted"
        return self._model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> dict[str, float]:
        assert self._model is not None, "Model not fitted"
        return dict(zip(self._model.feature_names_in_, self._model.feature_importances_))


LEARNER_REGISTRY: dict[str, type[BaseLearner]] = {
    "xgboost": XGBoostLearner,
    "catboost": CatBoostLearner,
    "lightgbm": LightGBMLearner,
}


def get_learner(name: str, params: dict[str, Any] | None = None) -> BaseLearner:
    if name not in LEARNER_REGISTRY:
        raise ValueError(f"Unknown learner: {name}. Available: {list(LEARNER_REGISTRY.keys())}")
    return LEARNER_REGISTRY[name](params)

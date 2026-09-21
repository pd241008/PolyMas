"""Base learner registry and training pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseLearner(ABC):
    """Abstract base class for ensemble base learners."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = params or {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def get_feature_importances(self) -> dict[str, float]: ...


def _preflight_learner(name: str) -> bool:
    """Probe whether a learner can actually train in this environment.

    Runs a minimal fit in a throwaway subprocess: some environments (e.g.
    sandboxes without GPU driver support) segfault inside xgboost's native
    data-ingestion code — a crash that a Python try/except in-process cannot
    catch. Returns True only if the subprocess exits cleanly.
    """
    import subprocess
    import sys

    import numpy as np
    import pandas as pd

    probe = (
        "import sys; sys.path.insert(0, %r)\n"
        "import numpy as np, pandas as pd\n"
        "from polymas_ml.models.base_learners import get_learner\n"
        "l = get_learner(%r)\n"
        "l.fit(pd.DataFrame(np.random.randn(8, 3)), pd.Series(np.random.randint(0, 2, 8)))\n"
        "print('ok')\n"
    ) % (
        str(__import__("pathlib").Path(__file__).parent.parent),
        name,
    )
    try:
        import os
        import pathlib

        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONPATH": str(pathlib.Path(__file__).parent.parent)},
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


class XGBoostLearner(BaseLearner):
    """XGBoost binary relevance base learner."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = params or {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05}
        self._model: Any = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        import xgboost as xgb

        self._model = xgb.XGBClassifier(**self._params)
        try:
            self._model.fit(X, y)
        except BaseException as e:
            # xgboost's native data-ingestion backend segfaults (not raises) in
            # some sandboxes, so a Python-level except cannot catch it; the
            # ensemble-level guard skips this learner if the process survives.
            raise RuntimeError(f"XGBoost fit failed: {e}") from e

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "Model not fitted"
        return self._model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> dict[str, float]:
        assert self._model is not None, "Model not fitted"
        return dict(
            zip(
                self._model.feature_names_in_,
                self._model.feature_importances_,
                strict=True,
            )
        )


class CatBoostLearner(BaseLearner):
    """CatBoost binary relevance base learner with native categorical support."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = (
            params or {"iterations": 500, "depth": 6, "learning_rate": 0.05, "verbose": 0}
        )
        self._model: Any = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        from catboost import CatBoostClassifier

        self._model = CatBoostClassifier(**self._params)
        self._model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "Model not fitted"
        return self._model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> dict[str, float]:
        assert self._model is not None, "Model not fitted"
        return dict(
            zip(
                self._model.feature_names_,
                self._model.feature_importances_,
                strict=True,
            )
        )


class LightGBMLearner(BaseLearner):
    """LightGBM binary relevance base learner."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = (
            params or {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "verbose": -1}
        )
        self._model: Any = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        import lightgbm as lgb

        params = dict(self._params)
        # Scale min_data_in_leaf down for small cohorts: the default (20)
        # prevents any meaningful splits when the training set is < ~100 rows,
        # collapsing LightGBM to near-constant base-rate predictions.
        if "min_child_samples" not in params:
            params["min_child_samples"] = max(1, min(20, len(X) // 10))
        self._model = lgb.LGBMClassifier(**params)
        self._model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "Model not fitted"
        return self._model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> dict[str, float]:
        assert self._model is not None, "Model not fitted"
        return dict(
            zip(
                self._model.feature_names_in_,
                self._model.feature_importances_,
                strict=True,
            )
        )


LEARNER_REGISTRY: dict[str, type[BaseLearner]] = {
    "xgboost": XGBoostLearner,
    "catboost": CatBoostLearner,
    "lightgbm": LightGBMLearner,
}


def get_learner(name: str, params: dict[str, Any] | None = None) -> BaseLearner:
    if name not in LEARNER_REGISTRY:
        raise ValueError(f"Unknown learner: {name}. Available: {list(LEARNER_REGISTRY.keys())}")
    return LEARNER_REGISTRY[name](params)

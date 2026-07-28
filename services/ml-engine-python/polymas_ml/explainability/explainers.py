"""SHAP and LIME explainability wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class TreeExplainerWrapper:
    """Exact SHAP values via TreeExplainer for tree-based base models."""

    def __init__(self, model: Any, feature_names: list[str] | None = None) -> None:
        self._model = model
        self._feature_names = feature_names or []

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        import shap

        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        return pd.DataFrame(shap_values, columns=self._feature_names or list(X.columns), index=X.index)

    def feature_importance(self, X: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        shap_df = self.explain(X)
        mean_abs = shap_df.abs().mean(axis=0).sort_values(ascending=False)
        return mean_abs.head(top_k).to_frame("mean_abs_shap")


class LIMEExplainerWrapper:
    """LIME explanations for ensemble output."""

    def __init__(self, predict_fn: Any, feature_names: list[str], num_samples: int = 1000) -> None:
        self._predict_fn = predict_fn
        self._feature_names = feature_names
        self._num_samples = num_samples

    def explain(self, X_row: pd.DataFrame, label_idx: int = 0) -> pd.DataFrame:
        from lime.lime_tabular import LimeTabularExplainer

        explainer = LimeTabularExplainer(
            training_data=np.zeros((1, len(self._feature_names))),
            feature_names=self._feature_names,
            mode="classification",
        )

        explanation = explainer.explain_instance(
            X_row.values[0],
            self._predict_fn,
            num_features=len(self._feature_names),
            top_labels=1,
        )

        as_list = explanation.as_list(label=label_idx)
        return pd.DataFrame(as_list, columns=["feature", "attribution"])

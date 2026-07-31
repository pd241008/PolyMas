"""Hierarchical clustering on risk-probability vectors with dendrogram generation."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, to_tree
from scipy.spatial.distance import pdist


class DiseaseRiskClusterer:
    """
    Clusters patients based on their multi-disease risk probability vectors.
    Generates dendrograms for comparison against the 1988 Type 1-4 classification.
    """

    def __init__(
        self,
        linkage_method: str = "ward",
        distance_metric: str = "euclidean",
        min_clusters: int = 3,
        max_clusters: int = 10,
    ) -> None:
        self._linkage_method = linkage_method
        self._distance_metric = distance_metric
        self._min_clusters = min_clusters
        self._max_clusters = max_clusters
        self._linkage_matrix: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def fit_predict(self, risk_matrix: pd.DataFrame, n_clusters: int | None = None) -> np.ndarray:
        """
        Fit hierarchical clustering on risk probability matrix.

        Args:
            risk_matrix: (n_patients x n_diseases) probability scores.
            n_clusters: Fixed number of clusters. If None, auto-select via gap heuristic.

        Returns:
            Cluster assignment array.
        """
        distances = pdist(risk_matrix.values, metric=self._distance_metric)
        self._linkage_matrix = linkage(distances, method=self._linkage_method)

        if n_clusters is None:
            n_clusters = self._select_k(risk_matrix.values)

        self._labels = fcluster(self._linkage_matrix, t=n_clusters, criterion="maxclust")
        return self._labels

    def dendrogram_json(self, patient_ids: list[str]) -> str:
        """Serialize dendrogram structure as JSON for frontend rendering."""
        if self._linkage_matrix is None:
            raise ValueError("Must call fit_predict before dendrogram_json")

        tree = to_tree(self._linkage_matrix)

        def _node_to_dict(node: Any) -> dict:
            if node.is_leaf():
                return {
                    "id": f"leaf_{node.id}",
                    "patient_id": patient_ids[node.id]
                    if node.id < len(patient_ids)
                    else str(node.id),
                    "distance": 0.0,
                }
            return {
                "id": f"node_{node.id}",
                "distance": node.dist,
                "count": node.count,
                "left": _node_to_dict(node.get_left()),
                "right": _node_to_dict(node.get_right()),
            }

        return json.dumps(_node_to_dict(tree), indent=2)

    def silhouette_score(self, risk_matrix: pd.DataFrame) -> float:
        """Compute silhouette score for the current clustering."""
        from sklearn.metrics import silhouette_score

        if self._labels is None:
            raise ValueError("Must call fit_predict before silhouette_score")

        return float(
            silhouette_score(risk_matrix.values, self._labels, metric=self._distance_metric)
        )

    def _select_k(self, X: np.ndarray) -> int:
        """Select number of clusters using a simple gap-like heuristic."""
        from sklearn.metrics import silhouette_score as _sil

        best_k = self._min_clusters
        best_score = -1.0

        for k in range(self._min_clusters, min(self._max_clusters + 1, X.shape[0])):
            labels = fcluster(self._linkage_matrix, t=k, criterion="maxclust")
            if len(set(labels)) < 2:
                continue
            score = _sil(X, labels, metric=self._distance_metric)
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

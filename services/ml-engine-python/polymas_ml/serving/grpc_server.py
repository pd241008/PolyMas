"""gRPC server for the ML Engine."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MLEngineServicer:
    """
    gRPC servicer wrapping the ML pipeline.

    TODO: Replace with auto-generated stubs from polymas/v1/services.proto.
    """

    def __init__(self) -> None:
        self._ensemble: Any = None  # Lazy init after model loading
        self._clusterer: Any = None

    def score_batch(self, profiles: list[dict]) -> list[dict]:
        """Score a batch of patient profiles across all disease labels."""
        from polymas_ml.models.ensemble import MultiLabelEnsemble

        if self._ensemble is None:
            self._ensemble = MultiLabelEnsemble()
            # TODO: Load trained weights from model registry

        feature_matrix = self._profiles_to_dataframe(profiles)
        probabilities = self._ensemble.predict_proba(feature_matrix)

        results = []
        for i, profile in enumerate(profiles):
            predictions = []
            for label in probabilities.columns:
                predictions.append({
                    "label": label,
                    "probability": float(probabilities.iloc[i][label]),
                    "confidence_lower": max(0.0, float(probabilities.iloc[i][label]) - 0.05),
                    "confidence_upper": min(1.0, float(probabilities.iloc[i][label]) + 0.05),
                    "model_version": "0.1.0",
                })
            results.append({
                "patient_id": profile.get("patient_id", ""),
                "predictions": predictions,
            })

        return results

    def explain_patient(
        self, profile: dict, method: str = "shap", target_disease: str = "T1D"
    ) -> dict:
        """Generate feature attributions for a single patient."""

        # TODO: Extract actual features and use the trained base learner
        return {
            "patient_id": profile.get("patient_id", ""),
            "target_disease": target_disease,
            "attributions": [],
            "explanation_json": "{}",
        }

    def cluster_predictions(
        self, scored_patients: list[dict], n_clusters: int | None = None
    ) -> dict:
        """Run hierarchical clustering on scored prediction vectors."""
        from polymas_ml.clustering.hierarchical import DiseaseRiskClusterer

        patient_ids = [sp["patient_id"] for sp in scored_patients]
        disease_labels = ["T1D", "T2D", "LADA", "GESTATIONAL_DM", "MONOGENIC_DIABETES"]

        risk_matrix = np.zeros((len(scored_patients), len(disease_labels)))
        for i, sp in enumerate(scored_patients):
            for pred in sp["predictions"]:
                if pred["label"] in disease_labels:
                    risk_matrix[i, disease_labels.index(pred["label"])] = pred["probability"]

        df = pd.DataFrame(risk_matrix, columns=disease_labels)

        self._clusterer = DiseaseRiskClusterer()
        labels = self._clusterer.fit_predict(df, n_clusters=n_clusters)

        clusters = []
        for cluster_id in sorted(set(labels)):
            members = [patient_ids[i] for i in range(len(labels)) if labels[i] == cluster_id]
            clusters.append({
                "cluster_id": f"cluster_{cluster_id}",
                "member_patient_ids": members,
                "distance_to_centroid": 0.0,
            })

        return {
            "clusters": clusters,
            "dendrogram_json": self._clusterer.dendrogram_json(patient_ids),
            "silhouette_score": self._clusterer.silhouette_score(df),
        }

    @staticmethod
    def _profiles_to_dataframe(profiles: list[dict]) -> pd.DataFrame:
        """Convert raw profile dicts to feature DataFrame."""
        # TODO: Implement full feature extraction pipeline
        rows = []
        for p in profiles:
            row = {"patient_id": p.get("patient_id", "")}
            for rs in p.get("risk_scores", []):
                row[f"rs_{rs['locus_id']}_score"] = rs.get("continuous_score", 0.0)
                row[f"rs_{rs['locus_id']}_zscore"] = rs.get("z_score", 0.0)
            rows.append(row)
        return pd.DataFrame(rows).set_index("patient_id")

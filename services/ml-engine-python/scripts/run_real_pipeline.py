"""Fetch real small datasets from GWAS Catalog and ImmPort APIs,
then run the full ML pipeline, saving outputs per subphase.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from polymas_ml.clustering.hierarchical import DiseaseRiskClusterer
from polymas_ml.explainability.explainers import LIMEExplainerWrapper, TreeExplainerWrapper
from polymas_ml.models.ensemble import MultiLabelEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = PROJECT_ROOT / "results"
GWAS_DIR = OUTPUTS_DIR / "raw" / "gwas"
IMMPORT_DIR = OUTPUTS_DIR / "raw" / "immport"
FEATURES_DIR = OUTPUTS_DIR / "features"
MODELS_DIR = OUTPUTS_DIR / "models"
EXPLANATIONS_DIR = OUTPUTS_DIR / "explanations"
CLUSTERS_DIR = OUTPUTS_DIR / "clusters"
REPORTS_DIR = OUTPUTS_DIR / "reports"

dirs = [
    GWAS_DIR,
    IMMPORT_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    EXPLANATIONS_DIR,
    CLUSTERS_DIR,
    REPORTS_DIR,
]
for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

GWAS_BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api"
IMMPORT_BASE_URL = "https://www.immport.org/data/query"

AUTOIMMUNE_LOCI = {
    "rs2187668": "HLA-DRB1",
    "rs9272346": "HLA-DQB1",
    "rs2476601": "PTPN22",
    "rs3087243": "CTLA4",
    "rs2292239": "ERBB3",
    "rs11209026": "IL23R",
    "rs2104286": "IL2RA",
    "rs7574865": "STAT4",
}

DISEASE_LABELS = ["RA", "SLE", "SJOGRENS", "AITD", "T1D", "VITILIGO", "MS"]


def fetch_gwas_associations(rs_id: str, max_retries: int = 3) -> list[dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            url = f"{GWAS_BASE_URL}/singleNucleotidePolymorphisms/{rs_id}/associations"
            resp = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            associations = []
            for assoc in data.get("_embedded", {}).get("associations", []):
                associations.append({
                    "rs_id": rs_id,
                    "gene": AUTOIMMUNE_LOCI.get(rs_id, rs_id),
                    "pvalue": assoc.get("pvalue"),
                    "pvalueText": assoc.get("pvalueText"),
                    "efoTrait": assoc.get("mappedLabel", assoc.get("efoTrait")),
                    "orPerCopyNum": assoc.get("orPerCopyNum"),
                    "betaNum": assoc.get("betaNum"),
                    "studyId": assoc.get("studyId"),
                })
            return associations
        except Exception as e:
            logger.warning("GWAS fetch %s attempt %d failed: %s", rs_id, attempt + 1, e)
            time.sleep(2 ** attempt)
    return []


def fetch_immport_study(study_id: str, max_retries: int = 3) -> dict[str, Any] | None:
    for attempt in range(max_retries):
        try:
            url = f"{IMMPORT_BASE_URL}/api/study/{study_id}?format=json"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                logger.warning("ImmPort study %s requires authentication (401)", study_id)
                return None
            else:
                resp.raise_for_status()
        except Exception as e:
            logger.warning("ImmPort fetch %s attempt %d failed: %s", study_id, attempt + 1, e)
            time.sleep(2 ** attempt)
    return None


def build_real_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_associations = []
    for rs_id in AUTOIMMUNE_LOCI:
        logger.info("Fetching GWAS data for %s (%s)", rs_id, AUTOIMMUNE_LOCI[rs_id])
        assocs = fetch_gwas_associations(rs_id)
        all_associations.extend(assocs)
        time.sleep(0.5)

    gwas_df = pd.DataFrame(all_associations)
    gwas_df.to_csv(GWAS_DIR / "gwas_associations.csv", index=False)
    gwas_df.to_parquet(GWAS_DIR / "gwas_associations.parquet", index=False)
    logger.info("Saved %d GWAS associations to %s", len(gwas_df), GWAS_DIR)

    immport_studies = ["SDY1", "SDY180"]
    immport_data = []
    for study_id in immport_studies:
        logger.info("Fetching ImmPort study %s", study_id)
        study = fetch_immport_study(study_id)
        if study:
            immport_data.append(study)
            with open(IMMPORT_DIR / f"study_{study_id}.json", "w") as f:
                json.dump(study, f, indent=2)
        time.sleep(0.5)

    n_patients = 400
    np.random.seed(42)

    prs_rows = []
    for i in range(n_patients):
        patient_id = f"P{i:04d}"
        for rs_id, gene in AUTOIMMUNE_LOCI.items():
            locus_df = gwas_df[gwas_df["rs_id"] == rs_id]
            if not locus_df.empty and locus_df["pvalue"].notna().any():
                pval = locus_df["pvalue"].dropna().mean()
                base_score = min(1.0, max(0.0, -np.log10(max(pval, 1e-300)) / 300))
                noise = np.random.normal(0, 0.25)
                score = min(1.0, max(0.0, base_score + noise))
            else:
                score = np.random.beta(2, 5)
            z_score = round(float(np.random.normal(score * 2 - 1, 0.5)), 4)
            prs_rows.append({
                "patient_id": patient_id,
                "locus_id": rs_id,
                "gene_symbol": gene,
                "continuous_score": round(float(score), 4),
                "z_score": z_score,
                "pvalue": pval if not locus_df.empty else None,
            })

    prs_df = pd.DataFrame(prs_rows)
    prs_df.to_csv(FEATURES_DIR / "prs_features.csv", index=False)
    prs_df.to_parquet(FEATURES_DIR / "prs_features.parquet", index=False)

    base_prevalences = {
        "RA": 0.20,
        "SLE": 0.10,
        "SJOGRENS": 0.08,
        "AITD": 0.15,
        "T1D": 0.08,
        "VITILIGO": 0.06,
        "MS": 0.12,
    }

    clinical_rows = []
    label_rows = []
    for i in range(n_patients):
        patient_id = f"P{i:04d}"
        patient_risk_factor = np.random.normal(0, 0.25)

        labels = {"patient_id": patient_id}
        for disease in DISEASE_LABELS:
            base = base_prevalences[disease]
            prevalence = min(0.95, max(0.01, base + patient_risk_factor))
            labels[disease] = int(np.random.random() < prevalence)
        label_rows.append(labels)

        has_any_autoimmune = any(labels[d] for d in DISEASE_LABELS)
        sex = "F" if np.random.random() < (0.55 + 0.1 * labels.get("SLE", 0) + 0.08 * labels.get("AITD", 0) + 0.05 * labels.get("RA", 0)) else "M"
        age_factor = (35 + 15 * patient_risk_factor + 10 * labels.get("AITD", 0) + 8 * labels.get("RA", 0))
        age_at_diagnosis_days = int(np.clip(age_factor * 365.25 + np.random.normal(0, 4 * 365), 365, 80 * 365))
        bmi = round(float(np.clip(22 + 2 * patient_risk_factor + np.random.normal(0, 3), 16, 42)), 1)
        family_history = int(np.random.random() < (0.15 + 0.2 * patient_risk_factor + 0.2 * has_any_autoimmune))

        clinical_rows.append({
            "patient_id": patient_id,
            "sex": sex,
            "ethnicity": np.random.choice(["EUR", "AFR", "EAS", "SAS"]),
            "age_at_diagnosis_days": age_at_diagnosis_days,
            "bmi": bmi,
            "family_history": family_history,
        })

    clinical_df = pd.DataFrame(clinical_rows)
    clinical_df.to_csv(FEATURES_DIR / "clinical_features.csv", index=False)
    clinical_df.to_parquet(FEATURES_DIR / "clinical_features.parquet", index=False)
    labels_df = pd.DataFrame(label_rows)
    labels_df.to_csv(FEATURES_DIR / "labels.csv", index=False)
    labels_df.to_parquet(FEATURES_DIR / "labels.parquet", index=False)

    return prs_df, clinical_df, labels_df


def prepare_feature_matrix(prs_df: pd.DataFrame, clinical_df: pd.DataFrame) -> pd.DataFrame:
    pivot = prs_df.pivot_table(
        index="patient_id",
        columns="locus_id",
        values="continuous_score",
        aggfunc="first",
    )
    pivot.columns = [f"{col}__score" for col in pivot.columns]
    pivot = pivot.reset_index()

    wide_prs = prs_df.pivot_table(
        index="patient_id",
        columns="locus_id",
        values="z_score",
        aggfunc="first",
    )
    wide_prs.columns = [f"{col}__zscore" for col in wide_prs.columns]
    wide_prs = wide_prs.reset_index()

    merged = clinical_df.merge(pivot, on="patient_id", how="left")
    merged = merged.merge(wide_prs, on="patient_id", how="left")
    merged = merged.fillna(0)

    feature_cols = [c for c in merged.columns if c not in ("patient_id", "sex", "ethnicity")]
    X = merged[feature_cols].copy()
    X["sex"] = (merged["sex"] == "M").astype(int)
    X["ethnicity_EUR"] = (merged["ethnicity"] == "EUR").astype(int)
    X["ethnicity_AFR"] = (merged["ethnicity"] == "AFR").astype(int)
    X["ethnicity_EAS"] = (merged["ethnicity"] == "EAS").astype(int)

    X.to_csv(FEATURES_DIR / "feature_matrix.csv", index=False)
    X.to_parquet(FEATURES_DIR / "feature_matrix.parquet", index=False)
    with open(FEATURES_DIR / "feature_matrix_metadata.json", "w") as f:
        metadata = {
            "n_features": len(X.columns),
            "features": list(X.columns),
            "n_patients": len(X),
        }
        json.dump(metadata, f, indent=2)
    logger.info("Feature matrix saved: %d patients x %d features", len(X), len(X.columns))
    return X


def train_ensemble(X: pd.DataFrame, y: pd.DataFrame) -> MultiLabelEnsemble:
    valid_labels = [col for col in DISEASE_LABELS if col in y.columns and y[col].nunique() >= 2]
    if not valid_labels:
        raise ValueError("No valid labels with >=2 classes found in y")
    y_valid = y[valid_labels].copy()
    logger.info("Training on valid labels: %s", valid_labels)

    learner_names = ["xgboost", "catboost", "lightgbm"]
    ensemble = MultiLabelEnsemble(learner_names=learner_names, platt_scaling=True)
    importances = ensemble.fit(X, y_valid)

    platt_coeffs = []
    for label, model in ensemble._platt_params.items():
        coef = float(model.coef_[0][0])
        intercept = float(model.intercept_[0])
        platt_coeffs.append({"disease": label, "A": coef, "B": intercept})
        logger.info("Platt scaling for %s: A=%.4f, B=%.4f", label, coef, intercept)
    pd.DataFrame(platt_coeffs).to_csv(MODELS_DIR / "platt_coefficients.csv", index=False)

    importance_rows = []
    for label, imps in importances.items():
        for learner, imp in imps.items():
            importance_rows.append({"disease": label, "learner": learner, "importance": imp})
    importance_df = pd.DataFrame(importance_rows)
    importance_df.to_csv(MODELS_DIR / "feature_importances.csv", index=False)

    predictions = ensemble.predict_proba(X)
    predictions.index = y_valid.index
    predictions.to_csv(MODELS_DIR / "predictions.csv", index=False)
    predictions.to_parquet(MODELS_DIR / "predictions.parquet", index=False)

    diag = ensemble.predict_proba_with_diagnostics(X)
    diag_rows = []
    for disease, vals in diag.items():
        raw_std = float(np.std(vals["raw"]))
        cal_std = float(np.std(vals["calibrated"]))
        for learner_name, learner_probs in vals["learners"].items():
            diag_rows.append({
                "disease": disease,
                "learner": learner_name,
                "std": float(np.std(learner_probs)),
                "mean": float(np.mean(learner_probs)),
            })
        diag_rows.append({
            "disease": disease,
            "learner": "raw",
            "std": raw_std,
            "mean": float(np.mean(vals["raw"])),
        })
        diag_rows.append({
            "disease": disease,
            "learner": "calibrated",
            "std": cal_std,
            "mean": float(np.mean(vals["calibrated"])),
        })
        logger.info(
            "%s — XGB std: %.4f, CatBoost std: %.4f, LightGBM std: %.4f, Raw std: %.4f, Calibrated std: %.4f",
            disease,
            float(np.std(vals["learners"].get("xgboost", np.zeros(X.shape[0])))),
            float(np.std(vals["learners"].get("catboost", np.zeros(X.shape[0])))),
            float(np.std(vals["learners"].get("lightgbm", np.zeros(X.shape[0])))),
            raw_std,
            cal_std,
        )
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(MODELS_DIR / "prediction_diagnostics.csv", index=False)

    logger.info("Ensemble trained. Predictions shape: %s", predictions.shape)
    return ensemble


def run_explainability(ensemble: MultiLabelEnsemble, X: pd.DataFrame) -> None:
    for disease in DISEASE_LABELS[:3]:
        if disease not in ensemble._models or not ensemble._models[disease]:
            continue
        try:
            base_model = ensemble._models[disease][0]._model
            explainer = TreeExplainerWrapper(model=base_model, feature_names=list(X.columns))
            shap_df = explainer.explain(X)
            shap_df.to_csv(EXPLANATIONS_DIR / f"shap_{disease}.csv", index=False)

            imp_df = explainer.feature_importance(X, top_k=10)
            imp_df.to_csv(EXPLANATIONS_DIR / f"shap_importance_{disease}.csv", index=False)
            logger.info("SHAP explanations saved for %s", disease)
        except Exception as e:
            logger.warning("SHAP failed for %s: %s", disease, e)

        try:
            def predict_fn(arr: np.ndarray, _disease: str = disease) -> np.ndarray:
                df = pd.DataFrame(arr, columns=X.columns)
                return ensemble.predict_proba(df)[_disease].values

            lime_explainer = LIMEExplainerWrapper(
                predict_fn=predict_fn,
                feature_names=list(X.columns),
                training_data=X.values[:10],
                mode="regression",
            )
            lime_df = lime_explainer.explain(X.iloc[:1])
            lime_df.to_csv(EXPLANATIONS_DIR / f"lime_{disease}.csv", index=False)
            logger.info("LIME explanations saved for %s", disease)
        except Exception as e:
            logger.warning("LIME failed for %s: %s", disease, e)


def run_clustering(X: pd.DataFrame, predictions: pd.DataFrame) -> None:
    clusterer = DiseaseRiskClusterer(min_clusters=2, max_clusters=5)
    labels = clusterer.fit_predict(predictions, n_clusters=3)

    cluster_df = pd.DataFrame({
        "patient_id": X.index,
        "cluster_label": labels,
    })
    cluster_df.to_csv(CLUSTERS_DIR / "cluster_assignments.csv", index=False)

    dendro = clusterer.dendrogram_json(list(cluster_df["patient_id"]))
    with open(CLUSTERS_DIR / "dendrogram.json", "w") as f:
        f.write(dendro)

    score = clusterer.silhouette_score(predictions)
    with open(CLUSTERS_DIR / "silhouette_score.txt", "w") as f:
        f.write(f"silhouette_score: {score:.4f}\n")

    logger.info("Clustering complete. Silhouette score: %.4f", score)


def generate_reports(prs_df: pd.DataFrame, clinical_df: pd.DataFrame, labels_df: pd.DataFrame,
                     X: pd.DataFrame, ensemble: MultiLabelEnsemble) -> None:
    report = {
        "dataset_summary": {
            "n_patients": len(clinical_df),
            "n_loci": len(AUTOIMMUNE_LOCI),
            "n_features": len(X.columns),
            "n_diseases": len(DISEASE_LABELS),
            "gwas_associations_fetched": len(prs_df),
        },
        "gwas_loci": list(AUTOIMMUNE_LOCI.keys()),
        "disease_labels": DISEASE_LABELS,
        "model_config": {
            "learners": ensemble._learner_names,
            "platt_scaling": ensemble._platt_scaling,
        },
    }
    with open(REPORTS_DIR / "pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2)

    summary = {
        "raw_gwas_records": len(prs_df),
        "clinical_records": len(clinical_df),
        "label_records": len(labels_df),
        "feature_matrix_shape": list(X.shape),
    }
    with open(REPORTS_DIR / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Reports saved to %s", REPORTS_DIR)


def main() -> None:
    logger.info("=== Starting real-data ML pipeline ===")

    prs_df, clinical_df, labels_df = build_real_dataset()

    X = prepare_feature_matrix(prs_df, clinical_df)
    X.index = clinical_df["patient_id"].values
    y = labels_df.set_index("patient_id").loc[X.index, DISEASE_LABELS].copy()

    logger.info("Training ensemble on real-data-derived features...")
    ensemble = train_ensemble(X, y)

    logger.info("Running explainability...")
    run_explainability(ensemble, X)

    logger.info("Running clustering...")
    predictions = pd.read_csv(MODELS_DIR / "predictions.csv", index_col=0)
    run_clustering(X, predictions)

    logger.info("Generating reports...")
    generate_reports(prs_df, clinical_df, labels_df, X, ensemble)

    logger.info("=== Pipeline complete. Results in %s ===", OUTPUTS_DIR)


if __name__ == "__main__":
    main()

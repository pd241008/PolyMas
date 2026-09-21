"""Fetch real small datasets from GWAS Catalog and ImmPort APIs,
then run the full ML pipeline, saving outputs per subphase.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "services" / "ml-engine-python"))

from polymas_ml.clustering.hierarchical import DiseaseRiskClusterer
from polymas_ml.data import (
    MODELED_DISEASES,
    NEGATIVE_SEARCH_RESULTS,
    assign_subjects,
    build_subject_pool,
    draw_patient_groups,
)
from polymas_ml.data.patients import DISEASE_LABELS, simulate_genotypes_prs, simulate_labels
from polymas_ml.explainability.explainers import LIMEExplainerWrapper, TreeExplainerWrapper
from polymas_ml.models.ensemble import MultiLabelEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

OUTPUTS_DIR = PROJECT_ROOT / "results"

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

# DISEASE_LABELS now imported from polymas_ml.data.patients (shared with System B)


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
    api_key = os.environ.get("IMMPORT_API_KEY", "")
    if not api_key:
        logger.warning("IMMPORT_API_KEY not set — ImmPort requests will be rejected (401)")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for attempt in range(max_retries):
        try:
            url = f"{IMMPORT_BASE_URL}/api/study/{study_id}?format=json"
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                logger.warning("ImmPort study %s requires authentication (401) — check IMMPORT_API_KEY", study_id)
                return None
            else:
                resp.raise_for_status()
        except Exception as e:
            logger.warning("ImmPort fetch %s attempt %d failed: %s", study_id, attempt + 1, e)
            time.sleep(2 ** attempt)
    return None


def build_real_dataset(
    n_patients: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
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

    # ---- REAL ImmPort subject demographics (subject-level API) ----
    subject_pool = build_subject_pool(cache_dir=IMMPORT_DIR)
    subject_pool.to_csv(IMMPORT_DIR / "subject_pool.csv", index=False)

    rng = np.random.default_rng(42)
    patient_groups = draw_patient_groups(subject_pool, n_patients, rng)
    assignments = assign_subjects(subject_pool, n_patients, patient_groups, rng)

    # ---- Shared patient simulation (genotypes -> PRS -> labels) ----
    prs_df, gen_rows = simulate_genotypes_prs(n_patients, AUTOIMMUNE_LOCI, patient_groups, rng)
    label_rows = simulate_labels(
        n_patients, patient_groups, [a["sex"] for a in assignments], gen_rows, AUTOIMMUNE_LOCI, rng
    )
    prs_df.to_csv(FEATURES_DIR / "prs_features.csv", index=False)
    prs_df.to_parquet(FEATURES_DIR / "prs_features.parquet", index=False)

    # ---- Clinical features: REAL demographics + modeled BMI/family_history ----
    clinical_rows = []
    for i, a in enumerate(assignments):
        pid = f"P{i:04d}"
        age_years = a["age_years"]
        try:
            age_days = int(float(age_years) * 365.25)
        except (TypeError, ValueError):
            age_days = int(rng.normal(40, 12) * 365.25)
        sex = a["sex"] if a["sex"] in ("F", "M") else ("F" if rng.random() < 0.55 else "M")
        clinical_rows.append({
            "patient_id": pid,
            "sex": sex,
            "ethnicity": a["ancestry"],
            "age_at_diagnosis_days": age_days,
            "bmi": round(float(np.clip(rng.normal(22 + 2 * rng.normal(0, 0.25), 3), 16, 42)), 1),
            "family_history": int(rng.random() < (0.15 + 0.25 * (a["assigned_group"] is not None))),
            "subject_accession": a["subject_accession"],
            "study_accession": a["study_accession"],
            "cohort": a["assigned_group"] if a["assigned_group"] else "BACKGROUND",
            "clinical_source": "immport_real",
            "bmi_source": "modeled",
            "family_history_source": "modeled",
        })

    clinical_df = pd.DataFrame(clinical_rows)
    clinical_df.to_csv(FEATURES_DIR / "clinical_features.csv", index=False)
    clinical_df.to_parquet(FEATURES_DIR / "clinical_features.parquet", index=False)

    labels_df = pd.DataFrame(label_rows)
    labels_df.to_csv(FEATURES_DIR / "labels.csv", index=False)
    labels_df.to_parquet(FEATURES_DIR / "labels.parquet", index=False)

    # ---- Provenance manifest ----
    from collections import Counter
    # Reuse disclosure per cohort: patients drawn from that cohort / unique
    # real subjects used for it. 1.0 = pure 1:1 mapping, >1 = within-cohort
    # reuse (each real subject backs that many simulated patients on average).
    cohort_patient_counts = Counter(a["assigned_group"] if a["assigned_group"] else "BACKGROUND" for a in assignments)
    cohort_unique_subjects: dict[str, set] = {}
    for a in assignments:
        key = a["assigned_group"] if a["assigned_group"] else "BACKGROUND"
        cohort_unique_subjects.setdefault(key, set())
        cohort_unique_subjects[key].add(a["subject_accession"])
    reuse_ratios = {
        k: round(cohort_patient_counts.get(k, 0) / max(len(v), 1), 3)
        for k, v in cohort_unique_subjects.items()
    }
    provenance = {
        "n_patients": n_patients,
        "n_unique_subjects": clinical_df["subject_accession"].nunique(),
        "subject_pool_size": len(subject_pool),
        "n_gwas_associations": len(gwas_df),
        "reuse_policy": "cohort-internal reuse after pool exhaustion; no cross-cohort borrowing",
        "cohort_counts": dict(Counter(clinical_df["cohort"])),
        "cohort_unique_subjects": {k: len(v) for k, v in cohort_unique_subjects.items()},
        "reuse_ratio": reuse_ratios,
        "study_counts": dict(Counter(clinical_df["study_accession"])),
        "modeled_diseases": MODELED_DISEASES,
        "real_cohort_diseases": [d for d in DISEASE_LABELS if d not in MODELED_DISEASES],
        "aitd_vitiligo_search": NEGATIVE_SEARCH_RESULTS,
        "field_sources": {
            "sex": "real (ImmPort demographic.gender)",
            "age": "real (ImmPort demographic.max_subject_age_in_years)",
            "ancestry": "real (ImmPort demographic.race, mapped)",
            "hispanic": "real (ImmPort demographic.ethnicity)",
            "bmi": "modeled",
            "family_history": "modeled",
            "genotypes": "simulated (shared with System B)",
            "labels": "simulated (cohort-informed)",
        },
    }
    with open(REPORTS_DIR / "data_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info(
        "Patient build: %d patients on %d unique ImmPort subjects; cohorts: %s",
        n_patients, provenance["n_unique_subjects"], provenance["cohort_counts"],
    )

    return prs_df, clinical_df, labels_df, len(gwas_df)


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

    # Provenance columns are metadata, not features — keep numeric feature
    # columns only (sex/ethnicity are re-added below as one-hots).
    provenance_cols = {
        "subject_accession", "study_accession", "cohort",
        "clinical_source", "bmi_source", "family_history_source",
    }
    feature_cols = [
        c for c in merged.columns
        if c not in ("patient_id", "sex", "ethnicity")
        and c not in provenance_cols
        and pd.api.types.is_numeric_dtype(merged[c])
    ]
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


def make_train_test_split(y: pd.DataFrame, test_size: float = 0.2, random_state: int = 123) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 80/20 patient split on a composite label.

    Stratifies on 'has any autoimmune disease' so both arms see positives
    for every disease at realistic rates. The same split is used for
    ensemble training (80%) and held-out metric evaluation (20%).
    """
    from sklearn.model_selection import train_test_split

    composite = (y[DISEASE_LABELS].sum(axis=1) > 0).astype(int)
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=test_size, stratify=composite, random_state=random_state
    )
    return idx_train, idx_test


def run_metrics(
    ensemble: MultiLabelEnsemble,
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_indices: np.ndarray,
    tag: str | None = None,
) -> pd.DataFrame:
    """Held-out discrimination metrics (AUROC/AUPRC/F1) per disease.

    Scores ONLY on the held-out 20% test split (the ensemble was fit on the
    complementary 80%). Labels for the modeled diseases (AITD, VITILIGO)
    are simulated without a real cohort, and their metrics carry a 'modeled'
    flag in the output. With tag set (e.g. 'genotype_only'), outputs are
    suffixed so ablation runs never overwrite the full-feature tables.
    """
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    suffix = f"_{tag}" if tag else ""
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    preds = ensemble.predict_proba(X_test)
    preds.index = X_test.index
    preds.to_csv(MODELS_DIR / f"test_predictions{suffix}.csv", index_label="patient_id")

    rows = []
    for disease in DISEASE_LABELS:
        if disease not in preds.columns or disease not in y_test.columns:
            continue
        y_true = y_test[disease].values
        p = preds[disease].values
        pred_bin = (p >= 0.5).astype(int)
        auroc = float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) >= 2 else float("nan")
        auprc = float(average_precision_score(y_true, p)) if y_true.sum() > 0 else float("nan")
        f1 = float(f1_score(y_true, pred_bin, zero_division=0))
        # Best-F1 threshold (swept on test — exploratory upper bound; the
        # calibrated output of an imbalanced problem often never crosses 0.5).
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, 91):
            f1_t = f1_score(y_true, (p >= t).astype(int), zero_division=0)
            if f1_t > best_f1:
                best_f1, best_t = float(f1_t), float(t)
        rows.append({
            "disease": disease,
            "n_test": int(len(y_true)),
            "n_positives": int(y_true.sum()),
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "f1": round(f1, 4),
            "f1_best": round(best_f1, 4),
            "best_threshold": round(best_t, 2),
            "modeled_label": disease in MODELED_DISEASES,
        })
        logger.info(
            "METRICS %s: AUROC=%.4f AUPRC=%.4f F1@0.5=%.4f F1@best(t=%.2f)=%.4f (n_pos=%d)",
            disease, auroc, auprc, f1, best_t, best_f1, int(y_true.sum()),
        )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(MODELS_DIR / f"per_disease_metrics{suffix}.csv", index=False)
    return metrics_df


def train_ensemble(
    X: pd.DataFrame, y: pd.DataFrame, train_indices: np.ndarray | None = None
) -> MultiLabelEnsemble:
    valid_labels = [col for col in DISEASE_LABELS if col in y.columns and y[col].nunique() >= 2]
    if not valid_labels:
        raise ValueError("No valid labels with >=2 classes found in y")
    y_valid = y[valid_labels].copy()
    logger.info("Training on valid labels: %s", valid_labels)

    learner_names = ["xgboost", "catboost", "lightgbm"]
    ensemble = MultiLabelEnsemble(learner_names=learner_names, platt_scaling=True)
    importances = ensemble.fit(X, y_valid, train_indices=train_indices)

    platt_coeffs = []
    cal_split_info = []
    for label, model in ensemble._platt_params.items():
        coef = float(model.coef_[0][0])
        intercept = float(model.intercept_[0])
        platt_coeffs.append({"disease": label, "A": coef, "B": intercept})
        n_cal = len(ensemble._calibration_indices.get(label, []))
        cal_split_info.append({"disease": label, "calibration_samples": n_cal})
        logger.info("Platt scaling for %s (n_cal=%d): A=%.4f, B=%.4f", label, n_cal, coef, intercept)
    pd.DataFrame(platt_coeffs).to_csv(MODELS_DIR / "platt_coefficients.csv", index=False)
    pd.DataFrame(cal_split_info).to_csv(MODELS_DIR / "calibration_split_info.csv", index=False)

    importance_rows = []
    for label, imps in importances.items():
        for learner, imp in imps.items():
            importance_rows.append({"disease": label, "learner": learner, "importance": imp})
    importance_df = pd.DataFrame(importance_rows)
    importance_df.to_csv(MODELS_DIR / "feature_importances.csv", index=False)

    predictions = ensemble.predict_proba(X)
    predictions.index = y_valid.index
    predictions.to_csv(MODELS_DIR / "predictions.csv", index_label="patient_id")
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


def build_cluster_disease_table(predictions: pd.DataFrame, cluster_assignments: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster mean calibrated probability per disease (3x7 table).

    This is the table that lets the Discussion compare data-driven clusters
    with Humbert-Dupond Type 1-3: each row is a cluster, each column a
    disease, entries are mean calibrated probability of that disease within
    the cluster. Also saves per-cluster sizes and the dominant-disease label.
    """
    merged = cluster_assignments.merge(
        predictions.reset_index().rename(columns={"index": "patient_id"}),
        on="patient_id",
        how="inner",
    )
    disease_cols = [c for c in predictions.columns if c in DISEASE_LABELS]
    profile = merged.groupby("cluster_label")[disease_cols].mean().round(4)
    sizes = merged.groupby("cluster_label").size().rename("n_patients")
    profile = profile.join(sizes)
    profile["dominant_disease"] = profile[disease_cols].idxmax(axis=1)
    profile["dominant_prob"] = profile[disease_cols].max(axis=1).round(4)
    profile.to_csv(CLUSTERS_DIR / "cluster_disease_profile.csv", index_label="cluster_label")
    logger.info("Cluster x disease profile:\n%s", profile)
    return profile


def generate_reports(prs_df: pd.DataFrame, clinical_df: pd.DataFrame, labels_df: pd.DataFrame,
                     X: pd.DataFrame, ensemble: MultiLabelEnsemble, n_gwas_associations: int) -> None:
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
        "raw_gwas_associations": n_gwas_associations,
        "prs_rows": len(prs_df),
        "clinical_records": len(clinical_df),
        "label_records": len(labels_df),
        "feature_matrix_shape": list(X.shape),
        "immport_unique_subjects": int(clinical_df["subject_accession"].nunique()) if "subject_accession" in clinical_df.columns else 0,
        "clinical_source": "immport_real_demographics + modeled bmi/family_history" if "subject_accession" in clinical_df.columns else "synthetic",
        "real_cohort_diseases": [d for d in DISEASE_LABELS if d not in MODELED_DISEASES],
        "modeled_diseases": MODELED_DISEASES,
    }
    with open(REPORTS_DIR / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Reports saved to %s", REPORTS_DIR)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PolyMas real-data ML pipeline (System A)")
    parser.add_argument("--n-patients", type=int, default=400, help="Number of synthetic patients to generate")
    args = parser.parse_args()
    n_patients = args.n_patients

    logger.info("=== Starting real-data ML pipeline (n_patients=%d) ===", n_patients)

    prs_df, clinical_df, labels_df, n_gwas = build_real_dataset(n_patients)

    n_gwas = n_gwas  # noqa: F841 — used in generate_reports

    X = prepare_feature_matrix(prs_df, clinical_df)
    X.index = clinical_df["patient_id"].values
    y = labels_df.set_index("patient_id").loc[X.index, DISEASE_LABELS].copy()

    idx_train, idx_test = make_train_test_split(y)
    logger.info("Patient split: %d train / %d held-out test (composite-stratified)", len(idx_train), len(idx_test))

    logger.info("Training ensemble on real-data-derived features (train split only)...")
    ensemble = train_ensemble(X, y, train_indices=idx_train)

    logger.info("Computing held-out discrimination metrics (AUROC/AUPRC/F1)...")
    run_metrics(ensemble, X, y, idx_test)

    logger.info("Running explainability...")
    run_explainability(ensemble, X)

    logger.info("Running clustering...")
    predictions = pd.read_csv(MODELS_DIR / "predictions.csv", index_col=0)
    run_clustering(X, predictions)

    cluster_assignments = pd.read_csv(CLUSTERS_DIR / "cluster_assignments.csv")
    build_cluster_disease_table(predictions, cluster_assignments)

    logger.info("Generating reports...")
    generate_reports(prs_df, clinical_df, labels_df, X, ensemble, n_gwas)

    logger.info("=== Pipeline complete. Results in %s ===", OUTPUTS_DIR)


if __name__ == "__main__":
    main()

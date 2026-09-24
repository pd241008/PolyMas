"""Post-pipeline statistical evaluation (run after run_real_pipeline.py).

Produces, from the saved outputs in results/:

1. Patient-level percentile bootstrap CIs for held-out AUROC/AUPRC
   (per_disease_metrics gives point estimates; this gives the intervals).
2. Permutation test for the clustering silhouette score (within-disease
   column shuffling null; preserves marginals, destroys patient-level
   co-occurrence).
3. Ancestry-stratified held-out metrics (EUR/AFR/EAS) + stratified
   bootstrap CIs on the pooled test split.
4. Label co-occurrence structure report (MAS diagnostics of the generated
   cohort: overdispersion, polyautoimmunity rates, pairwise phi).

Outputs land in results/stats/ and results/reports/.

Run with:
  PYTHONPATH=services/ml-engine-python services/ml-engine-python/.venv/bin/python \
      scripts/run_stats_eval.py --n-boot 2000 --n-permutations 500
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
CLUSTERS_DIR = RESULTS_DIR / "clusters"
FEATURES_DIR = RESULTS_DIR / "features"
STATS_DIR = RESULTS_DIR / "stats"
REPORTS_DIR = RESULTS_DIR / "reports"

DISEASE_LABELS = ["RA", "SLE", "SJOGRENS", "AITD", "T1D", "VITILIGO", "MS"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_stats_eval")


def load_pipeline_outputs():
    """Load the saved pipeline artifacts needed for the statistical eval."""
    preds_all = pd.read_csv(MODELS_DIR / "predictions.csv", index_col=0)
    clinical = pd.read_csv(FEATURES_DIR / "clinical_features.csv").set_index("patient_id")
    labels = pd.read_csv(FEATURES_DIR / "labels.csv").set_index("patient_id")
    assignments = pd.read_csv(CLUSTERS_DIR / "cluster_assignments.csv")

    # Held-out split = patients with saved test predictions (written by
    # run_metrics on the composite-stratified 20% split).
    test_preds_path = MODELS_DIR / "test_predictions.csv"
    if test_preds_path.exists():
        test_ids = pd.read_csv(test_preds_path, index_col=0).index.astype(str)
    else:
        logger.warning("test_predictions.csv missing - falling back to all patients")
        test_ids = preds_all.index.astype(str)

    preds_all.index = preds_all.index.astype(str)
    clinical.index = clinical.index.astype(str)
    labels.index = labels.index.astype(str)
    assignments["patient_id"] = assignments["patient_id"].astype(str)

    return preds_all, clinical, labels, assignments, test_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="PolyMas post-pipeline statistical evaluation")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    from polymas_ml.data.patients import label_structure_report
    from polymas_ml.evaluation.ancestry import (
        ancestry_bootstrap_intervals,
        ancestry_stratified_metrics,
    )
    from polymas_ml.evaluation.stats import (
        pairwise_phi_pvalues,
        per_disease_bootstrap_table,
        silhouette_permutation_test,
    )

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    preds_all, clinical, labels, assignments, test_ids = load_pipeline_outputs()

    y_test = labels.loc[test_ids, [d for d in DISEASE_LABELS if d in labels.columns]]
    preds_test = preds_all.loc[test_ids]
    ancestry_test = clinical.loc[test_ids, "ethnicity"]
    diseases = [d for d in DISEASE_LABELS if d in preds_test.columns]
    logger.info("Held-out split: %d patients; diseases: %s", len(test_ids), diseases)

    # ---- 1. Bootstrap CIs for AUROC / AUPRC --------------------------------
    for metric in ("auroc", "auprc"):
        table = per_disease_bootstrap_table(
            y_test, preds_test, diseases, metric=metric, n_boot=args.n_boot, seed=args.seed
        )
        out = STATS_DIR / f"bootstrap_ci_{metric}.csv"
        table.to_csv(out, index=False)
        logger.info("Bootstrap %s CIs -> %s", metric.upper(), out)
        print(table.to_string(index=False))

    # ---- 2. Silhouette permutation test ------------------------------------
    clusterer_labels = (
        assignments.set_index("patient_id").loc[preds_all.index, "cluster_label"].to_numpy()
    )
    perm = silhouette_permutation_test(
        preds_all.to_numpy(dtype=float),
        clusterer_labels,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    (STATS_DIR / "silhouette_permutation_test.json").write_text(json.dumps(perm, indent=2))
    logger.info("Silhouette permutation test: %s", perm)

    # ---- 3. Ancestry-stratified results ------------------------------------
    strat = ancestry_stratified_metrics(y_test, preds_test, ancestry_test, diseases)
    strat.to_csv(STATS_DIR / "ancestry_stratified_metrics.csv", index=False)
    logger.info("Ancestry-stratified metrics -> %s", STATS_DIR / "ancestry_stratified_metrics.csv")
    print(strat.to_string(index=False))

    strat_ci = ancestry_bootstrap_intervals(
        y_test, preds_test, ancestry_test, diseases, n_boot=args.n_boot, seed=args.seed
    )
    strat_ci.to_csv(STATS_DIR / "bootstrap_ci_ancestry_stratified.csv", index=False)

    # ---- 4. Co-occurrence structure of the generated cohort ----------------
    labels_all = labels.loc[preds_all.index]
    structure = label_structure_report(labels_all)
    phi_table = pairwise_phi_pvalues(labels_all, diseases)
    phi_table.to_csv(STATS_DIR / "pairwise_phi_pvalues.csv", index=False)
    cooc = {
        "label_structure": structure,
        "phi_table_rows": int(len(phi_table)),
        "n_phi_below_0.05_bonferroni": (
            int((phi_table["p_bonferroni"] < 0.05).sum()) if not phi_table.empty else 0
        ),
    }
    (STATS_DIR / "cooccurrence_structure.json").write_text(json.dumps(cooc, indent=2))
    logger.info("Co-occurrence structure -> %s", STATS_DIR / "cooccurrence_structure.json")
    print(json.dumps(structure, indent=2)[:1200])

    # ---- Summary manifest ---------------------------------------------------
    summary = {
        "n_boot": args.n_boot,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "n_test": int(len(test_ids)),
        "silhouette_permutation": perm,
        "mas_rate_3plus": structure.get("rate_3plus_mas"),
        "overdispersion_ratio": structure.get("overdispersion_ratio"),
        "outputs": [
            "bootstrap_ci_auroc.csv",
            "bootstrap_ci_auprc.csv",
            "silhouette_permutation_test.json",
            "ancestry_stratified_metrics.csv",
            "bootstrap_ci_ancestry_stratified.csv",
            "pairwise_phi_pvalues.csv",
            "cooccurrence_structure.json",
        ],
    }
    (REPORTS_DIR / "stats_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Stats summary -> %s", REPORTS_DIR / "stats_summary.json")


if __name__ == "__main__":
    main()

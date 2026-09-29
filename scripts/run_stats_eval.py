"""Post-pipeline statistical evaluation (run after run_real_pipeline.py).

Produces, from the saved outputs in results/system_a_run_current/:

1. Patient-level percentile bootstrap CIs for held-out AUROC/AUPRC
   (per_disease_metrics gives point estimates; this gives the intervals).
2. Permutation test for the clustering silhouette score (within-disease
   column shuffling null; preserves marginals, destroys patient-level
   co-occurrence).
3. Ancestry-stratified held-out metrics (EUR/AFR/EAS) + stratified
   bootstrap CIs on the pooled test split.
4. Label co-occurrence structure report (MAS diagnostics of the generated
   cohort: overdispersion, polyautoimmunity rates, pairwise phi).
5. Split-conformal coverage tables (F-11b wiring, 2026-09-28): per-disease
   empirical coverage at the nominal level + descriptive ancestry
   stratification. Calibration uses a seeded half of the HELD-OUT block
   (never the in-sample predictions.csv); the F-11 pre-registered verdict
   itself stands from the OOF-calibration run (f11_conformal.py).

Outputs land in results/system_a_run_current/stats/ and results/system_a_run_current/reports/.

Run with:
  PYTHONPATH=services/ml-engine-python services/ml-engine-python/.venv/bin/python \
      scripts/run_stats_eval.py --n-boot 2000 --n-permutations 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "services" / "ml-engine-python"))
# Results root; override with POLYMAS_RESULTS_DIR to read/write a specific run folder.
RESULTS_DIR = Path(os.environ.get("POLYMAS_RESULTS_DIR", PROJECT_ROOT / "results" / "system_a_run_current"))
MODELS_DIR = RESULTS_DIR / "models"
CLUSTERS_DIR = RESULTS_DIR / "clusters"
FEATURES_DIR = RESULTS_DIR / "features"
STATS_DIR = RESULTS_DIR / "stats"
REPORTS_DIR = RESULTS_DIR / "reports"

DISEASE_LABELS = ["RA", "SLE", "SJOGRENS", "AITD", "T1D", "VITILIGO", "MS"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_stats_eval")


def load_pipeline_outputs():
    """Load the saved pipeline artifacts needed for the statistical eval.

    TWO prediction matrices exist in models/ and they are NOT the same
    thing (reconciliation note, 2026-09-26):
      - test_predictions.csv: HELD-OUT probabilities from the ensemble fit
        on the train split (the numbers per_disease_metrics.csv reports).
        All held-out inference (bootstrap CIs, ancestry stratification)
        MUST use this.
      - predictions.csv: IN-SAMPLE full-cohort probabilities from a
        separate full-data refit, used for clustering/SHAP. Its rows are
        optimistic for the fitting patients and must never be mixed into
        held-out statistics.
    """
    preds_all = pd.read_csv(MODELS_DIR / "predictions.csv", index_col=0)
    clinical = pd.read_csv(FEATURES_DIR / "clinical_features.csv").set_index("patient_id")
    labels = pd.read_csv(FEATURES_DIR / "labels.csv").set_index("patient_id")
    assignments = pd.read_csv(CLUSTERS_DIR / "cluster_assignments.csv")

    # Held-out split = patients with saved test predictions (written by
    # run_metrics on the composite-stratified 20% split).
    test_preds_path = MODELS_DIR / "test_predictions.csv"
    test_preds = None
    if test_preds_path.exists():
        test_preds = pd.read_csv(test_preds_path, index_col=0)
        test_preds.index = test_preds.index.astype(str)
        test_ids = test_preds.index
    else:
        logger.warning("test_predictions.csv missing - falling back to all patients")
        test_ids = preds_all.index.astype(str)

    preds_all.index = preds_all.index.astype(str)
    clinical.index = clinical.index.astype(str)
    labels.index = labels.index.astype(str)
    assignments["patient_id"] = assignments["patient_id"].astype(str)

    return preds_all, test_preds, clinical, labels, assignments, test_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="PolyMas post-pipeline statistical evaluation")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--only-perm", action="store_true",
                        help="run ONLY the silhouette permutation test (it needs ~1.1 s/perm "
                             "at n=5000; split into its own call so it can finish)")
    parser.add_argument("--skip-perm", action="store_true",
                        help="reuse the existing silhouette_permutation_test.json")
    parser.add_argument("--conformal-alpha", type=float, default=0.10,
                        help="nominal miscoverage level for the conformal section (default 0.10)")
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
    preds_all, preds_test, clinical, labels, assignments, test_ids = load_pipeline_outputs()

    y_test = labels.loc[test_ids, [d for d in DISEASE_LABELS if d in labels.columns]]
    if preds_test is None:
        preds_test = preds_all.loc[test_ids]
    preds_test = preds_test.loc[test_ids]
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
    perm_path = STATS_DIR / "silhouette_permutation_test.json"
    if args.skip_perm and perm_path.exists():
        perm = json.loads(perm_path.read_text())
        logger.info("Silhouette permutation test: reused %s", perm_path)
    else:
        clusterer_labels = (
            assignments.set_index("patient_id").loc[preds_all.index, "cluster_label"].to_numpy()
        )
        perm = silhouette_permutation_test(
            preds_all.to_numpy(dtype=float),
            clusterer_labels,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )
        perm_path.write_text(json.dumps(perm, indent=2))
        logger.info("Silhouette permutation test: %s", perm)
    if args.only_perm:
        return

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

    # ---- 5. Split-conformal coverage (F-11b wiring) ------------------------
    # Honest calibration without an OOF matrix in the standard artifacts:
    # split the HELD-OUT block into seeded halves, calibrate on one,
    # evaluate on the other. The in-sample predictions.csv is never used
    # (reconciliation note above). This is the split-halves variant of
    # split conformal; the F-11 pre-registered verdict stands from the
    # OOF-calibration run (f11_conformal.py) and is NOT re-adjudicated here.
    from polymas_ml.evaluation.conformal import evaluate_coverage, stratified_coverage

    alpha = args.conformal_alpha
    rng_c = np.random.default_rng(args.seed)
    perm_c = rng_c.permutation(len(test_ids))
    half = len(perm_c) // 2
    cal_ids = [test_ids[i] for i in perm_c[:half]]
    eval_ids = [test_ids[i] for i in perm_c[half:]]
    logger.info("Conformal: %d calibration / %d evaluation patients (alpha=%.2f)",
                len(cal_ids), len(eval_ids), alpha)

    coverage: dict[str, dict] = {}
    strat_cov: dict[str, dict] = {}
    ancestry_eval = clinical.loc[eval_ids, "ethnicity"].to_numpy()
    for d in diseases:
        coverage[d] = evaluate_coverage(
            y_test.loc[eval_ids, d].to_numpy(),
            preds_test.loc[eval_ids, d].to_numpy(),
            y_test.loc[cal_ids, d].to_numpy(),
            preds_test.loc[cal_ids, d].to_numpy(),
            alpha=alpha,
        )
        strat_cov[d] = stratified_coverage(
            y_test.loc[eval_ids, d].to_numpy(),
            preds_test.loc[eval_ids, d].to_numpy(),
            ancestry_eval,
            y_test.loc[cal_ids, d].to_numpy(),
            preds_test.loc[cal_ids, d].to_numpy(),
            alpha=alpha,
        )
    conf = {
        "method": "split conformal (held-out block split in seeded halves; calibration/evaluation)",
        "alpha": alpha,
        "n_cal": len(cal_ids),
        "n_eval": len(eval_ids),
        "seed": args.seed,
        "per_disease": coverage,
        "ancestry_stratified_descriptive": strat_cov,
        "gate_reference": {
            "note": "F-11 verdict stands from the pre-registered OOF run; this automated section is not a re-adjudication",
            "tolerance": "coverage >= alpha - 0.02 per disease",
            "n_pass_at_tolerance": int(sum(c["coverage"] >= alpha - 0.02 for c in coverage.values())),
            "n_diseases": len(diseases),
        },
    }
    (STATS_DIR / "conformal_coverage.json").write_text(json.dumps(conf, indent=2))
    logger.info("Conformal coverage -> %s", STATS_DIR / "conformal_coverage.json")
    print(pd.DataFrame(coverage).T[["coverage", "mean_set_size", "frac_empty"]].to_string())

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
            "conformal_coverage.json",
        ],
    }
    (REPORTS_DIR / "stats_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Stats summary -> %s", REPORTS_DIR / "stats_summary.json")


if __name__ == "__main__":
    main()

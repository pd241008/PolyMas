"""F-11: split-conformal prediction sets on the canonical run (ROADMAP Phase 4).

Pre-registered claim (ROADMAP): split-conformal sets achieve >= 88%
empirical coverage at the nominal 90% level on held-out data, per
disease (tolerance: coverage >= 0.90 - 0.02). Miscoverage reported per
disease AND per ancestry (descriptive, pre-registered as NOT a gate —
the guarantee is marginal, not group-conditional).

Calibration-honesty protocol (pre-registered):
  - Conformal thresholds must come from probabilities the model did NOT
    fit on. Two honest sources exist on this run:
      (a) test_predictions.csv rows (held-out, Platt-calibrated) — using
          them as BOTH calibration and evaluation splits the test set
          50/50: calibrate on half, evaluate on the other half (seeded
          deterministic split). Fully honest; halves the eval n.
      (b) predictions.csv val rows — IN-SAMPLE for the base learners
          (documented in the 2026-09-26 reconciliation); REJECTED as a
          calibration source. Listed here so the choice is visible.
  - We use (a): 500 calibrate / 500 evaluate per disease, seeded
    (numpy Generator 123, split on sorted patient ids — deterministic
    and reproducible). Eval-side coverage is the pre-registered metric;
    the calibration half is never touched for evaluation.
  - Platt-calibrated probs are the system's actual outputs; conformal
    wraps whatever the base system produces (marginal coverage holds
    under exchangeability regardless of the model's quality).

Evidence: <results_root>/f11_conformal/
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "ml-engine-python"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("f11_conformal")

from polymas_ml.data.patients import DISEASE_LABELS  # noqa: E402
from polymas_ml.evaluation.conformal import evaluate_coverage, stratified_coverage  # noqa: E402

RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "results/results_final_20260926"))
OUT = RESULTS / "f11_conformal"
ALPHA = 0.10
GATE = 0.88
SEED = 123


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    test_preds = pd.read_csv(RESULTS / "models" / "test_predictions.csv", index_col=0)
    test_preds.index = test_preds.index.astype(str)
    labels = pd.read_csv(RESULTS / "features" / "labels.csv").set_index("patient_id")
    clinical = pd.read_csv(RESULTS / "features" / "clinical_features.csv").set_index("patient_id")
    test_ids = sorted(test_preds.index)
    # Deterministic 50/50 calibration/evaluation split of the test ids.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(test_ids))
    cal_ids = [test_ids[i] for i in perm[: len(perm) // 2]]
    eval_ids = [test_ids[i] for i in perm[len(perm) // 2 :]]
    ancestry = clinical.loc[eval_ids, "ethnicity"].to_numpy()
    logger.info("calibration n=%d, evaluation n=%d (seed %d)", len(cal_ids), len(eval_ids), SEED)

    rows, strat_rows = [], []
    for d in DISEASE_LABELS:
        y = labels.loc[test_ids, d]
        p = test_preds[d]
        res = evaluate_coverage(
            y.loc[cal_ids].to_numpy(), p.loc[cal_ids].to_numpy(),
            y.loc[eval_ids].to_numpy(), p.loc[eval_ids].to_numpy(),
            alpha=ALPHA,
        )
        rows.append({"disease": d, **res})
        strat = stratified_coverage(
            y.loc[eval_ids].to_numpy(), p.loc[eval_ids].to_numpy(), ancestry,
            y.loc[cal_ids].to_numpy(), p.loc[cal_ids].to_numpy(), alpha=ALPHA,
        )
        for g, v in strat.items():
            strat_rows.append({"disease": d, "ancestry": g, **v})
        logger.info("%s: coverage=%.4f (gate %.2f) mean_set=%.3f q_pos=%.3f q_neg=%.3f",
                    d, res["coverage"], GATE, res["mean_set_size"], res["q_pos"], res["q_neg"])

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "f11_coverage_by_disease.csv", index=False)
    sdf = pd.DataFrame(strat_rows)
    sdf.to_csv(OUT / "f11_coverage_by_ancestry.csv", index=False)

    n_pass = int((df["coverage"] >= GATE).sum())
    summary = {
        "alpha": ALPHA,
        "nominal_coverage": 1 - ALPHA,
        "gate": GATE,
        "calibration_source": "test_predictions.csv, deterministic seeded 50/50 "
                              "calibration/evaluation split of the held-out test set "
                              "(predictions.csv REJECTED as in-sample; see protocol header)",
        "n_cal": len(cal_ids), "n_eval": len(eval_ids),
        "gates": {"all_diseases_ge_0.88": {"pass": bool(n_pass == len(df)), "n_pass": n_pass}},
        "verdict": "PASS" if n_pass == len(df) else "FAIL",
        "note": "coverage guarantee is marginal; ancestry miscoverage is "
                "descriptive only (pre-registered)",
    }
    (OUT / "f11_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("F-11 verdict: %s\n%s", summary["verdict"], df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

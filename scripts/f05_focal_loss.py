"""F-05: focal / cost-sensitive loss ablation (ROADMAP Phase 3).

Pre-registered claim (ROADMAP F-05): class-imbalance-aware loss improves
AUPRC for low-prevalence diseases without degrading high-prevalence ones.
Null bound (pre-registered): AUPRC improvements < 0.005 count as null;
no disease may lose > 0.01 AUROC.

Protocol (pre-registered before the run):
  - Data: the ADR-006-corrected canonical run, its exact F-12 split.
  - Design: SINGLE-FAMILY paired ablation — LightGBM only (500 trees,
    depth 6, lr 0.05, seed 42), so the ONLY differing variable is the
    loss. (Running the full heterogeneous ensemble would confound the
    loss effect with learner composition.)
      control  : binary logloss (LightGBM default)
      focal    : focal loss, alpha=0.25, gamma=2.0 (paper defaults,
                 pre-registered — NOT tuned)
      cost-sens: logloss with scale_pos_weight = n_neg/n_pos (train)
  - Metrics on the F-12 test split: AUROC and AUPRC per disease.
  - Arms: all three; gates evaluated on the focal arm (the claim's
    subject), with the cost-sensitive arm reported alongside as
    context (the classic remedy focal competes with).
  - Low-prevalence set: the 3 lowest test-split prevalence diseases
    (deterministic rule, computed from the test labels — no picking).
  - Gates (focal arm):
      G1 AUPRC gain: mean AUPRC delta vs control over the 3
         lowest-prevalence diseases > +0.005 (the null bound)
      G2 AUROC floor: min per-disease AUROC delta vs control >= -0.01
  - Custom-objective contract: LightGBM returns RAW MARGINS from
    predict_proba under a custom objective; the runner applies the
    sigmoid link (pinned by tests/test_focal.py).

Evidence: <results_root>/f05_focal_loss/
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
logger = logging.getLogger("f05_focal")

from polymas_ml.data.patients import DISEASE_LABELS  # noqa: E402
from polymas_ml.models.focal import cost_sensitive_objective, focal_objective  # noqa: E402

RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "results/results_final_20260926"))
OUT = RESULTS / "f05_focal_loss"
SEED = 42
NULL_BOUND = 0.005
AUROC_FLOOR = -0.01
LGB_PARAMS = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "verbose": -1}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def main() -> int:
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    OUT.mkdir(parents=True, exist_ok=True)
    probe = "--probe" in sys.argv

    X = pd.read_csv(RESULTS / "features" / "feature_matrix.csv")
    y = pd.read_csv(RESULTS / "features" / "labels.csv")
    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    pos = {pid: i for i, pid in enumerate(y["patient_id"])}
    splits = {s: np.array([pos[p] for p in split.loc[split["split"] == s, "patient_id"]])
              for s in ("train", "val", "test")}
    tr, te = splits["train"], splits["test"]
    if probe:
        tr, te = tr[:600], te[:400]

    Yte = y.loc[te, DISEASE_LABELS]
    prev = Yte.mean()
    low_prev = list(prev.sort_values(ascending=False).index[-3:])  # 3 lowest
    logger.info("test prevalences: %s", {d: round(float(v), 4) for d, v in prev.items()})
    logger.info("low-prevalence set (3 lowest): %s", low_prev)

    rows = []
    for arm in ("control", "focal", "costsens"):
        for d in DISEASE_LABELS:
            ytr = y.loc[tr, d].to_numpy()
            spw = float((1 - ytr).sum() / max(ytr.sum(), 1))
            model = lgb.LGBMClassifier(random_state=SEED, **LGB_PARAMS)
            if arm == "focal":
                model = lgb.LGBMClassifier(
                    random_state=SEED, objective=focal_objective(0.25, 2.0), **LGB_PARAMS)
            elif arm == "costsens":
                model = lgb.LGBMClassifier(
                    random_state=SEED, objective=cost_sensitive_objective(spw), **LGB_PARAMS)
            model.fit(X.iloc[tr], ytr)
            raw = np.asarray(model.predict_proba(X.iloc[te])).squeeze()
            p = _sigmoid(raw) if arm != "control" else (
                raw if raw.ndim == 1 else raw[:, 1])
            rows.append({
                "arm": arm, "disease": d,
                "auroc": round(float(roc_auc_score(Yte[d], p)), 4),
                "auprc": round(float(average_precision_score(Yte[d], p)), 4),
                "test_prevalence": round(float(prev[d]), 4),
            })
            logger.info("%s %s: AUROC=%.4f AUPRC=%.4f", arm, d, rows[-1]["auroc"], rows[-1]["auprc"])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "f05_comparison.csv", index=False)

    pivot_auc = df.pivot(index="disease", columns="arm", values="auroc")
    pivot_pr = df.pivot(index="disease", columns="arm", values="auprc")
    d_focal = {
        "auprc_delta": (pivot_pr["focal"] - pivot_pr["control"]).round(4),
        "auroc_delta": (pivot_auc["focal"] - pivot_auc["control"]).round(4),
    }
    d_cost = {
        "auprc_delta": (pivot_pr["costsens"] - pivot_pr["control"]).round(4),
        "auroc_delta": (pivot_auc["costsens"] - pivot_auc["control"]).round(4),
    }
    g1 = float(d_focal["auprc_delta"].loc[low_prev].mean())
    g2 = float(d_focal["auroc_delta"].min())
    g1_cost = float(d_cost["auprc_delta"].loc[low_prev].mean())
    g2_cost = float(d_cost["auroc_delta"].min())

    summary = {
        "protocol": "single-family LightGBM paired ablation; only the loss differs; "
                    "focal alpha=0.25 gamma=2 (paper defaults, not tuned)",
        "low_prevalence_set": low_prev,
        "null_bound_auprc": NULL_BOUND,
        "auroc_floor": AUROC_FLOOR,
        "focal_arm": {
            "mean_auprc_delta_low_prev": round(g1, 4),
            "min_auroc_delta": round(g2, 4),
            "gates": {"G1_auprc_gain_pass": bool(g1 > NULL_BOUND),
                      "G2_auroc_floor_pass": bool(g2 >= AUROC_FLOOR)},
        },
        "costsens_arm_context": {
            "mean_auprc_delta_low_prev": round(g1_cost, 4),
            "min_auroc_delta": round(g2_cost, 4),
        },
        "per_disease": {
            d: {"focal_auprc_delta": float(d_focal["auprc_delta"].loc[d]),
                "focal_auroc_delta": float(d_focal["auroc_delta"].loc[d]),
                "costsens_auprc_delta": float(d_cost["auprc_delta"].loc[d])}
            for d in DISEASE_LABELS
        },
    }
    verdict = "PASS" if all(summary["focal_arm"]["gates"].values()) else "FAIL"
    summary["verdict"] = verdict
    (OUT / "f05_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("F-05 verdict: %s\n%s", verdict, json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

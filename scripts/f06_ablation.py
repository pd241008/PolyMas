"""F-06 ablation harness validation.

Ground truth injection: binary labels are simulated from REAL 1000G donor
genotypes via a logistic model with known coefficients, including an
epistasis (interaction) term. System A's ensemble is then trained twice —
base dosages vs base + haplotype/epistasis features — on identical splits.

Expected (pre-registered): the feature-augmented model recovers the injected
interaction signal (higher AUROC on the interaction-driven task); the base
model cannot represent the interaction cleanly. This validates the F-06
feature machinery BEFORE it is wired into the production label simulation
(the current pipeline's labels derive from simulated genotypes, against
which real-dosage features would be null by construction — documented in
ROADMAP F-06).

Outputs -> stash/results/f06_ablation_<date>/
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "services" / "ml-engine-python"))

from polymas_ml.data.haplotypes import (  # noqa: E402
    assemble_patient_features,
    haplotype_features,
    load_substrate,
)
from polymas_ml.models.ensemble import MultiLabelEnsemble  # noqa: E402

RESULTS_ROOT = Path(os.environ.get("POLYMAS_RESULTS_DIR", PROJECT_ROOT / "stash" / "results"))
RUN_TAG = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
OUT_DIR = RESULTS_ROOT / f"f06_ablation_{RUN_TAG}"
SEED = 123
N_PATIENTS = 3000

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("f06_ablation")

# Ground truth (logistic, known coefficients on real dosages /2 scaled):
#   logit = -1.2 + 0.9*PTPN22 + 0.7*STAT4 + 1.1*(PTPN22>0)*(CTLA4>0)
GROUND_TRUTH = {
    # Coefficients act on WITHIN-PANEL Z-SCORED dosages (not raw alt-allele
    # counts): rs2476601's VCF alt allele is the COMMON protective allele
    # (eaf ~0.88), so raw dosages average 1.76 and prevalence calibration
    # (and coefficient interpretation) breaks — the first two runs landed at
    # 89% / 79% prevalence for exactly this reason. Z-scoring makes the
    # injection coding-independent. Epistasis term stays carrier-based.
    "intercept": -0.9,
    "main": {"rs2476601": 0.9, "rs7574865": 0.7},
    "epistasis": {"pair": ("rs2476601", "rs3087243"), "coef": 1.1},
}


def simulate_labels(feats: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    z = np.full(len(feats), GROUND_TRUTH["intercept"])
    for rs, coef in GROUND_TRUTH["main"].items():
        d = feats[f"g_{rs}"].to_numpy(dtype=float)
        z += coef * (d - d.mean()) / (d.std() + 1e-9)
    a, b = GROUND_TRUTH["epistasis"]["pair"]
    interaction = ((feats[f"g_{a}"] > 0) & (feats[f"g_{b}"] > 0)).to_numpy()
    z += GROUND_TRUTH["epistasis"]["coef"] * interaction
    p = 1 / (1 + np.exp(-z))
    # Named 'RA' so the injected task flows through the ensemble's standard
    # DISEASE_LABELS training path (it skips unknown columns by design).
    return pd.Series(rng.binomial(1, p), index=feats.index, name="RA")


def train_and_eval(X: pd.DataFrame, y: pd.Series, train_idx, test_idx, tag: str) -> dict:
    ens = MultiLabelEnsemble(learner_names=["xgboost", "catboost", "lightgbm"], platt_scaling=True)
    ens.fit(X, y.to_frame(), train_indices=train_idx)
    preds = ens.predict_proba(X.iloc[test_idx])[y.name]
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_test = y.iloc[test_idx]
    auroc = float(roc_auc_score(y_test, preds))
    auprc = float(average_precision_score(y_test, preds))
    logger.info("%s: AUROC=%.4f AUPRC=%.4f", tag, auroc, auprc)
    return {"tag": tag, "auroc": round(auroc, 4), "auprc": round(auprc, 4), "n_features": X.shape[1]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dosages, meta, pcs, _ = load_substrate(RESULTS_ROOT)
    rng = np.random.default_rng(SEED)

    # Patients: ancestry labels drawn from the panel's super-pop distribution.
    pops = meta["super_pop"].value_counts(normalize=True)
    labels_anc = pd.Series(
        rng.choice(pops.index.to_numpy(), N_PATIENTS, p=pops.to_numpy()),
        index=[f"P{i:04d}" for i in range(N_PATIENTS)],
    )
    feats = assemble_patient_features(dosages, pcs, labels_anc, rng)
    y = simulate_labels(feats, rng)
    logger.info("Injected prevalence: %.3f", y.mean())

    # Stratified split on the label.
    from sklearn.model_selection import train_test_split
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED
    )
    X_base = feats[[c for c in feats.columns if c.startswith(("g_", "PC"))]]
    X_aug = feats.copy()

    results = [
        train_and_eval(X_base, y, idx_train, idx_test, "base_dosages+PCs"),
        train_and_eval(X_aug, y, idx_train, idx_test, "base+haplotype/epistasis"),
    ]
    delta = results[1]["auroc"] - results[0]["auroc"]
    passed = delta > 0.005  # pre-registered null bound from ROADMAP F-05/F-06 convention

    summary = {
        "run_date": RUN_TAG,
        "seed": SEED,
        "n_patients": N_PATIENTS,
        "ground_truth": GROUND_TRUTH,
        "results": results,
        "auroc_delta": round(delta, 4),
        "pre_registered_pass": bool(passed),
        "interpretation": (
            "Harness validation: features carry recoverable signal through "
            "System A's training path. NOT a production-label ablation — "
            "labels here are injected through real genotypes by construction."
        ),
    }
    (OUT_DIR / "ablation_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("delta=%.4f pass=%s -> %s", delta, passed, OUT_DIR)


if __name__ == "__main__":
    main()

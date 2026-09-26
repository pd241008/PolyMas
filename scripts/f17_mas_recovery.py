"""F-17: MAS-recovery ablation (ROADMAP Phase 2).

Pre-registered claim (ROADMAP F-17): trained models recover the embedded MAS
pairwise odds direction; per-pair sign agreement between (a) the phi
correlation of MODEL PREDICTIONS and (b) the embedded MAS_PAIRWISE_ODDS log-OR
sign, tested against chance with a binomial test at p < 0.05, Bonferroni
across the 21 evaluated pairs (tolerance pre-registered in ROADMAP §registry).

Design notes (honesty rules):
  - The model never saw the labels of the test split; phi is computed on
    test-split predictions only.
  - All 21 pairs of the 7-disease panel are evaluated (excluded MAS pairs
    are expected to be near-zero or negative and are reported as-is).
  - Pairs where either side has no variance are skipped and counted as
    non-recoveries (no silent drops: they appear in the output).
  - The embedded table is read from polymas_ml.data.patients (single source
    of truth), NOT re-declared here.

Evidence: <results_root>/f17_mas_recovery/
"""
from __future__ import annotations

import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "ml-engine-python"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("f17_mas")

from polymas_ml.data.patients import (  # noqa: E402
    DISEASE_LABELS,
    MAS_EXCLUSIONS,
    MAS_PAIRWISE_ODDS,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "stash/results_final_20260926"))
ALPHA = 0.05
N_PAIRS_EXPECTED = 21


def phi(x: np.ndarray, y: np.ndarray) -> float:
    vx, vy = x.std(), y.std()
    if vx == 0 or vy == 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (vx * vy))


def _significance_from_phi(phi_val: float, n: int) -> float:
    """Two-sided binomial-style p-value for phi != 0 via the normal
    approximation (standard for phi; Fisher's exact is equivalent here)."""
    from scipy import stats as sps

    if np.isnan(phi_val) or n < 10:
        return float("nan")
    z = phi_val * np.sqrt(n - 1)
    return float(2.0 * sps.norm.sf(abs(z)))


def main() -> int:
    out_dir = RESULTS / "f17_mas_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    test_ids = split.loc[split["split"] == "test", "patient_id"].tolist()
    preds = pd.read_csv(RESULTS / "models" / "test_predictions.csv").set_index("patient_id")
    preds = preds.loc[test_ids]
    labels = pd.read_csv(RESULTS / "features" / "labels.csv").set_index("patient_id").loc[test_ids]

    n = len(test_ids)
    rows = []
    for a, b in combinations(DISEASE_LABELS, 2):
        embedded = MAS_PAIRWISE_ODDS.get((a, b), MAS_PAIRWISE_ODDS.get((b, a)))
        excluded = any(
            {a, b} == set(pair) for pair in MAS_EXCLUSIONS
        )
        phi_pred = phi(preds[a].to_numpy(dtype=float), preds[b].to_numpy(dtype=float))
        phi_label = phi(labels[a].to_numpy(dtype=float), labels[b].to_numpy(dtype=float))
        p_val = _significance_from_phi(phi_pred, n)
        recovered = (
            embedded is not None
            and not np.isnan(phi_pred)
            and np.sign(phi_pred) == np.sign(embedded)
            and p_val == p_val and p_val < ALPHA
        )
        rows.append({
            "pair": f"{a}|{b}",
            "embedded_log_or": embedded if embedded is not None else 0.0,
            "mas_excluded": excluded,
            "phi_predictions": None if np.isnan(phi_pred) else round(phi_pred, 4),
            "phi_labels": None if np.isnan(phi_label) else round(phi_label, 4),
            "p_value": None if p_val != p_val else p_val,
            "sign_agrees": (embedded is not None and not np.isnan(phi_pred)
                            and np.sign(phi_pred) == np.sign(embedded)),
            "recovered": bool(recovered),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "mas_recovery_pairs.csv", index=False)

    # Pre-registered test: > chance sign agreement across all embedded pairs
    # (binomial, Bonferroni x21). Chance for sign agreement is 0.5.
    embedded_rows = df[~df["mas_excluded"]]
    n_sign_ok = int(embedded_rows["sign_agrees"].sum())
    n_embedded = len(embedded_rows)
    from scipy import stats as sps
    p_sign = float(sps.binomtest(n_sign_ok, n_embedded, 0.5, alternative="greater").pvalue)
    n_recovered = int(df["recovered"].sum())
    p_bonf = min(1.0, p_sign * N_PAIRS_EXPECTED)

    verdict = {
        "n_pairs_evaluated": len(df),
        "n_embedded_pairs": n_embedded,
        "n_sign_agreement": n_sign_ok,
        "sign_agreement_rate": round(n_sign_ok / max(n_embedded, 1), 4),
        "binomial_p_vs_chance": p_sign,
        "bonferroni_x21_p": p_bonf,
        "pass_pre_registered": bool(p_sign < ALPHA and p_bonf < ALPHA),
        "n_full_recovery_sign_and_significant": n_recovered,
        "tolerance": "sign agreement > chance at binomial p < 0.05, Bonferroni x21",
    }
    (out_dir / "mas_recovery_summary.json").write_text(json.dumps(verdict, indent=2))
    logger.info("F-17 verdict: %s", json.dumps(verdict, indent=1))
    logger.info("\n%s", df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

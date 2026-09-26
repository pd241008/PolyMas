"""F-19: sample-size scaling curves (ROADMAP Phase 2).

Pre-registered claim: performance vs n (1k/2.5k/5k/10k) quantifies data
efficiency for System A; the deliverable is the scaling figure + fitted
exponent in the report (R3).

Protocol (pre-registered):
  - Points: full real-mode pipeline runs (F-12 split, val-swept thresholds)
    at n in {1000, 2500, 5000, 10000}, seed 42, identical code path — the
    5k point IS the canonical Phase-2 run (no re-run drift).
  - Metric: held-out test AUROC per disease from each run's
    models/per_disease_metrics.csv, plus macro.
  - Fit: log-log OLS of macro-AUROC shortfall vs n: AUROC_max - AUROC(n)
    ~ n^(-alpha); alpha is the reported data-efficiency exponent (larger =
    faster gains from more data). AUROC_max taken as the 10k point
    (largest n), per the pre-registered rule — stated because with 4
    points the choice is visible, not hidden.

Evidence: <results_root>/f19_scaling_sweep/
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
logger = logging.getLogger("f19_scaling")

from polymas_ml.data.patients import DISEASE_LABELS  # noqa: E402

CANONICAL = "stash/results_final_20260926"
POINTS = [
    (1000, "stash/results_scaling_n1000"),
    (2500, "stash/results_scaling_n2500"),
    (5000, CANONICAL),
    (10000, "stash/results_scaling_n10000"),
]


def main() -> int:
    results = Path(os.environ.get(
        "POLYMAS_RESULTS_DIR", ROOT / CANONICAL))
    out_dir = results / "f19_scaling_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, folder in POINTS:
        f = ROOT / folder / "models" / "per_disease_metrics.csv"
        if not f.exists():
            logger.warning("missing %s — point skipped (listed, not silent)", f)
            continue
        m = pd.read_csv(f)
        for _, r in m.iterrows():
            rows.append({"n": n, "disease": r["disease"], "auroc": r["auroc"]})
        macro = float(m["auroc"].mean())
        rows.append({"n": n, "disease": "MACRO", "auroc": round(macro, 4)})
        logger.info("n=%d macro AUROC=%.4f", n, macro)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "scaling_curves.csv", index=False)

    # Fit the power law on the macro curve: AUROC_max - AUROC(n) ~ n^-alpha.
    macro = df[df["disease"] == "MACRO"].sort_values("n")
    auroc_max = float(macro["auroc"].iloc[-1])
    shortfall = (auroc_max - macro["auroc"]).to_numpy()
    n = macro["n"].to_numpy(dtype=float)
    ok = shortfall > 0
    alpha = None
    if ok.sum() >= 2:
        slope, _ = np.polyfit(np.log(n[ok]), np.log(shortfall[ok]), 1)
        alpha = round(float(-slope), 3)

    per_disease_exponent = {}
    for disease, g in df[df["disease"] != "MACRO"].groupby("disease"):
        g = g.sort_values("n")
        if len(g) < 3:
            continue
        d_max = float(g["auroc"].iloc[-1])
        sh = (d_max - g["auroc"]).to_numpy()
        good = sh > 0
        if good.sum() >= 2:
            s, _ = np.polyfit(np.log(g["n"].to_numpy(dtype=float)[good]),
                              np.log(sh[good]), 1)
            per_disease_exponent[disease] = round(float(-s), 3)

    summary = {
        "points": {str(n): f for n, f in POINTS},
        "auroc_max_reference": "n=10000 (largest point, pre-registered)",
        "macro_curve": {str(int(r.n)): float(r.auroc) for r in macro.itertuples()},
        "macro_exponent_alpha": alpha,
        "per_disease_exponent": per_disease_exponent,
        "note": "alpha = decay rate of the AUROC shortfall with n "
                "(AUROC_max - AUROC(n) ~ n^-alpha); larger = faster saturation",
    }
    (out_dir / "scaling_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("F-19 summary:\n%s", json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""F-17a: anchor-pair-restricted MAS recovery (deferred from the ⏸ register).

Pre-registered claim (docs/ROADMAP.md ⏸ register, written before this run):
sign agreement tested ONLY on the own-anchor pair set — pairs (A, B) sharing
at least one of A's own anchor loci (DISEASE_RISK_LOCI, the ADR-006 P2
restriction that produced F-20's 4/4 verdict). Motivation: F-17's overall
9/19 (p = 0.68) aggregates weakly-anchored pairs with the strongly-anchored
ones; the label generator plants co-occurrence signal through the anchor
loci dosage term, so anchor-restricted pairs are where recovery is possible
in principle. This is a NEW, narrower claim — not a repair of F-17.

Protocol (pre-registered):
  - Pair set: unordered {A, B} such that DISEASE_RISK_LOCI[A] ∩ {loci of B's
    anchors} ≠ ∅ in EITHER direction (A's anchor is B's anchor or vice versa).
    MAS-excluded pairs are reported but not counted (same rule as F-17).
  - Statistic: binomial test vs 0.5, greater alternative, p < 0.05 with
    Bonferroni over the anchor pair-set size.
  - Inputs: the EXISTING F-17 evidence (mas_recovery_pairs.csv) — no model
    is re-evaluated; the phi values are unchanged, only the tested subset.

This script does NOT overwrite any F-17 artifact. Evidence:
<results_root>/f17a_anchor_recovery/
"""
from __future__ import annotations

import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "ml-engine-python"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("f17a")

RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "results/results_final_20260926"))
OUT = RESULTS / "f17a_anchor_recovery"
ALPHA = 0.05


def anchor_pair_set() -> list[tuple[str, str]]:
    """Own-anchor pairs: {A,B} shares at least one anchor locus either way.

    Read from polymas_ml.data.patients (single source of truth). A pair
    qualifies if ANY anchor locus of A is also an anchor locus of B, or
    vice versa (the sets overlap in either direction).
    """
    from polymas_ml.data.patients import DISEASE_LABELS, DISEASE_RISK_LOCI

    pairs = []
    for a, b in combinations(DISEASE_LABELS, 2):
        la, lb = set(DISEASE_RISK_LOCI[a]), set(DISEASE_RISK_LOCI[b])
        if la & lb:
            pairs.append((a, b))
    return pairs


def main() -> int:
    from scipy import stats as sps

    OUT.mkdir(parents=True, exist_ok=True)
    pairs_csv = RESULTS / "f17_mas_recovery" / "mas_recovery_pairs.csv"
    if not pairs_csv.exists():
        logger.error("F-17 evidence missing: %s", pairs_csv)
        return 1
    df = pd_read(pairs_csv)

    anchor_pairs = anchor_pair_set()
    anchor_keys = {f"{a}|{b}" for a, b in anchor_pairs}

    # Pair order in the F-17 table is alphabetical (combinations of the
    # sorted-free DISEASE_LABELS order); match in both orientations.
    def key_for(row_pair: str) -> str:
        a, b = row_pair.split("|")
        return f"{a}|{b}" if (a, b) in anchor_pairs else f"{b}|{a}" if (b, a) in anchor_pairs else ""

    sel = []
    for _, row in df.iterrows():
        if key_for(row["pair"]):
            sel.append(row)
    sub = pd_read_rows(sel)

    sub.to_csv(OUT / "anchor_pairs.csv", index=False)

    n_tested = int((~sub["mas_excluded"]).sum())
    n_ok = int(sub.loc[~sub["mas_excluded"], "sign_agrees"].sum())
    p_sign = float(sps.binomtest(n_ok, max(n_tested, 1), 0.5,
                                 alternative="greater").pvalue) if n_tested else float("nan")
    p_bonf = min(1.0, p_sign * len(anchor_pairs)) if p_sign == p_sign else float("nan")

    # Post-hoc decomposition of the FULL F-17 table (embedded non-excluded
    # pairs split by whether a co-occurrence direction was planted at all).
    emb = df[~df["mas_excluded"]]
    planted_mask = emb["embedded_log_or"] != 0.0
    n_planted = int(planted_mask.sum())
    n_planted_ok = int(emb.loc[planted_mask, "sign_agrees"].sum())
    n_zero = int((~planted_mask).sum())
    n_zero_ok = int(emb.loc[~planted_mask, "sign_agrees"].sum())
    p_planted = (float(sps.binomtest(n_planted_ok, max(n_planted, 1), 0.5,
                                     alternative="greater").pvalue)
                 if n_planted else float("nan"))

    verdict = {
        "claim": "F-17a: sign agreement on own-anchor pairs only (DISEASE_RISK_LOCI overlap, either direction)",
        "protocol": "re-scoring of existing F-17 phi table; no model re-evaluation",
        "anchor_pair_set": sorted(anchor_keys),
        "n_anchor_pairs": len(anchor_pairs),
        "n_pairs_tested": n_tested,
        "n_sign_agreement": n_ok,
        "sign_agreement_rate": round(n_ok / max(n_tested, 1), 4),
        "binomial_p_vs_chance": p_sign,
        "bonferroni_xN_p": p_bonf,
        "pre_registered_pass": bool(p_sign == p_sign and p_sign < ALPHA and p_bonf < ALPHA),
        "tolerance": "sign agreement > chance at binomial p < 0.05, Bonferroni x n_anchor_pairs",
        # Descriptive stratification (post-hoc, NOT a second gate): anchor
        # pairs with embedded_log_or == 0.0 have no planted co-occurrence
        # direction, so sign agreement vs them is mechanically undefined
        # (sign(0) can never match a +-phi). Reported to make the verdict
        # interpretable, not to replace it.
        "descriptive_stratification": {
            "note": "post-hoc, reported alongside the pre-registered verdict",
            "planted_affinity_pairs": [
                {"pair": r["pair"], "embedded_log_or": r["embedded_log_or"],
                 "phi_predictions": r["phi_predictions"], "sign_agrees": bool(r["sign_agrees"])}
                for _, r in sub.iterrows()
                if r["embedded_log_or"] != 0.0],
            "zero_affinity_pairs": [
                r["pair"] for _, r in sub.iterrows() if r["embedded_log_or"] == 0.0],
            "planted_n": int((sub["embedded_log_or"] != 0.0).sum()),
            "planted_agree": int(sub.loc[sub["embedded_log_or"] != 0.0, "sign_agrees"].sum()),
            "excluded_pairs_note": (
                "MAS-excluded pairs (T1D|MS, RA|MS) are NOT in the anchor set "
                "(no shared anchor locus); their strong negative-phi agreement "
                "is separate F-17 evidence and must not be counted here."),
        },
        # Full-F-17-table decomposition (same post-hoc caveat): sign agreement
        # restricted to pairs with planted nonzero affinity vs zero-affinity
        # pairs. sign(0) can never match a +-/phi, so zero-affinity pairs are
        # coin-flip noise inside the F-17 aggregate; this quantifies how much
        # of the 9/19 headline they manufactured.
        "full_table_decomposition": {
            "note": ("post-hoc descriptive split of the F-17 aggregate: "
                     "pairs with planted nonzero affinity vs zero-affinity "
                     "pairs. For zero-affinity pairs sign(embedded)=0 can "
                     "NEVER equal sign(phi) in {+1,-1}, so their "
                     "sign_agrees=False is deterministic, not empirical — "
                     "7 structurally-impossible items inside the 9/19 "
                     "denominator"),
            "planted_pairs": {
                "n": n_planted,
                "n_sign_agree": n_planted_ok,
                "rate": round(n_planted_ok / max(n_planted, 1), 4),
            },
            "zero_affinity_pairs": {
                "n": n_zero,
                "n_sign_agree": n_zero_ok,
                "rate": round(n_zero_ok / max(n_zero, 1), 4),
            },
            "planted_binomial_p_vs_chance": p_planted,
        },
        "per_pair": [
            {"pair": r["pair"], "embedded_log_or": r["embedded_log_or"],
             "phi_predictions": r["phi_predictions"], "phi_labels": r["phi_labels"],
             "sign_agrees": bool(r["sign_agrees"]), "mas_excluded": bool(r["mas_excluded"])}
            for _, r in sub.iterrows()
        ],
    }
    (OUT / "f17a_summary.json").write_text(json.dumps(verdict, indent=2))
    logger.info("F-17a verdict: %s", json.dumps(
        {k: verdict[k] for k in ("n_pairs_tested", "n_sign_agreement",
                                 "sign_agreement_rate", "binomial_p_vs_chance",
                                 "pre_registered_pass")},
        indent=1))
    return 0


def pd_read(path: Path):
    import pandas as pd
    return pd.read_csv(path)


def pd_read_rows(rows: list):
    import pandas as pd
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())

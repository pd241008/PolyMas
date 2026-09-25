"""F-20: external direction-agreement validation (ROADMAP Phase 2).

Pre-registered claim (ROADMAP F-20): simulated effect directions match real
published associations. Operationalized on the canonical Phase-2 run as:
for each (disease, panel locus) pair where the disease's published dataset
has a genome-wide-significant association (p < 5e-8), compare

  - the MODEL's learned direction: Spearman correlation between the patient
    per-locus feature value (features/prs_features.csv z_score, which in
    real mode is a monotone function of the patient's real donor dosage at
    that locus) and the model's disease prediction on the F-12 held-out
    test split;
  - the PUBLISHED direction: sign of the OpenGWAS beta aligned to the VCF
    alt allele (same alignment machinery as ADR-005/F-16).

Pre-registered tolerance (ROADMAP §registry): >= 70% sign agreement on
p < 5e-8 associations counts as a pass. Pairs with zero-variance features
are skipped and listed (no silent drops). AITD/VITILIGO have no
genome-significant panel associations in their datasets and are reported
as coverage gaps, not failures.

Evidence: <results_root>/f20_external_validation/
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "ml-engine-python"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("f20_extval")

from polymas_ml.data import coupling as coup_mod  # noqa: E402
from polymas_ml.data.haplotypes import load_substrate  # noqa: E402
from polymas_ml.data.patients import DISEASE_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "stash/results_phase2_20260925"))
GWS_P = 5e-8
PASS_RATE = 0.70


def main() -> int:
    out_dir = RESULTS / "f20_external_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    dosages, _, _, _ = load_substrate(RESULTS)
    panel_af = dosages.mean() / 2.0
    ensembl_freqs = coup_mod.fetch_ensembl_allele_freqs(
        dosages.columns.tolist(), RESULTS / "raw" / "ensembl")
    vcf_alt = coup_mod.identify_vcf_alt(panel_af, ensembl_freqs)

    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    test_ids = split.loc[split["split"] == "test", "patient_id"].tolist()
    preds = pd.read_csv(RESULTS / "models" / "test_predictions.csv").set_index("patient_id").loc[test_ids]
    zscores = pd.read_csv(RESULTS / "features" / "prs_features.csv")
    z_wide = zscores.pivot_table(index="patient_id", columns="locus_id",
                                 values="z_score", aggfunc="first").loc[test_ids]

    rows = []
    for disease in DISEASE_LABELS:
        dataset = coup_mod.DISEASE_DATASETS.get(disease, "")
        if dataset.startswith("local:"):
            rows.append({"disease": disease, "locus": None,
                         "note": "local curated anchors: no published p-values; coverage gap"})
            continue
        assoc = coup_mod.load_assoc_rows(RESULTS, "20260925", dataset)
        gws = {r["rsid"]: r for r in assoc
               if r.get("p") is not None and float(r["p"]) < GWS_P and r.get("beta") is not None}
        coup = coup_mod.build_disease_coupling(
            disease, panel_af, ensembl_freqs, vcf_alt, RESULTS, "20260925")
        direction = coup.directions()
        if not gws:
            rows.append({"disease": disease, "locus": None,
                         "note": "no genome-wide panel associations; coverage gap"})
            continue
        for rs_id, r in sorted(gws.items()):
            if rs_id not in z_wide.columns:
                rows.append({"disease": disease, "locus": rs_id, "note": "locus not in patient features"})
                continue
            feat = z_wide[rs_id].to_numpy(dtype=float)
            if feat.std() == 0:
                rows.append({"disease": disease, "locus": rs_id,
                             "note": "zero-variance feature (skipped, counted)"})
                continue
            from scipy import stats as sps
            rho, _ = sps.spearmanr(feat, preds[disease].to_numpy(dtype=float))
            beta = float(r["beta"])
            # Align published beta to the VCF-alt direction of the feature.
            aligned_beta = beta if direction.get(rs_id) == "alt" else -beta
            rows.append({
                "disease": disease, "locus": rs_id,
                "published_p": float(r["p"]), "published_beta": beta,
                "aligned_to": direction.get(rs_id),
                "model_rho": round(float(rho), 4),
                "sign_agrees": bool(np.sign(rho) == np.sign(aligned_beta)),
                "note": "",
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "external_direction_pairs.csv", index=False)

    tested = df[df["sign_agrees"].notna()]
    n_ok = int(tested["sign_agrees"].sum())
    n_tested = len(tested)
    rate = n_ok / max(n_tested, 1)
    verdict = {
        "n_pairs_tested": n_tested,
        "n_sign_agreement": n_ok,
        "sign_agreement_rate": round(rate, 4),
        "pre_registered_pass": bool(rate >= PASS_RATE and n_tested >= 10),
        "tolerance": ">= 70% sign agreement on p<5e-8 associations",
        "coverage_gaps": df[df["note"].str.contains("coverage gap", na=True)]
            .get("disease", pd.Series(dtype=str)).dropna().unique().tolist(),
        "interpretation": (
            "FAIL recorded as-is. Root cause identified, not patched: the real-mode "
            "label polygenic term in simulate_labels uses UNALIGNED VCF dosages, "
            "which is sign-inverted vs published biology at loci where the VCF alt "
            "is the common protective allele (PTPN22 rs2476601, STAT4 rs7574865 — "
            "the ADR-004 calibration trap, still present in the label term). The "
            "models faithfully learned the inverted simulation signal. Fixing the "
            "label term changes the simulation generative model and requires a new "
            "ADR cycle; the rejected ADR-005 label design (allele-aligned published "
            "betas) already contained the correct mechanism."
        ),
    }
    (out_dir / "external_validation_summary.json").write_text(json.dumps(verdict, indent=2))
    logger.info("F-20 verdict: %s", json.dumps(verdict, indent=1))
    logger.info("\n%s", tested.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

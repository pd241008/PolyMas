"""F-16: classic PRS baseline (C+T clumping-and-thresholding style) on the
canonical Phase-2 split.

A per-disease polygenic risk score built ONLY from published OpenGWAS
summary statistics (no cohort fitting), evaluated on exactly the same
held-out test patients as the models (models/split_indices.csv). This is
the external baseline row for the results tables: if the learned models
cannot beat a zero-training published-stats PRS, that is reported as-is.

Construction (pre-registered here, before evaluation):
  - Betas: OpenGWAS curated log-odds datasets (ADR-005 probe cache), the
    same per-disease datasets used for the coupling decision
    (coupling.DISEASE_DATASETS). VITILIGO uses the local Jin-2016 anchors.
  - Allele alignment: identical machinery to ADR-005
    (identify_vcf_alt via Ensembl 1000GENOMES:phase_3:ALL frequency match).
  - C+T: keep p < 5e-8 loci ("clumping" proxy: our panel is sparse, ~91
    variants, no MHC fine-mapping possible; documented honestly). When a
    disease has fewer than 3 genome-wide loci, fall back to p < 1e-4
    (flagged in the manifest as the C+T threshold actually used).
  - Score: PRS = sum_loci beta * aligned_dosage, per-SD (analytic
    population SD as in coupling.label_prs_term).
  - Evaluation: AUROC/AUPRC on the F-12 held-out test split; reported
    alongside the models regardless of outcome (R3 comparison row).

Evidence: <results_root>/f16_prs_baseline/
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
logger = logging.getLogger("f16_prs")

from polymas_ml.data import coupling as coup_mod  # noqa: E402
from polymas_ml.data.haplotypes import load_substrate  # noqa: E402
from polymas_ml.data.patients import DISEASE_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "stash/results_final_20260926"))
GWS_P = 5e-8
TOPUP_P = 1e-4
MIN_GWS = 3


def build_prs_per_disease(
    dosages: pd.DataFrame, panel_af: pd.Series
) -> tuple[dict[str, pd.Series], dict]:
    """Per-disease per-SD PRS over ALL 2,504 donors (donor-indexed)."""
    ensembl_freqs = coup_mod.fetch_ensembl_allele_freqs(
        dosages.columns.tolist(), RESULTS / "raw" / "ensembl")
    vcf_alt = coup_mod.identify_vcf_alt(panel_af, ensembl_freqs)
    prs_by_disease: dict[str, pd.Series] = {}
    manifest: dict = {"thresholds": {}, "n_loci": {}, "dropped": {}}
    for disease in DISEASE_LABELS:
        coup = coup_mod.build_disease_coupling(
            disease, panel_af, ensembl_freqs, vcf_alt, RESULTS, "20260925")
        # Local curated anchors (AITD/VITILIGO) carry no published p-values;
        # pre-registered convention: include all their anchor loci (the C+T
        # threshold does not apply — flagged in the manifest).
        if not coup.loci:
            logger.warning("%s: no aligned loci — PRS undefined (NaN row)", disease)
            manifest["thresholds"][disease] = None
            manifest["n_loci"][disease] = 0
            manifest["dropped"][disease] = coup.dropped
            continue
        gws = [l for l in coup.loci if l.pvalue is not None and l.pvalue < GWS_P]
        no_pvals = all(l.pvalue is None for l in coup.loci)
        if no_pvals:
            selected, threshold = list(coup.loci), "curated anchors (no published p-values)"
        elif len(gws) >= MIN_GWS:
            selected, threshold = gws, "p<5e-8"
        else:
            selected = [l for l in coup.loci
                        if l.pvalue is not None and l.pvalue < TOPUP_P]
            threshold = "p<1e-4 (fallback: <3 genome-wide loci)"
        if not selected:
            logger.warning("%s: no loci under either threshold", disease)
            manifest["thresholds"][disease] = threshold
            manifest["n_loci"][disease] = 0
            manifest["dropped"][disease] = coup.dropped
            manifest.setdefault("notes", {})[disease] = (
                "no panel association under C+T thresholds; curated anchors are "
                "HLA-proxy loci (allele-unresolvable onto panel variants)"
                if disease == "AITD" else "no loci under thresholds")
            continue
        # Score all donors (aligned to effect allele), then per-SD.
        aligned = coup_mod.aligned_dosage_matrix(dosages, coup)
        score = pd.Series(0.0, index=dosages.index)
        var = 0.0
        for locus in selected:
            col = aligned[locus.rs_id]
            p_eff = (1.0 - float(panel_af[locus.rs_id])
                     if locus.aligned_to == "ref" else float(panel_af[locus.rs_id]))
            score += locus.beta * col
            var += locus.beta ** 2 * 2.0 * p_eff * (1.0 - p_eff)
        sd = float(np.sqrt(var)) if var > 0 else 1.0
        prs_by_disease[disease] = score / sd
        manifest["thresholds"][disease] = threshold
        manifest["n_loci"][disease] = len(selected)
        manifest["dropped"][disease] = coup.dropped
        logger.info("%s: PRS over %d loci (%s)", disease, len(selected), threshold)
    return prs_by_disease, manifest


def main() -> int:
    out_dir = RESULTS / "f16_prs_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    dosages, meta, _, _ = load_substrate(RESULTS)
    panel_af = dosages.mean() / 2.0

    logger.info("building published-betas PRS per disease...")
    prs_by_disease, manifest = build_prs_per_disease(dosages, panel_af)
    prs_wide = pd.DataFrame(prs_by_disease)
    prs_wide.to_csv(out_dir / "donor_prs.csv", index_label="donor_id")

    # Patient <- donor map persisted by the canonical run (R2 provenance:
    # F-10 wiring writes donor_map.csv so cross-script evaluations score the
    # SAME patients' donors).
    donor_map_path = RESULTS / "donor_map.csv"
    if not donor_map_path.exists():
        raise FileNotFoundError(
            "donor_map.csv missing from the run folder — rerun the pipeline "
            "(the F-10 wiring persists it)")
    patient_donor = pd.read_csv(donor_map_path, index_col=0).iloc[:, 0]

    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    test_ids = split.loc[split["split"] == "test", "patient_id"].tolist()

    labels = pd.read_csv(RESULTS / "features" / "labels.csv").set_index("patient_id")
    labels = labels.loc[split["patient_id"].tolist()]

    rows = []
    for disease in DISEASE_LABELS:
        if disease not in prs_by_disease:
            rows.append({"disease": disease, "auroc": None, "auprc": None,
                         "note": "no published loci under threshold"})
            continue
        prs_patient = prs_by_disease[disease].reindex(patient_donor.loc[test_ids].to_numpy())
        prs_patient.index = test_ids
        y = labels.loc[test_ids, disease].to_numpy()
        from sklearn.metrics import average_precision_score, roc_auc_score
        auroc = (float(roc_auc_score(y, prs_patient.to_numpy()))
                 if len(np.unique(y)) >= 2 else None)
        auprc = (float(average_precision_score(y, prs_patient.to_numpy()))
                 if y.sum() > 0 else None)
        rows.append({"disease": disease, "auroc": None if auroc is None else round(auroc, 4),
                     "auprc": None if auprc is None else round(auprc, 4),
                     "n_loci": manifest["n_loci"][disease],
                     "threshold": manifest["thresholds"][disease]})
        logger.info("F16 %s: AUROC=%s AUPRC=%s", disease,
                    rows[-1]["auroc"], rows[-1]["auprc"])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "prs_baseline_metrics.csv", index=False)
    manifest["results_root"] = str(RESULTS)
    (out_dir / "prs_baseline_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Comparison row vs the models (same split).
    m = pd.read_csv(RESULTS / "models" / "per_disease_metrics.csv")
    comp = m[["disease", "auroc", "auprc"]].merge(df[["disease", "auroc", "auprc"]],
                                                  on="disease", suffixes=("_model", "_prs"))
    comp["auroc_delta_model_minus_prs"] = (
        comp["auroc_model"] - comp["auroc_prs"]).round(4)
    comp.to_csv(out_dir / "model_vs_prs_comparison.csv", index=False)
    logger.info("model vs PRS comparison:\n%s", comp.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

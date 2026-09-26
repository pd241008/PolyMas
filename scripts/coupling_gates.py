"""ADR-005 gate evaluation: decide adopt vs fallback for published coupling.

Runs the 5,000-patient coupled patient build IN-PROCESS (no model training),
then evaluates the pre-registered gates from ADR-005:

  G1  enrichment realism: cohort-vs-background aligned-dosage shift at each
      disease's genome-wide loci within [1/3, 3] x the first-order
      published-OR prediction (2 p(1-p) beta); SKIP when predicted < 0.01
      or the disease has no genome-wide loci (top-5 p<1e-4 used instead,
      flagged).
  G2  label structure preserved: overdispersion >= 1.05 and 2+ rate >= 0.20.
  G3  marginals preserved: per-disease prevalence within +/-0.03 of the
      no-coupling real run (stash/results_real_20260925).

Adoption rule (ADR-005): G1 PASS for >= 3 of {RA, SLE, T1D, MS, SJOGRENS}
(remainder PASS or SKIP) AND G2 AND G3.

Writes stash/results_real_20260925/adr005_gates/gate_report.json (R2).

Usage:
    POLYMAS_RESULTS_DIR=stash/results_real_20260925 \
        services/ml-engine-python/.venv/bin/python scripts/coupling_gates.py
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("coupling_gates")

from polymas_ml.data import coupling as coup_mod  # noqa: E402
from polymas_ml.data.immport import (  # noqa: E402
    assign_subjects,
    build_subject_pool,
    draw_patient_groups,
)
from polymas_ml.data.patients import (  # noqa: E402
    DISEASE_LABELS,
    label_structure_report,
    simulate_genotypes_prs,
    simulate_labels,
)
from polymas_ml.data.haplotypes import load_substrate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("POLYMAS_RESULTS_DIR", ROOT / "stash/results_real_20260925"))
N_PATIENTS = 5000
SEED = 42
COHORT_PREV = {"RA": 0.50, "SLE": 0.45, "SJOGRENS": 0.40, "T1D": 0.45, "MS": 0.40}
ANCHOR_DISEASES = ["RA", "SLE", "T1D", "MS", "SJOGRENS"]
GWS_P = 5e-8
TOPUP_P = 1e-4
PREDICTED_MIN = 0.01
G1_BAND = 3.0
G2_MIN_OVERDISPERSION = 1.05
G2_MIN_RATE_2PLUS = 0.20
G3_TOL = 0.03
G1_MIN_PASS = 3


def build_coupled_cohort():
    """Mirror build_real_dataset's real+coupling branch with full introspection."""
    from polymas_ml.data.immport import build_subject_pool

    subject_pool = build_subject_pool(cache_dir=RESULTS / "raw" / "immport")
    rng = np.random.default_rng(SEED)
    patient_groups = draw_patient_groups(subject_pool, N_PATIENTS, rng)
    assignments = assign_subjects(subject_pool, N_PATIENTS, patient_groups, rng)

    dosages, kg_meta, _, _ = load_substrate(RESULTS)
    ancestry_labels = pd.Series(
        [a["ancestry"] for a in assignments],
        index=[f"P{i:04d}" for i in range(N_PATIENTS)],
    )
    known = set(kg_meta["super_pop"].unique())
    ancestry_labels = ancestry_labels.where(ancestry_labels.isin(known), "EUR")

    probe_dir = RESULTS / "adr005_probe_20260925"
    probe_dir.mkdir(parents=True, exist_ok=True)
    if not (probe_dir / "dataset_picks.json").exists():
        import shutil
        shutil.copytree(RESULTS / "adr005_probe", probe_dir, dirs_exist_ok=True)

    panel_af = dosages.mean() / 2.0
    ensembl_freqs = coup_mod.fetch_ensembl_allele_freqs(
        dosages.columns.tolist(), RESULTS / "raw" / "ensembl")
    vcf_alt = coup_mod.identify_vcf_alt(panel_af, ensembl_freqs)
    couplings = {
        d: coup_mod.build_disease_coupling(
            d, panel_af, ensembl_freqs, vcf_alt, RESULTS, "20260925")
        for d in coup_mod.DISEASE_DATASETS
    }
    aligned_by_disease = {
        d: coup_mod.aligned_dosage_matrix(dosages, c) for d, c in couplings.items()
    }
    coup_mod.save_coupling_manifest(RESULTS / "coupling_20260925", couplings, vcf_alt, {})

    anchored = coup_mod.sample_anchored_donors(
        couplings, aligned_by_disease, dosages, kg_meta,
        ancestry_labels, patient_groups, rng,
    )
    anchored = anchored.loc[[f"P{i:04d}" for i in range(N_PATIENTS)]]
    donor_gt = dosages.loc[anchored.to_numpy()].copy()
    donor_gt.index = anchored.index
    label_prs_terms = {
        d: coup_mod.label_prs_term(donor_gt, c, panel_af)
        for d, c in couplings.items()
    }

    _, gen_rows, donor_ids = simulate_genotypes_prs(
        N_PATIENTS, {}, patient_groups, rng,
        donor_dosages=donor_gt, donor_map=anchored,
    )
    labels_df = simulate_labels(
        N_PATIENTS, patient_groups, [a["sex"] for a in assignments],
        gen_rows, {}, rng, genotype_mode="real",
        label_prs_terms=label_prs_terms,
    )
    return {
        "patient_groups": patient_groups,
        "assignments": assignments,
        "couplings": couplings,
        "aligned_by_disease": aligned_by_disease,
        "donor_gt": donor_gt,
        "gen_rows": gen_rows,
        "labels_df": labels_df,
        "panel_af": panel_af,
    }


def gate_g1(state: dict) -> dict:
    """Enrichment realism per anchor disease."""
    dosages = state["donor_gt"]
    groups = pd.Series(state["patient_groups"],
                       index=[f"P{i:04d}" for i in range(N_PATIENTS)])
    out: dict[str, dict] = {}
    for disease in ANCHOR_DISEASES:
        coup = state["couplings"][disease]
        # Patient-level aligned dosages (donor_gt is patient-indexed).
        aligned = coup_mod.aligned_dosage_matrix(state["donor_gt"], coup)
        # Locus set: GWS, else top-5 with p < 1e-4 (flagged), else SKIP.
        loci = [l for l in coup.loci if l.pvalue is not None and l.pvalue < GWS_P]
        mode = "genome_wide"
        if not loci:
            loci = sorted(
                [l for l in coup.loci if l.pvalue is not None and l.pvalue < TOPUP_P],
                key=lambda l: l.pvalue)[:5]
            mode = "top5_p1e4"
        if not loci:
            out[disease] = {"status": "SKIP", "reason": "no usable loci",
                            "n_loci": len(coup.loci)}
            continue
        cohort_mask = groups == disease
        if cohort_mask.sum() < 50:
            out[disease] = {"status": "SKIP", "reason": "cohort too small",
                            "n_cohort": int(cohort_mask.sum())}
            continue
        bg_mask = groups.isna()
        realized_shifts, predicted_shifts = [], []
        for locus in loci:
            p_eff = (1.0 - float(state["panel_af"][locus.rs_id])
                     if locus.aligned_to == "ref"
                     else float(state["panel_af"][locus.rs_id]))
            predicted = 2.0 * p_eff * (1.0 - p_eff) * locus.beta
            col = aligned[locus.rs_id]
            realized = float(col[cohort_mask].mean() - col[bg_mask].mean())
            realized_shifts.append(realized)
            predicted_shifts.append(predicted)
        realized_mean = float(np.mean(realized_shifts))
        predicted_mean = float(np.mean(predicted_shifts))
        if abs(predicted_mean) < PREDICTED_MIN:
            out[disease] = {"status": "SKIP", "reason": "predicted enrichment < 0.01",
                            "predicted": predicted_mean, "realized": realized_mean,
                            "mode": mode, "n_loci": len(loci)}
            continue
        # "Enrichment" means shifted in the PUBLISHED direction: locus sets
        # can net negative betas (protective alleles), so the sign guard is
        # direction consistency with the prediction, not positivity.
        ratio = realized_mean / predicted_mean
        direction_ok = (realized_mean * predicted_mean) > 0
        band_ok = (1.0 / G1_BAND <= abs(ratio) <= G1_BAND)
        out[disease] = {
            "status": "PASS" if (direction_ok and band_ok) else "FAIL",
            "realized": round(realized_mean, 5),
            "predicted": round(predicted_mean, 5),
            "ratio": round(ratio, 3),
            "mode": mode,
            "n_loci": len(loci),
            "n_cohort": int(cohort_mask.sum()),
            "per_locus": [
                {"rs": l.rs_id, "beta": l.beta, "p": l.pvalue,
                 "realized": round(r, 5), "predicted": round(pr, 5)}
                for l, r, pr in zip(loci, realized_shifts, predicted_shifts)
            ],
        }
    return out


def gate_g2(labels_df: pd.DataFrame) -> dict:
    report = label_structure_report(labels_df)
    ok = (report["overdispersion_ratio"] >= G2_MIN_OVERDISPERSION
          and report["rate_2plus"] >= G2_MIN_RATE_2PLUS)
    return {
        "status": "PASS" if ok else "FAIL",
        "overdispersion": report["overdispersion_ratio"],
        "rate_2plus": report["rate_2plus"],
        "rate_3plus_mas": report["rate_3plus_mas"],
        "mean_diseases_per_patient": report["mean_diseases_per_patient"],
    }


def gate_g3(labels_df: pd.DataFrame) -> dict:
    ref_path = RESULTS / "features" / "labels.csv"
    if not ref_path.exists():
        return {"status": "SKIP", "reason": f"no reference labels at {ref_path}"}
    ref = pd.read_csv(ref_path).set_index("patient_id")
    per_disease = {}
    ok = True
    for d in DISEASE_LABELS:
        if d not in ref.columns:
            continue
        delta = float(labels_df[d].mean() - ref[d].mean())
        per_disease[d] = {
            "coupled": round(float(labels_df[d].mean()), 4),
            "no_coupling": round(float(ref[d].mean()), 4),
            "delta": round(delta, 4),
        }
        if abs(delta) > G3_TOL:
            ok = False
    return {"status": "PASS" if ok else "FAIL",
            "tolerance": G3_TOL, "per_disease": per_disease}


def main() -> int:
    logger.info("building the coupled cohort (n=%d, seed=%d)...", N_PATIENTS, SEED)
    state = build_coupled_cohort()

    logger.info("evaluating gates...")
    g1 = gate_g1(state)
    g2 = gate_g2(state["labels_df"])
    g3 = gate_g3(state["labels_df"])

    n_pass = sum(1 for d in ANCHOR_DISEASES if g1.get(d, {}).get("status") == "PASS")
    n_fail = sum(1 for d in ANCHOR_DISEASES if g1.get(d, {}).get("status") == "FAIL")
    adopt = (n_pass >= G1_MIN_PASS and n_fail == 0
             and g2["status"] == "PASS" and g3["status"] == "PASS")

    report = {
        "date": "2026-09-25",
        "n_patients": N_PATIENTS,
        "seed": SEED,
        "g1_enrichment": g1,
        "g2_structure": g2,
        "g3_marginals": g3,
        "adoption_rule": f"G1 PASS >= {G1_MIN_PASS} of {ANCHOR_DISEASES} (no FAIL) AND G2 AND G3",
        "g1_pass_count": n_pass,
        "verdict": "ADOPT" if adopt else "FALLBACK (no-coupling stays canonical, ADR-004)",
    }
    out_dir = RESULTS / "adr005_gates"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_report.json").write_text(json.dumps(report, indent=2))

    for d in ANCHOR_DISEASES:
        v = g1.get(d, {})
        logger.info("G1 %-10s %s realized=%s predicted=%s ratio=%s mode=%s",
                    d, v.get("status"), v.get("realized"), v.get("predicted"),
                    v.get("ratio"), v.get("mode", "-"))
    logger.info("G2 %s overdispersion=%s 2+rate=%s",
                g2["status"], g2["overdispersion"], g2["rate_2plus"])
    logger.info("G3 %s", g3["status"])
    logger.info("VERDICT: %s", report["verdict"])
    print(json.dumps({k: report[k] for k in
                      ("g1_pass_count", "verdict")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

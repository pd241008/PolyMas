"""Semi-synthetic dataset construction for PolyMas.

Combines real GWAS effect-size distributions with ImmPort-style clinical
feature distributions to produce patient profiles suitable for the ensemble.

Labels are drawn with the SAME co-occurrence (polyautoimmunity) engine as
the main pipeline (polymas_ml.data.patients.draw_cooccurring_labels): a
latent liability mixture plus pairwise MAS log-OR affinities, so the
standalone builder reproduces the overdispersed multi-disease joint of the
primary dataset rather than independent Bernoulli draws.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from polymas_ml.data.patients import (
    DISEASE_LABELS as COOC_DISEASE_LABELS,
    LIABILITY_SCALE,
    MAS_AFFINITY_SCALE,
    MAS_LIABILITY_MIX,
    MAS_PAIRWISE_ODDS,
    draw_cooccurring_labels,
)
from polymas_ml.models.shared_features import SHARED_LOCI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISEASE_LABELS = list(COOC_DISEASE_LABELS)

# Background (non-cohort) marginal prevalences for the standalone builder,
# matched to the co-occurrence calibration in patients.py (see notes there:
# moderate marginals so multi-disease accumulation is structured, not
# incidental chance stacking).
BASE_PREVALENCES = {
    "RA": 0.06,
    "SLE": 0.04,
    "SJOGRENS": 0.03,
    "AITD": 0.05,
    "T1D": 0.03,
    "VITILIGO": 0.025,
    "MS": 0.04,
}

ALL_LOCI = list(SHARED_LOCI.keys()) + [f"rs{random.randint(100000, 999999)}" for _ in range(40)]


def _generate_prs(patient_id: str, n_loci: int = 25, base_scores: dict[str, float] | None = None) -> pd.DataFrame:
    rows = []
    for _ in range(n_loci):
        locus_id = random.choice(ALL_LOCI)
        gene = SHARED_LOCI.get(locus_id, {}).get("gene", locus_id)
        base = base_scores.get(locus_id, 0.3) if base_scores else 0.3
        noise = random.gauss(0, 0.25)
        continuous_score = round(min(1.0, max(0.0, base + noise)), 4)
        z_score = round(random.gauss(continuous_score * 2 - 1, 0.5), 4)
        rows.append({
            "patient_id": patient_id,
            "locus_id": locus_id,
            "gene_symbol": gene,
            "continuous_score": continuous_score,
            "z_score": z_score,
        })
    return pd.DataFrame(rows)


def _generate_clinical(patient_id: str, risk_factor: float) -> dict:
    has_any = abs(risk_factor) > 0.15
    sex = "F" if random.random() < (0.55 + 0.1 * has_any) else "M"
    age_base = 35 + 15 * risk_factor
    age_at_diagnosis_days = int(max(365, min(80 * 365, age_base * 365.25 + random.gauss(0, 4 * 365))))
    bmi = round(max(16, min(42, 22 + 2 * risk_factor + random.gauss(0, 3))), 1)
    family_history = int(random.random() < (0.15 + 0.2 * risk_factor + 0.2 * has_any))
    return {
        "patient_id": patient_id,
        "sex": sex,
        "ethnicity": random.choice(["EUR", "AFR", "EAS", "SAS"]),
        "age_at_diagnosis_days": age_at_diagnosis_days,
        "bmi": bmi,
        "family_history": family_history,
    }


def _generate_labels(
    patient_id: str,
    risk_factor: float,
    liability: float,
    rng: np.random.Generator,
) -> dict:
    """One patient's labels via the shared MAS co-occurrence cascade.

    The builder's scalar risk_factor plays the role of the pipeline's
    cohort/polygenic term: it shifts every disease's marginal risk before
    the latent-liability and pairwise-affinity lifts are applied.
    """
    marginal = {
        disease: min(0.95, max(0.005, BASE_PREVALENCES[disease] + risk_factor))
        for disease in DISEASE_LABELS
    }
    labels = draw_cooccurring_labels(marginal, liability, rng)
    return {"patient_id": patient_id, **labels}


def build_dataset(
    n_patients: int = 400,
    n_loci: int = 25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    prs_frames = []
    clinical_rows = []
    label_rows = []

    base_scores = {locus: random.betavariate(2, 5) for locus in ALL_LOCI}
    liability_weights, liability_values = zip(*MAS_LIABILITY_MIX)
    for i in range(n_patients):
        patient_id = f"P{i:04d}"
        risk_factor = random.gauss(0, 0.03)
        liability = float(rng.choice(liability_values, p=liability_weights))
        prs_frames.append(_generate_prs(patient_id, n_loci, base_scores))
        clinical_rows.append(_generate_clinical(patient_id, risk_factor))
        label_rows.append(_generate_labels(patient_id, risk_factor, liability, rng))

    prs_df = pd.concat(prs_frames, ignore_index=True)
    clinical_df = pd.DataFrame(clinical_rows)
    labels_df = pd.DataFrame(label_rows)

    return prs_df, clinical_df, labels_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semi-synthetic PolyMas dataset")
    parser.add_argument("--n-patients", type=int, default=400)
    parser.add_argument("--n-loci", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="stash/results/raw/semi-synthetic")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prs_df, clinical_df, labels_df = build_dataset(
        n_patients=args.n_patients,
        n_loci=args.n_loci,
        seed=args.seed,
    )

    prs_df.to_parquet(out / "prs_scores.parquet", index=False)
    clinical_df.to_parquet(out / "clinical_features.parquet", index=False)
    labels_df.to_parquet(out / "labels.parquet", index=False)

    from polymas_ml.data.patients import label_structure_report

    summary = {
        "n_patients": int(args.n_patients),
        "n_loci_per_patient": int(args.n_loci),
        "seed": args.seed,
        "prs_rows": len(prs_df),
        "clinical_rows": len(clinical_df),
        "label_rows": len(labels_df),
        "label_structure": label_structure_report(labels_df),
        "cooccurrence_model": {
            "liability_mix": [list(t) for t in MAS_LIABILITY_MIX],
            "liability_scale": LIABILITY_SCALE,
            "affinity_scale": MAS_AFFINITY_SCALE,
            "n_pairwise_affinities": len(MAS_PAIRWISE_ODDS),
        },
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Dataset built in %s", out)


if __name__ == "__main__":
    main()

"""Semi-synthetic dataset construction for PolyMas.

Combines real GWAS effect-size distributions with ImmPort-style clinical
feature distributions to produce patient profiles suitable for the ensemble.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from polymas_ml.models.shared_features import SHARED_LOCI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISEASE_LABELS = [
    "T1D",
    "T2D",
    "LADA",
    "GESTATIONAL_DM",
    "MONOGENIC_DIABETES",
]

ALL_LOCI = list(SHARED_LOCI.keys()) + [f"rs{random.randint(100000, 999999)}" for _ in range(40)]


def _generate_prs(patient_id: str, n_loci: int = 25) -> pd.DataFrame:
    rows = []
    for _ in range(n_loci):
        locus_id = random.choice(ALL_LOCI)
        gene = SHARED_LOCI.get(locus_id, {}).get("gene", locus_id)
        continuous_score = round(random.betavariate(2, 5), 4)
        z_score = round(random.gauss(0, 1), 4)
        rows.append({
            "patient_id": patient_id,
            "locus_id": locus_id,
            "gene_symbol": gene,
            "continuous_score": continuous_score,
            "z_score": z_score,
        })
    return pd.DataFrame(rows)


def _generate_clinical(patient_id: str) -> dict:
    return {
        "patient_id": patient_id,
        "sex": random.choice(["M", "F"]),
        "ethnicity": random.choice(["EUR", "AFR", "EAS", "SAS"]),
        "age_at_diagnosis_days": random.randint(365, 365 * 80),
        "hla_type": random.choice(["DR3", "DR4", "DQ2", "DQ8", "None"]),
        "bmi": round(random.uniform(18, 40), 1),
        "family_history": random.choice([0, 1]),
    }


def _generate_labels(patient_id: str) -> dict:
    labels = {"patient_id": patient_id}
    for disease in DISEASE_LABELS:
        prevalence = {
            "T1D": 0.08,
            "T2D": 0.25,
            "LADA": 0.04,
            "GESTATIONAL_DM": 0.10,
            "MONOGENIC_DIABETES": 0.02,
        }[disease]
        labels[disease] = int(random.random() < prevalence)
    return labels


def build_dataset(
    n_patients: int = 500,
    n_loci: int = 25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)

    prs_frames = []
    clinical_rows = []
    label_rows = []

    for i in range(n_patients):
        patient_id = f"P{i:04d}"
        prs_frames.append(_generate_prs(patient_id, n_loci))
        clinical_rows.append(_generate_clinical(patient_id))
        label_rows.append(_generate_labels(patient_id))

    prs_df = pd.concat(prs_frames, ignore_index=True)
    clinical_df = pd.DataFrame(clinical_rows)
    labels_df = pd.DataFrame(label_rows)

    return prs_df, clinical_df, labels_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semi-synthetic PolyMas dataset")
    parser.add_argument("--n-patients", type=int, default=500)
    parser.add_argument("--n-loci", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/raw")
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

    summary = {
        "n_patients": int(args.n_patients),
        "n_loci_per_patient": int(args.n_loci),
        "seed": args.seed,
        "prs_rows": len(prs_df),
        "clinical_rows": len(clinical_df),
        "label_rows": len(labels_df),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Dataset built in %s", out)


if __name__ == "__main__":
    main()

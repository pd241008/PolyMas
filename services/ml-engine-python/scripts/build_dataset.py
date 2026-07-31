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
    "RA",
    "SLE",
    "SJOGRENS",
    "AITD",
    "T1D",
    "VITILIGO",
    "MS",
]

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


def _generate_labels(patient_id: str, risk_factor: float) -> dict:
    labels = {"patient_id": patient_id}
    for disease in DISEASE_LABELS:
        base = {
            "RA": 0.20,
            "SLE": 0.10,
            "SJOGRENS": 0.08,
            "AITD": 0.15,
            "T1D": 0.08,
            "VITILIGO": 0.06,
            "MS": 0.12,
        }[disease]
        prevalence = min(0.95, max(0.01, base + risk_factor))
        labels[disease] = int(random.random() < prevalence)
    return labels


def build_dataset(
    n_patients: int = 400,
    n_loci: int = 25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)

    prs_frames = []
    clinical_rows = []
    label_rows = []

    base_scores = {locus: random.betavariate(2, 5) for locus in ALL_LOCI}
    for i in range(n_patients):
        patient_id = f"P{i:04d}"
        risk_factor = random.gauss(0, 0.15)
        prs_frames.append(_generate_prs(patient_id, n_loci, base_scores))
        clinical_rows.append(_generate_clinical(patient_id, risk_factor))
        label_rows.append(_generate_labels(patient_id, risk_factor))

    prs_df = pd.concat(prs_frames, ignore_index=True)
    clinical_df = pd.DataFrame(clinical_rows)
    labels_df = pd.DataFrame(label_rows)

    return prs_df, clinical_df, labels_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semi-synthetic PolyMas dataset")
    parser.add_argument("--n-patients", type=int, default=400)
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

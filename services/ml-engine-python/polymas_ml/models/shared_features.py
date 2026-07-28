"""Shared cross-disease feature ingestion for common autoimmune loci."""

from __future__ import annotations

import pandas as pd

# Loci known to be shared across autoimmune diseases (HLA region, etc.)
SHARED_LOCI = {
    "HLA_DR3": {"gene": "HLA-DRB1", "region": "6p21.3", "diseases": ["T1D", "LADA"]},
    "HLA_DR4": {"gene": "HLA-DRB1", "region": "6p21.3", "diseases": ["T1D", "LADA"]},
    "HLA_DQ2": {"gene": "HLA-DQB1", "region": "6p21.3", "diseases": ["T1D", "T2D", "LADA"]},
    "HLA_DQ8": {"gene": "HLA-DQB1", "region": "6p21.3", "diseases": ["T1D", "LADA"]},
    "CTLA4": {"gene": "CTLA4", "region": "2q33.2", "diseases": ["T1D", "LADA", "T2D"]},
    "PTPN22": {"gene": "PTPN22", "region": "1p13.2", "diseases": ["T1D", "LADA"]},
    "IL2RA": {"gene": "IL2RA", "region": "10p15.1", "diseases": ["T1D", "LADA"]},
    "INS_VNTR": {"gene": "INS", "region": "11p15.5", "diseases": ["T1D"]},
    "IFIH1": {"gene": "IFIH1", "region": "2q24.2", "diseases": ["T1D", "LADA"]},
    "TCF7L2": {"gene": "TCF7L2", "region": "10q25.2", "diseases": ["T2D", "GESTATIONAL_DM"]},
    "KCNJ11": {"gene": "KCNJ11", "region": "11p15.1", "diseases": ["T2D", "MONOGENIC_DIABETES"]},
    "PPARG": {"gene": "PPARG", "region": "3p25.2", "diseases": ["T2D"]},
}


def extract_shared_features(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and pivot shared cross-disease loci into wide-form feature matrix.

    Expected input columns: patient_id, locus_id, gene_symbol, continuous_score, z_score.
    Returns: DataFrame with patient_id as index, one column per shared locus feature.
    """
    shared_locus_ids = set(SHARED_LOCI.keys())
    filtered = profiles_df[profiles_df["locus_id"].isin(shared_locus_ids)].copy()

    # Pivot to wide form: rows = patient_id, columns = locus_id__metric
    pivot = filtered.pivot_table(
        index="patient_id",
        columns="locus_id",
        values=["continuous_score", "z_score"],
        aggfunc="first",
    )

    # Flatten multi-index columns
    pivot.columns = [f"{col[1]}__{col[0]}" for col in pivot.columns]
    return pivot.reset_index()

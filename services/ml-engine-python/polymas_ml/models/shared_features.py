"""Shared cross-disease feature ingestion for common autoimmune loci."""

from __future__ import annotations

import pandas as pd

# Loci known to be shared across autoimmune diseases (HLA region, etc.)
SHARED_LOCI = {
    "rs2187668": {"gene": "HLA-DRB1", "region": "6p21.3", "diseases": ["RA", "SLE", "T1D", "AITD"]},
    "rs9272346": {"gene": "HLA-DQB1", "region": "6p21.3", "diseases": ["RA", "SLE", "T1D", "AITD"]},
    "rs2476601": {"gene": "PTPN22", "region": "1p13.2", "diseases": ["RA", "SLE", "T1D", "AITD"]},
    "rs3087243": {"gene": "CTLA4", "region": "2q33.2", "diseases": ["RA", "T1D", "AITD", "VITILIGO"]},
    "rs2292239": {"gene": "ERBB3", "region": "12q13", "diseases": ["T1D"]},
    "rs11209026": {"gene": "IL23R", "region": "1p31.3", "diseases": ["RA", "SLE", "SJOGRENS"]},
    "rs2104286": {"gene": "IL2RA", "region": "10p15.1", "diseases": ["RA", "SLE", "T1D"]},
    "rs7574865": {"gene": "STAT4", "region": "2q32.2", "diseases": ["SLE", "RA", "SJOGRENS"]},
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

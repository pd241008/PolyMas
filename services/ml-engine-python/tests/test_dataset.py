"""Tests for the dataset construction script."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_dataset import build_dataset


@pytest.fixture
def dataset(tmp_path: Path):
    prs_df, clinical_df, labels_df = build_dataset(
        n_patients=100,
        n_loci=15,
        seed=7,
    )
    return prs_df, clinical_df, labels_df


def test_output_schemas(dataset):
    prs_df, clinical_df, labels_df = dataset

    assert "patient_id" in prs_df.columns
    assert "locus_id" in prs_df.columns
    assert "continuous_score" in prs_df.columns
    assert "z_score" in prs_df.columns

    assert "patient_id" in clinical_df.columns
    assert "sex" in clinical_df.columns
    assert "ethnicity" in clinical_df.columns

    for col in ["T1D", "T2D", "LADA", "GESTATIONAL_DM", "MONOGENIC_DIABETES"]:
        assert col in labels_df.columns


def test_score_bounds(dataset):
    prs_df, _, _ = dataset
    assert prs_df["continuous_score"].between(0, 1).all()


def test_row_counts(dataset):
    prs_df, clinical_df, labels_df = dataset
    n_patients = 100
    n_loci = 15
    assert len(prs_df) == n_patients * n_loci
    assert len(clinical_df) == n_patients
    assert len(labels_df) == n_patients


def test_shared_loci_present(dataset):
    from polymas_ml.models.shared_features import SHARED_LOCI

    prs_df, _, _ = dataset
    present = set(prs_df["locus_id"].unique())
    assert bool(present & set(SHARED_LOCI.keys()))


def test_reproducibility(tmp_path: Path):
    prs1, _, _ = build_dataset(n_patients=50, n_loci=10, seed=1)
    prs2, _, _ = build_dataset(n_patients=50, n_loci=10, seed=1)
    pd.testing.assert_frame_equal(prs1, prs2)

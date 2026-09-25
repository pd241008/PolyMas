"""Tests for the Phase-1 data-substrate modules (F-10/F-07/F-08/F-06)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polymas_ml.data.genotypes import (
    SUPER_POPS,
    compute_ld_r2,
    sample_donor_genotypes,
    vcf_url,
)
from polymas_ml.data.haplotypes import (
    HLA_PAIR,
    assemble_patient_features,
    haplotype_features,
    load_substrate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_DIR = PROJECT_ROOT / "stash" / "results" / "real_genotypes_20260925"


def _synthetic_dosages(n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    dosages = pd.DataFrame(
        {
            "rsA": rng.integers(0, 3, n),
            "rsB": rng.integers(0, 3, n),
            "rsC": rng.integers(0, 3, n),
        },
        index=[f"HG{i:05d}" for i in range(n)],
    )
    meta = pd.DataFrame(
        {
            "pop": ["EUR"] * n,
            "super_pop": ["EUR"] * n,
        },
        index=dosages.index,
    )
    return dosages, meta


def test_vcf_url_naming() -> None:
    """Phase-3 release: autosomes v5b, chrX v1c (learned the hard way)."""
    assert "v5b" in vcf_url("1")
    assert "v1c" in vcf_url("X")


def test_ld_r2_identical_columns_is_one() -> None:
    dosages, _ = _synthetic_dosages()
    corr = compute_ld_r2(dosages[["rsA"]].assign(rsA_copy=dosages["rsA"]))
    assert corr.iloc[0, 1] == pytest.approx(1.0)


def test_sample_donor_genotypes_matches_ancestry() -> None:
    dosages, meta = _synthetic_dosages()
    meta.loc[meta.index[:30], "super_pop"] = "AFR"
    labels = pd.Series(["EUR"] * 5 + ["AFR"] * 5,
                       index=[f"P{i}" for i in range(10)])
    rng = np.random.default_rng(1)
    out = sample_donor_genotypes(dosages, meta, labels, rng)
    assert out.shape == (10, 3)
    for pid, anc in labels.items():
        donor = meta.loc[meta["super_pop"] == anc].index
        assert dosages.loc[donor].eq(out.loc[pid]).any(axis=1).any()


def test_haplotype_features_shape_and_values() -> None:
    dosages, _ = _synthetic_dosages()
    dosages.columns = pd.Index(list(HLA_PAIR) + ["rsC"])
    feats = haplotype_features(dosages)
    assert "hla_drb1_dqb1_both_carrier" in feats.columns
    assert (feats["hla_drb1_dqb1_both_carrier"].isin([0, 1])).all()


def test_assemble_patient_features_alignment() -> None:
    dosages, meta = _synthetic_dosages(40)
    rng = np.random.default_rng(2)
    pcs = pd.DataFrame(
        {"PC1": rng.normal(size=40), "PC2": rng.normal(size=40),
         "PC3": rng.normal(size=40), "PC4": rng.normal(size=40),
         "PC5": rng.normal(size=40), "PC6": rng.normal(size=40),
         "super_pop": meta["super_pop"]},
        index=dosages.index,
    )
    labels = pd.Series(["EUR"] * 10, index=[f"P{i}" for i in range(10)])
    feats = assemble_patient_features(dosages, pcs, labels, rng)
    assert len(feats) == 10
    assert sum(c.startswith("g_") for c in feats) == dosages.shape[1]
    assert "donor_ids" in feats.attrs
    # Each patient's g_* block must equal its donor's dosage row.
    donor_of = feats.attrs["donor_ids"]
    for pid in feats.index:
        np.testing.assert_array_equal(
            feats.loc[pid, [f"g_{c}" for c in dosages.columns]].to_numpy(dtype=float),
            dosages.loc[donor_of[pid]].to_numpy(dtype=float),
        )


def test_substrate_outputs_if_present() -> None:
    """The real-genotype substrate run (F-10) must satisfy the pre-registered
    LD validation gate before anything downstream consumes it."""
    if not (SUBSTRATE_DIR / "manifest.json").exists():
        pytest.skip("real genotype substrate not built yet")
    manifest = json.loads((SUBSTRATE_DIR / "manifest.json").read_text())
    ld = manifest["ld_validation"]
    assert ld["pass"] is True
    assert ld["n_within_tolerance"] / ld["n_pairs"] >= 0.9
    assert manifest["n_variants_typed"] >= 50

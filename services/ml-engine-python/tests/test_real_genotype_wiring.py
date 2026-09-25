"""Tests for the real-donor genotype wiring (F-10 pipeline half, ADR-004)."""

import numpy as np
import pandas as pd
import pytest

from polymas_ml.data.patients import simulate_genotypes_prs, simulate_labels

LOCI = {
    "rs2187668": "HLA-DRB1", "rs9272346": "HLA-DQB1", "rs2476601": "PTPN22",
    "rs3087243": "CTLA4", "rs2292239": "ERBB3", "rs11209026": "IL23R",
    "rs2104286": "IL2RA", "rs7574865": "STAT4",
}


def _donors(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {rs: rng.integers(0, 3, n) for rs in LOCI},
        index=[f"HG{i:05d}" for i in range(n)],
    )


def test_real_mode_returns_donor_map() -> None:
    donors = _donors()
    rng = np.random.default_rng(0)
    prs, gen, donor_ids = simulate_genotypes_prs(
        20, LOCI, [None] * 20, rng,
        donor_dosages=donors,
        donor_map=pd.Series(donors.index[:20], index=[f"P{i:04d}" for i in range(20)]),
    )
    assert donor_ids is not None and len(donor_ids) == 20
    # Every patient's genotype row must equal its donor's dosage row exactly.
    for i in range(20):
        donor = donor_ids.iloc[i]
        row = gen.set_index("patient_id").iloc[i]
        assert row["rs2476601"] == int(donors.at[donor, "rs2476601"])
        assert row["rs7574865"] == int(donors.at[donor, "rs7574865"])


def test_real_mode_prs_contract_unchanged() -> None:
    donors = _donors()
    rng = np.random.default_rng(1)
    prs, gen, _ = simulate_genotypes_prs(
        10, LOCI, [None] * 10, rng, donor_dosages=donors,
        donor_map=pd.Series(donors.index[:10], index=[f"P{i:04d}" for i in range(10)]),
    )
    assert set(prs.columns) == {"patient_id", "locus_id", "gene_symbol",
                                "continuous_score", "z_score", "pvalue", "genotype"}
    assert len(prs) == 10 * len(LOCI)


def test_real_mode_missing_locus_raises() -> None:
    donors = _donors().drop(columns=["rs11209026"])
    with pytest.raises(ValueError, match="missing panel loci"):
        simulate_genotypes_prs(5, LOCI, [None] * 5, np.random.default_rng(2),
                               donor_dosages=donors)


def test_legacy_simulated_mode_unchanged() -> None:
    rng = np.random.default_rng(3)
    prs, gen, donor_ids = simulate_genotypes_prs(30, LOCI, [None] * 30, rng)
    assert donor_ids is None
    # Simulated rs2476601 dosages are rare (q=0.08): mean must be well below 1.
    assert gen["rs2476601"].mean() < 1.0


def test_labels_real_mode_calibrated_by_panel_mean() -> None:
    """With common-allele donors (mean dosage ~2), the real-mode polygenic
    term must stay near the base prevalence instead of exploding — the
    raw-0.15 centering would clip everything to 0.95."""
    rng = np.random.default_rng(4)
    n = 60
    # Fixture covers the full legacy 8-locus panel: the label model reads
    # every disease's risk loci for every patient (SLE reads rs3087243,
    # MS reads rs2104286/rs2292239, etc.).
    donors = pd.DataFrame(
        {rs: [2] * n for rs in LOCI},
        index=[f"HG{i:05d}" for i in range(n)],
    )
    prs, gen, _ = simulate_genotypes_prs(
        n, LOCI, [None] * n, rng, donor_dosages=donors,
        donor_map=pd.Series(donors.index, index=[f"P{i:04d}" for i in range(n)]),
    )
    labels = simulate_labels(
        n, [None] * n, ["F"] * n, gen, LOCI, rng, genotype_mode="real"
    )
    # RA's base prevalence is 0.10; a common-allele donor pool must not push
    # it to the 0.95 clip (the historical centering would).
    assert labels["RA"].mean() < 0.5

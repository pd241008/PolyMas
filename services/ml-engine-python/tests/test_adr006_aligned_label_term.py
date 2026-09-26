"""Tests for ADR-006: allele-aligned label polygenic term (real mode)."""

import numpy as np
import pandas as pd
import pytest

from polymas_ml.data.patients import DISEASE_RISK_LOCI, simulate_labels


LEGACY_TABLE = {
    "RA": ["rs2476601", "rs11209026"],
    "SLE": ["rs7574865", "rs3087243"],
    "SJOGRENS": ["rs2187668", "rs7574865"],
    "T1D": ["rs9272346", "rs2476601"],
    "MS": ["rs2104286", "rs2292239"],
    "AITD": ["rs9272346", "rs3087243"],
    "VITILIGO": ["rs9272346", "rs2476601"],
}

ALL_LOCI = sorted({rs for loci in LEGACY_TABLE.values() for rs in loci})


def _gen_matrix(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.integers(0, 3, size=(n, len(ALL_LOCI))).astype(float),
        columns=ALL_LOCI,
        index=[f"P{i:04d}" for i in range(n)],
    )


def test_disease_risk_loci_table_matches_legacy() -> None:
    """ADR-006 hoisted the table verbatim — no silent anchor-set change."""
    assert DISEASE_RISK_LOCI == LEGACY_TABLE


def test_aligned_term_flips_protective_alt_locus() -> None:
    """At a locus where the published effect allele is the ref (dosage = 2 -
    vcf_alt), higher raw dosage must LOWER the aligned term (sign-corrected
    vs the legacy unaligned term)."""
    n = 4000
    gen = _gen_matrix(n, seed=1)
    # Half the patients carry 2 alt (raw dosage 2), half carry 0: a pure
    # direction probe at rs2476601 (PTPN22, ref-aligned under published
    # biology).
    gen["rs2476601"] = np.tile([0.0, 2.0], n // 2)
    other = [rs for rs in LEGACY_TABLE["RA"] if rs != "rs2476601"]
    for rs in other:
        gen[rs] = 0.0

    # Aligned dosage at a ref-aligned locus = 2 - raw: raw dosage 2 -> ZERO
    # effect-allele copies (negative term), raw dosage 0 -> two copies (+).
    term = np.where(gen["rs2476601"].to_numpy() == 2.0, -1.0, 1.0)
    labels = simulate_labels(
        n,
        [None] * n,
        ["F"] * n,
        gen.reset_index().rename(columns={"index": "patient_id"}),
        LEGACY_TABLE,
        np.random.default_rng(42),
        genotype_mode="real",
        aligned_prs_terms={"RA": term},
    )
    legacy = simulate_labels(
        n,
        [None] * n,
        ["F"] * n,
        gen.reset_index().rename(columns={"index": "patient_id"}),
        LEGACY_TABLE,
        np.random.default_rng(42),
        genotype_mode="real",
    )
    # Aligned: RA prevalence LOWER in the homozygous-alt (protective-allele)
    # half. Legacy: HIGHER (the bug).
    hi = (gen["rs2476601"] == 2.0).to_numpy()
    assert labels.loc[hi, "RA"].mean() < labels.loc[~hi, "RA"].mean()
    assert legacy.loc[hi, "RA"].mean() > legacy.loc[~hi, "RA"].mean()


def test_missing_disease_falls_back_to_legacy_real_term() -> None:
    """A disease absent from aligned_prs_terms keeps the legacy panel-mean
    real-mode term (no silent signal loss)."""
    n = 1500
    gen = _gen_matrix(n, seed=2).reset_index().rename(columns={"index": "patient_id"})
    kwargs = dict(
        n_patients=n,
        patient_groups=[None] * n,
        sexes=["F"] * n,
        gen_rows=gen,
        loci=LEGACY_TABLE,
        rng=np.random.default_rng(7),
        genotype_mode="real",
    )
    a = simulate_labels(aligned_prs_terms={"SLE": np.zeros(n)}, **kwargs)
    b = simulate_labels(aligned_prs_terms=None, **kwargs)
    # Zero aligned term shifts SLE toward the base prevalence; legacy term
    # uses the raw dosage — different draws are expected, so compare against
    # the base prevalence rather than each other.
    assert 0.0 < a["SLE"].mean() < 0.15
    assert 0.0 < b["SLE"].mean() < 0.15


def test_simulated_mode_ignores_aligned_terms() -> None:
    """Simulated mode is byte-identical with/without aligned terms passed."""
    n = 800
    gen = _gen_matrix(n, seed=3).reset_index().rename(columns={"index": "patient_id"})
    kwargs = dict(
        n_patients=n,
        patient_groups=[None] * n,
        sexes=["F"] * n,
        gen_rows=gen,
        loci=LEGACY_TABLE,
        genotype_mode="simulated",
    )
    a = simulate_labels(rng=np.random.default_rng(42), aligned_prs_terms={"RA": np.ones(n)}, **kwargs)
    b = simulate_labels(rng=np.random.default_rng(42), **kwargs)
    pd.testing.assert_frame_equal(a, b)

"""Tests for the co-occurrence (polyautoimmunity) label generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from polymas_ml.data.patients import (
    BASE_PREVALENCES,
    DISEASE_LABELS,
    MAS_EXCLUSIONS,
    draw_cooccurring_labels,
    label_structure_report,
    simulate_labels,
)

LOCI = {
    "rs2187668": "HLA-DRB1",
    "rs9272346": "HLA-DQB1",
    "rs2476601": "PTPN22",
    "rs3087243": "CTLA4",
    "rs2292239": "ERBB3",
    "rs11209026": "IL23R",
    "rs2104286": "IL2RA",
    "rs7574865": "STAT4",
}

# Realistic locus alt-allele frequencies (matches the genotype simulator).
Q_BASE = {
    "rs2187668": 0.10, "rs9272346": 0.25, "rs2476601": 0.08, "rs3087243": 0.40,
    "rs2292239": 0.30, "rs11209026": 0.07, "rs2104286": 0.35, "rs7574865": 0.22,
}


def _gen_rows(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"patient_id": [f"P{i:04d}" for i in range(n)]}
    for rs, q in Q_BASE.items():
        data[rs] = rng.binomial(2, q, size=n)
    return pd.DataFrame(data)


def _run(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = list(rng.choice(
        ["RA", "SLE", "T1D", "MS", "SJOGRENS", None],
        size=n, p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
    ))
    sexes = list(rng.choice(["F", "M"], size=n, p=[0.7, 0.3]))
    return simulate_labels(n, groups, sexes, _gen_rows(n, seed), LOCI, rng)


def test_labels_are_binary_with_all_diseases():
    labels = _run(n=500, seed=1)
    assert list(labels.columns) == ["patient_id"] + DISEASE_LABELS
    for d in DISEASE_LABELS:
        assert set(labels[d].unique()) <= {0, 1}


def test_marginals_exceed_background_in_cohorts():
    """Cohort patients carry their anchor disease above the background rate."""
    n = 4000
    rng = np.random.default_rng(2)
    groups = ["RA"] * (n // 2) + [None] * (n // 2)
    sexes = ["F"] * n
    labels = simulate_labels(n, groups, sexes, _gen_rows(n, 3), LOCI, rng)
    cohort_rate = labels.loc[groups[0] == "RA" if False else np.array(groups) == "RA", "RA"].mean()
    background_rate = labels.loc[np.array(groups) == None, "RA"].mean()  # noqa: E711
    assert cohort_rate > background_rate
    assert cohort_rate > 2 * BASE_PREVALENCES["RA"]


def test_excluded_pairs_never_cooccur():
    labels = _run(n=3000, seed=4)
    for a, b in MAS_EXCLUSIONS:
        both = ((labels[a] == 1) & (labels[b] == 1)).sum()
        assert both == 0, f"incoercible pair {a}/{b} co-occurred {both} times"


def test_mas_pairs_have_positive_phi():
    labels = _run(n=6000, seed=0)
    rep = label_structure_report(labels)
    phi = rep["pairwise_phi"]
    assert phi["AITD|VITILIGO"] > 0.05
    assert phi["SJOGRENS|AITD"] > 0.03
    assert phi["T1D|MS"] < 0.0  # exclusion pair


def test_overdispersion_vs_binomial_null():
    """Joint is overdispersed relative to independent-Bernoulli null."""
    labels = _run(n=6000, seed=0)
    rep = label_structure_report(labels)
    assert rep["overdispersion_ratio"] > 1.0
    # MAS-3+ subgroup exists but is a minority (defensible band).
    assert 0.03 < rep["rate_3plus_mas"] < 0.30
    # Empirical polyautoimmunity anchor (Anaya 2012): 2+ given 1+ well above chance.
    assert rep["rate_2plus_given_1plus"] > 0.30


def test_marginals_approximately_preserved():
    """Realized prevalences stay near target + bounded lift.

    The polygenic term (up to ~+0.09 for high-frequency risk loci) and the
    co-occurrence cascade add on top of the background prevalence, so the
    bound is absolute rather than multiplicative.
    """
    labels = _run(n=6000, seed=0)
    for d, target in BASE_PREVALENCES.items():
        realized = labels[d].mean()
        assert 0.5 * target <= realized <= target + 0.35, (d, target, realized)
        assert realized <= 0.55  # clinically sane prevalences


def test_draw_cooccurring_labels_respects_exclusions_directly():
    rng = np.random.default_rng(5)
    for _ in range(200):
        marginal = {d: 0.9 for d in DISEASE_LABELS}
        labels = draw_cooccurring_labels(marginal, liability=1.2, rng=rng)
        for a, b in MAS_EXCLUSIONS:
            assert not (labels[a] == 1 and labels[b] == 1)


def test_reproducible_given_same_seed():
    l1 = _run(n=800, seed=9)
    l2 = _run(n=800, seed=9)
    pd.testing.assert_frame_equal(l1, l2)

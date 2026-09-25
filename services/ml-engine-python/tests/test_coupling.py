"""Tests for ADR-005 published-effect-size coupling (polymas_ml.data.coupling)."""

import numpy as np
import pandas as pd
import pytest

from polymas_ml.data.coupling import (
    AlignedLocus,
    DiseaseCoupling,
    aligned_dosage_matrix,
    build_disease_coupling,
    donor_weights_for_anchor,
    identify_vcf_alt,
    label_prs_term,
)


# ---------------------------------------------------------------------------
# identify_vcf_alt: frequency matching identifies the VCF alt allele
# ---------------------------------------------------------------------------

def _freqs(rs: str, f_a: float) -> dict:
    return {rs: {"freqs_1kg_all": {"A": f_a, "G": round(1 - f_a, 6)}}}


def test_identify_vcf_alt_matches_panel_af() -> None:
    # Panel alt AF 0.094 -> matches allele A (Ensembl ALL 0.09), not G (0.91).
    panel_af = pd.Series({"rs2476601": 0.094})
    alt = identify_vcf_alt(panel_af, _freqs("rs2476601", 0.0944))
    assert alt["rs2476601"] == "A"


def test_identify_vcf_alt_drops_ambiguous() -> None:
    # Panel AF 0.50 matches both alleles within the ambiguity tolerance.
    panel_af = pd.Series({"rsX": 0.50})
    alt = identify_vcf_alt(panel_af, _freqs("rsX", 0.51))
    assert alt["rsX"] == ""


def test_identify_vcf_alt_drops_unresolvable() -> None:
    # Panel AF far from both Ensembl frequencies -> unresolvable.
    panel_af = pd.Series({"rsY": 0.30})
    alt = identify_vcf_alt(panel_af, _freqs("rsY", 0.05))
    assert alt["rsY"] == ""


# ---------------------------------------------------------------------------
# build_disease_coupling: allele alignment flips dosage direction correctly
# ---------------------------------------------------------------------------

def test_coupling_aligns_effect_allele_to_ref() -> None:
    # ea = G but VCF alt = A (the PTPN22 trap): aligned_to must be "ref".
    panel_af = pd.Series({"rs2476601": 0.094})
    freqs = _freqs("rs2476601", 0.0944)
    vcf_alt = {"rs2476601": "A"}
    coup = build_disease_coupling(
        "RA", panel_af, freqs, vcf_alt, None, "test")
    assert len(coup.loci) == 1
    assert coup.loci[0].aligned_to == "ref"
    assert coup.loci[0].effect_allele == "G"
    assert coup.loci[0].beta == pytest.approx(-0.593327)


def test_coupling_aligns_effect_allele_to_alt() -> None:
    panel_af = pd.Series({"rs11209026": 0.30})
    freqs = _freqs("rs11209026", 0.30)
    vcf_alt = {"rs11209026": "A"}  # ea == vcf alt for the IL23R fixture row
    coup = build_disease_coupling(
        "RA", panel_af, freqs, vcf_alt, None, "test")
    assert len(coup.loci) == 1
    assert coup.loci[0].aligned_to == "alt"


def test_coupling_eaf_audit_rejects_mismatch() -> None:
    panel_af = pd.Series({"rs11209026": 0.30})
    freqs = _freqs("rs11209026", 0.30)
    vcf_alt = {"rs11209026": "A"}
    # Published eaf 0.9 vs ensembl 0.30 -> locus must be dropped.
    coup = build_disease_coupling(
        "RA", panel_af, freqs, vcf_alt, None, "test",
        ea_audit={"rs11209026": "0.9"})
    assert coup.loci == [] and "rs11209026" in coup.dropped


@pytest.fixture
def _patch_assoc_loader(monkeypatch):
    """Route load_assoc_rows to an in-memory fixture."""
    from polymas_ml.data import coupling

    def fake_load(results_root, tag, dataset):
        return FAKE_ROWS.get(dataset, [])

    monkeypatch.setattr(coupling, "load_assoc_rows", fake_load)


# Route the RA tests through the fixture (build_disease_coupling uses the
# module-level loader; monkeypatching it avoids any filesystem fixture).
for _name in (
    "test_coupling_aligns_effect_allele_to_ref",
    "test_coupling_aligns_effect_allele_to_alt",
    "test_coupling_eaf_audit_rejects_mismatch",
):
    globals()[_name] = pytest.mark.usefixtures("_patch_assoc_loader")(
        globals()[_name])


FAKE_ROWS = {
    "ieu-a-833": [
        {"rsid": "rs2476601", "beta": -0.593327, "p": 1e-149, "ea": "G", "nea": "A"},
        {"rsid": "rs11209026", "beta": 0.10, "p": 5e-9, "ea": "A", "nea": "G"},
    ],
}


# ---------------------------------------------------------------------------
# Donor weighting: Bayes-consistent case enrichment
# ---------------------------------------------------------------------------

def _coupling(rs: str, beta: float, aligned_to: str) -> DiseaseCoupling:
    return DiseaseCoupling(
        disease="TEST", dataset="x",
        loci=[AlignedLocus(rs, beta, "A", 1e-20, aligned_to)],
    )


def test_weights_enrich_high_dosage_donors() -> None:
    rng = np.random.default_rng(0)
    dos = pd.DataFrame(
        {"rsX": rng.integers(0, 3, 200).astype(float)},
        index=[f"D{i}" for i in range(200)],
    )
    coup = _coupling("rsX", beta=0.6, aligned_to="alt")
    aligned = aligned_dosage_matrix(dos, coup)
    pool = dos.index.to_numpy()
    w = donor_weights_for_anchor(coup, aligned, pool)
    assert w.sum() == pytest.approx(1.0)
    # Donors with aligned dosage 2 must get more weight than dosage 0.
    w2 = w[dos["rsX"].to_numpy() == 2].mean()
    w0 = w[dos["rsX"].to_numpy() == 0].mean()
    assert w2 > w0


def test_weights_flip_with_ref_alignment() -> None:
    rng = np.random.default_rng(1)
    dos = pd.DataFrame(
        {"rsX": rng.integers(0, 3, 200).astype(float)},
        index=[f"D{i}" for i in range(200)],
    )
    coup = _coupling("rsX", beta=0.6, aligned_to="ref")
    aligned = aligned_dosage_matrix(dos, coup)
    # Aligned dosage must be the mirror: 2 - vcf dosage.
    assert aligned["rsX"].iloc[0] == pytest.approx(2.0 - dos["rsX"].iloc[0])
    pool = dos.index.to_numpy()
    w = donor_weights_for_anchor(coup, aligned, pool)
    # Now LOW vcf dosage donors carry the effect allele -> heavier weight.
    w0 = w[dos["rsX"].to_numpy() == 0].mean()
    w2 = w[dos["rsX"].to_numpy() == 2].mean()
    assert w0 > w2


def test_weights_uniform_without_loci() -> None:
    dos = pd.DataFrame({"rsX": [0.0, 1.0, 2.0]}, index=["a", "b", "c"])
    coup = DiseaseCoupling(disease="T", dataset="x", loci=[])
    aligned = aligned_dosage_matrix(dos, coup)
    w = donor_weights_for_anchor(coup, aligned, dos.index.to_numpy())
    assert np.allclose(w, 1 / 3)


def test_weighted_sampling_recovers_expected_enrichment() -> None:
    """End-to-end sanity: weighted draws must shift cohort dosage by roughly
    the published-OR prediction (the G1 gate logic in miniature)."""
    rng = np.random.default_rng(2)
    n = 4000
    dos = pd.DataFrame(
        {"rsX": rng.binomial(2, 0.10, n).astype(float)},  # p=0.10
        index=[f"D{i}" for i in range(n)],
    )
    beta = 0.6  # OR ~1.82 per allele
    coup = _coupling("rsX", beta, "alt")
    aligned = aligned_dosage_matrix(dos, coup)
    pool = dos.index.to_numpy()
    w = donor_weights_for_anchor(coup, aligned, pool)
    draws = rng.choice(pool, size=2000, p=w)
    case_mean = dos.loc[draws, "rsX"].mean()
    bg_mean = dos["rsX"].mean()
    # Prediction under the log-additive model: E[G|case] - E[G] ~
    # 2p(1-p)*2*beta / (1 + ...) — just assert a positive, material shift
    # and that it is bounded (no fake enrichment).
    assert case_mean > bg_mean
    assert case_mean - bg_mean < 1.0  # dosage scale guard


# ---------------------------------------------------------------------------
# Label PRS term: per-SD published PRS from patient genotypes
# ---------------------------------------------------------------------------

def test_label_prs_term_zero_centered_and_scaled() -> None:
    rng = np.random.default_rng(3)
    n = 3000
    gen = pd.DataFrame({"rsX": rng.binomial(2, 0.10, n).astype(float)})
    panel_af = pd.Series({"rsX": gen["rsX"].mean() / 2})
    coup = _coupling("rsX", 0.6, "alt")
    prs = label_prs_term(gen, coup, panel_af)
    # Centered on the panel => mean ~ 0 (panel AF == empirical AF here).
    assert abs(prs.mean()) < 0.05
    # Per-SD => std ~ 1
    assert prs.std() == pytest.approx(1.0, abs=0.15)


def test_label_prs_term_respects_ref_alignment() -> None:
    gen = pd.DataFrame({"rsX": [0.0, 2.0]})
    panel_af = pd.Series({"rsX": 0.10})
    coup_ref = _coupling("rsX", 0.6, "ref")
    coup_alt = _coupling("rsX", 0.6, "alt")
    prs_ref = label_prs_term(gen, coup_ref, panel_af)
    prs_alt = label_prs_term(gen, coup_alt, panel_af)
    # Patient with vcf dosage 2: aligned 2 under alt, 0 under ref.
    assert prs_alt[1] > prs_ref[1]
    # And the two are exact mirrors.
    assert prs_ref[1] == pytest.approx(-prs_alt[1], abs=1e-9)

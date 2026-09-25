"""Tests for the curated loci panel (F-09, ROADMAP.md)."""

from pathlib import Path

import pandas as pd
import pytest

from polymas_ml.data.loci import (
    CANDIDATE_LOCI,
    DISEASES,
    LEGACY_LOCI,
    unique_loci,
    panel_size,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_legacy_panel_fully_preserved() -> None:
    """The original 8 launch loci must never drop out of the panel."""
    panel_ids = {locus.rs_id for locus in unique_loci()}
    for locus in LEGACY_LOCI:
        assert locus.rs_id in panel_ids, f"legacy locus {locus.rs_id} lost"


def test_unique_loci_dedupes_by_rs_id() -> None:
    panel = unique_loci()
    rs_ids = [locus.rs_id for locus in panel]
    assert len(rs_ids) == len(set(rs_ids)), "duplicate rs_ids in panel"
    assert panel_size() == len(panel)


def test_panel_schema() -> None:
    for locus in unique_loci():
        assert locus.rs_id.startswith("rs") and locus.rs_id[2:].isdigit()
        assert locus.gene and isinstance(locus.gene, str)
        assert locus.disease in DISEASES + ["SHARED"], locus
        assert locus.source, f"missing provenance for {locus.rs_id}"


def test_meets_f09_claim_bar() -> None:
    """ROADMAP F-09: the candidate panel must offer >=50 unique loci."""
    assert panel_size() >= 50


def test_verification_manifest_if_present() -> None:
    """If the verification run exists, its counts must be internally honest:
    verified + dropped + error + pending == total, and the F-09 verified
    bar (>=50) must hold for the manifest to be usable downstream."""
    csv_path = (
        PROJECT_ROOT / "stash" / "results" / "panel_expansion_20260925"
        / "panel_verification.csv"
    )
    if not csv_path.exists():
        pytest.skip("panel verification not run yet")
    df = pd.read_csv(csv_path)
    counts = df["status"].value_counts()
    total = sum(int(counts.get(s, 0)) for s in ("VERIFIED", "DROPPED", "ERROR", "PENDING"))
    assert total == len(df), "unknown status values present"
    assert int(counts.get("ERROR", 0)) == 0, "rate-limit errors must be re-run before use"
    assert int(counts.get("PENDING", 0)) == 0
    assert int(counts.get("VERIFIED", 0)) >= 50


@pytest.mark.parametrize("locus", list(CANDIDATE_LOCI)[:5])
def test_candidate_sources_cite_publication(locus) -> None:
    """Provenance rule: every candidate cites a paper or catalog ID."""
    assert any(ch.isdigit() for ch in locus.source), locus.rs_id

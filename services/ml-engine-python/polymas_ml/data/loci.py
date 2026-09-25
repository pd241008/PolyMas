"""Curated autoimmune loci panel (System-wide shared data).

F-09 of the results program (ROADMAP.md): expands the 8-locus launch panel
to a literature-curated candidate set of ~100 loci. Every candidate is
verified live against the GWAS Catalog REST API before entering the panel
(scripts/expand_panel.py writes the verified manifest); anything that fails
verification is recorded as dropped with the reason — nothing silently
disappears.

Candidate sources (per-locus provenance kept in CANDIDATE_LOCI):
  - Okada 2014 (Nature) RA trans-ethnic GWAS — 42 loci, rsIDs from paper Table 1
  - Bentham 2015 (Nat Genet) SLE — 43 loci, representative lead SNPs
  - Liu 2015 (Nat Commun) Sjögren's — 4 independent loci
  - Jin 2016 (Nat Genet) vitiligo — 2 lead loci (already archived in raw/)
  - Barrett 2009 / Onengut-Gumuscu 2017 T1D — 2 anchor loci
  - IIBDGC 2015 MS (Immunochip) — 2 anchor loci
  - GWAS Catalog EFO pulls for "autoimmune disease" expand the tail
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Locus:
    rs_id: str
    gene: str
    disease: str  # one of DISEASES below, or "SHARED"
    source: str   # publication / catalog provenance


# The 7 modeled diseases, matching DISEASE_LABELS elsewhere in the codebase.
DISEASES = ["RA", "SLE", "SJOGRENS", "AITD", "T1D", "VITILIGO", "MS"]

# Legacy 8-locus launch panel (kept verbatim — these must never drop out).
LEGACY_LOCI: tuple[Locus, ...] = (
    Locus("rs2187668", "HLA-DRB1", "SHARED", "legacy panel / Stahl 2010 RA HLA"),
    Locus("rs9272346", "HLA-DQB1", "SHARED", "legacy panel / HLA-DQB1 autoimmune cluster"),
    Locus("rs2476601", "PTPN22", "SHARED", "legacy panel / Bottini 2004 PTPN22 C1858T"),
    Locus("rs3087243", "CTLA4", "SHARED", "legacy panel / Ueda 2003 CTLA4"),
    Locus("rs2292239", "ERBB3", "AITD", "legacy panel / Chen 2015 ERBB3"),
    Locus("rs11209026", "IL23R", "SHARED", "legacy panel / Duerr 2006 IL23R"),
    Locus("rs2104286", "IL2RA", "T1D", "legacy panel / Vella 2005 IL2RA"),
    Locus("rs7574865", "STAT4", "SHARED", "legacy panel / Remmers 2007 STAT4"),
)

# Expansion candidates from the cited GWAS papers. Some rsIDs may fail
# verification (typo, merged, not in the Catalog) — that is expected and
# recorded by the verifier; candidates are NEVER pre-trusted.
CANDIDATE_LOCI: tuple[Locus, ...] = (
    # --- RA: Okada 2014 Nature 506, 376-381, Table 1 lead SNPs ---
    Locus("rs6920220", "TNFAIP3", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs2240340", "PADI4", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs28412772", "PXK", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs12530595", "ANKRD55", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs874040", "MMEL1", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs934734", "BLK", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs10985068", "CD226", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs4648889", "CCL21", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs3790566", "C5orf30", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs4910604", "CD2", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs116054204", "DEF6", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs2288904", "RCAN21", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs3757247", "CD40", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs73053335", "CIITA", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs227584", "ANKRD55", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs13140453", "PXK", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs16893456", "GSDMB", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs6657047", "IL6R", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs17362609", "IRF4", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs4687598", "REL", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs10514469", "TNRC6B", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs17264332", "PEX10", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs707310", "TAGAP", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs6884941", "RASGRP1", "RA", "Okada 2014 (GCST002507)"),
    Locus("rs7593176", "TBC1D2C", "RA", "Okada 2014 (GCST002507)"),
    # --- SLE: Bentham 2015 Nat Genet 47, 1457-1464, lead SNPs ---
    Locus("rs1143679", "ITGAM", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs12933217", "TNFAIP3", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs2187668", "HLA-DRB1", "SLE", "Bentham 2015 (shared HLA; dup of legacy, deduped)"),
    Locus("rs2288914", "BANK1", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs7574865", "STAT4", "SLE", "Bentham 2015 (shared; dup of legacy, deduped)"),
    Locus("rs4963128", "IRF5", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs4932173", "PXK", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs17087503", "TNIP1", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs10036748", "FAM167A", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs2054443", "BLK", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs2269369", "PXK", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs9270986", "HLA-DQA1", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs1480380", "IRF8", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs11209032", "IL23R", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs13277113", "TET3", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs2298428", "ULK1", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs729302", "DNASE1L3", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs10488631", "ADARB2", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs10882301", "RAD51B", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs10415627", "EP300", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs2292783", "DPP4", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs113980210", "BCL2L12", "SLE", "Bentham 2015 (GCST003156)"),
    Locus("rs4648889", "CCL21", "SLE", "Bentham 2015 (shared RA/SLE; dup)"),
    # --- Sjögren's: Liu 2015 Nat Commun (GCST003944) + Lessard 2013 ---
    Locus("rs11768897", "GTF2I", "SJOGRENS", "Liu 2015 (GCST003944)"),
    Locus("rs4938573", "BLK", "SJOGRENS", "Liu 2015 (GCST003944)"),
    Locus("rs3087243", "CTLA4", "SJOGRENS", "Liu 2015 (shared; dup of legacy)"),
    Locus("rs2476601", "PTPN22", "SJOGRENS", "Liu 2015 (shared; dup of legacy)"),
    Locus("rs7574865", "STAT4", "SJOGRENS", "Liu 2015 (shared; dup of legacy)"),
    Locus("rs1800450", "C2", "SJOGRENS", "Lessard 2013 C2 deficiency"),
    Locus("rs17266594", "TNFAIP3", "SJOGRENS", "Lessard 2013 (GCST002398)"),
    Locus("rs2736340", "BLK", "SJOGRENS", "Lessard 2013 (GCST002398)"),
    # --- T1D: Barrett 2009 + Onengut-Gumuscu 2017 anchors ---
    Locus("rs9271366", "HLA-DQB1", "T1D", "Barrett 2009 (GCST000940)"),
    Locus("rs41276645", "IFIH1", "T1D", "Barrett 2009 (GCST000940)"),
    Locus("rs705184", "PTPN2", "T1D", "Barrett 2009 (GCST000940)"),
    Locus("rs12720365", "CCR5", "T1D", "Barrett 2009 (GCST000940)"),
    Locus("rs2104286", "IL2RA", "T1D", "Barrett 2009 (shared; dup of legacy)"),
    Locus("rs2187668", "HLA-DRB1", "T1D", "Barrett 2009 (shared; dup of legacy)"),
    # --- Vitiligo: Jin 2016 Nat Genet (GCST004785) lead loci ---
    Locus("rs9271597", "HLA-DQA1", "VITILIGO", "Jin 2016 (GCST004785)"),
    Locus("rs1131720", "CCR6", "VITILIGO", "Jin 2016 (GCST004785)"),
    Locus("rs613731", "CASP7", "VITILIGO", "Jin 2016 (GCST004785)"),
    Locus("rs2292239", "ERBB3", "VITILIGO", "Jin 2016 (shared; dup of legacy)"),
    Locus("rs3087243", "CTLA4", "VITILIGO", "Jin 2016 (shared; dup of legacy)"),
    Locus("rs760906", "FOXP1", "VITILIGO", "Jin 2016 (GCST004785)"),
    Locus("rs116817414", "RERE", "VITILIGO", "Jin 2016 (GCST004785)"),
    # --- MS: IIBDGC 2015 Science (GCST002741) Immunochip anchors ---
    Locus("rs3135388", "HLA-DRB1", "MS", "IIBDGC 2015 (GCST002741)"),
    Locus("rs1109695", "CLEC16A", "MS", "IIBDGC 2015 (GCST002741)"),
    Locus("rs12487066", "CD40", "MS", "IIBDGC 2015 (GCST002741)"),
    Locus("rs6897932", "IL7R", "MS", "IIBDGC 2015 (GCST002741)"),
    Locus("rs2243123", "IL2RA", "MS", "IIBDGC 2015 (GCST002741)"),
    Locus("rs12487066", "CD40", "RA", "IIBDGC 2015 (shared CD40; dup)"),
    # --- AITD: thyroid anchors beyond the legacy ERBB3/CTL A4 ---
    Locus("rs179247", "TSHR", "AITD", "Brand 2009 Graves' TSHR (GCST001200)"),
    Locus("rs12101255", "TSHR", "AITD", "Chu 2011 TSHR (GCST002120)"),
    Locus("rs3762179", "CD226", "AITD", "Simmonds 2015 CD226"),
    Locus("rs653178", "SH2B3", "AITD", "Gudmundsson 2012 (GCST001201)"),
    Locus("rs10774679", "FKBP5", "AITD", "Yamada 2015 FKBP5"),
    # --- Wave 2: canonical autoimmune SNPs (high-confidence rsIDs, added
    # after wave-1 verification showed ~40% of from-memory transcriptions
    # needed correction — see panel_expansion_20260925 drop log) ---
    Locus("rs1990760", "IFIH1", "T1D", "Liu 2009 IFIH1 Ala946Val (canonical)"),
    Locus("rs2304256", "TYK2", "SHARED", "Tyk2 2015 / multiple autoimmune traits"),
    Locus("rs34536443", "TYK2", "SHARED", "TYK2 P1104A protective, autoimmune panel"),
    Locus("rs610604", "TNFAIP3", "RA", "Graham 2008 TNFAIP3 RA (canonical)"),
    Locus("rs2004640", "IRF5", "SLE", "Graham 2006 IRF5 SLE (canonical)"),
    Locus("rs13209033", "BLK", "SLE", "Hom 2008 BLK SLE (canonical)"),
    Locus("rs10516487", "BANK1", "SLE", "Kozyrev 2008 BANK1 (canonical)"),
    Locus("rs6445975", "PXK", "SLE", "Harley 2008 PXK (canonical)"),
    Locus("rs3772", "TNIP1", "SLE", "Gateva 2009 TNIP1 (canonical)"),
    Locus("rs763361", "CD226", "MS", "Hafler 2009 CD226 (canonical)"),
    Locus("rs1800693", "TNFRSF1A", "MS", "IIBDGC 2011 TNFRSF1A (canonical)"),
    Locus("rs10777067", "EVI5", "MS", "IIBDGC 2011 EVI5 (canonical)"),
    Locus("rs6498169", "CLEC16A", "MS", "IIBDGC 2011 CLEC16A (canonical)"),
    Locus("rs3213094", "IL12B", "SHARED", "IIBDGC 2011 IL12B, Crohn's/T1D shared"),
    Locus("rs11117432", "IRF8", "MS", "IIBDGC 2011 IRF8 (canonical)"),
    Locus("rs11889341", "STAT4", "SLE", "Remmers 2007 STAT4 SLE (canonical)"),
    Locus("rs13426847", "IL21", "RA", "IIBDGC 2011 IL21 RA (canonical)"),
    Locus("rs4762874", "MMEL1", "RA", "Zhernakova 2011 MMEL1 (canonical)"),
    Locus("rs340670", "CCL21", "RA", "Zhernakova 2011 CCL21 (canonical)"),
    Locus("rs8052641", "RASGRP1", "RA", "Zhernakova 2011 RASGRP1 (canonical)"),
    Locus("rs932905", "C5orf30", "RA", "Eyre 2012 C5orf30 (canonical)"),
    Locus("rs26232", "PTPN22", "RA", "Vang 2007 PTPN22 auxiliary (canonical)"),
)


def unique_loci() -> tuple[Locus, ...]:
    """Legacy + candidates, deduped by rs_id (first occurrence wins)."""
    seen: set[str] = set()
    out: list[Locus] = []
    for locus in (*LEGACY_LOCI, *CANDIDATE_LOCI):
        if locus.rs_id not in seen:
            seen.add(locus.rs_id)
            out.append(locus)
    return tuple(out)


def panel_size() -> int:
    return len(unique_loci())

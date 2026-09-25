"""Phase-1 derived features (F-06, F-07, F-08).

F-06: DRB1/DQB1 haplotype proxies + pairwise epistasis terms for System A.
      With array data (no phase info) the haplotype signal enters as
      co-carriage features — dosage pairs at the two HLA loci — which is
      what a GBM can actually consume without phasing.
F-07: ancestry PCs from the real 1000G genotype matrix (computed in
      scripts/build_real_genotype_substrate.py; this module just loads them
      and provides the patient-aligned covariate frame).
F-08: per-locus functional annotation (consequence + gene via Ensembl
      overlap lookup) joined onto the panel.

Feature frames are donor-indexed: each simulated patient inherits the
real genotype vector of a matched-ancestry 1000G donor (F-10), so all
features here describe REAL variation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

ENSEMBL_OVERLAP_URL = "https://grch37.rest.ensembl.org/overlap/region/human/{region}?feature=gene;content-type=application/json"

# The two classic HLA class-II loci on the panel (F-06).
HLA_PAIR = ("rs2187668", "rs9271366")  # DRB1-associated, DQB1-associated


def haplotype_features(genotypes: pd.DataFrame) -> pd.DataFrame:
    """F-06: HLA co-carriage columns + pairwise epistasis terms.

    genotypes: index=patient, columns=rs_id, values 0/1/2 dosages.
    """
    out = pd.DataFrame(index=genotypes.index)
    rs_a, rs_b = HLA_PAIR
    if rs_a in genotypes and rs_b in genotypes:
        a, b = genotypes[rs_a], genotypes[rs_b]
        out["hla_drb1_dqb1_dosage_sum"] = a + b
        out["hla_drb1_dqb1_both_carrier"] = ((a > 0) & (b > 0)).astype(int)
        out["hla_drb1_dqb1_double_dose"] = ((a == 2) & (b == 2)).astype(int)
    # Epistasis pairs among the shared autoimmune anchors (top literature pairs).
    epistasis_pairs = [
        ("rs2476601", "rs3087243"),  # PTPN22 x CTLA4 (T-cell activation axis)
        ("rs7574865", "rs2104286"),  # STAT4 x IL2RA
        ("rs11209026", "rs7574865"), # IL23R x STAT4
    ]
    for a, b in epistasis_pairs:
        if a in genotypes and b in genotypes:
            tag = f"{a}_x_{b}"
            out[f"epi_{tag}"] = genotypes[a] * genotypes[b]
            out[f"epi_{tag}_carrier"] = ((genotypes[a] > 0) & (genotypes[b] > 0)).astype(int)
    return out


def gene_annotation(chrom: str, start: int, end: int | None = None) -> dict:
    """F-08: nearest/overlapping gene via Ensembl GRCh37 overlap (one region,
    symbol + biotype only — keeps the provenance chain simple)."""
    end = end or start
    region = f"{chrom}:{start}-{max(end, start + 1)}"
    try:
        resp = requests.get(
            ENSEMBL_OVERLAP_URL.format(region=region),
            headers={"Content-Type": "application/json"}, timeout=30,
        )
        resp.raise_for_status()
        genes = resp.json()
    except (requests.RequestException, ValueError):
        return {}
    if not genes:
        return {}
    g = genes[0]
    return {"gene_id": g.get("id"), "gene_symbol": g.get("external_name"),
            "gene_biotype": g.get("biotype")}


def annotate_variants(variant_annotations: pd.DataFrame, throttle: float = 0.35) -> pd.DataFrame:
    """F-08: attach overlapping-gene annotation to the variant table from
    the real-genotype substrate run."""
    rows = []
    for i, row in variant_annotations.iterrows():
        info = gene_annotation(str(row["chrom"]), int(row["pos_grch37"]))
        rows.append({**row.to_dict(), **info})
        if (i + 1) % 10 == 0:
            logger.info("annotated %d/%d", i + 1, len(variant_annotations))
        import time
        time.sleep(throttle)
    return pd.DataFrame(rows)


def assemble_patient_features(
    dosages: pd.DataFrame,
    pcs: pd.DataFrame,
    ancestry_labels: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One frame, one row per simulated patient: donor dosages (F-10) +
    donor PCs (F-07) + haplotype/epistasis features (F-06).

    Donors are sampled per patient, matched on super-population; the donor's
    real dosage vector AND the donor's PCs both attach to the patient, so
    every feature column describes real measured variation.
    """
    pools = {
        sp: pcs[pcs["super_pop"] == sp].index.to_numpy()
        for sp in pcs["super_pop"].unique()
    }
    fallback = pcs.index.to_numpy()

    donor_ids = {}
    for pid, anc in ancestry_labels.items():
        pool = pools.get(str(anc))
        if pool is None or len(pool) == 0:
            pool = fallback
        donor_ids[pid] = rng.choice(pool)

    donor_series = pd.Series(donor_ids)              # patient -> donor id
    donor_gt = dosages.loc[donor_series.values]      # aligned to patient order
    donor_gt.index = donor_series.index
    donor_pcs = pcs.loc[donor_series.values, [f"PC{i}" for i in range(1, 7)]]
    donor_pcs.index = donor_series.index

    feats = pd.concat(
        [donor_gt.add_prefix("g_"), donor_pcs, haplotype_features(donor_gt)],
        axis=1,
    )
    feats.attrs["donor_ids"] = donor_series          # provenance: patient -> donor
    return feats


def load_substrate(results_root: Path, run_tag: str = "20260925") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the real-genotype substrate outputs (dosages, meta, PCs, annotations)."""
    base = results_root / f"real_genotypes_{run_tag}"
    dosages = pd.read_csv(base / "genotype_dosages.csv", index_col=0)
    meta = dosages[["pop", "super_pop"]]
    dosages = dosages.drop(columns=["pop", "super_pop"])
    pcs = pd.read_csv(base / "ancestry_pcs.csv", index_col=0)
    annotations = pd.read_csv(base / "variant_annotations.csv")
    return dosages, meta, pcs, annotations

"""Real genotype substrate from 1000 Genomes phase 3 (F-10).

Replaces the z-score->genotype simulation's *background* with real phased
haplotypes sampled from the 2504-sample 1000 Genomes release, while the
cohort-informed label simulation stays exactly as it was. Patients sample
donor haplotypes from an ancestry group; LD becomes real by construction,
and ancestry labels for stratified evaluation come from the donor's
population (population-label provenance instead of modeled continents).

Data sources (all public, no credentials):
  - VCF: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/
    ALL.chrN.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz
    (GRCh37, phased, 2504 samples)
  - Sample panel: integrated_call_samples_v3.20130502.ALL.panel
  - rsID -> (chrom, pos37): Ensembl POST /variation/human/ids (batched)
  - LD (r2) validation: Ensembl REST /ld/human/{rs}/{pop}

Per the ledger: real genotypes (R2 substrate), labels stay simulated
(cohort-informed), so metrics remain R3 statistical.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

VCF_TEMPLATE = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v{vtag}.20130502.genotypes.vcf.gz"
)


def vcf_url(chrom: str) -> str:
    """Phase-3 release naming: autosomes are v5b, chrX is v1c."""
    return VCF_TEMPLATE.format(chrom=chrom, vtag="1c" if chrom == "X" else "5b")
PANEL_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "integrated_call_samples_v3.20130502.ALL.panel"
)
# The 1000G phase-3 VCFs are GRCh37; the default REST mirror serves GRCh38
# coordinates, so position resolution MUST use the GRCh37 mirror or every
# VCF lookup silently targets the wrong locus.
VARIATION_URL = "https://grch37.rest.ensembl.org/variation/human/ids"
LD_URL = "https://rest.ensembl.org/ld/human/{rs}/{pop}"

SUPER_POPS = ["EUR", "AFR", "EAS", "SAS", "AMR"]

ENSEMBL_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


@dataclass(frozen=True)
class Variant:
    rs_id: str
    chrom: str
    pos: int
    ref: str
    alt: str
    consequence: str | None = None
    gene: str | None = None


def load_population_panel(cache_dir: Path) -> pd.DataFrame:
    """1000G sample -> (pop, super_pop); cached locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel_path = cache_dir / "kg_panel_20130502.txt"
    if not panel_path.exists():
        resp = requests.get(PANEL_URL, timeout=60)
        resp.raise_for_status()
        panel_path.write_text(resp.text)
        logger.info("Downloaded 1000G sample panel (%d samples)", resp.text.count("\n") - 1)
    df = pd.read_csv(panel_path, sep="\t")
    df = df[df["sample"].notna()][["sample", "pop", "super_pop"]]
    return df.reset_index(drop=True)


def resolve_positions(rs_ids: list[str], cache_dir: Path) -> dict[str, Variant]:
    """Batch-resolve rsIDs to GRCh37 coordinates + consequence (Ensembl
    GRCh37 mirror — coordinates must match the 1000G phase-3 VCF build)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "variant_positions_grch37.json"
    cache: dict[str, dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    missing = [rs for rs in rs_ids if rs not in cache]
    for start in range(0, len(missing), 200):
        batch = missing[start : start + 200]
        for attempt in range(3):
            try:
                resp = requests.post(
                    VARIATION_URL, headers=ENSEMBL_HEADERS, json={"ids": batch}, timeout=60
                )
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", "10"))
                    logger.warning("Ensembl 429 — sleeping %ss", retry_after)
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                logger.warning("Ensembl resolve attempt %d failed: %s", attempt + 1, e)
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"Ensembl batch resolve failed for {len(batch)} variants")
        payload = resp.json()
        for rs, entry in payload.items():
            if "error" in entry or not entry.get("mappings"):
                cache[rs] = None
                continue
            # Prefer the PRIMARY GRCh37 mapping: alternate-contig mappings
            # (alt haplotypes like HSCHR6_MHC_*) share the assembly name and
            # would corrupt coordinates (this bit rs227584 -> "chr17").
            def _is_primary(m: dict) -> bool:
                name = str(m.get("seq_region_name", ""))
                return name.isdigit() or name in {"X", "Y", "MT"}

            mapping = next(
                (m for m in entry["mappings"]
                 if m.get("assembly_name") == "GRCh37" and _is_primary(m)),
                None,
            )
            if mapping is None:
                # No primary GRCh37 mapping: refuse to guess — mixing builds
                # or alt contigs silently would poison the dosage extraction.
                cache[rs] = None
                continue
            allele_string = entry.get("allele_string") or ""
            alleles = allele_string.split("/")
            cache[rs] = {
                "rs_id": rs,
                "chrom": str(mapping["seq_region_name"]),
                "pos": int(mapping["start"]),
                "ref": alleles[0] if len(alleles) >= 1 else "",
                "alt": alleles[1] if len(alleles) == 2 else "",
                "consequence": entry.get("most_severe_consequence"),
            }
        logger.info("Resolved %d/%d variants", min(start + 200, len(missing)), len(missing))
        time.sleep(0.3)

    cache_path.write_text(json.dumps(cache, indent=2))
    out: dict[str, Variant] = {}
    for rs, entry in cache.items():
        if entry is None:
            logger.warning("Unresolvable variant (skipped): %s", rs)
            continue
        out[rs] = Variant(**entry)
    return out


def fetch_genotypes(
    variants: dict[str, Variant],
    panel: pd.DataFrame,
    cache_dir: Path,
    super_pops: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract per-variant dosage genotypes (0/1/2 alt copies) for the given
    super-populations.

    Returns:
      dosages: DataFrame (samples x variants), values in {0,1,2}, index=sample
      meta: DataFrame index=sample with pop / super_pop columns
    """
    import pysam

    super_pops = super_pops or SUPER_POPS
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "kg_dosages.parquet"
    meta_path = cache_dir / "kg_sample_meta.parquet"
    if cache_path.exists() and meta_path.exists():
        return pd.read_parquet(cache_path), pd.read_parquet(meta_path)

    keep_samples = panel[panel["super_pop"].isin(super_pops)]["sample"].tolist()
    meta = panel.set_index("sample").loc[keep_samples, ["pop", "super_pop"]]

    by_chrom: dict[str, list[Variant]] = {}
    for v in variants.values():
        by_chrom.setdefault(v.chrom, []).append(v)

    columns: dict[str, np.ndarray] = {}
    for chrom, vs in sorted(by_chrom.items(), key=lambda kv: len(kv[1]), reverse=True):
        vcf_path = vcf_url(chrom)
        logger.info("Opening %s for %d variant(s)", vcf_path.split("/")[-1], len(vs))
        try:
            vcf = pysam.VariantFile(vcf_path)
        except (OSError, ValueError) as e:
            logger.error("  chrom %s: cannot open VCF (%s) — skipping %d variant(s)",
                         chrom, e, len(vs))
            continue
        try:
            for v in vs:
                # Targeted region query — never scan the whole chromosome over
                # HTTP (that cost us a 10-minute timeout on the first run).
                try:
                    records = list(vcf.fetch(contig=chrom, start=v.pos - 1, end=v.pos))
                except (ValueError, OSError) as e:
                    logger.warning("  %s region fetch failed: %s", v.rs_id, e)
                    continue
                record = next((r for r in records if r.pos == v.pos), None)
                if record is None or len(record.alts or ()) != 1:
                    if record is None:
                        logger.warning("  chrom %s: variant not found in VCF: %s", chrom, v.rs_id)
                    else:
                        logger.warning("  %s skipped (multiallelic)", v.rs_id)
                    continue
                col = np.empty(len(keep_samples), dtype=np.int16)
                for i, sample in enumerate(keep_samples):
                    gt = record.samples[sample]["GT"]
                    col[i] = sum(a for a in gt if a is not None and a > 0)
                columns[v.rs_id] = col
        finally:
            vcf.close()
        logger.info("  chrom %s: typed %d/%d", chrom, sum(1 for v in vs if v.rs_id in columns), len(vs))
        missing = [v.rs_id for v in vs if v.rs_id not in columns]
        for rs in missing:
            logger.warning("  chrom %s: variant not typed: %s", chrom, rs)

    dosages = pd.DataFrame(columns, index=keep_samples)
    dosages = dosages.loc[:, [v.rs_id for v in variants.values() if v.rs_id in dosages.columns]]
    # chrX male genotypes are hemizygous (single allele -> 0/1); documented
    # rather than silently mixed into the 0/1/2 coding.
    dosages.to_parquet(cache_path)
    meta.to_parquet(meta_path)
    return dosages, meta


def compute_ld_r2(dosages: pd.DataFrame) -> pd.DataFrame:
    """Pairwise r2 across the panel variants ( Pearson on dosage, unbiased
    enough for r2 at n>=500; matches Ensembl LD's computation for biallelic
    SNPs)."""
    corr = dosages.corr(method="pearson")
    return corr ** 2


def validate_ld_against_ensembl(
    dosages: pd.DataFrame, meta: pd.DataFrame, super_pop: str, rs_ids: list[str], cache_dir: Path
) -> pd.DataFrame:
    """F-10's R2 check: our computed within-super-pop r2 vs Ensembl's
    published 1000G phase-3 LD. Pre-registered tolerance in ROADMAP:
    |delta r2| <= 0.05 on checked pairs."""
    rows = []
    sub = dosages.loc[meta[meta["super_pop"] == super_pop].index]
    for rs in rs_ids:
        if rs not in sub.columns:
            continue
        url = LD_URL.format(rs=rs, pop=f"1000GENOMES:phase_3:{super_pop}")
        try:
            resp = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
            resp.raise_for_status()
            pairs = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("LD fetch %s %s failed: %s", rs, super_pop, e)
            continue
        for pair in pairs:
            other = pair.get("variation2")
            if other not in sub.columns:
                continue
            our_r2 = float(compute_ld_r2(sub[[rs, other]]).iloc[0, 1])
            rows.append({
                "rs1": rs,
                "rs2": other,
                "super_pop": super_pop,
                "ensembl_r2": float(pair["r2"]),
                "our_r2": round(our_r2, 6),
                "abs_delta": round(abs(float(pair["r2"]) - our_r2), 6),
            })
    return pd.DataFrame(rows)


def sample_donor_genotypes_with_ids(
    dosages: pd.DataFrame, meta: pd.DataFrame, ancestry_labels: pd.Series, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.Series]:
    """Draw each simulated patient's genotype vector from a real 1000G donor
    whose super-population matches the patient's ancestry label, and return
    the donor assignment (patient -> 1000G sample id) alongside the dosage
    matrix. The donor ids are required by downstream external-anchoring
    evaluations (F-16 PRS baseline, F-20 external validation) so the same
    donors' published statistics can be scored against the same patients.

    Patients without a usable label fall back to the pooled panel.
    Returns (dosage DataFrame aligned to ancestry_labels.index with columns
    = rs_ids, donor id Series aligned to ancestry_labels.index).
    """
    fallback = dosages.index.to_numpy()
    pools = {sp: meta[meta["super_pop"] == sp].index.to_numpy() for sp in SUPER_POPS}
    out = {}
    donor_ids = {}
    for pid, anc in ancestry_labels.items():
        pool = pools.get(str(anc), fallback)
        if len(pool) == 0:
            pool = fallback
        donor = rng.choice(pool)
        donor_ids[pid] = str(donor)
        out[pid] = dosages.loc[donor]
    return pd.DataFrame(out).T, pd.Series(donor_ids)


def sample_donor_genotypes(
    dosages: pd.DataFrame, meta: pd.DataFrame, ancestry_labels: pd.Series, rng: np.random.Generator
) -> pd.DataFrame:
    """Dosage rows only (see sample_donor_genotypes_with_ids)."""
    dos, _ = sample_donor_genotypes_with_ids(dosages, meta, ancestry_labels, rng)
    return dos

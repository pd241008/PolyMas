"""Published-effect-size coupling for real-donor mode (ADR-005).

Restores a *legitimate* genotype-label link in `--genotypes real` mode:
real autoimmune patients are enriched at risk loci relative to background
(that is what published GWAS odds ratios measure - ascertainment). We
reproduce that epidemiology Bayes-consistently:

- Donor side: a patient whose anchor cohort is disease D samples their 1000G
  donor from the ancestry-matched pool with weight
      w(donor) proportional to exp(beta_D . aligned_dosage(donor))
  i.e. the posterior over genotypes given disease status under a
  log-additive model (Bayes consistency: P(G|case)/P(G) = P(case|G)/P(case)
  up to normalization). Background patients sample uniformly.

- Label side: the polygenic term in simulate_labels uses the SAME published
  betas (per-SD standardized), replacing the legacy 0.50-centering guess.

Allele alignment: our dosage matrix is VCF-alt-indexed; published betas are
per *effect allele* (OpenGWAS `ea`). The VCF alt allele is identified by
matching panel alt frequencies (dosage mean / 2) against Ensembl
1000GENOMES:phase_3:ALL allele frequencies (|df| < 0.04; loci where both
alleles match within 0.02 are dropped as ambiguous). Aligned dosage is then
the dosage of the PUBLISHED EFFECT allele:

    aligned = vcf_alt_dosage          if ea == vcf_alt
    aligned = 2 - vcf_alt_dosage      if ea == vcf_ref
    (locus excluded from that disease's term when unresolvable)

Everything is cached under <results_root>/coupling_<tag>/ for provenance (R2).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Per-disease OpenGWAS/local datasets selected by the 2026-09-25 probe
# (scripts/ogwas_probe.py, evidence adr005_probe/dataset_picks.json).
DISEASE_DATASETS: dict[str, str] = {
    "RA": "ieu-a-833",           # Okada 2014, ncase=19234
    "SLE": "ebi-a-GCST003156",   # ncase=5201
    "SJOGRENS": "finn-b-M13_SJOGREN",  # FinnGen, ncase=1290
    "AITD": "ieu-a-1082",        # Graves', ncase=649
    "T1D": "ebi-a-GCST005536",   # ncase=6683
    "MS": "ieu-a-1025",          # ncase=14498
    # VITILIGO has no curated log-odds dataset on OpenGWAS; use the local
    # Jin-2016 curated anchors (same file the legacy simulator reads).
    "VITILIGO": "local:jin2016",
}

VITILIGO_LOCAL_BETAS: dict[str, float] = {
    # From polymas_ml MODELED_DISEASE_LOCUS_EFFECTS / GCST004785 (Jin 2016).
    "rs9272346": 0.572,
    "rs2476601": 0.324,
}

AITD_LOCAL_BETAS: dict[str, float] = {
    # Same file's Graves' anchors (GCST001200): rs6457617 HLA-DRB1/DQB1
    # OR=1.40 -> our rs9272346; rs1024161 CD28/CTLA4 OR=1.30 -> our rs3087243.
    "rs9272346": 0.336,
    "rs3087243": 0.262,
}

LOCAL_BETAS = {"VITILIGO": VITILIGO_LOCAL_BETAS, "AITD": AITD_LOCAL_BETAS}

ENSEMBL_VARIATION_URL = "https://grch37.rest.ensembl.org/variation/human/{rs}"
ENSEMBL_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

AF_MATCH_TOL = 0.04       # |panel alt AF - Ensembl ALL AF| to call the alt allele
AF_AMBIGUITY_TOL = 0.02   # both alleles matching within this => drop locus
EAF_AUDIT_TOL = 0.05      # cross-check for datasets that publish eaf


@dataclass
class AlignedLocus:
    rs_id: str
    beta: float          # published log-OR per effect allele
    effect_allele: str   # published ea
    pvalue: float | None
    aligned_to: str      # "alt" or "ref" (dosage direction)


@dataclass
class DiseaseCoupling:
    disease: str
    dataset: str
    loci: list[AlignedLocus] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)  # rs_id -> reason

    @property
    def rs_ids(self) -> list[str]:
        return [l.rs_id for l in self.loci]

    def betas(self) -> dict[str, float]:
        return {l.rs_id: l.beta for l in self.loci}

    def directions(self) -> dict[str, str]:
        return {l.rs_id: l.aligned_to for l in self.loci}


def load_assoc_rows(results_root: Path, tag: str, dataset: str) -> list[dict]:
    """Read the cached OpenGWAS association rows fetched by the probe."""
    if dataset.startswith("local:"):
        return []
    path = results_root / f"adr005_probe_{tag}" / f"assoc_rows_{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing probe cache {path}; run scripts/ogwas_probe.py first"
        )
    rows = json.loads(path.read_text())
    return rows if isinstance(rows, list) else rows.get("associations", [])


def fetch_ensembl_allele_freqs(rs_ids: list[str], cache_dir: Path) -> dict[str, dict]:
    """Per-rsID Ensembl VARIATION record: 1000GENOMES:phase_3:ALL allele
    frequencies + mappings (to identify ref/alt). Cached to JSON."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ensembl_allele_freqs.json"
    cache: dict[str, dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [rs for rs in rs_ids if rs not in cache]
    for i, rs in enumerate(missing):
        try:
            resp = requests.get(
                ENSEMBL_VARIATION_URL.format(rs=rs) + "?pops=1",
                headers=ENSEMBL_HEADERS, timeout=30,
            )
            if resp.status_code != 200:
                cache[rs] = {"error": resp.status_code}
                continue
            data = resp.json()
            freqs: dict[str, float] = {}
            for p in data.get("populations", []) or []:
                if p.get("population") == "1000GENOMES:phase_3:ALL":
                    freqs[str(p["allele"]).upper()] = float(p["frequency"])
            mappings = []
            for m in data.get("mappings", []) or []:
                # primary contigs only (ADR-004 trap: alt-contigs share the
                # assembly name)
                seq = str(m.get("seq_region_name", ""))
                if not (seq.isdigit() or seq in ("X", "Y", "MT")):
                    continue
                if m.get("assembly_name") == "GRCh37":
                    mappings.append({
                        "allele_string": m.get("allele_string"),
                        "seq_region": seq,
                        "start": m.get("start"),
                    })
            cache[rs] = {"freqs_1kg_all": freqs, "mappings": mappings,
                         "minor_allele": data.get("minor_allele"),
                         "maf": data.get("MAF")}
        except requests.RequestException as e:
            logger.warning("Ensembl variation fetch %s failed: %s", rs, e)
            cache[rs] = {"error": str(e)}
        if (i + 1) % 20 == 0:
            logger.info("allele freqs: %d/%d", i + 1, len(missing))
    cache_path.write_text(json.dumps(cache))
    return cache


def identify_vcf_alt(panel_af: pd.Series, ensembl_freqs: dict[str, dict]) -> dict[str, str]:
    """Identify each locus's VCF alt allele by matching panel alt frequency
    (dosage mean / 2) against Ensembl 1000GENOMES:phase_3:ALL frequencies.

    Returns rs_id -> alt allele string. Ambiguous/unresolvable loci map to
    "" with the reason logged (caller drops them)."""
    out: dict[str, str] = {}
    for rs_id, af in panel_af.items():
        rec = ensembl_freqs.get(rs_id, {})
        freqs = rec.get("freqs_1kg_all") or {}
        if len(freqs) != 2:
            out[rs_id] = ""
            continue
        alleles = sorted(freqs)
        diffs = {a: abs(float(af) - freqs[a]) for a in alleles}
        best = min(diffs, key=diffs.get)  # type: ignore[arg-type]
        other = alleles[1] if best == alleles[0] else alleles[0]
        if diffs[best] > AF_MATCH_TOL:
            out[rs_id] = ""  # unresolvable (frequency mismatch)
        elif abs(diffs[best] - diffs[other]) < AF_AMBIGUITY_TOL:
            out[rs_id] = ""  # both alleles plausibly match -> ambiguous
        else:
            out[rs_id] = best
        # note: 'other' is implicitly the ref allele
    return out


def build_disease_coupling(
    disease: str,
    panel_af: pd.Series,
    ensembl_freqs: dict[str, dict],
    vcf_alt: dict[str, str],
    results_root: Path,
    tag: str,
    ea_audit: dict[str, str] | None = None,
) -> DiseaseCoupling:
    """Align a disease's published betas to our dosage matrix orientation.

    ea_audit: optional rs_id -> published eaf string (from datasets that
    publish eaf, e.g. AITD/MS) used for the R2 cross-check."""
    dataset = DISEASE_DATASETS.get(disease, "")
    coup = DiseaseCoupling(disease=disease, dataset=dataset)
    if dataset.startswith("local:"):
        for rs_id, beta in LOCAL_BETAS.get(disease, {}).items():
            if rs_id not in vcf_alt or not vcf_alt[rs_id]:
                coup.dropped[rs_id] = "vcf alt unidentified"
                continue
            # Jin-2016 betas are given on the risk allele; identify the risk
            # allele from Ensembl ALL frequencies as the MINOR allele for
            # these classic hits (all have MAF < 0.5) and match against the
            # two candidate alleles.
            rec = ensembl_freqs.get(rs_id, {})
            freqs = rec.get("freqs_1kg_all") or {}
            if not freqs:
                coup.dropped[rs_id] = "no ensembl freqs"
                continue
            risk_allele = min(freqs, key=lambda a: freqs[a])  # type: ignore[arg-type]
            aligned_to = "alt" if risk_allele == vcf_alt[rs_id] else "ref"
            coup.loci.append(AlignedLocus(rs_id, beta, risk_allele, None, aligned_to))
        return coup

    for row in load_assoc_rows(results_root, tag, dataset):
        rs_id = str(row.get("rsid", ""))
        beta = row.get("beta")
        pval = row.get("p")
        ea = str(row.get("ea", "")).upper()
        if rs_id not in panel_af.index or beta is None or not ea:
            continue
        if rs_id not in vcf_alt or not vcf_alt[rs_id]:
            coup.dropped[rs_id] = "vcf alt unidentified"
            continue
        alt = vcf_alt[rs_id].upper()
        rec = ensembl_freqs.get(rs_id, {})
        freqs = rec.get("freqs_1kg_all") or {}
        # Cross-check: published eaf (if any) should match Ensembl ALL freq
        # of the effect allele.
        if ea_audit and rs_id in ea_audit:
            try:
                pub_eaf = float(ea_audit[rs_id])
                ens_eaf = freqs.get(ea)
                if ens_eaf is not None and abs(pub_eaf - ens_eaf) > EAF_AUDIT_TOL:
                    coup.dropped[rs_id] = (
                        f"eaf audit failed: published {pub_eaf:.3f} vs ensembl {ens_eaf:.3f}")
                    continue
            except (TypeError, ValueError):
                pass
        if ea == alt:
            aligned_to = "alt"
        elif ea in freqs and alt in freqs:
            # ea is the other allele (ref) -> dosage flips
            aligned_to = "ref"
        else:
            coup.dropped[rs_id] = f"effect allele {ea} not in ensembl freqs"
            continue
        coup.loci.append(AlignedLocus(
            rs_id, float(beta), ea,
            float(pval) if pval is not None else None, aligned_to,
        ))
    return coup


def restrict_to_anchor_loci(
    coup: DiseaseCoupling, allowed_rs_ids: list[str]
) -> DiseaseCoupling:
    """ADR-006: keep only the disease's OWN anchor loci (DISEASE_RISK_LOCI)
    from a full coupling. The label term must not read other diseases'
    anchors; loci dropped by alignment stay absent (caller falls back to the
    legacy term when a disease has no remaining anchor loci)."""
    keep = set(allowed_rs_ids)
    return DiseaseCoupling(
        disease=coup.disease,
        dataset=coup.dataset,
        loci=[l for l in coup.loci if l.rs_id in keep],
        dropped=dict(coup.dropped),
    )


def aligned_dosage_matrix(
    dosages: pd.DataFrame, coup: DiseaseCoupling
) -> pd.DataFrame:
    """Dosage of the PUBLISHED EFFECT allele for the disease's loci."""
    cols: dict[str, np.ndarray] = {}
    for locus in coup.loci:
        col = dosages[locus.rs_id].to_numpy(dtype=float)
        cols[locus.rs_id] = col if locus.aligned_to == "alt" else 2.0 - col
    return pd.DataFrame(cols, index=dosages.index)


def donor_weights_for_anchor(
    coup: DiseaseCoupling,
    aligned: pd.DataFrame,
    pool_donor_ids: np.ndarray,
) -> np.ndarray:
    """Bayes-consistent case weights over the ancestry pool:
    w(donor) proportional to exp(beta . aligned_dosage), normalized.
    Zero-variance pools (all weights equal) fall back to uniform."""
    if not coup.loci:
        return np.full(len(pool_donor_ids), 1.0 / max(len(pool_donor_ids), 1))
    sub = aligned.loc[pool_donor_ids, coup.rs_ids].to_numpy(dtype=float)
    beta_vec = np.array([coup.betas()[rs] for rs in coup.rs_ids])
    log_w = sub @ beta_vec
    log_w -= log_w.max()
    w = np.exp(log_w)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(pool_donor_ids), 1.0 / max(len(pool_donor_ids), 1))
    w /= total
    return w


def sample_anchored_donors(
    couplings: dict[str, DiseaseCoupling],
    aligned_by_disease: dict[str, pd.DataFrame],
    dosages: pd.DataFrame,
    meta: pd.DataFrame,
    ancestry_labels: pd.Series,
    patient_groups: list[str | None],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw each patient's donor: uniform for background/modeled-anchor
    patients (no published-coupling info is abused), weighted by the anchor
    disease's Bayes posterior within the ancestry-matched pool.

    Returns patient -> donor sample id (same contract as
    genotypes.sample_donor_genotypes)."""
    fallback = dosages.index.to_numpy()
    pools = {sp: meta[meta["super_pop"] == sp].index.to_numpy() for sp in ["EUR", "AFR", "EAS", "SAS", "AMR"]}
    out: dict[str, str] = {}
    weight_audit: dict[str, dict] = {}
    for i, pid in enumerate(ancestry_labels.index):
        anc = str(ancestry_labels.iloc[i])
        pool = pools.get(anc, fallback)
        if len(pool) == 0:
            pool = fallback
        anchor = patient_groups[i]
        coup = couplings.get(str(anchor)) if anchor else None
        if coup is None or not coup.loci:
            donor = str(rng.choice(pool))
        else:
            aligned = aligned_by_disease[str(anchor)]
            w = donor_weights_for_anchor(coup, aligned, pool)
            # Effective sample size audit (weights degenerate -> suspicious)
            ess = float(1.0 / np.sum(w ** 2)) if w.sum() > 0 else 0.0
            weight_audit.setdefault(str(anchor), []).append(ess)
            donor = str(rng.choice(pool, p=w))
        out[pid] = donor
    for disease, esses in weight_audit.items():
        logger.info(
            "coupling weight ESS for %s: mean=%.1f of pool n=%d (n_patients=%d)",
            disease, float(np.mean(esses)), len(pools.get("EUR", fallback)),
            len(esses),
        )
    return pd.Series(out)


def aligned_anchor_prs(
    gen_matrix: pd.DataFrame,
    coup: DiseaseCoupling,
) -> np.ndarray:
    """ADR-006 label-term PRS at the LEGACY TERM'S SCALE (do not confuse
    with label_prs_term, which is per-SD standardized for ADR-005's +0.10/SD
    coupling coefficient).

    Pre-registered ADR-006 formula: prs_d = mean over the disease's anchor
    loci of beta_locus * aligned_effect_dosage, empirically centered on the
    patient panel (zero-mean by construction, same magnitude convention as
    the legacy raw-dosage term so `p += 0.50 * term` keeps marginals
    calibrated). Sign-corrected: aligned_effect_dosage is the dosage of the
    PUBLISHED effect allele, so beta * aligned_dosage is the correct log-OR
    contribution per locus regardless of which allele the VCF calls alt."""
    if not coup.loci:
        return np.zeros(len(gen_matrix))
    total = np.zeros(len(gen_matrix))
    for locus in coup.loci:
        col = gen_matrix[locus.rs_id].to_numpy(dtype=float)
        aligned = col if locus.aligned_to == "alt" else 2.0 - col
        total += locus.beta * aligned
    total /= len(coup.loci)
    return total - total.mean()


def label_prs_term(
    gen_matrix: pd.DataFrame,
    coup: DiseaseCoupling,
    panel_af: pd.Series,
) -> np.ndarray:
    """Per-patient published PRS (per SD) for one disease, from the PATIENT's
    genotypes: sum_loci beta * (aligned dosage - 2 * panel_AF), scaled by the
    analytic population SD sqrt(sum beta^2 * 2p(1-p)) (LD ignored; rescaling
    only). Returns a 1-D array aligned to gen_matrix rows."""
    if not coup.loci:
        return np.zeros(len(gen_matrix))
    total = np.zeros(len(gen_matrix))
    var = 0.0
    for locus in coup.loci:
        col = gen_matrix[locus.rs_id].to_numpy(dtype=float)
        aligned = col if locus.aligned_to == "alt" else 2.0 - col
        # Center on the EFFECT-allele frequency: panel_af is the VCF-alt AF,
        # so ref-aligned loci center on 1 - panel_af (keeps the term exactly
        # zero-centered under either orientation).
        p_alt = float(panel_af[locus.rs_id])
        p_eff = (1.0 - p_alt) if locus.aligned_to == "ref" else p_alt
        total += locus.beta * (aligned - 2.0 * p_eff)
        var += locus.beta ** 2 * 2.0 * p_eff * (1.0 - p_eff)
    sd = float(np.sqrt(var)) if var > 0 else 1.0
    return total / sd


def save_coupling_manifest(
    out_dir: Path,
    couplings: dict[str, DiseaseCoupling],
    vcf_alt: dict[str, str],
    ea_audit: dict[str, dict[str, str]],
) -> None:
    """R2 provenance: everything needed to audit the coupling."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "datasets": {d: c.dataset for d, c in couplings.items()},
        "loci": {
            d: [
                {"rs_id": l.rs_id, "beta": l.beta, "effect_allele": l.effect_allele,
                 "pvalue": l.pvalue, "aligned_to": l.aligned_to}
                for l in c.loci
            ]
            for d, c in couplings.items()
        },
        "dropped": {d: c.dropped for d, c in couplings.items()},
        "vcf_alt_alleles": vcf_alt,
        "ea_audit_datasets": ea_audit,
    }
    (out_dir / "coupling_manifest.json").write_text(json.dumps(payload, indent=2))

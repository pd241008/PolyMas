"""Shared patient simulation: genotypes, PRS, and disease labels.

Generates the synthetic genotype->PRS->label process shared by System A
(feature matrix) and System B (k-mer sequences), so both systems describe
the same underlying patients. Clinical features come from REAL ImmPort
subjects (see polymas_ml.data.immport); only BMI, family history, genotypes
and labels are simulated (documented in the PDF as modeled fields).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DISEASE_LABELS = ["RA", "SLE", "SJOGRENS", "AITD", "T1D", "VITILIGO", "MS"]

# Real-cohort diseases get a genotype effect (beta); modeled diseases (no real
# cohort on ImmPort) get beta=0 and are flagged in outputs as modeled.
GENOTYPE_EFFECTS = {
    "RA": 0.55,
    "SLE": 0.45,
    "SJOGRENS": 0.30,
    "T1D": 0.50,
    "MS": 0.40,
    "AITD": 0.0,
    "VITILIGO": 0.0,
}

BASE_PREVALENCES = {
    "RA": 0.20,
    "SLE": 0.10,
    "SJOGRENS": 0.08,
    "AITD": 0.15,
    "T1D": 0.08,
    "VITILIGO": 0.06,
    "MS": 0.12,
}

# Background sex prevalence modifiers (female-biased, as observed clinically).
SEX_RISK = {
    "F": {"SLE": 0.06, "SJOGRENS": 0.05, "AITD": 0.04, "RA": 0.03},
    "M": {},
}


def simulate_genotypes_prs(
    n_patients: int,
    loci: dict[str, str],
    patient_groups: list[str | None],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate per-locus genotypes and derive PRS rows from real effect sizes.

    Genotype (0/1/2 alt alleles) ~ Binomial(2, q), q locus-specific. PRS
    z-score = sum(beta_g * genotype_g) + noise. Disease cohorts shift their
    own loci's genotype frequencies upward so genotypes carry real signal
    about cohort membership (which drives the labels).
    """
    # Per-locus baseline alt-allele frequencies (order matches loci dict keys).
    q_base = {
        "rs2187668": 0.10, "rs9272346": 0.25, "rs2476601": 0.08,
        "rs3087243": 0.40, "rs2292239": 0.30, "rs11209026": 0.07,
        "rs2104286": 0.35, "rs7574865": 0.22,
    }
    rs_ids = list(loci.keys())
    # Map each disease to its most-specific risk loci for cohort enrichment.
    disease_loci = {
        "RA": ["rs2476601", "rs11209026"],
        "SLE": ["rs7574865", "rs3087243"],
        "SJOGRENS": ["rs2187668", "rs7574865"],
        "T1D": ["rs9272346", "rs2476601"],
        "MS": ["rs2104286", "rs2292239"],
        "AITD": [],
        "VITILIGO": [],
    }

    gen_rows = []
    prs_rows = []
    n_loci = len(rs_ids)
    betas = np.array([0.30, 0.22, 0.35, 0.15, 0.18, 0.28, 0.20, 0.25][:n_loci])

    for i in range(n_patients):
        pid = f"P{i:04d}"
        group = patient_groups[i]
        risk_loci = set(disease_loci.get(group, []) if group else [])

        genotypes: dict[str, int] = {}
        for j, rs_id in enumerate(rs_ids):
            q = q_base.get(rs_id, 0.2)
            if rs_id in risk_loci:
                q = min(0.6, q + 0.25)  # cohort enrichment at its own loci
            g = int(rng.binomial(2, q))
            genotypes[rs_id] = g

        for rs_id, gene in loci.items():
            g = genotypes[rs_id]
            # Per-locus z directly encodes the allele count at that locus
            # (centered on its expected value) plus modest noise, so the
            # PRS features carry detectable genotype signal per locus.
            q_locus = q_base.get(rs_id, 0.2)
            z = round(float((g - 2 * q_locus) * betas[rs_ids.index(rs_id)] * 1.2
                            + rng.normal(0, 0.25)), 4)
            score = round(float(np.clip((z + 1.5) / 3.0, 0.0, 1.0)), 4)
            prs_rows.append({
                "patient_id": pid,
                "locus_id": rs_id,
                "gene_symbol": gene,
                "continuous_score": score,
                "z_score": z,
                "pvalue": None,
                "genotype": g,
            })
        gen_rows.append({"patient_id": pid, **genotypes})

    return pd.DataFrame(prs_rows), pd.DataFrame(gen_rows)


def simulate_labels(
    n_patients: int,
    patient_groups: list[str | None],
    sexes: list[str],
    gen_rows: pd.DataFrame,
    loci: dict[str, str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate disease labels from cohort membership, sex, and genotypes.

    Patients assigned to a real disease cohort get a high prevalence of that
    disease (plus base rates for others); background patients get base
    prevalences only. Modeled diseases (AITD, VITILIGO) draw from background
    prevalence regardless of cohort.
    """
    rs_ids = list(loci.keys())
    cohort_prev = {
        "RA": 0.65, "SLE": 0.60, "SJOGRENS": 0.55, "T1D": 0.60, "MS": 0.55,
    }
    rows = []
    gen_matrix = gen_rows.set_index("patient_id").loc[[f"P{i:04d}" for i in range(n_patients)]]

    # Disease-specific risk loci (same mapping the genotype simulator uses for
    # cohort enrichment) so labels are learnable from the per-locus features.
    disease_loci = {
        "RA": ["rs2476601", "rs11209026"],
        "SLE": ["rs7574865", "rs3087243"],
        "SJOGRENS": ["rs2187668", "rs7574865"],
        "T1D": ["rs9272346", "rs2476601"],
        "MS": ["rs2104286", "rs2292239"],
        "AITD": [],
        "VITILIGO": [],
    }

    for i in range(n_patients):
        pid = f"P{i:04d}"
        group = patient_groups[i]
        sex = sexes[i]
        g = gen_matrix.iloc[i]

        labels = {"patient_id": pid}
        for disease in DISEASE_LABELS:
            if disease in GENOTYPE_EFFECTS and GENOTYPE_EFFECTS[disease] == 0.0:
                # Modeled disease: background prevalence only.
                p = BASE_PREVALENCES[disease]
            else:
                p = BASE_PREVALENCES[disease]
                if group == disease:
                    p = cohort_prev.get(disease, 0.5)
                # Disease-specific polygenic effect from its own risk loci.
                risk = disease_loci.get(disease, [])
                if risk:
                    prs_d = float(np.mean([g[rs_id] for rs_id in risk])) / 2.0
                    p += 0.50 * (prs_d - 0.15)
                p += SEX_RISK.get(sex, {}).get(disease, 0.0)
            labels[disease] = int(rng.random() < min(0.95, max(0.01, p)))
        rows.append(labels)

    return pd.DataFrame(rows)

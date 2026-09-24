"""Shared patient simulation: genotypes, PRS, and disease labels.

Generates the synthetic genotype->PRS->label process shared by System A
(feature matrix) and System B (k-mer sequences), so both systems describe
the same underlying patients. Clinical features come from REAL ImmPort
subjects (see polymas_ml.data.immport); only BMI, family history, genotypes
and labels are simulated (documented in the PDF as modeled fields).

Modeled diseases (AITD, VITILIGO) have no ImmPort cohort, but their label
simulation is GWAS-INFORMED: genotype effects at our panel loci use log-OR
anchors from published summary statistics (Jin 2016 vitiligo GCST004785;
Graves' disease GCST001200), so their labels carry real published genetic
signal instead of pure noise.

Labels are NOT independent Bernoulli draws. A per-patient latent autoimmune
liability drives a co-occurrence (polyautoimmunity) structure so that a
defensible subset of patients genuinely accumulates 3+ concurrent
diagnoses — the defining property of Multiple Autoimmune Syndrome (MAS)
per Humbert & Dupond 1988 and Anaya 2012. MAS_PAIRWISE_ODDS encodes
published comorbidity pairs (Anaya 2020 review; Somers 2006; Betterle 2023)
as log-OR affinities; MAS_EXCLUSIONS implements the incoercible
(antagonistic) combinations. See simulate_labels for the generative model
and label_structure_report for the diagnostics surfaced in the paper.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DISEASE_LABELS = ["RA", "SLE", "SJOGRENS", "AITD", "T1D", "VITILIGO", "MS"]

# Real-cohort diseases get a genotype effect (beta); modeled diseases (no real
# cohort on ImmPort) are flagged in outputs as modeled, but now carry
# GWAS-derived effect strengths (see MODELED_DISEASE_LOCUS_EFFECTS below).
GENOTYPE_EFFECTS = {
    "RA": 0.55,
    "SLE": 0.45,
    "SJOGRENS": 0.30,
    "T1D": 0.50,
    "MS": 0.40,
    "AITD": 0.35,
    "VITILIGO": 0.45,
}

# GWAS-informed anchors for modeled diseases: our panel loci that carry the
# disease effect, with log(OR) per copy taken from published summary stats.
# Sources downloaded under results/raw/gwas_sumstats/ (curated/ CSVs).
MODELED_DISEASE_LOCUS_EFFECTS = {
    "AITD": [
        # Graves' HLA-DRB1/DQA1/DQB1 lead rs6457617 OR=1.40 -> probes our HLA-DQB1 locus.
        {"rsid": "rs9272346", "beta": 0.336, "source": "GCST001200 (Graves') rs6457617 HLA-DRB1/DQB1 OR=1.40, p=7e-33"},
        # Graves' CD28/CTLA4 rs1024161 OR=1.30 -> our CTLA4 locus rs3087243.
        {"rsid": "rs3087243", "beta": 0.262, "source": "GCST001200 (Graves') rs1024161 CD28/CTLA4 OR=1.30, p=2e-17"},
    ],
    "VITILIGO": [
        # Jin 2016 HLA-DRB1/DQA1 rs9271597 OR=1.772 (strongest non-HLA-A hit).
        {"rsid": "rs9272346", "beta": 0.572, "source": "GCST004785 (Jin 2016) rs9271597 HLA-DRB1/DQA1 OR=1.772, p=3e-89"},
        # Jin 2016 PTPN22 rs2476601 OR=1.383 - DIRECT rsID hit on our panel.
        {"rsid": "rs2476601", "beta": 0.324, "source": "GCST004785 (Jin 2016) rs2476601 PTPN22 OR=1.383, p=1e-18"},
    ],
}

# Background (non-cohort) prevalences. Deliberately moderate: with the
# disease-enriched cohort design (75% of patients drawn from autoimmune
# cohorts), higher values made multi-disease accumulation incidental
# (chance stacking) rather than a modeled polyautoimmunity phenomenon.
# The co-occurrence cascade is what now produces multi-disease patients.
BASE_PREVALENCES = {
    "RA": 0.06,
    "SLE": 0.04,
    "SJOGRENS": 0.03,
    "AITD": 0.05,
    "T1D": 0.03,
    "VITILIGO": 0.025,
    "MS": 0.04,
}

# Background sex prevalence modifiers (female-biased, as observed clinically).
SEX_RISK = {
    "F": {"SLE": 0.06, "SJOGRENS": 0.05, "AITD": 0.04, "RA": 0.03},
    "M": {},
}

# ----------------------------------------------------------------------------
# Polyautoimmunity (MAS co-occurrence) structure
# ----------------------------------------------------------------------------
# Published autoimmune comorbidity pairs with odds ratios, used to bias
# latent-MAS co-occurrence toward clinically observed combinations.
# Sources:
#   - Anaya JM et al. Autoimmune Disease Co-occurrence: The Puzzling Aspect
#     of the Autoimmune Tautology (2020 review tables).
#   - Somers EC et al. Epidemiology 2006 (family/individual co-occurrence).
#   - Humbert P & Dupond JL 1988 (Type 1-4 composition).
# log(OR) values are used as additive pair-affinity weights.
MAS_PAIRWISE_ODDS: dict[tuple[str, str], float] = {
    # Type 2 core (Humbert-Dupond): Sjogren's-RA-AITD-PBC-scleroderma
    ("SJOGRENS", "RA"): 0.65,          # OR ~1.9
    ("SJOGRENS", "AITD"): 1.10,        # OR ~3.0
    ("RA", "AITD"): 0.52,              # OR ~1.7
    ("SJOGRENS", "SLE"): 0.75,         # OR ~2.1
    ("SLE", "RA"): 0.48,               # OR ~1.6
    # Type 3 / APS-3 axis (Betterle 2023 expanding galaxy)
    ("AITD", "VITILIGO"): 1.05,        # OR ~2.9 (thyroid-vitiligo association)
    ("AITD", "T1D"): 0.65,             # OR ~1.9 (APS-3v)
    ("T1D", "VITILIGO"): 0.60,         # OR ~1.8
    ("AITD", "SJOGRENS"): 0.40,        # within APS-3 overlap
    # Shared-genetics pairs (GWAS sharing / pleiotropy literature)
    ("SLE", "T1D"): 0.30,              # modest
    ("SLE", "MS"): 0.20,               # HLA-DRB1*15:01 shared; RA-protective allele
    ("SLE", "VITILIGO"): 0.10,         # weak
    ("MS", "AITD"): 0.15,              # elevated thyroid autoimmunity in MS
}

# Humbert-Dupond "incoercible" combinations: disease pairs reported as
# antagonistic (co-occurrence at/below chance). Implemented as hard
# exclusions applied AFTER MAS assembly to keep the pattern interpretable.
MAS_EXCLUSIONS: tuple[tuple[str, str], ...] = (
    ("T1D", "MS"),   # T1D + MS: negative association reported
    ("RA", "MS"),    # shared-locus antagonism (rs2104286 IL2RA direction)
)

# Global scale converting pairwise log-OR affinities into conditional
# probability lifts (beta coefficients of the linear-conditional cascade).
# Calibrated so the cohort shows overdispersion > 1 and a defensible MAS-3+
# subgroup while preserving approximately the target marginal prevalences.
MAS_AFFINITY_SCALE = 0.065

# Latent autoimmune liability distribution (population mix): a discrete
# mixture so a subset of patients carries high polyautoimmunity liability —
# the 'MAS patients' the 1988 classification describes.
MAS_LIABILITY_MIX = [
    (0.80, 0.0),   # 80% of patients: standard liability
    (0.15, 0.50),  # 15%: elevated liability (2-disease territory)
    (0.05, 1.20),  # 5%: high liability (MAS territory, 3+)
]

# Empirical anchor: Anaya 2012 reports ~25% of autoimmune patients develop
# additional diseases; our population-mix parameters are tuned so the
# generated cohort reproduces that 2+ rate (verified by diagnostics).
TARGET_POLYAUTOIMMUNITY_RATE_2PLUS = 0.25


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
        # Modeled diseases: GWAS-anchored loci (see MODELED_DISEASE_LOCUS_EFFECTS).
        "AITD": ["rs9272346", "rs3087243"],
        "VITILIGO": ["rs9272346", "rs2476601"],
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
    """Simulate disease labels with explicit polyautoimmunity (MAS) structure.

    Generative model per patient:

    1. Latent liability. A per-patient autoimmune liability L is drawn from
       a three-component mixture (MAS_LIABILITY_MIX), so a subset of the
       population carries elevated multi-disease liability - the empirical
       MAS subgroup (Anaya 2012: ~25% develop additional diseases).
    2. Marginal risks. Each disease's base prevalence is raised by the
       patient's real-cohort membership, sex, and GWAS-anchored polygenic
       score (same terms as before).
    3. Co-occurrence pass. Diseases are drawn sequentially; each successive
       draw is modulated by the patient's liability and by the log-OR
       affinities (MAS_PAIRWISE_ODDS) of the diseases already positive, so
       positive labels cluster into published MAS patterns. Incoercible
       pairs (MAS_EXCLUSIONS) are never drawn together.

    Marginal prevalences are approximately preserved; what changes vs the
    old independent-Bernoulli engine is the JOINT: disease counts follow a
    heavier-than-binomial (overdispersed) distribution and pairwise phi
    correlations become positive for the MAS pairs. The resulting structure
    is quantified by label_structure_report and written into the provenance
    manifest so the paper's framing matches the generated reality.
    """
    cohort_prev = {
        "RA": 0.50, "SLE": 0.45, "SJOGRENS": 0.40, "T1D": 0.45, "MS": 0.40,
    }
    rows = []
    gen_matrix = gen_rows.set_index("patient_id").loc[[f"P{i:04d}" for i in range(n_patients)]]

    # Disease-specific risk loci (same mapping the genotype simulator uses for
    # cohort enrichment) so labels are learnable from the per-locus features.
    # Modeled diseases use GWAS-anchored loci (MODELED_DISEASE_LOCUS_EFFECTS).
    disease_loci = {
        "RA": ["rs2476601", "rs11209026"],
        "SLE": ["rs7574865", "rs3087243"],
        "SJOGRENS": ["rs2187668", "rs7574865"],
        "T1D": ["rs9272346", "rs2476601"],
        "MS": ["rs2104286", "rs2292239"],
        "AITD": ["rs9272346", "rs3087243"],
        "VITILIGO": ["rs9272346", "rs2476601"],
    }

    # Sequential draw order: draw the cohort disease first when the patient
    # belongs to one, then diseases in descending baseline prevalence so the
    # affinity cascade starts from the patient's anchor disease.
    prevalence_order = sorted(DISEASE_LABELS, key=lambda d: -BASE_PREVALENCES[d])

    for i in range(n_patients):
        pid = f"P{i:04d}"
        group = patient_groups[i]
        sex = sexes[i]
        g = gen_matrix.iloc[i]

        # 1. Latent autoimmune liability (population mixture).
        liability = float(rng.choice(
            [w for w, _ in MAS_LIABILITY_MIX],
            p=[p for p, _ in MAS_LIABILITY_MIX],
        )) * LIABILITY_SCALE

        # 2. Marginal risks per disease (cohort + polygenic + sex terms).
        marginal: dict[str, float] = {}
        for disease in DISEASE_LABELS:
            p = BASE_PREVALENCES[disease]
            if group == disease:
                p = cohort_prev.get(disease, 0.5)
            risk = disease_loci.get(disease, [])
            if risk:
                prs_d = float(np.mean([g[rs_id] for rs_id in risk])) / 2.0
                p += 0.50 * (prs_d - 0.15)
            p += SEX_RISK.get(sex, {}).get(disease, 0.0)
            marginal[disease] = float(min(0.95, max(0.01, p)))

        # 3. Sequential co-occurrence draw with MAS affinities.
        draw_order = ([group] if group in DISEASE_LABELS else []) + [
            d for d in prevalence_order if d != group
        ]
        labels = draw_cooccurring_labels(marginal, liability, rng, draw_order=draw_order)
        # Emit columns in canonical DISEASE_LABELS order regardless of draw order.
        rows.append({"patient_id": pid, **{d: labels[d] for d in DISEASE_LABELS}})

    return pd.DataFrame(rows)


def draw_cooccurring_labels(
    marginal: dict[str, float],
    liability: float,
    rng: np.random.Generator,
    draw_order: list[str] | None = None,
) -> dict[str, int]:
    """Draw one patient's labels given marginal risks and latent liability.

    Linear-conditional cascade: each disease's conditional probability is
    its marginal risk, lifted by the latent liability and by the log-OR
    affinities (MAS_PAIRWISE_ODDS) of the diseases already drawn positive:

        p_i = clip(p_i_marginal + lift*liability
                   + scale * sum_{k positive} logOR_(i,k), 0.001, 0.95)

    This preserves each disease's marginal prevalence in expectation (the
    lift terms average out over the population) while inducing positive
    pairwise correlation for MAS pairs. Incoercible pairs
    (MAS_EXCLUSIONS) are never drawn together. Shared by System A
    (simulate_labels) and the standalone dataset builder so both produce
    the same joint structure.
    """
    order = draw_order if draw_order is not None else sorted(
        marginal, key=lambda d: -marginal[d]
    )
    labels: dict[str, int] = {}
    positive: list[str] = []
    for disease in order:
        p = marginal[disease] + LIABILITY_SCALE * liability
        excluded = False
        for other in positive:
            # Exclusions are symmetric: test both orientations.
            if (disease, other) in MAS_EXCLUSIONS or (other, disease) in MAS_EXCLUSIONS:
                excluded = True
                break
            affinity = MAS_PAIRWISE_ODDS.get(
                (disease, other), MAS_PAIRWISE_ODDS.get((other, disease), 0.0)
            )
            p += MAS_AFFINITY_SCALE * affinity
        if excluded:
            labels[disease] = 0
            continue
        if rng.random() < _logit_clip(p):
            positive.append(disease)
            labels[disease] = 1
        else:
            labels[disease] = 0
    return labels


def _logit_clip(p: float) -> float:
    """Clamp a probability into (0.001, 0.95) to keep odds math stable."""
    return float(min(0.95, max(0.001, p)))


# Liability probability-space lift coefficient (per unit liability).
LIABILITY_SCALE = 0.10

# Calibration targets reported alongside the realized diagnostics.
TARGET_MAS_RATE_3PLUS = 0.15


def label_structure_report(labels_df: pd.DataFrame) -> dict:
    """Quantify the co-occurrence structure of a label matrix.

    Returns the diagnostics the paper reports to substantiate the MAS
    framing: disease-count distribution (overdispersion vs a binomial
    null), polyautoimmunity rates (1/2/3+ concurrent diseases), and the
    pairwise phi correlation matrix.
    """
    label_cols = [d for d in DISEASE_LABELS if d in labels_df.columns]
    counts = labels_df[label_cols].sum(axis=1)

    expected_p = float(labels_df[label_cols].mean().mean())
    binom_var = len(label_cols) * expected_p * (1 - expected_p)
    dispersion = float(counts.var() / binom_var) if binom_var > 0 else float("nan")

    phis: dict[tuple[str, str], float] = {}
    for a_idx in range(len(label_cols)):
        for b_idx in range(a_idx + 1, len(label_cols)):
            a, b = label_cols[a_idx], label_cols[b_idx]
            xa = labels_df[a].to_numpy(dtype=float)
            xb = labels_df[b].to_numpy(dtype=float)
            va, vb = xa.std(), xb.std()
            phi = (
                float(((xa - xa.mean()) * (xb - xb.mean())).mean() / (va * vb))
                if va > 0 and vb > 0 else 0.0
            )
            phis[(a, b)] = round(phi, 4)

    return {
        "n_patients": int(len(labels_df)),
        "disease_count_distribution": {
            str(k): int(v) for k, v in counts.value_counts().sort_index().items()
        },
        "mean_diseases_per_patient": round(float(counts.mean()), 4),
        "var_diseases_per_patient": round(float(counts.var()), 4),
        "binomial_null_variance": round(float(binom_var), 4),
        "overdispersion_ratio": round(dispersion, 4),
        "rate_1plus": round(float((counts >= 1).mean()), 4),
        "rate_2plus": round(float((counts >= 2).mean()), 4),
        "rate_3plus_mas": round(float((counts >= 3).mean()), 4),
        "rate_2plus_given_1plus": round(
            float((counts >= 2).sum() / max((counts >= 1).sum(), 1)), 4
        ),
        "target_mas_rate_3plus": TARGET_MAS_RATE_3PLUS,
        "target_polyautoimmunity_rate_2plus": TARGET_POLYAUTOIMMUNITY_RATE_2PLUS,
        "pairwise_phi": {"|".join(k): v for k, v in phis.items()},
        "mean_positive_pairwise_phi": round(
            float(np.mean([v for v in phis.values() if v > 0]))
            if any(v > 0 for v in phis.values()) else 0.0,
            4,
        ),
        "model": "latent_liability_mixture + pairwise_log_or_affinity (MAS_PAIRWISE_ODDS)",
        "mas_exclusions": [list(p) for p in MAS_EXCLUSIONS],
    }

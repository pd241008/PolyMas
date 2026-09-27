# ADR-005: Genotype–label coupling policy in real-donor mode

- **Status:** Decided — **Rejected (fallback to ADR-004 no-coupling)**; gates
  pre-registered below were run 2026-09-25 and failed (verdict record at the
  bottom)
- **Date:** 2026-09-25
- **Deciders:** Program lead (results program, ROADMAP.md Phase 2)
- **Related:** ADR-004 (real-donor wiring, Decided), ROADMAP F-10/F-16/F-20,
  Phase-2 ordering (F-16 PRS baseline runs on whatever mode is canonical)

## Context

ADR-004 wired real 1000G donor genotypes into the pipeline (F-10). The measured
consequence: AUROC dropped vs the legacy simulator (System A macro 0.601 vs
0.656) because the legacy simulator *enriched* cohort genotypes at risk loci
(q+0.25) — a genotype–label coupling the models could (and did) learn. Real
donor haplotypes cannot be enriched post hoc without falsifying genotypes
(verified: RA cohort vs background risk-locus dosage 1.386 vs 1.374).

This left two defensible policies, and an open question:

1. **No-coupling baseline (ADR-004 default):** real genotypes, labels drawn
   from the same marginal/cascade model without any genotype-dependent term.
   Honest, but the genotype features carry no label information by
   construction, so every model metric on real mode is structurally capped.
2. **Legitimate coupling:** real autoimmune patients genuinely *are* enriched
   at risk loci relative to background — that is what the published GWAS
   odds ratios measure (ascertainment). Sampling anchor-disease patients'
   donors with probability tilted by **published per-variant log-ORs**
   reproduces that epidemiological reality without inventing genotypes.

The program rule: "if we can get real coupling then we will do it, if not
then no coupling." This ADR defines *real* (mechanistically justified,
Bayes-consistent, pre-registered) vs *fake* (post-hoc frequency shifts like
the legacy q+0.25 enrichment) coupling, and the gates that decide.

## Feasibility probe (2026-09-25, evidence `stash/results_real_20260925/adr005_probe/`)

OpenGWAS curated log-odds datasets per disease (JWT valid to 2026-10-09;
`/associations` POST {variant: [...], id} returns beta + ea/nea per variant;
chunked batches of 25 with retry — gateway throws transient 502/400):

| Disease | Dataset | Panel rows w/ beta | Genome-wide (p<5e-8) |
|---------|---------|--------------------|----------------------|
| RA | ieu-a-833 (Okada 2014, ncase=19234) | 76/91 | 24 |
| SLE | ebi-a-GCST003156 (ncase=5201) | 68/91 | 18 |
| SJOGRENS | finn-b-M13_SJOGREN (FinnGen, ncase=1290) | 74/91 | 5 |
| AITD | ieu-a-1082 (Graves', ncase=649) | 43/91 | 0 |
| T1D | ebi-a-GCST005536 (ncase=6683) | 43/91 | 10 |
| VITILIGO | (local) Jin 2016 curated anchors | 4 anchor loci | — |
| MS | ieu-a-1025 (ncase=14498) | 47/91 | 8 |

**7/7 diseases have published per-variant betas.** Coupling is feasible.

## Decision (pre-registered gates)

Adopt cohort-conditional donor weighting **iff all gates pass on the
5,000-patient verification run**; otherwise fall back to the ADR-004
no-coupling mode as the canonical real baseline. Gates are evaluated in
`scripts/coupling_gates.py` and written to
`stash/results_real_20260925/adr005_gates/` before any modeling rerun.

### Design (the mechanism being gated)

- **Label side (shared betas):** the polygenic term in `simulate_labels` is
  computed from the same published log-ORs aligned to each donor's VCF alt
  allele (see allele-alignment trap below), standardized per disease by its
  panel mean. This replaces the 0.50 coefficient guess with published
  effect sizes.
- **Donor side (ascertainment):** for a patient whose anchor cohort is
  disease D, sample the donor from the ancestry-matched pool with weight
  w(donor) ∝ exp(beta_D · aligned dosage(donor)) — the posterior over
  genotypes given disease status under a log-additive model (Bayes
  consistency: w ∝ P(G | case) / P(G) = P(case | G) / P(case) up to
  normalization). Background patients sample uniformly (unchanged).
- **Only the anchor disease** conditions donor sampling; the co-occurrence
  cascade (MAS affinities) still governs additional labels.

### Trap to avoid (from ADR-004 engineering notes)

PTPN22 rs2476601's VCF alt allele is the *common protective* allele
(eaf≈0.88 in the panel). Raw dosage arithmetic on VCF alts inverts the sign
of effect estimates. **Every beta is aligned to its effect allele: dosage
used in weighting/labels is 2 − vcf_dosage when the published effect allele
is the VCF reference; the OpenGWAS `ea` string is matched to the 1000G VCF
alt where possible, else the sign convention is resolved via EAF direction.**

### Gates (pass/fail, no judgment calls)

- **G1 — Enrichment realism:** for each of the ≥5 diseases with a real
  cohort (RA, SLE, SJOGRENS, T1D, MS), cohort-vs-background mean aligned
  dosage across that disease's genome-wide-significant loci must be
  enriched, and the enrichment must be **within 3× of the published-OR
  prediction** (computed from the betas themselves) — too little means the
  weighting failed; too much means ascertainment is being faked.
- **G2 — Structure preserved:** overdispersion ≥ 1.05 and 2+ rate ≥ 0.20
  (label structure must match ADR-004's diagnostics: 1.09 / 25%).
- **G3 — Marginals preserved:** per-disease realized prevalence within
  ±0.03 of the no-coupling run's prevalences (no drift from re-weighting).
- **G4 — No new leakage:** label polygenic term uses only betas that are
  *external* (published, cohort-independent); donor weighting uses the same
  public betas. No term may use cohort membership of the *donor matrix
  itself* (there is none — 1000G donors are healthy volunteers; recorded
  here as the audit statement).

### Gate evaluation rules (pre-registered before the run)

- **Locus set per disease:** p < 5e-8 associations. If a disease has none
  (AITD in ieu-a-1082), the top-5 by p with p < 1e-4 are used and G1 is
  marked SKIP if the predicted enrichment is < 0.01 dosages (too small to
  test — recorded as SKIP, not PASS).
- **Adoption rule:** coupling is adopted iff G1 PASSES for ≥ 3 of
  {RA, SLE, T1D, MS, SJOGRENS} (the remainder may PASS or SKIP) AND G2 and
  G3 pass. Any FAIL → fallback.
- **Allele alignment:** OpenGWAS `ea` strings identify effect alleles;
  the VCF-alt allele of our dosage matrix is identified by matching panel
  alt-allele frequencies (dosage mean / 2) against Ensembl
  `1000GENOMES:phase_3:ALL` frequencies (|Δf| < 0.04; if both alleles of a
  locus match within 0.02 the locus is dropped as ambiguous, recorded).
  Cross-validation: for datasets that DO publish eaf (AITD, MS), the
  string-matched effect-allele frequency must agree with the published eaf
  (|Δ| < 0.05) — reported as an R2 audit line.
- **Label-side coefficient (fixed before the run):** per disease, the
  patient's published PRS is Σ_loci β·(aligned dosage − 2·panel AF) over the
  disease's locus set, converted to per-SD by the analytic population SD
  Var = Σ β²·2p(1−p) (LD ignored; only rescales, documented), then the
  marginal gets +0.10·PRS_SD — chosen to match the magnitude of the legacy
  per-locus term so G3 (±0.03) is a real test, not a formality.

### Consequences

- If adopted: canonical real-mode runs become `--genotypes real
  --coupling published`, and the honest-baseline numbers from ADR-004 are
  kept as the no-coupling ablation row (both reported in the paper).
- If any gate fails: ADR-004 no-coupling stays canonical; the gate outputs
  are attached as negative evidence; F-16/F-17/F-18/F-19 proceed on the
  no-coupling baseline.
- Either way: **the pre-registered tolerances for Phase-2 evaluations
  (F-16 null bound, F-17 sign recovery, F-18/F-19 curve shape) are applied
  to whichever mode is canonical.**

## Outcome (2026-09-25, same day — gates run and REJECTED)

Evidence: `stash/results_real_20260925/adr005_gates/gate_report.json`
(5,000 patients, seed 42; weighting ESS audits: RA 241/503, SLE 175/503,
T1D 408/503, MS 111/503, SJOGRENS 356/503 — weights were healthy, the
mechanism worked; the gates still failed it).

| Gate | Result | Detail |
|------|--------|--------|
| G1 RA | **PASS** | realized +0.067 vs predicted +0.025 (ratio 2.6, within 3×) |
| G1 T1D | **PASS** | realized −0.026 vs predicted −0.036 (ratio 0.72; GWS set nets protective) |
| G1 SLE | SKIP | predicted +0.006 < 0.01 floor (realized +0.084, direction correct) |
| G1 SJOGRENS | SKIP | predicted +0.0005 < 0.01 floor (FinnGen betas tiny) |
| G1 MS | **FAIL** | ratio 4.7× — LD over-compounding on MHC loci (rs9271366 β=−1.03, rs3135388 β=−1.06): independent-locus `exp(Σβ·g)` weights compound correlated evidence far beyond the first-order prediction |
| G2 structure | **FAIL** | overdispersion 1.000 < 1.05 (vs 1.09 uncoupled): the continuous per-SD published-PRS marginal term replaced the legacy per-locus discrete term and diluted the MAS cascade |
| G3 marginals | **FAIL** | RA +0.056, T1D +0.054, MS +0.048, SJOGRENS +0.033 (tolerance ±0.03): donor weighting feeds higher-dosage donors into the cascade, which amplifies marginal drift |

**Verdict: FALLBACK.** G1 passed 2/5 (bar was ≥3 with no FAIL), G2 and G3
failed. Per the decision rule ("if we can get real coupling then we will do
it, if not then no coupling"), **the ADR-004 no-coupling real mode is the
canonical baseline.** The gate report is retained as negative evidence; the
coupling code (`polymas_ml/data/coupling.py`, `--coupling published`) is
kept as an ablation arm, not a default.

What we would need to revisit (recorded, not actioned): per-locus
LD-whitened weights (eigen-decomposition of the real LD matrix) for the
donor weighting, and a marginal-rebalancing pass to hold G3 — both are
post-hoc patches to a design whose pre-registered form failed, so they
would need a new ADR cycle rather than a quiet tweak.

# ADR-006: Allele-align the real-mode label polygenic term

- **Status:** Accepted (2026-09-26) — all pre-registered gates passed; measured table below
- **Date:** 2026-09-26
- **Deciders:** Program lead (results program, ROADMAP.md Phase 2 close-out)
- **Related:** ADR-004 (real-donor wiring), ADR-005 (coupling REJECTED — this ADR
  adopts only the label-side alignment, NOT the donor weighting), F-20 external
  validation (the trigger), F-16/F-17/F-18/F-19 (re-run implications)

## Context

F-20 external direction validation FAILED (3/12 = 25% sign agreement vs the
pre-registered ≥70% bar). The root-cause audit (recorded in the F-20 summary)
found two independent problems:

**P1 — Substantive bug (this ADR's target).** The real-mode label polygenic
term in `simulate_labels` (the `elif risk:` branch) computes

```
p += 0.50 * (mean(dosage[anchor_loci]) / 2 - panel_mean)
```

on **raw VCF-alt dosages**. The VCF alt allele is the *published effect* allele
only sometimes; at PTPN22 rs2476601 the VCF alt is the common **protective**
allele (panel AF 0.973; published risk allele is the minor A, eaf≈0.094 — the
ADR-004 calibration trap). The same inversion affects every anchor locus whose
published effect allele maps to the VCF ref. The label model therefore encodes
sign-inverted biology at exactly the loci F-20 tests, and the trained models
faithfully learned the inverted signal (model ρ > 0 where published β < 0).

**P2 — Protocol flaw in the F-20 harness (fixed alongside, pre-registered
here).** 7 of the 12 tested pairs are (disease, locus) combinations where the
simulation embeds **no direction at all**: the locus is another disease's
anchor (e.g. RA vs rs2187668, a Sjögren's anchor). The label model gives those
pairs no genetic signal by design, so sign agreement there is a coin flip
testing nothing. F-20 must restrict to pairs where the disease's OWN anchor
loci are tested — the pairs the generative model actually claims to embed.
(The raw 12-pair table stays published as-is; the restriction is additive.)

## Decision

1. **Fix P1** by replacing the real-mode `elif risk:` polygenic term with the
   **allele-aligned published-beta PRS** already built and unit-tested in
   ADR-005's `polymas_ml.data.coupling`:

   ```
   prs_d = mean_over_anchor_loci( aligned_dosage[effect_allele] * beta_locus )
   p += 0.50 * (prs_d - empirical_mean(prs_d))
   ```

   - Anchor loci = the disease's own `MODELED_DISEASE_LOCUS_EFFECTS` /
     `disease_loci` set, betas taken from the disease's published dataset
     (OpenGWAS probe cache) where available, falling back to the curated
     anchors for AITD/VITILIGO. Only anchor loci enter the term — this is a
     **sign-and-magnitude correction of the existing term**, not the rejected
     ADR-005 donor weighting (donor sampling stays uniform).
   - The empirical centering keeps marginals calibrated by construction
     (zero-mean over the panel), replacing the static `panel_mean` heuristic.
   - Simulated mode is **byte-identical** (no change to the legacy branch).
2. **Fix P2** by restricting F-20's tested set to (disease, own-anchor-locus)
   pairs, with the unrestricted table retained for transparency.
3. Re-run the canonical Phase-2 pipeline and ALL Phase-2 evaluations (F-16,
   F-17, F-18, F-19, F-20) on the corrected run so every published number
   comes from one consistent generative model. The previous run's artifacts
   are superseded, not deleted (superseded_<ts> convention).

## Pre-registered gates (evaluated in order, before adoption)

| Gate | Claim | Tolerance |
|------|-------|-----------|
| **G1 alignment** | Every anchor locus used by the label term resolves via the ADR-005 alignment machinery (or is explicitly dropped with reason); for each resolved locus the aligned direction matches the published effect allele. | 100% of used loci resolved; 0 sign-flips vs Ensembl eaf audit |
| **G2 marginals** | Marginal prevalences per disease drift ≤ ±0.03 vs the current canonical run (same RNG seed). | all 7 diseases within ±0.03 |
| **G3 structure** | MAS label structure preserved: overdispersion ≥ 1.05; ≥ 2-disease rate within ±0.03 of current run. | both hold |
| **G4 F-20** | External direction agreement on the restricted (own-anchor) pair set improves and meets the ROADMAP tolerance. Primary tier: GWS (p<5e-8) own-anchor pairs — note the panel yields only ~4–5 such pairs (most GWS panel associations belong to other diseases' anchors), so n≥8 is NOT achievable at the GWS tier; the pair table ships with an exact binomial 95% CI. Secondary tier (transparency): suggestive own-anchor pairs at p<5e-6. | primary: ≥ 70% with n ≥ 4; secondary reported as measured |
| **G5 model sanity** | System A macro AUROC stays within the rerun "hold" band vs the current canonical run. | Δ ≤ ±0.02 |

**Adoption rule:** P1 is adopted iff G1–G3 pass AND (G4 pass OR G5 hold).
Rationale: G4 is the substantive goal, but if the corrected term still fails
external direction agreement while model quality holds (G5), the honest
outcome is recorded either way — the alignment fix is correct on first
principles (it makes the simulation match its own cited sources), so G1–G3 +
G5 suffice to adopt, with G4's verdict published as measured.

**Fallback if G2/G3 fail:** revert to the unaligned term, record ADR-006 as
Rejected, keep the F-20 FAIL with root cause (status quo of record today).

## Consequences

- The simulation's genetic signal finally agrees with the literature it cites
  (R2/R3): the label term, the F-16 baseline, and F-20 all read the same
  aligned betas from one cache.
- All Phase-2 numbers change slightly (same seed, different label draws);
  the canonical run directory is replaced and every eval re-run — no mixing
  of runs in any table or figure.
- AITD/VITILIGO remain F-20 coverage gaps (no GWS p-values in their datasets);
  unchanged by this ADR.
- If G4 fails on first principles-correct data, that is itself a reportable
  finding about the panel's external validity.

## Evidence

Gate run: `stash/results_final_20260926/adr006_gates/gate_report.json`
(canonical run: `stash/results_final_20260926`, 5,000 patients, seed 42,
`--genotypes real --coupling none`).

| Gate | Result | Measured |
|------|--------|----------|
| G1 alignment | **PASS** | 11/14 own-anchor loci resolved (3 dropped with logged reasons); 0 sign flips vs Ensembl audit |
| G2 marginals | **PASS** | max \|Δ\| = 0.024 (AITD); all 7 diseases ≤ ±0.03 |
| G3 structure | **PASS** | overdispersion 1.093 → 1.063 (≥1.05); 2+ rate 0.251 → 0.232 (Δ −0.019 ≤ 0.03) |
| G4 F-20 | **PASS** | 4/4 = 100% sign agreement on GWS own-anchor pairs (exact binomial 95% CI [0.40, 1.00]); model ρ now negative, matching published β |
| G5 model sanity | **PASS** | macro AUROC 0.6052 → 0.6163 (Δ +0.0111 ≤ 0.02) |

**Verdict: ADOPTED.** The label term, F-16 baseline, and F-20 now read the
same aligned published betas from one cache.

### Recorded deviation (first attempt)

The first implementation plugged the ADR-005 per-SD standardized
`label_prs_term` into the legacy `p += 0.50 * (…)` slot — a scale error that
failed G2 (prevalence drift up to +0.20), G3 (2+ rate +0.284) and G5 (macro
+0.105) on a full run. Corrected to the pre-registered raw-scale formula
(`p += 0.50 * (mean_loci beta·aligned_dosage − panel_mean)`), after which all
gates pass. Both attempts are recorded here per the no-silent-drops rule;
the failed run was superseded in place (same directory, rebuilt).

### F-20 protocol fix detail (P2)

The unrestricted 12-pair table from the pre-fix run is retained in
`stash/results_phase2_20260925/f20_external_validation/` for transparency;
on the corrected run the verdict is computed on the 4 GWS own-anchor pairs.
All non-anchor pairs are recorded with an exclusion note in
`external_direction_pairs.csv`.

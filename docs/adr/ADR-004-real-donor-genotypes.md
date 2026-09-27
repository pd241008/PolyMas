# 📜 ADR-004: Real-donor genotype wiring (F-10 → production pipeline)

> **Status:** `Decided`
> **Date:** September 2026
> **Items:** F-10 (pipeline-wiring half), Phase 2 kickoff

---

## 🌎 Context

F-10's core (Phase 1) produced a validated real-genotype substrate: 91 panel
variants typed on 2,504 1000G phase-3 donors, LD gate passed 13/13. The
production pipeline (`run_real_pipeline.py` → `simulate_genotypes_prs` →
`simulate_labels`) still generates genotypes via Binomial(2, q) draws with
ad-hoc cohort enrichment. Reviewers read fully-simulated genotypes as the
weakest link; real haplotype background removes that objection while the
cohort-informed label model (MAS structure) remains the honest phenotype bridge.

Constraints: the PRS/k-mer contracts (`prs_rows`, `gen_rows`, per-locus
z-scores feeding System B's genotype tokens) must not change shape — Systems
A and B both consume them — and the label model's prevalence calibration
assumed simulated allele frequencies.

## 🛤️ Options Considered

1. **Replace simulation wholesale with real donors + real labels** — impossible
   ethically/practically: no paired genotype+autoimmune-label cohort is
   accessible at this stage (All of Us DURA pending), and fabricating one is
   out of scope.
2. **Real donor genotypes, cohort-informed simulated labels (chosen)** —
   patients inherit real dosage vectors; the MAS label model stays, with its
   polygenic term recalibrated for real MAFs. Honest framing: "real genetic
   background, modeled phenotypes."
3. **Real genotypes only for a validation subset** — adds a second cohort
   schema without removing the simulated objection; defers the benefit.

---

## 🎯 Decision

> [!IMPORTANT]
> **Add `--genotypes {simulated,real}` to the pipeline (default simulated,
> legacy behavior byte-identical). In real mode, each patient's genotype
> vector is a REAL 1000G donor's dosage, sampled ancestry-matched from the
> F-10 substrate; the label model's polygenic term standardizes by the
> empirical panel mean instead of the hardcoded 0.15. Cohort enrichment at
> risk loci is disabled in real mode (donor haplotypes are what they are).**

## 🧠 Reasoning

Ancestry-matched donor sampling keeps the existing ancestry-stratified
evaluation meaningful (donor super-population ≈ ImmPort-mapped ancestry), and
reusing the same `rng` stream keeps runs reproducible. The polygenic-term
recalibration is required, not cosmetic: rs2476601's VCF alt allele is the
common protective allele (eaf 0.88), so raw-dosage arithmetic shifts
prevalence out of calibration — the same trap the F-06 ablation hit. Keeping
simulated mode as the default preserves every historical result and the
R2/R3 claims already registered against it.

## ⚖️ Consequences

- **Good:** 🟢 Real LD by construction across patients; genotype features and
  System B genotype tokens now describe measured variation; the "fully
  synthetic" objection weakens to "modeled phenotypes on real genetics."
- **Bad:** 🔴 Two genotype modes exist and results are mode-specific — every
  results table must state its mode; the 91-variant substrate covers the old
  8-locus panel plus expansions, but legacy enrichment behavior is not
  reproducible in real mode (by design).
- **Neutral:** ⚪ Donor ids are recorded in provenance for full traceability.

## 🔄 Revisit When

- An individual-level real cohort becomes accessible (switch labels to real,
  retire the simulation entirely).
- The panel grows again (re-run the substrate; the loader keys off
  `real_genotypes_<date>` folders).
- System B's k-mer builder consumes donor dosages (next integration commit).

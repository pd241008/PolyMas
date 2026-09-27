# PM-008: F-20 external validation — allele-sign inversion (FIXED)

- **Status:** Closed (root-caused, fixed via ADR-006, re-validated PASS — 2026-09-26)
- **Item:** F-20 — external direction validation (first pass)
- **Category:** 🐛 Real bug — the one outright code defect in the program
- **Related:** ADR-006 (the fix), ADR-004 (the calibration trap), F-16/F-17/F-18/F-19 (re-run implications), 🧯 Failure Taxonomy in docs/ROADMAP.md

## Impact

The first external-validation pass **FAILED its pre-registered bar**:
direction agreement 3/12 = 25% vs the ≥70% requirement. Because this failure
was taken seriously rather than tuned around, it exposed a substantive bug
in the label model that affected every real-mode result — the highest-value
failure of the program.

## Timeline

- 2026-09-25 — First pass runs: 3/12 = 25% sign agreement. FAIL recorded.
- 2026-09-25 — Root-cause audit (recorded in the F-20 summary) finds two
  independent problems: **P1** a substantive bug (allele-sign inversion at
  protective-alt loci in the real-mode label polygenic term) and **P2** a
  protocol flaw (non-anchor pairs diluting the test).
- 2026-09-25/26 — ADR-006 drafted with pre-registered gates G1–G5: allele-
  align the label term to the published effect strand + restrict the
  re-test to the pre-registered-narrower own-anchor pair set.
- 2026-09-26 — Corrected re-run: **all 5 gates PASS**, macro AUROC
  0.605 → 0.616; own-anchor sign agreement **4/4 = 100%**, CI [0.40, 1.00].
  Everything downstream re-run; the failed first-pass tables preserved under
  `superseded_first_pass_20260925/`.

## Root cause

**P1 — the bug.** The real-mode label polygenic term computed the PRS
contribution on **raw VCF-alt dosages**. The VCF alt allele is the published
effect allele only sometimes: at PTPN22 rs2476601 the VCF alt is the common
**protective** allele (panel AF 0.973; the published risk allele is the
minor A, eaf ≈ 0.094 — the trap ADR-004 had already flagged for donor
calibration). The same inversion affected every anchor locus whose published
effect allele mapped to the VCF ref strand. The label model therefore
encoded **sign-inverted biology at exactly the loci F-20 tests**, and the
trained models faithfully learned the inverted signal (model ρ > 0 where
published β < 0).

**P2 — the protocol flaw.** The 12-pair test included pairs with no
individual-level anchor, diluting a signal that was partly present.

## Contributing factors

- Strand/effect-allele identity lives in different sources (GWAS Catalog
  effect alleles vs VCF ref/alt) with no cross-check at ingestion.
- The simulator's genotype-enrichment (ADR-004's q+0.25) had made the
  inversion partially invisible in earlier synthetic-only modes — real
  donors exposed it. **The bug was caught by its own downstream
  consequence**, which is the strongest argument for the external
  validation gate existing at all.

## What went well

- The failure was treated as information: no threshold adjustment, no
  pair-dropping until the mechanism was understood.
- The fix was pre-registered as an ADR with gates BEFORE the re-run; the
  failed first-pass artifacts were preserved (superseded, not deleted).
- The reviewer caught two related reporting inconsistencies later; both
  reconciled (see the Reconciliation log).

## Lessons

1. **Allele alignment is a correctness requirement, not a preprocessing
   detail** — effect strand must be asserted (and tested) wherever dosages
   meet published effect sizes.
2. External validation gates earn their cost precisely when they fail.
3. Never delete failed first-pass evidence: the before/after pair is what
   makes the fix auditable.

## Action items

- ✅ ADR-006 adopted; gates G1–G5 all PASS; F-16/F-17/F-18/F-19 re-run on the
  corrected canonical run.
- Allele-alignment unit tests live in
  `tests/test_adr006_aligned_label_term.py`; the loader asserts effect-strand
  provenance for every locus.

# PM-009: Benign nulls — F-15 MC-dropout arm & F-06 haplotype features

- **Status:** Closed (recorded; no action warranted — 2026-09-25/27)
- **Items:** F-15 MC-dropout sub-gate; F-06 haplotype/epistasis negative result
- **Category:** ➖ Benign nulls (mechanism known, bound understood)
- **Related:** F-15 deep-ensemble PASS (marginal), F-09/F-10 substrate, 🧯 Failure Taxonomy in docs/ROADMAP.md

These two are grouped because neither represents a defect or a surprise: both
are pre-registered nulls whose mechanisms were measured, and neither has a
cost-justified follow-up.

---

## F-15 — MC-dropout sub-gate (ΔNLL +0.0000 vs required < 0)

**What was measured:** MC-dropout (T=30) matches but does not beat the single
model on NLL (0.3791 vs 0.3791, macro) — the deep-ensemble arm (ΔNLL −0.0009)
carries the F-15 PASS; the ensemble also halves ECE (0.0187 → 0.0087).

**Root cause:** dropout-at-inference adds predictive *variance* without adding
*information*. The five ensemble members are near-identical (member val
macro 0.522–0.560) — the model family has almost no functional diversity, and
dropout perturbations of the same basin cannot manufacture any. T=30 is also
modest, but the binding constraint is diversity, not samples.

**Lesson:** uncertainty-quantification gains are bounded by member diversity;
pre-register a diversity check (member disagreement) before claiming
calibration improvements from ensembling/dropout.

**Action:** none. T=100 + temperature scaling might shave the rounding hair —
declined (not worth GPU time; would not change the structural reading).

---

## F-06 — haplotype/epistasis features (ΔAUROC 0.0005, null bar 0.005)

**What was measured:** ground-truth injection through real 1000G dosages
(main + epistasis coefficients, 47% prevalence, n=3,000) proved the harness
recovers planted signal (AUROC 0.665 ≫ 0.5) — but augmented-vs-base ΔAUROC on
real labels was 0.0005: **no tree-model gain**.

**Root cause:** GBMs learn pairwise interactions natively; explicit
interaction features are redundant for this model family. The harness PASS +
feature null is the pre-registered outcome, not a broken pipeline.

**Lesson:** feature-engineering claims must specify the model family they
target — interaction features have a different value proposition for linear
models (F-16's PRS baseline) than for GBMs.

**Action:** machinery retained and validated (it powers F-16's linear
baseline and interpretability); production-label wiring remains deferred
(Phase-1 legacy row in the ⏸ register) — *decide later*.

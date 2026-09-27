# PM-003: F-04 gated fusion — parent dominance

- **Status:** Closed (negative result recorded, 2026-09-26)
- **Item:** F-04 — clinical+genetic gated late fusion
- **Category:** 🚧 Parent dominance
- **Related:** F-03 (System B checkpoint used), F-13 (tuned System A context), 🧯 Failure Taxonomy in docs/ROADMAP.md

## Impact

The claim — *gated late fusion beats the better parent system* — failed
unconditionally: fused macro 0.5738 vs best-parent 0.6210; System A wins on
6/7 diseases; worst fused disease −0.139 (MS). The macro gate fails under
ANY α — even α=1 (pure System A) yields 0.6163 < 0.6210.

## Timeline

- 2026-09-26 — Claim pre-registered: per-disease mixing weight α fit on val
  by bounded logloss grid; fusion must exceed max(System A, System B).
- 2026-09-26 — First run used unbounded val-logloss α; the fit blew up
  (|α| ≫ 1, exploiting separable val errors) — estimator fixed to a bounded
  [0, 1] grid before the verdict was recorded.
- 2026-09-26 — Bounded re-run confirms FAIL; amended note recorded.

## Root cause

**Parent dominance**: System B's test probabilities (from the F-03 flat
checkpoint) trail System A on 6/7 diseases. A convex combination of a
dominated system and its dominator cannot exceed the dominator — the
geometry of the claim makes it unpassable while dominance holds.

Estimator caveat (recorded, not load-bearing): System A's val rows are
in-sample for the base learners, so val logloss under-weights A; even
granting the caveat, B's dominance failure is decisive on test.

## Contributing factors

- The claim was pre-registered before F-03's per-disease table made B's
  weakness per-disease explicit; a dominance pre-check would have flagged
  the claim as near-unpassable at this B quality.

## What went well

- The α-fit pathology was caught by inspecting the fitted weights (α≈0,
  |α|>1 artifacts) and fixed transparently — the amendment is in the ledger,
  not silently patched.
- The "fails under any α" analysis turned a tuning complaint into a proof.

## Lessons

1. Fusion claims need a **dominance pre-check**: if one parent loses on most
   labels, late fusion is geometry, not learning.
2. Unbounded mixture weights on separable validation errors extrapolate;
   constrain estimators to the operationally meaningful range (mixing
   weights ∈ [0, 1]).

## Action items

- Recorded negative; sequence-side (feature-level) fusion mooted by the same
  dominance.
- Revisit ONLY if System B becomes competitive (its val ceiling ~0.55–0.575
  is a data/architecture limit per F-13/F-19) — **decide later**.

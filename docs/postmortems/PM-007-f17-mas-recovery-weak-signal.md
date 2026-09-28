# PM-007: F-17 MAS-pair recovery — weak signal vs noise

- **Status:** Closed (negative result recorded, 2026-09-26); **amended 2026-09-28** — F-17a executed (FAIL) and the aggregate metric shown to be structurally biased
- **Item:** F-17 — MAS pairwise-structure recovery
- **Category:** 📉 Regime absence (weak signal)
- **Related:** F-20 (own-anchor protocol, ADR-006), ⏸ Deferred register (F-17a), 🧯 Failure Taxonomy in docs/ROADMAP.md

## Impact

The claim — *model-confusion structure recovers the published MAS pairwise
co-occurrence affinities* — failed its pre-registered bar: **9/19 sign
agreement (47.4%)**, binomial p = 0.68 vs the ≥70% bar. Chance-level overall.

## Timeline

- 2026-09-26 — Pre-registered on the corrected canonical run (post-ADR-006);
  per-pair confusion/phi tables written to
  `results/results_final_20260926/f17_mas_recovery/`.
- 2026-09-26 — Verdict FAIL recorded the same day, with the per-pair table
  retained (it drives the deferred re-test).

## Root cause

**The co-occurrence signal in the label generator is too weak relative to
sampling noise at n = 5,000** for the model's error structure to encode all
19 pairwise affinities. The aggregate hides a sharp split:

- The **incoercible pairs** (T1D–MS, RA–MS — exclusions the generator
  structurally suppresses) were recovered **sharply and consistently**.
- Weakly-anchored pairs (small log-ORs) sit inside the noise band and agree
  or disagree effectively at random — they dominate the 19-pair denominator
  and drag the fraction to chance.

So the failure is not "the model learned nothing about MAS structure" but
"the test demanded uniform recovery across pairs of wildly different anchor
strength".

### 2026-09-28 amendment (F-17a): the aggregate was structurally biased

The F-17a re-test (own-anchor restriction, 3/7 = 42.9%, p = 0.77 — FAIL)
surfaced something better than the anchor-strength story: **7 of the 19
embedded pairs have embedded_log_or = 0.0**, and `sign(0)` can never equal
`sign(phi)` in {+1,−1} — those 7 "disagreements" are DETERMINISTIC artifacts
of the scoring rule, not evidence about the model. F-17's effective
denominator was 12, not 19. Restricted to planted-affinity pairs: **9/12 =
75%, binomial p = 0.073** — consistent with the pre-registered bar rate but
still short of the 0.05 gate. Also corrected in this pass: T1D–MS / RA–MS
are NOT own-anchor pairs (no shared anchor locus); the sharp negative-phi
recovery stands as separate F-17 evidence.

## Contributing factors

- The pre-registration counted all 19 pairs equally; no anchor-strength
  weighting or stratification was specified, though the odds table's dynamic
  range was known at registration time.

## What went well

- Per-pair tables were retained exactly so a better-specified re-test could
  reuse them — no re-training needed.
- The incoercible-pair recovery was surfaced in the same verdict rather than
  buried (honest reporting of a mixed result).

## Lessons

1. Agreement tests over heterogeneous effect sizes should pre-register
   stratification (or weighting) by anchor strength; equal-weight counts
   dilute the detectable signal.
2. "Recovered vs not" is often "strong-anchor vs weak-anchor" — report the
   split even when the aggregate fails.

## Action items

- Recorded negative.
- **F-17a EXECUTED 2026-09-28**: pre-registered gate FAIL (3/7 anchor pairs,
  p = 0.77), but the post-hoc decomposition (clearly labeled descriptive in
  `f17a_summary.json`) converts this from "model at chance" to "metric
  biased by construction + signal recoverable where planted". Manuscript
  framing: report the planted-affinity stratum (9/12 = 75%, p = 0.073) as a
  descriptive result alongside the pre-registered failures — never as a
  gate pass. Evidence: `results/results_final_20260926/f17a_anchor_recovery/`.
- New guardrail adopted: agreement tests must assert that every counted item
  has a defined agreement target (nonzero embedded direction) at
  registration time, not at analysis time.

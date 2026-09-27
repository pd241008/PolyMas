# PM-002: F-02 disease-graph head — stacking bottleneck

- **Status:** Closed (negative result recorded, 2026-09-26)
- **Item:** F-02 — MAS-aware disease-graph head
- **Category:** 🚧 Information bottleneck
- **Related:** F-17 (MAS odds table), ADR-006 canonical run, 🧯 Failure Taxonomy in docs/ROADMAP.md

## Impact

The claim — *a learned disease-disease adjacency (MAS-odds init) improves
co-occurrence calibration at no AUROC cost* — failed on both gates: test
macro 0.5751 vs System A 0.6163 (floor −0.01) and 0.5751 vs its own
matched-capacity no-graph control 0.5796 (−0.0046).

## Timeline

- 2026-09-26 — Claim pre-registered: graph head vs control on 3-fold OOF
  base predictions (heads never see in-sample scores); phi recovery and
  AUROC both gated.
- 2026-09-26 — Both arms trained; adjacency report + per-disease table
  written (`results/results_final_20260926/f02_disease_graph/`).
- 2026-09-26 — Verdict: FAIL on AUROC gates; phi recovery 8/19 (vs control
  6/19, System A 9/19) — direction positive, within noise.

## Root cause

A **7-probability stacking bottleneck**: every graph head sees only what the
ensemble compresses into 7 numbers, while System A decides from 23 features.
Every head trails System A (worst: AITD −0.104). No adjacency prior can
recover information the input never contained.

Two secondary observations sharpen the mechanism:

- The graph path itself was NOT the liability — it beat its own control on
  4/7 diseases, and the learned adjacency stayed near its MAS-odds init
  (`adjacency_report.json`): the data did not contradict the published
  structure.
- The bottleneck, not the graph, is the binding constraint.

## Contributing factors

- Stacking-on-probabilities was chosen to guarantee OOF hygiene; the same
  hygiene starved the heads of feature signal. The control arm (same
  bottleneck, no graph) isolates this cleanly — the design did its job.

## What went well

- The matched-capacity control converted "graph head loses" into the sharper
  "bottleneck loses, graph path mildly helps" — a much better finding.
- Learned-adjacency-vs-init comparison gave a free plausibility check on the
  MAS odds table.

## Lessons

1. In stacked designs, the input width of the meta-learner is the ceiling —
   gate claims that assume the meta-learner can find signal the base layer
   already discarded.
2. Controls should differ in exactly one structural element (here: graph vs
   no-graph) so failure attribution is mechanical.

## Action items

- Recorded negative; phi-recovery direction noted for the discussion.
- Feature-hybrid head (graph head sees OOF probabilities AND features):
  candidate for a future cycle — **decide later**; it is a new protocol, not
  a patch of this claim.

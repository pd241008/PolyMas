# PM-001: F-01 LD-GNN — structural mismatch between claim and panel design

- **Status:** Closed (negative result recorded, 2026-09-26)
- **Item:** F-01 — LD-aware GNN (System C)
- **Category:** 🏗️ Structural contradiction
- **Related:** F-09 (panel expansion, ADR-002), F-10 (real genotypes), 🧯 Failure Taxonomy in docs/ROADMAP.md

## Impact

The claim — *message passing over real LD-r² edges matches the GBM ensemble
(tolerance ±0.02 macro AUROC)* — failed decisively: test macro 0.5401 vs
System A 0.6163 (Δ −0.076). No per-disease win except SJOGRENS (+0.036).

## Timeline

- 2026-09-25 — F-09 panel expansion accepted (ADR-002): 91 loci curated with
  an explicit LD-pruning criterion to avoid redundant correlated features.
- 2026-09-26 — F-01 pre-registered (edge matrix to be validated vs published
  HLA LD before training); `polymas_ml/graph/ld_gnn.py` + runner built.
- 2026-09-26 — 2-round message-passing model trained on the canonical split;
  LD edge audit run as part of the pre-registered validation.
- 2026-09-26 — Verdict: FAIL with structural root cause recorded the same day.

## Root cause

The panel was **deliberately LD-pruned at selection** (the F-09 design goal),
so the graph the claim needs does not exist in the data:

- `results/results_final_20260926/f01_ld_gnn/ld_edge_audit.json`: only
  **7 edges ≥ r²0.2 among all 91 substrate loci** (MHC block present, max
  r² = 0.846 — the pruning kept one representative per block).
- With a near-edgeless graph, message passing reduces to a per-node MLP; the
  model cannot out-leverage System A's 23 engineered features.

The claim and the panel design are mutually exclusive **by construction** —
the failure is a property of the design pairing, not of training.

## Contributing factors

- The F-01 claim was written against the generic LD-GNN literature, not
  re-checked against the specific panel's LD budget before pre-registration.
- The edge audit WAS pre-registered (good) — but only as a validation gate,
  not as a go/no-go feasibility check before spending GPU time.

## What went well

- The pre-registered edge audit turned the failure into a *quantified
  structural finding* (7 edges) instead of a vague "GNN underperforms".
- Same-split, same-labels comparison made the verdict unambiguous.

## Lessons

1. Any graph-structure claim needs a **structure-feasibility check** (edge
   budget) against the actual substrate BEFORE pre-registration.
2. Deliberate design choices (LD pruning) create hidden exclusivity clauses
   on downstream claims — cross-check new claims against prior design ADRs.

## Action items

- Recorded negative; SJOGRENS anomaly noted for the discussion section.
- LD-rich panel variant: **declined for now** — requires a new ADR + panel
  rebuild; revisit only if a graph claim becomes manuscript-load-bearing.
- Feasibility-check-first rule adopted for future structural claims (see
  PM-005's equivalent for HPO budgets).

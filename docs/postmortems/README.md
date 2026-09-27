# Postmortems

Blameless postmortems for every failed pre-registered gate in the results
program (Phases 1–4). One document per failure; benign nulls grouped. Format
follows the ADR conventions in `docs/adr/` (status header, evidence paths,
no silent patches). The one-line version of each lives in the
**🧯 Failure Taxonomy** section of [ROADMAP.md](../ROADMAP.md).

| PM | Item | Category | Disposition |
|----|------|----------|-------------|
| [PM-001](PM-001-f01-ld-gnn-structural-mismatch.md) | F-01 LD-GNN | 🏗️ Structural contradiction | Recorded negative; LD-rich panel = new ADR (decide later) |
| [PM-002](PM-002-f02-disease-graph-stacking-bottleneck.md) | F-02 disease-graph head | 🚧 Information bottleneck | Recorded negative; hybrid head = new protocol |
| [PM-003](PM-003-f04-fusion-parent-dominance.md) | F-04 gated fusion | 🚧 Parent dominance | Recorded negative; moot until System B competitive |
| [PM-004](PM-004-f05-focal-loss-regime-absence.md) | F-05 focal/cost-sensitive loss | 📉 Regime absence | Recorded negative; no fix applicable here |
| [PM-005](PM-005-f13-hpo-selection-budget-mismatch.md) | F-13 System B HPO | 🧪 Protocol artifact | Deferred as F-13a (decide later) |
| [PM-006](PM-006-f14-ssl-objective-misallocation.md) | F-14 SSL pretraining | 🎯 Objective misallocation | Deferred as F-14a (decide later) |
| [PM-007](PM-007-f17-mas-recovery-weak-signal.md) | F-17 MAS recovery | 📉 Regime absence (weak signal) | Recorded negative; F-17a candidate (decide later) |
| [PM-008](PM-008-f20-allele-sign-inversion.md) | F-20 external validation (first pass) | 🐛 Real bug — FIXED | ✅ ADR-006; re-run 4/4 = 100% |
| [PM-009](PM-009-benign-nulls.md) | F-15 MC-dropout arm + F-06 haplotypes | ➖ Benign nulls | Recorded; no action |

## Format

Each postmortem answers, in order:

1. **Impact** — what claim was being tested, what the gate required, what was measured.
2. **Timeline** — when the claim was pre-registered, when it ran, when the verdict landed.
3. **Root cause** — the falsifiable mechanism, with the evidence file that proves it.
4. **Contributing factors** — what made the failure likely (or made it look like something else).
5. **What went well** — the guardrail, control, or pre-registration that caught it honestly.
6. **Lessons** — transferable program-level lessons.
7. **Action items** — done / deferred (with the register link) / declined (with reason).

## Rules

- Blameless: no "should have known"; only mechanisms and guardrails.
- No silent patches: every fix points to an ADR or a tracked deferral.
- Evidence paths are relative to `results/` (the tracked JSONs reproduce every number cited).

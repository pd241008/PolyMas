# 📜 ADR-001: Adopt a 24-feature results program with typed verification

> **Status:** `Decided`
> **Date:** September 2026

---

## 🌎 Context

PolyMas has a stable, reproducible two-system pipeline (System A GBM ensemble,
System B Mamba over k-mers) verified end-to-end on 2026-09-25. A backlog of 24
candidate features has accumulated across five categories: model architectures,
features/inputs, training/calibration, evaluation/rigor, and productization.
Executing them ad hoc risks (a) silent scope dropping, (b) unmeasured model
claims, and (c) new models being compared against evaluation rails that are
themselves still moving (e.g. the test-swept best-F1 threshold, item F-12).

Per Design-Dungeons §ml-and-research (honesty-first policy, artifact-design
principles P1–P6), research programs need a navigable claim map and typed
reproducibility guarantees, not a feature list.

## 🛤️ Options Considered

1. **Ad-hoc feature queue (issue list only)** — cheap, but nothing forces a
   verification standard per item; silent scope drops remain invisible.
2. **Full REVIEWER_GUIDE/CLAIM_MAP/VERIFY.md artifact suite now** — the
   Design-Dungeons end-state, but premature before canonical results exist to
   map claims onto; would be documentation debt on day one.
3. **Single tracked ledger (ROADMAP.md) + lightweight ADRs, artifact suite
   deferred** — claim-map discipline now, reviewer suite generated when the
   program stabilizes.

---

## 🎯 Decision

> [!IMPORTANT]
> **Adopt a single tracked ledger (`ROADMAP.md`) holding all 24 items as claims
> with R1–R4 verification levels and pre-registered R3 tolerances, plus
> lightweight ADRs in `docs/adr/` for decisions that change claims or
> tolerances.**

## 🧠 Reasoning

The ledger converts the backlog into claim form, which forces every item to
state upfront what would count as success, failure, or a null result — the
negative-results discipline from Design-Dungeons applies to model features too.
Typed levels (R1–R4) prevent the most common verification failure: a reviewer
(or future us) expecting bit-exact reproduction from a statistical result.
Pre-registering tolerances before running the checks converts "close enough"
into pass/fail and is only honest if stated in advance.

Option 2 was rejected as premature: the reviewer documentation suite is the
*output* of a completed program, not its input. Deferring it is recorded here so
the deferral is a decision, not an oversight.

## ⚖️ Consequences

- **Good:** 🟢 Nothing can be silently skipped; every feature lands as pass,
  fail, or documented null; comparisons across future runs are pre-registered.
- **Bad:** 🔴 Ledger upkeep is mandatory (status + evidence columns on every
  change); tolerances are now commitments — loosening one requires an ADR.
- **Neutral:** ⚪ `docs/` is gitignored here, so ADRs live locally while
  ROADMAP.md is the tracked source of truth.

## 🔄 Revisit When

- The program completes Phase 2 (then generate the reviewer artifact suite per
  Design-Dungeons artifact-design.md).
- Any tolerance proves consistently miscalibrated (e.g. ±0.02 AUROC too tight
  for fresh-seed Mamba reruns).
- External validation (F-20) reaches individual-level cohorts — the claim map
  will need a second tier for real-data claims.

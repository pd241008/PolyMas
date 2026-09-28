# ADR-007: Correct the MLM masking semantics (targets = original ids at masked positions)

- **Status:** Accepted (2026-09-28) — fix unit-covered; F-14 evidence annotated, not re-run
- **Date:** 2026-09-28
- **Deciders:** Program lead (during ⏸-register execution of F-14a)
- **Related:** F-14 (the affected evidence), F-14a (the follow-up that exposed it),
  PM-006 (SSL objective misallocation — addendum required), PM-008 (the
  precedent: bug found by a downstream gate, failed evidence preserved)

## Context

While building F-14a (context-only masking variant), two newly written unit
tests failed with losses ≈ ln(4100) ≈ 8.3 where a zero-guard should have
returned 0.0. Root cause: in `polymas_ml/sequence/ssl.py`, BOTH maskers
computed

```python
targets = input_ids.masked_fill(mask, -100)
```

which puts `-100` **at masked** positions and original ids elsewhere — the
inverse of the documented convention in the same docstring ("targets hold
ORIGINAL ids at masked positions and -100 elsewhere, so F.cross_entropy with
ignore_index=-100 scores exactly the masked set"). Downstream, `pretrain_loss`
selects scored positions via `sel = targets != -100`, which therefore selected
the **kept (visible)** tokens.

Consequence for the F-14 run: its MLM objective was not "predict the original
id at masked positions from context" but **"predict the token that is still
visible at that position"** — a degenerate identity-copy task. The recorded
mechanism (context memorization) remains real — the loss curve collapsed and
transfer hurt — but it was compounded: the encoder was never trained on the
documented masked-reconstruction task at all.

This also means the F-14 ledger row's mlm_loss numbers (4.84 → 0.015) measure
visible-token copying, not masked reconstruction.

## Decision

1. Fix both maskers (`mask_tokens`, `mask_context_tokens`) to the documented
   semantics: `masked_fill(~mask, -100)`. The F-14a pre-registered protocol
   runs on the corrected code.
2. Add a graph-connected zero guard in `pretrain_loss` for the (possible
   under context-only masking) empty scored set — plain CE would return NaN.
3. **Do not re-run F-14 silently.** Its evidence is annotated (ledger row,
   PM-006 addendum); re-running would require a fresh pre-registration and
   would erase the audit trail. The F-14 verdict (FAIL, −0.0189) stands as
   measured, with the mechanism amended to "objective misallocation **plus**
   degenerate identity-copy objective."
4. F-14a's verdict stands on its own pre-registered gates (FAIL by 0.0006,
   mean Δ +0.0094). The F-14 ↔ F-14a contrast is NOT attributable to the
   masking change alone and must not be quoted as a clean A/B.

## Consequences

- `tests/test_ssl.py` grows to 16 tests: corrected-convention assertions,
  context-only eligibility/never-score-genotype checks, zero-mask NaN guard
  (backward stays finite), and a default-path equivalence test proving the
  unflagged route matches manual masked CE.
- Any future SSL arm (locus-dropout, external-corpus) inherits the corrected
  semantics; its pre-registration must state that it is NOT comparable to the
  F-14 numbers.
- Manuscript: the SSL section reports F-14 as executed with a documented
  objective-semantics defect, mechanism amended; F-14a as the corrected-
  semantics follow-up that still missed its gate by 0.0006.

## Measured (recorded as run, 2026-09-28)

| Quantity | Value |
|---|---|
| F-14a mean paired val Δ (pre − scratch) | **+0.0094** (gate ≥ +0.01) |
| F-14a worst-seed Δ | +0.0026 (G2 passes) |
| F-14a pretrain mlm_loss (epochs 1→8) | 0.73 → 0.037 (corrected semantics) |
| F-14a verdict | **FAIL** (G1 missed by 0.0006; direction flipped vs F-14) |
| Evidence | `results/results_final_20260926/f14a_ssl/` |

## Addendum (2026-09-28, later): F-14b — the corrected-semantics mask-all cell

The missing ablation cell was pre-registered (commit `5ca1e38`, timestamped
before the run, with an explicit mechanistic expectation) and executed:

| Quantity | Value |
|---|---|
| F-14b mean paired val Δ | **+0.0213 — PASS** (G1 ≥ +0.01 ✓, worst seed +0.0087 ✓) |
| Scratch-arm vals | bit-identical to F-14a's (0.5323/0.5402/0.5379) — paired design verified |
| Pre-registered expectation | F-14b ≈ F-14a; **FALSIFIED** (\|Δ\| = 0.0119 ≫ seed noise) |
| Evidence | `results/results_final_20260926/f14b_ssl/` |

Consequence: the SSL story is no longer "a null either way." The falsified
expectation localizes the transferable signal: I(genotype→context) = 0 still
holds, but **I(genotype→genotype) > 0** — the generator's shared-liability
mixture and MAS cascade correlate dosages across loci, so uniform masking
(involuntarily) trains a small locus-dropout task. This is why F-14b
(+0.0213) beats F-14a (+0.0094): F-14a never masked genotypes. F-14's row
stands as executed; F-14b is NOT a re-run of F-14 (different pretrain seed
path, corrected objective, own pre-registration). Follow-up claim F-14c
(explicit locus-dropout) is deferred in the ⏸ register.

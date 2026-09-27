# PM-004: F-05 focal / cost-sensitive loss — regime absence

- **Status:** Closed (negative result recorded, 2026-09-26)
- **Item:** F-05 — class-imbalance-aware losses
- **Category:** 📉 Regime absence
- **Related:** ADR-006 canonical run, 🧯 Failure Taxonomy in docs/ROADMAP.md

## Impact

The claim — *focal loss lifts low-prevalence AUPRC (≥ +0.005 on the three
lowest-prevalence diseases) without costing AUROC* — failed in the cleanest
possible design: focal ΔAUPRC **−0.0045** on the target diseases (below the
+0.005 null bound) and an AUROC floor violation (−0.023, SJOGRENS). The
cost-sensitive context arm was worse (−0.0102).

## Timeline

- 2026-09-26 — Claim pre-registered (paper-default α=0.25, γ=2 — never tuned)
  with the single-family ablation design: only the loss differs.
- 2026-09-26 — `polymas_ml/models/focal.py` implemented; gradient/Hessian
  derivation caught twice by finite-difference tests before any run
  (component chain-rule form matches d(grad)/dz to 1e-11; 36 tests).
- 2026-09-26 — 21 LightGBM fits; verdict FAIL same day.

## Root cause

**The benefit regime does not arise in this cohort.** Prevalences here are
9–20%; logloss GBMs already cope well there. Focal loss's advantage case is
EXTREME imbalance (sub-1% positives), where easy-background gradients swamp
the rare-positive signal. The signature finding: focal's largest "gain" was
on **high**-prevalence RA (+0.005) — the opposite end of the claim, i.e. the
loss re-weighted effectively random direction because there was no imbalance
problem to solve.

## Contributing factors

- None mechanistic — the claim was plausible from literature and cleanly
  falsified. The implementation difficulty (negative focal Hessian at small
  p for y=1, h≈−0.01 at p=0.01) was a real discovery but did not drive the
  verdict (floored at 1e-6, visible per the no-silent-patches rule).

## What went well

- Single-family, single-difference ablation makes the verdict airtight.
- FD-verified custom objectives (36 tests) mean the negative result cannot
  be attributed to a broken loss.

## Lessons

1. Loss-function claims should pre-register the **regime check** (here: is
   prevalence actually extreme?) before the ablation.
2. A null from a verified implementation is a *stronger* finding than a
   positive from an unverified one.

## Action items

- Recorded negative; no fix applicable in this cohort (would require
  genuinely extreme-imbalance data).
- The FD-tested focal/cost-sensitive objectives stay in the codebase as
  reusable, verified components.

# PM-005: F-13 System B HPO — selection-budget mismatch

- **Status:** Closed (gate FAIL recorded, follow-up deferred — 2026-09-27)
- **Item:** F-13 — Optuna sweep, System B arm
- **Category:** 🧪 Protocol artifact (measurement wrong, idea untested)
- **Related:** F-03 (default config + 12-epoch protocol), F-19 (scaling curve), ⏸ Deferred register (F-13a), 🧯 Failure Taxonomy in docs/ROADMAP.md

## Impact

The 30-trial sweep "won" at its 2-epoch trial budget (+0.0183 vs the default
config at 2 epochs) — but the pre-registered full-budget retrain (12 epochs,
F-03 protocol) **lost**: val macro 0.5601 vs 0.5750 (Δ −0.0149); test 0.5363
vs 0.5464. The sweep's signal evaporated at the budget that matters.
(Stability gate passed — no NaN/divergence.) The System A arm of F-13 PASSED
decisively (+0.0392 val, +0.045 test) — the mismatch is specific to
early-plateauing iterative models, not to HPO per se.

## Timeline

- 2026-09-27 — Sweep pre-registered: 30 trials, TPESampler(seed=42),
  sqlite-resumable, **2 epochs per trial** (rationale: Phase-3 curves showed
  rank-stability by epoch 2–3 — this turned out to be wrong for HPO
  ranking); full-budget retrain of the best config pre-registered as the
  verdict gate.
- 2026-09-27 — Sweep completed (best val 0.5624 at 2 epochs; the F-03
  default d_model=64/2-layer/d_state=8 config was IN-SPACE and rejected).
- 2026-09-27 — Full-budget retrain (12 epochs, resume-hardened): tuned
  config peaks at 0.5601 (epoch ~6), drifts to 0.544 by epoch 10–12; verdict
  FAIL recorded with `f13_system_b_retrain.json`.

## Root cause

**Selection-budget mismatch**: at 2 epochs, val AUROC ranks configs by
early-training *speed*, not converged quality. TPE therefore selected a
fast-converging, low-capacity config (d_model 96, 1 local layer, d_state 4,
lr 1.4e-3) that plateaus by epoch 4 and drifts downward — while the slower
default keeps climbing to 0.5750. The decisive evidence: **the default was
inside the search space and rejected for the wrong reason.** HPO never
evaluated the thing the gate measures.

## Contributing factors

- Rank-stability by epoch 2–3 was extrapolated from *one config's* training
  curve (F-03's default) to all configs — a hidden assumption, not a
  measured property.
- GPU budget pressure encouraged the cheap-trial protocol; the cost was
  silently paid in ranking fidelity instead.

## What went well

- The full-budget retrain was pre-registered as the gate BEFORE the sweep
  ran — the protocol artifact was caught by design, not by luck.
- Training stability was separately gated and passed, cleanly separating
  "unstable" from "mis-selected".

## Lessons

1. HPO trial budgets must match (or provably rank-preserve against) the
   final budget. For early-plateauing architectures, use ASHA/successive
   halving with promotion at multiple budgets — never a single
   short-budget screen.
2. "Rank-stable by epoch k" is a per-config-family claim; verify it across
   the search space before building a protocol on it.

## Action items

- Gate FAIL recorded honestly; no re-run inside F-13 (budget fixed at 30
  trials by pre-registration).
- **F-13a deferred** (⏸ register): ASHA/successive-halving retune or a
  2-stage protocol (2-epoch screen → top-5 retrained to 12 epochs),
  ~45–60 min GPU, ~40% chance of flipping the verdict — *decide later*.

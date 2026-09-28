# PM-006: F-14 SSL pretraining — objective misallocation

- **Status:** Closed (gate FAIL recorded, follow-up deferred — 2026-09-27); **amended 2026-09-28** — F-14a executed (FAIL by 0.0006) and an objective-semantics bug discovered in the original F-14 code (ADR-007)
- **Item:** F-14 — masked k-mer SSL pretraining (System B)
- **Category:** 🎯 Objective misallocation
- **Related:** F-03 (FT schedule/protocol), Phase-3 token attribution, ADR-007 (masking semantics), ⏸ Deferred register (F-14a — executed), 🧯 Failure Taxonomy in docs/ROADMAP.md

## Impact

Masked k-mer MLM pretraining (8 epochs, train-split genotypes only) then
paired fine-tune vs scratch under an IDENTICAL 6-epoch schedule (seeds
1/2/3, classifier head at FT-seed init in both arms): SSL actively **hurt**
— mean paired val Δ **−0.0189** (gates required ≥ +0.01), worst seed
−0.0285, scratch wins 3/3 (0.526/0.544/0.538 vs 0.521/0.520/0.510).

## Timeline

- 2026-09-27 — Protocol pre-registered (gates G1/G2 written before any run);
  `polymas_ml/sequence/ssl.py` + runner built; 10 unit tests incl.
  masked-only CE equivalence vs full-grid CE.
- 2026-09-27 — Pretrain run: mlm_loss collapses **4.84 → 0.015** over 8
  epochs (observed mid-run; recorded as the mechanism, not treated as
  success).
- 2026-09-27 — 6 paired FT runs completed (resume-hardened); finalize gates:
  both FAIL.

## Root cause

**Objective misallocation — the MLM task is trivially solvable in the wrong
place.** 496 of 504 tokens per patient are context k-mers from shared
flanking sequence; the encoder reaches loss 0.015 by memorizing local
context composition. The 8 genotype tokens — the only disease-relevant
signal (consistent with Phase-3 attribution) — are ~1.6% of the masking
signal, so the encoder learns *context identity*, not *genotype structure*.
Fine-tuning then starts from a misaligned representation that 6 epochs
cannot recover; the pretrained init is a worse basin than a fresh one.

### 2026-09-28 amendment (ADR-007): the objective was ALSO degenerate

Unit tests written for F-14a exposed a semantics bug in `mask_tokens`:
`masked_fill(mask, -100)` put −100 AT masked positions, so `pretrain_loss`
selected the **visible** tokens — the F-14 MLM was a **copy-the-visible-
token** task, not masked reconstruction. The mechanism above is therefore
"objective misallocation **plus** an identity-copy objective". The F-14
verdict stands as measured (never re-run silently); F-14a ran on corrected
semantics and its result is NOT a clean A/B against F-14.

## Contributing factors

- Mask-rate arithmetic (15% of 504 tokens) made the token-class imbalance of
  the objective foreseeable; the check "what fraction of scored tokens carry
  the signal?" was not pre-registered.
- No external corpus was available by design (cohort-only SSL), so the
  objective had to manufacture its own difficulty — it didn't.

## What went well

- The pretrain curve was logged per epoch, making the failure mechanism
  *visible in the evidence* (a collapsing loss is the tell) rather than
  inferred after the fact.
- The paired design (identical FT schedule, head at FT-seed init in both
  arms) means the negative cannot be attributed to schedule asymmetry. An
  early bug (checkpoint's head row leaking into the pretrained arm) was
  caught by an assertion and fixed before any run.

## Lessons

1. Self-supervised objectives need a **signal-fraction pre-check**: what
   fraction of the training signal encodes the quantity downstream tasks
   need? Below some threshold, pretraining is structurally decorative.
2. A collapsing pretraining loss is not success — log it, but gate on the
   downstream paired comparison (as done here).
3. On a 6 GB GPU, per-batch host→device copies and large eval batches can
   destabilize custom scan kernels (`cudaErrorNotReady`); the workarounds
   (one-time upload, batch-64 eval, expandable_segments) are inline-commented
   in `scripts/f14_ssl_pretrain.py`.

## Action items

- Gate FAIL recorded honestly with the mechanism.
- **F-14a EXECUTED 2026-09-28** (context-only masking, corrected semantics,
  2 paired seeds per the register's costing): **FAIL by 0.0006** — mean Δ
  **+0.0094** (gate ≥ +0.01), worst seed +0.0026 (G2 passes). The direction
  FLIPPED vs F-14: pretraining now helps on both seeds (0.5485/0.5428 vs
  0.5323/0.5402). The obvious fix moved SSL from harmful to (marginally)
  helpful — a stronger, more complete negative, and the manuscript's SSL
  section can say the root-cause-motivated fix was tried. Evidence:
  `results/results_final_20260926/f14a_ssl/`.
- Locus-dropout (predict one locus's genotype from the other 7) remains the
  scientifically better-matched task; external-corpus (1000G) pretraining is
  a different, transfer-learning claim. Both = new pre-registrations.
- New guardrail adopted: mask-semantics equivalence tests (masked-CE vs
  manual, zero-mask NaN guard) are now part of the SSL suite (16 tests).

# 🗺️ PolyMas Results Program — 24-Feature Ledger

> **Status:** `Active` · **Started:** 2026-09-25 · **Convention:** [Design-Dungeons](https://github.com/pd241008/Design-Dungeons)
>
> The full feature marathon: every planned capability, tracked as a claim with a
> typed verification level (R1–R4), a pre-registered tolerance where results are
> statistical, and an honest status. **Nothing is skipped, nothing is silently
> dropped** — negative or null results get logged in the notes and, where they
> change a claim, get their own ADR.
>
> Living rules (from Design-Dungeons §ml-and-research):
> - **Honesty-first:** every number traces to a real file from a real run.
> - **P3 historical ≠ canonical:** superseded runs live under `stash/results_pilot_*`; the canonical run is whatever `RUN_INFO.md` in the newest `stash/results_*` folder names as such.
> - **P4 typed reproducibility:** every verification below carries an R1–R4 label and, for R3, a tolerance **pre-registered before the check runs**.
> - **Fresh outputs never overwrite archives:** new runs target `POLYMAS_RESULTS_DIR=stash/results_<name>_<date>`.

---

## 📊 Ledger Status

| Category | Items | Done | In progress | Not started |
|----------|:----:|:----:|:-----------:|:-----------:|
| A. Model architectures | 5 | 0 | 0 | 5 |
| B. Features & inputs | 5 | 2 | 3 | 0 |
| C. Training & calibration | 5 | 0 | 0 | 5 |
| D. Evaluation & rigor | 5 | 0 | 2 | 3 |
| E. Productization | 4 | 0 | 1 | 3 |
| **Total** | **24** | **2** | **6** | **16** |

> Last updated: 2026-09-25 — Phase 1 (data substrate) essentially complete:
> F-09 PASSED (54/99 verified), F-10 core PASSED (LD gate 13/13, 91 variants
> × 2,504 real donors), F-06 harness PASSED with a pre-registered negative
> tree-model result, F-07 PCs validated, F-08 gene mapping done (pathway
> scores open). All pipeline-integration commits deliberately deferred and
> tracked.

---

## ✅ The 24-Item Checklist

### A. Model architectures

| ID | Feature | Goal (claim form) | Verification | Files | Status |
|----|---------|-------------------|--------------|-------|--------|
| **F-01** | **LD-aware GNN (System C)** | *A message-passing GNN over a locus graph with real LD-r² edges matches or beats the GBM ensemble on held-out AUROC (R3).* | Compare macro-AUROC vs System A on the same split; **pre-registered tolerance: within ±0.02 AUROC counts as "matches"**; beats if strictly above. Edge matrix validated against published HLA LD before training (R2). | `polymas_ml/graph/` (new), `scripts/train_system_c.py` (new) | 🔶 **EVALUATED 2026-09-26 (honest FAIL with structural root cause)** — LD-GNN (2-round message passing, r²≥0.2 edges from the real 1000G substrate, 14 typed nodes + relays, best-val ckpt on the canonical split): test macro 0.5401 vs System A 0.6163 (Δ −0.076 ≪ −0.02 bar). **Root cause is structural, not trainable:** the LD-edge audit (`f01_ld_gnn/ld_edge_audit.json`) found only **7 edges ≥ 0.2 among all 91 substrate loci** (MHC max r²=0.846 — the block is there) — the F-09 panel was deliberately LD-pruned at selection, so an LD-GNN has no structure to exploit; the graph degenerates to a small MLP. Claim and panel design are mutually exclusive by construction; an LD-GNN claim would require an LD-rich panel variant (new ADR). SJOGRENS alone beats System A (+0.036). Evidence: `stash/results_final_20260926/f01_ld_gnn/` |
| **F-02** | **MAS-aware disease-graph head** | *A multi-label head with a learned disease-disease adjacency (init = MAS odds table) improves co-occurrence calibration vs independent heads (R3).* | Compare pairwise-phi recovery and per-label AUROC vs current independent-head ensemble; tolerance: ≥ neutral on AUROC (−0.01 floor) AND improved phi recovery. | `polymas_ml/models/ensemble.py`, new `polymas_ml/models/disease_graph.py` | 🔶 **EVALUATED 2026-09-26 (honest FAIL recorded)** — graph head (learned 7×7 adjacency, MAS-log-OR init, exclusion pairs softplus-pinned negative) trained on 3-fold OOF base predictions with a matched-capacity no-graph control (same inputs/split/seed): test macro 0.5751 vs System A 0.6163 (Δ −0.0412 ≪ −0.01 floor) and vs its own control 0.5796 (−0.0046) — both gates fail; phi recovery 8/19 vs control 6/19 and System A's 9/19 (direction positive, within noise). Root cause (recorded, not patched): a 7-probability stacking bottleneck discards the residual signal the 23-feature ensemble uses — every head AUROC trails System A (worst AITD −0.104); the graph path itself is not the liability (graph ≥ control on 4/7 diseases). Learned adjacency stays near init (`adjacency_report.json`). A wider/feature-hybrid head would be a new protocol, not a patch. Evidence: `stash/results_final_20260926/f02_disease_graph/` |
| **F-03** | **Hierarchical Mamba (per-locus → patient)** | *Per-locus local encoders pooled by attention beat the flat 8×65-token Mamba on val AUROC and scale linearly with panel size (R3).* | Same protocol as the 2026-09-25 e2e check: fresh-seed reruns, per-disease AUROC within noise of flat Mamba or better; tolerance ±0.01 vs flat Mamba to call "equal". | `polymas_ml/sequence/model.py`, `scripts/train_system_b.py` | ✅ **PASSED 2026-09-26** — `HierarchicalMamba` (shared per-locus 2×SSM encoder → attention pool → cross-locus SSM → head-pool; parameter-matched flat mode in the same class) on the corrected-labels k-mer dataset, canonical F-12 split, identical training (seed 42, 12 epochs, best-val ckpt): **val macro 0.5750 vs flat 0.5517 (Δ +0.0233 = "beats" bar)**; test macro 0.5365 vs 0.5418 (Δ −0.0053, inside the ±0.01 equal band, recorded); params identical (352,137) and O(1) in n_loci by construction — linear-scaling claim holds structurally. Evidence: `stash/results_final_20260926/f03_hierarchical_mamba/` |
| **F-04** | **Clinical+genetic fusion model** | *Late fusion (gated) of System A features and System B sequence embeddings improves held-out AUROC over either system alone (R3).* | Paired comparison on identical splits; fusion must exceed max(System A, System B) per disease or the claim is recorded as negative with the numbers. | `polymas_ml/models/fusion.py` (new), `scripts/train_fusion.py` | 🔶 **EVALUATED 2026-09-26 (honest FAIL)** — decision-level late fusion (per-disease α on System A vs System B test probs; α fit on val by bounded logloss grid; System B probs from the F-03 flat checkpoint, same artifact): fused macro 0.5738 vs best-parent 0.6210 — fails all three gates (wins 1/7; worst disease −0.139 MS; macro gate fails under ANY α since even α=1 gives 0.6163 < 0.6210). Estimator caveat recorded: System A's val rows are in-sample for the base learners, so val logloss under-weights A; but System B is dominated on 6/7 diseases (F-03 test), so no honest fusion beats the better parent. Sequence-side (feature-level) fusion mooted by the same dominance. Evidence: `stash/results_final_20260926/f04_fusion/` |
| **F-05** | **Focal / cost-sensitive loss** | *Class-imbalance-aware loss improves AUPRC for low-prevalence diseases (VITILIGO, SJOGRENS) without degrading high-prevalence ones (R3).* | AUPRC delta on the 3 lowest-prevalence diseases > 0; no disease loses > 0.01 AUROC. Tolerance pre-registered: improvements < 0.005 AUPRC count as null. | `polymas_ml/models/base_learners.py`, `polymas_ml/sequence/train.py` | 🔶 **EVALUATED 2026-09-26 (honest FAIL recorded, System A arm)** — single-family LightGBM paired ablation on the ADR-006-corrected canonical split (only the loss differs; focal α=0.25 γ=2 pre-registered, not tuned): focal shows NO low-prevalence AUPRC gain (mean Δ on the 3 lowest-prevalence diseases = −0.0045, below the +0.005 null bound) and violates the AUROC floor (min Δ −0.023, SJOGRENS); largest focal "gain" lands on high-prevalence RA (+0.0047 AUPRC) — opposite of the claim. Cost-sensitive arm (context) is worse still (−0.0102). Interpretation: at 9–20% prevalence, logloss GBMs already handle the imbalance; focal's benefit case (extreme imbalance) does not arise here. Objectives derived + FD-verified in `tests/test_focal.py` (36 tests; Hessian chain-rule form matches d(grad)/dz to 1e-11; the exact focal Hessian's negative small-p region documented and floored at 1e-6). Evidence: `stash/results_final_20260926/f05_focal_loss/` |

### B. Features & inputs

| ID | Feature | Goal (claim form) | Verification | Files | Status |
|----|---------|-------------------|--------------|-------|--------|
| **F-06** | **Haplotype + epistasis features** | *DRB1–DQB1 haplotype features and pairwise interaction terms improve System A AUROC vs one-hots alone (R3).* | 🔶 **Negative result, harness PASSED 2026-09-25** — ground-truth injection through real 1000G dosages (main + epistasis coefficients, 47% prevalence, n=3000): both arms recover the signal (AUROC 0.665 ≫ 0.5), but ΔAUROC(base→augmented) = 0.0005 < 0.005 null bar → **no tree-model gain** (GBMs learn pairwise interactions natively). Feature machinery validated for the linear F-16 PRS baseline + interpretability. Evidence: `stash/results/f06_ablation_20260925/`. Wiring into production labels deferred until F-10 integration. | `polymas_ml/data/haplotypes.py`, `scripts/f06_ablation.py` | 🔶 Harness done — production integration pending |
| **F-07** | **Ancestry PCs as covariates** | *Genotype-derived PCs correct population structure; ancestry stratified metrics improve in calibration (slope closer to 1) (R3).* | 🔶 **PCs built & validated 2026-09-25** — SVD PCA on 91 real dosages × 2,504 donors: correct structure (PC1 8.8% ≫ PC2 5.1%; AFR separated on PC1), wired into patient feature assembly, calibration-slope eval pending System A integration. Evidence: `stash/results/real_genotypes_20260925/ancestry_pcs.csv`. | `polymas_ml/data/genotypes.py`, `polymas_ml/data/haplotypes.py` | 🔶 In progress — pipeline integration pending |
| **F-08** | **Functional annotations** | *eQTL/gene mapping + pathway scores per locus are attached to every panel locus and appear as model features without breaking provenance (R2).* | 🔶 **Gene mapping done 2026-09-25** — 95 loci annotated via Ensembl GRCh37 overlap: consequences (13 missense, 41 intron, 8 regulatory…) + overlapping gene for 61/95 (MHC alt-contig variants honestly NaN). Canonical anchors verified: PTPN22→PTPN22, IL23R→IL23R, STAT4→STAT4, TSHR→TSHR, IL7R→IL7R. Pathway scores still open. Evidence: `stash/results/real_genotypes_20260925/variant_annotations_genes.csv`. | `polymas_ml/data/haplotypes.py`, `variant_annotations_genes.csv` | 🔶 In progress — pathway scores open |
| **F-09** | **Panel expansion 8 → 50–100 loci** | *The curated autoimmune panel covers ≥50 GWAS-catalog-validated loci with per-locus provenance (R2).* | ✅ **PASSED 2026-09-25** — 99 unique candidates verified live: 54 VERIFIED, 45 DROPPED with recorded reasons (404s & zero-assoc), 0 errors, 0 pending. Evidence: `stash/results/panel_expansion_20260925/panel_verification.csv` + `panel_manifest.json`. Regression tests: `tests/test_loci.py` (10 passed). Coverage uneven (SJOGRENS 4 / T1D 3 / VITILIGO 1) — wave-3 curation flagged in ADR-002. Pipeline integration is a separate change. | `polymas_ml/data/loci.py`, `scripts/expand_panel.py`, `tests/test_loci.py` | ✅ Done (verification) — pipeline wiring pending |
| **F-10** | **Real genotype backgrounds (1000G)** | *Patient genotypes are sampled from real 1000 Genomes haplotypes; simulated labels stay cohort-informed; LD becomes real (R2 for pipeline, R3 for metrics).* | ✅ **Core PASSED 2026-09-25** — 91 panel variants typed on 2,504 1000G phase-3 donors, **LD gate 13/13 within ±0.05**. **Wiring PASSED 2026-09-25 (ADR-004)**: `--genotypes real` runs end-to-end (System A 5,000 patients + System B 20 epochs). Expected consequence, pre-declared in ADR-004: AUROC drops vs simulated mode (System A macro 0.601 vs 0.656) because the legacy simulator *enriched* cohort genotypes at risk loci (q+0.25) — coupling the labels could learn; real donor haplotypes cannot be enriched (verified: cohort vs background risk-locus dosage 1.386 vs 1.374 for RA). The simulated-mode numbers were partially leakage-flattered; real-mode numbers are the honest baseline. Evidence: `stash/results_real_20260925/`. | `polymas_ml/data/patients.py`, `scripts/run_real_pipeline.py`, ADR-004 | ✅ Done (both halves) — enrichment question → ADR-005 **Decided: no-coupling stands** (published-OR donor weighting failed pre-registered gates G2/G3 + MS G1: LD over-compounding; gate report `stash/results_real_20260925/adr005_gates/`) |

### C. Training & calibration

| ID | Feature | Goal (claim form) | Verification | Files | Status |
|----|---------|-------------------|--------------|-------|--------|
| **F-11** | **Conformal prediction sets** | *Split-conformal sets achieve ≥90% empirical coverage at the nominal level on held-out data, per disease (R3; tolerance: coverage ≥ 0.90 − 0.02).* | Coverage audit script over the held-out split; miscoverage reported per disease + ancestry. | `polymas_ml/evaluation/conformal.py` (new), `scripts/run_stats_eval.py` | ✅ **EVALUATED 2026-09-26 (PASS)** — split-conformal on the canonical held-out split with HONEST calibration probabilities: quantiles from out-of-fold ensemble predictions (train-only refit machinery, per the Phase-2 reconciliation lesson that in-sample `predictions.csv` is optimistic), sets evaluated on `test_predictions.csv` rows. ALL 7 diseases meet the ≥0.88 gate (coverage 0.888–0.926, mean set size 1.70–1.75, zero empty sets). Ancestry-stratified miscoverage recorded (AFR/EAS dips to ~0.81–0.83 on some diseases — subgroup n=53–72, wide intervals; reported as-is, flagged for the manuscript). 7 unit tests (quantile index math, exchangeability edge cases, monotonicity). Evidence: `stash/results_final_20260926/f11_conformal/` |
| **F-12** | **Validation-swept thresholds** | *Per-disease thresholds are swept on validation only; test best-F1 becomes an honest estimate (fixes the current test-swept leakage-adjacent metric) (R2).* | `make_train_test_val` three-way split; metrics regenerated; the test best-F1 in new runs is within noise of val-chosen threshold F1 (no test sweep anywhere). | `scripts/run_real_pipeline.py`, `polymas_ml/models/ensemble.py` | ✅ **DONE 2026-09-25** — three-way 70/10/20 split, thresholds swept on validation only, applied unchanged to test; test-swept oracle kept as a clearly-marked verification column only. Verification on the ADR-006-corrected canonical run: mean |F1_val-chosen − F1_oracle| = 0.0370 across 7 diseases (SJOGRENS worst at 0.1295 — flat validation F1 curve picks threshold 0.16 vs oracle 0.12; recorded as-is, nothing downstream sweeps on test). First-run value 0.0172 (`stash/results_phase2_20260925/`) superseded; evidence: `stash/results_final_20260926/models/threshold_verification.csv` |
| **F-13** | **Optuna sweep (ensemble + Mamba)** | *Tuned configs beat defaults on val AUROC, and the tuned Mamba reruns stably under the guarded GPU wrapper (R3).* | Sweep with fixed budget (e.g. 50 trials), best-vs-default on val; final config retrained on GPU with the guarded wrapper end-to-end. | `scripts/tune_*.py` (new), `polymas_ml/config/` | 🔶 **EVALUATED 2026-09-26 (split verdict: A PASS / B honest FAIL)** — System A: 50-trial TPE sweep (seed 42, canonical split) lifts val macro AUROC 0.6054 → 0.6446 (+0.0392) and the test delta (+0.045) confirms real gain, not selection noise; the tuned single LightGBM (test 0.6486) beats the full 3-learner ensemble (0.6163) — recorded as a substantive result. System B: 30-trial sweep over Mamba hyperparameters WINS at the 2-epoch trial budget (+0.0183 vs default-at-2-epochs) but the pre-registered full-budget retrain (12 epochs, same protocol as F-03) LOSES to the F-03 default (val 0.5601 vs 0.5750, Δ −0.0149; test 0.5363 vs 0.5464): **short-budget HPO does not transfer for the SSM at this scale** — the honest headline. Training was stable (no NaN/divergence), so the stability gate passes while the performance gate fails. Evidence: `stash/results_final_20260926/f13_optuna/` (both studies sqlite-resumable, `f13_system_b_retrain.json` for the full-budget verdict) |
| **F-14** | **Self-supervised pretraining (System B)** | *Masked k-mer pretraining on unlabeled patients improves fine-tuned val AUROC vs training from scratch (R3).* | Paired runs (pretrain+FT vs scratch), same seeds ×3; tolerance: mean ΔAUROC > 0 to claim, < 0.005 = null recorded. | `polymas_ml/sequence/pretrain.py` (new), `scripts/train_system_b.py` | 🔶 **EVALUATED 2026-09-27 (honest FAIL)** — masked k-mer MLM (15% mask rate, causal GPT-style) on TRAIN-split genotypes only (no transductive leakage), 8 epochs; then paired fine-tune vs scratch under an IDENTICAL 6-epoch schedule (seeds 1/2/3; classifier head at FT-seed init in both arms, only the encoder init differs). Result: SSL pretraining HURT — mean paired val Δ = −0.0189 (gates: ≥ +0.01), worst seed −0.0285, scratch wins 3/3 (scratch vals 0.526/0.544/0.538 vs pretrained 0.521/0.520/0.510). Mechanism (visible in `pretrain_history.json`): mlm_loss collapses 4.84 → 0.015 because 496/504 context k-mers per patient are trivially predictable from shared flanking sequence — encoder capacity goes to modeling cheap local context, not the 8 genotype tokens that carry disease signal (consistent with the Phase-3 finding that genotype tokens dominate). Protocol deviations (recorded): batch 64 not 128 (128 OOMs at 6 GB with the 4100-way MLM head), eval batch 64, pretrain seed fixed at 42 as pre-registered. 10 unit tests incl. masked-only CE equivalence. Evidence: `stash/results_final_20260926/f14_ssl/` |
| **F-15** | **Deep ensembles + MC-dropout uncertainty** | *Ensemble-of-5 Mamba seeds + MC-dropout give calibrated uncertainty; NLL improves vs single model (R3).* | 5-seed ensemble NLL/ECE vs single model on val; tolerance for claim: ΔNLL < 0 recorded. | `polymas_ml/sequence/train.py`, `scripts/train_system_b.py` | ✅ **EVALUATED 2026-09-26 (PASS, marginal)** — 4 fresh-seed flat-Mamba trainings (seeds 1–4, canonical F-03 recipe) join the seed-42 model as ensemble members; NLL/ECE/Brier added in `polymas_ml/evaluation/uncertainty.py` (14 tests). Deep-ensemble gate: ΔNLL −0.0009 < 0 vs single seed-42 (val macro 0.3782 vs 0.3791) — in the pre-registered direction, though small. MC-dropout (T=30) gate fails by a rounding hair (ΔNLL +0.0000). Secondary signal is cleaner: ensemble ECE 0.0087 vs single 0.0187 (halves). Honest reading: seeds disagree very little (member vals 0.522–0.560), so uncertainty gains are bounded — the members are near-identical, not diverse. Evidence: `stash/results_final_20260926/f15_uncertainty/` |

### D. Evaluation & rigor

| ID | Feature | Goal (claim form) | Verification | Files | Status |
|----|---------|-------------------|--------------|-------|--------|
| **F-16** | **Classic PRS baseline (LDpred2-style)** | *C+T clumping-thresholding PRS built from real GWAS Catalog/OpenGWAS stats serves as an external baseline row in the results tables (R2 for construction, R3 for comparison).* | Baseline AUROC computed on the same held-out split; reported alongside models regardless of outcome. | `scripts/prs_baseline.py` (new) | ✅ **DONE — re-run on the ADR-006-corrected canonical run (2026-09-26)** — C+T PRS from published OpenGWAS betas (RA/SLE/SJOGRENS/T1D/MS; p<5e-8, fallback p<1e-4 flagged) + local curated anchors (VITILIGO); allele-aligned; scored on the same F-12 test split via persisted donor map. Models beat PRS on RA (+0.107), T1D (+0.153), MS (+0.136), SJOGRENS (+0.062); **PRS beats the model on SLE (0.644 vs 0.634) and VITILIGO (0.685 vs 0.542, honest)**; AITD = coverage gap (no panel association under thresholds; HLA-proxy anchors allele-unresolvable). Prior-run table retained on disk. Evidence: `stash/results_final_20260926/f16_prs_baseline/` |
| **F-17** | **MAS-recovery ablation** | *Trained models recover the embedded MAS pairwise odds direction/significance; recovery is reported as an honest pipeline sanity claim (R3).* | Pairwise phi from model predictions vs embedded OR sign agreement; per-pair sign agreement > chance (binomial p < 0.05, Bonferroni across 21 pairs — tolerance pre-registered). | `scripts/run_stats_eval.py`, `polymas_ml/evaluation/stats.py` | 🔶 **EVALUATED on ADR-006-corrected run (honest negative, 2026-09-26)** — 21 pairs (19 embedded, 2 incoercible reported): sign agreement 9/19 = 47.4%, binomial p=0.68 vs the pre-registered >chance bar; 9/19 full recovery (sign + significance, up from 8). Incoercible T1D|MS prediction-phi −0.27 and RA|MS −0.16 (models resolve the antagonism sharply); label-level phi recovers the embedded axes; models resolve RA/SjS/AITD/vitiligo axes but not SLE-T1D/weak pairs. Evidence: `stash/results_final_20260926/f17_mas_recovery/` |
| **F-18** | **Label-noise robustness curves** | *Performance degrades gracefully and monotonically under 5/10/20% label flipping for both systems (R3; curve shape is the deliverable).* | Sweep script flipping labels at fixed seeds; AUC-vs-noise curves saved as canonical figures. | `scripts/noise_sweep.py` (new) | ✅ **DONE (System A, re-run on ADR-006-corrected canonical 2026-09-26)** — 5/7 curves monotone within tolerance at 0/5/10/20% flips (seed 42, canonical split/features; T1D 0.760→0.568, total drop 0.192); SLE and VITILIGO wiggle above tolerance mid-curve (SLE 0.569→0.587, VITILIGO 0.491→0.524) with all total drops positive — recorded as-is. **System B arm COMPLETE (Phase-3 re-scope, 2026-09-26)**: flat Mamba retrained per rate on corrected labels (6 epochs, seed 42; 0% point reuses the F-03 flat arm); System B macro degrades 0.5418→0.5255 (0→20%) — monotone-within-tolerance on 5/7 diseases, same shape as System A at a lower level. Both systems merged in `noise_curves.csv` (system column). Evidence: `stash/results_final_20260926/f18_noise_sweep/` + `system_b_curves/` |
| **F-19** | **Sample-size scaling curves** | *Performance vs n (1k/2.5k/5k/10k) quantifies data efficiency for both systems (R3).* | Fixed-seed runs per n; scaling figure + fitted exponent in the report. | `scripts/scaling_sweep.py` (new) | ✅ **DONE (System A, re-run on ADR-006-corrected code 2026-09-26)** — macro AUROC 0.5896/0.5895/0.6163/0.6180 at n=1k/2.5k/5k/10k (identical code path; 5k point = corrected canonical run); fitted exponent α=1.658 (macro); per-disease exponents mixed (MS 5.13, SJOGRENS 2.35, VITILIGO 0.32; RA/SLE/T1D negative — weak-signal small-n luck, reported as-is). **System B arm COMPLETE (Phase-3 re-scope, 2026-09-26)**: macro 0.4877/0.5124/0.5418/0.5359 at 1k/2.5k/5k/10k — rising then flat-to-slight-decline at 10k (the 5k point reuses the F-03 flat arm; smaller-n points use the protocol-shape split within first-n). Both systems in `scaling_curves.csv` (system column). Evidence: `stash/results_final_20260926/f19_scaling_sweep/` + `system_b_curves/` + `stash/results_scaling_n*/` |
| **F-20** | **External validation on real data** | *Locus-level model evidence is externally anchored: simulated effect directions match real FinnGen/OpenGWAS associations for the panel, and (stretch) one real individual-level cohort is scored (R3 for direction agreement; R4 for the stretch).* | Per-locus × per-disease direction agreement vs OpenGWAS phewas; **pre-registered: ≥70% sign agreement on significant (p<5e-8) associations counts as a pass**. | `scripts/external_validation.py` (new) | ✅ **PASSED after ADR-006 fix (2026-09-26)** — history: first evaluation FAILED honestly (3/12 = 25% vs the ≥70% bar; root cause: label term used unaligned VCF dosages, sign-inverted at PTPN22/STAT4 where the VCF alt is the common protective allele; the unrestricted pair set also included non-anchor pairs that embed no direction). ADR-006 allele-aligned the label term and restricted the pair set to own-anchor loci (pre-registered protocol fix); corrected run: **4/4 = 100% sign agreement** on GWS own-anchor pairs (model ρ now negative, matching published β; exact binomial 95% CI [0.40, 1.00]); AITD/VITILIGO remain coverage gaps. Failed-run table retained for transparency. Evidence: `stash/results_final_20260926/f20_external_validation/` (+ superseded `stash/results_phase2_20260925/f20_external_validation/`) |

### E. Productization

| ID | Feature | Goal (claim form) | Verification | Files | Status |
|----|---------|-------------------|--------------|-------|--------|
| **F-21** | **Dashboard integration** | *The Next.js dashboard renders live per-disease metrics, SHAP beeswarms, and the cluster explorer from canonical run artifacts (R4 — re-derived from archived results).* | Dashboard pages read canonical `stash/results_*` exports; numbers match `per_disease_metrics.csv` (exact for R4). | `apps/dashboard-nextjs/` | ⬜ Not started |
| **F-22** | **gRPC serving** | *The trained ensemble serves predictions through the existing proto/gRPC scaffolding with a round-trip contract test (R2).* | `proto/polymas/v1` serving test: request → prediction matches offline `predict_proba` within float tolerance (1e-6). | `polymas_ml/serving/grpc_server.py`, `services/` | ⬜ Not started |
| **F-23** | **Experiment tracking (MLflow)** | *Every run (configs, metrics, artifacts) is logged; any two runs are diffable (e.g. the 09-24 vs 09-25 e2e comparison becomes automatic) (R2).* | Two recorded runs diffed via MLflow API; seeds/config/metrics present for both. | `polymas_ml/tracking.py` (new), all run scripts | ⬜ Not started |
| **F-24** | **One-command reproducibility (Makefile e2e)** | *`make e2e RESULTS_DIR=stash/results_<name>_<date>` runs the full chain (A → k-mer → B → stats → figures → PDF) into a fresh folder; `make verify` does the fast R4 re-check (R2 for the target, R3 for the comparison).* | Fresh-folder e2e via make matches the 2026-09-25 run: System A bit-identical (R1), System B within pre-registered tolerance (R3, per-disease AUROC ±0.02); figures regenerate. | `Makefile`, `scripts/run_e2e.sh` (new) | 🔶 In progress — `POLYMAS_RESULTS_DIR` override landed (2026-09-25) and verified by the e2e run; make targets + verify path remain |

---

## 🔗 Execution Order (dependency-aware)

```
Phase 0  ✅ repo hygiene, e2e verification, API access (OpenGWAS + GWAS Catalog)
Phase 1  F-09 panel expansion → F-10 real genotypes → F-07 PCs → F-06 haplotypes → F-08 annotations
Phase 2  F-16 PRS baseline → F-12 thresholds → F-20 external validation → F-17 MAS recovery → F-18 noise → F-19 scaling
Phase 3  F-02 MAS heads → F-05 focal loss → F-03 hierarchical Mamba (+ System B arms of F-18/F-19 on the retrained Mamba) → F-01 LD-GNN → F-04 fusion
Phase 4  F-11 conformal → F-15 uncertainty → F-13 Optuna → F-14 SSL pretraining
Phase 5  F-23 MLflow → F-24 make e2e/verify → F-22 gRPC → F-21 dashboard
```

Rationale: data substrate first (Phase 1 unblocks half the ledger), honesty
layer before new models (Phase 2 hardens evaluation so model claims in Phase 3
are measured against fixed rails), productization last so it wraps canonical
artifacts that already exist. **Everything verified in Phases 2+ lands in the
paper; Phase 3 features that miss their pre-registered bar are reported as
negative results, not dropped.**

---

## 📐 Pre-Registered Tolerances (R3 registry)

> Stated before the corresponding verification runs. Changing one requires an ADR.

| Check | Tolerance |
|-------|-----------|
| Fresh-seed reruns "hold" (like the 09-25 e2e check) | per-disease AUROC within ±0.02 of the canonical run |
| F-01 GNN "matches" System A | macro-AUROC within ±0.02 |
| F-05 focal loss null bound | ΔAUPRC < 0.005 = null |

## Reconciliation log

| Date | Finding | Resolution |
|------|---------|------------|
| 2026-09-26 | External review flagged `models/per_disease_metrics.csv` vs `stats/bootstrap_ci_auroc.csv` point estimates disagreeing by up to 0.12 AUROC (AITD). Root-caused: the stats layer bootstrapped `models/predictions.csv` rows — an IN-SAMPLE full-cohort refit matrix (clustering/SHAP), not the held-out `test_predictions.csv`. | `scripts/run_stats_eval.py` now bootstraps the held-out test predictions exclusively; `stats/` regenerated: bootstrap points match the direct test AUROCs to 0.000000. The pre-fix ancestry table had been anchoring to the wrong matrix — the corrected subgroup table shows real heterogeneity (EUR AITD 0.5804 vs pooled 0.5832), superseding the earlier spurious 0.46. F-16/F-19 headlines were built on `per_disease_metrics.csv`/`test_predictions.csv` (verified correct) — no re-runs needed. |
| 2026-09-26 | Same review flagged `clusters/silhouette_score.txt` (0.2683) vs `stats/silhouette_permutation_test.json` observed (0.0068). Root-caused: 0.2683 was a stale value not reproducible from any current artifact (recomputed direct silhouette on `predictions.csv` + assignments = 0.0068, exactly the permutation test's observed). | `silhouette_score.txt` regenerated from artifacts (0.0068) with a pointer to the permutation test as authoritative; the pipeline writer now embeds the same note. Substantive reading unchanged and honest: Ward k=3 finds NO structure beyond marginals (observed 0.0068 vs null 0.462, z=−16.5) — the risk-vector continuum does not cluster. |
| F-11 conformal coverage | ≥ 0.90 − 0.02 at nominal 90% |
| F-14 SSL pretrain gain | mean ΔAUROC > 0 across 3 seeds; < 0.005 = null |
| F-17 MAS sign recovery | > chance at binomial p < 0.05, Bonferroni ×21 |
| F-20 external direction agreement | ≥ 70% sign agreement on p < 5e-8 associations |
| F-22 gRPC contract | prediction match within 1e-6 |
| F-10 1000G LD vs published r² | ±0.05 on ≥3 checked pairs |

---

## 📜 ADRs (this program)

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](docs/adr/ADR-001-results-program.md) | Adopt a 24-feature results program with typed verification | Decided |
| [ADR-002](docs/adr/ADR-002-panel-expansion.md) | Panel expansion source & curation method (F-09) | Decided |
| [ADR-003](docs/adr/ADR-003-external-validation-scope.md) | External validation scope: summary-stats anchoring before individual-level cohorts | Draft |
| [ADR-004](docs/adr/ADR-004-real-donor-genotypes.md) | Real-donor genotype wiring: modes, calibration, enrichment tradeoff (F-10) | Decided |
| [ADR-005](docs/adr/ADR-005-genotype-label-coupling.md) | Genotype–label coupling policy: published-OR donor weighting **tested and rejected** by pre-registered gates (F-10) | Decided (fallback) |
| [ADR-006](docs/adr/ADR-006-label-term-allele-alignment.md) | Allele-aligned label polygenic term (real mode): sign fix at protective-alt loci + F-20 own-anchor pair restriction — gates G1–G5 all PASS | Decided (adopted) |

> Note: `docs/` is gitignored in this repo (NPA rule from `.gitignore`); ADRs live
> in `docs/adr/` locally per Design-Dungeons layout, while this ledger at the repo
> root is the tracked source of truth.

---

## 🧾 Completed-This-Session (context for the ledger)

| Date | Change | Verification |
|------|--------|--------------|
| 2026-09-25 | Honest renaming: pilot runs + `superseded_<ts>` + `*_full.csv`; zero "backup" naming left | `grep` clean; R2 |
| 2026-09-25 | `POLYMAS_RESULTS_DIR` override in all 9 run/reader scripts + guarded wrapper | e2e run into `stash/results_e2e_20260925`; R1 System A (bit-identical), R3 System B (within ±0.02) |
| 2026-09-25 | Full e2e verification run (System A → k-mer → Mamba → stats → figures → PDF) | macro AUROC 0.5776 vs 0.5807; hamming/F1 identical |
| 2026-09-25 | Modular commits: label-model fix, stash consolidation, results-root feature | 3 commits, no footers; tests pass |
| 2026-09-25 | External API access: GWAS Catalog REST verified live; OpenGWAS JWT stored in `.env` (expires 2026-10-09), `/associations` + `/phewas` validated | PTPN22 phewas reproduces published pleiotropy; R2 |

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
| **F-01** | **LD-aware GNN (System C)** | *A message-passing GNN over a locus graph with real LD-r² edges matches or beats the GBM ensemble on held-out AUROC (R3).* | Compare macro-AUROC vs System A on the same split; **pre-registered tolerance: within ±0.02 AUROC counts as "matches"**; beats if strictly above. Edge matrix validated against published HLA LD before training (R2). | `polymas_ml/graph/` (new), `scripts/train_system_c.py` (new) | ⬜ Not started — **requires F-09** |
| **F-02** | **MAS-aware disease-graph head** | *A multi-label head with a learned disease-disease adjacency (init = MAS odds table) improves co-occurrence calibration vs independent heads (R3).* | Compare pairwise-phi recovery and per-label AUROC vs current independent-head ensemble; tolerance: ≥ neutral on AUROC (−0.01 floor) AND improved phi recovery. | `polymas_ml/models/ensemble.py`, new `polymas_ml/models/disease_graph.py` | ⬜ Not started |
| **F-03** | **Hierarchical Mamba (per-locus → patient)** | *Per-locus local encoders pooled by attention beat the flat 8×65-token Mamba on val AUROC and scale linearly with panel size (R3).* | Same protocol as the 2026-09-25 e2e check: fresh-seed reruns, per-disease AUROC within noise of flat Mamba or better; tolerance ±0.01 vs flat Mamba to call "equal". | `polymas_ml/sequence/model.py`, `scripts/train_system_b.py` | ⬜ Not started |
| **F-04** | **Clinical+genetic fusion model** | *Late fusion (gated) of System A features and System B sequence embeddings improves held-out AUROC over either system alone (R3).* | Paired comparison on identical splits; fusion must exceed max(System A, System B) per disease or the claim is recorded as negative with the numbers. | `polymas_ml/models/fusion.py` (new), `scripts/train_fusion.py` | ⬜ Not started |
| **F-05** | **Focal / cost-sensitive loss** | *Class-imbalance-aware loss improves AUPRC for low-prevalence diseases (VITILIGO, SJOGRENS) without degrading high-prevalence ones (R3).* | AUPRC delta on the 3 lowest-prevalence diseases > 0; no disease loses > 0.01 AUROC. Tolerance pre-registered: improvements < 0.005 AUPRC count as null. | `polymas_ml/models/base_learners.py`, `polymas_ml/sequence/train.py` | ⬜ Not started |

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
| **F-11** | **Conformal prediction sets** | *Split-conformal sets achieve ≥90% empirical coverage at the nominal level on held-out data, per disease (R3; tolerance: coverage ≥ 0.90 − 0.02).* | Coverage audit script over the held-out split; miscoverage reported per disease + ancestry. | `polymas_ml/evaluation/conformal.py` (new), `scripts/run_stats_eval.py` | ⬜ Not started — builds on existing Platt calibration |
| **F-12** | **Validation-swept thresholds** | *Per-disease thresholds are swept on validation only; test best-F1 becomes an honest estimate (fixes the current test-swept leakage-adjacent metric) (R2).* | `make_train_test_val` three-way split; metrics regenerated; the test best-F1 in new runs is within noise of val-chosen threshold F1 (no test sweep anywhere). | `scripts/run_real_pipeline.py`, `polymas_ml/models/ensemble.py` | ✅ **DONE 2026-09-25** — three-way 70/10/20 split, thresholds swept on validation only, applied unchanged to test; test-swept oracle kept as a clearly-marked verification column only. Verification: mean |F1_val-chosen − F1_oracle| = 0.0172 across 7 diseases (`stash/results_phase2_20260925/models/threshold_verification.csv`); nothing downstream sweeps on test |
| **F-13** | **Optuna sweep (ensemble + Mamba)** | *Tuned configs beat defaults on val AUROC, and the tuned Mamba reruns stably under the guarded GPU wrapper (R3).* | Sweep with fixed budget (e.g. 50 trials), best-vs-default on val; final config retrained on GPU with the guarded wrapper end-to-end. | `scripts/tune_*.py` (new), `polymas_ml/config/` | ⬜ Not started |
| **F-14** | **Self-supervised pretraining (System B)** | *Masked k-mer pretraining on unlabeled patients improves fine-tuned val AUROC vs training from scratch (R3).* | Paired runs (pretrain+FT vs scratch), same seeds ×3; tolerance: mean ΔAUROC > 0 to claim, < 0.005 = null recorded. | `polymas_ml/sequence/pretrain.py` (new), `scripts/train_system_b.py` | ⬜ Not started |
| **F-15** | **Deep ensembles + MC-dropout uncertainty** | *Ensemble-of-5 Mamba seeds + MC-dropout give calibrated uncertainty; NLL improves vs single model (R3).* | 5-seed ensemble NLL/ECE vs single model on val; tolerance for claim: ΔNLL < 0 recorded. | `polymas_ml/sequence/train.py`, `scripts/train_system_b.py` | ⬜ Not started |

### D. Evaluation & rigor

| ID | Feature | Goal (claim form) | Verification | Files | Status |
|----|---------|-------------------|--------------|-------|--------|
| **F-16** | **Classic PRS baseline (LDpred2-style)** | *C+T clumping-thresholding PRS built from real GWAS Catalog/OpenGWAS stats serves as an external baseline row in the results tables (R2 for construction, R3 for comparison).* | Baseline AUROC computed on the same held-out split; reported alongside models regardless of outcome. | `scripts/prs_baseline.py` (new) | ✅ **DONE 2026-09-25** — C+T PRS from published OpenGWAS betas (RA/SLE/SJOGRENS/T1D/MS; p<5e-8, fallback p<1e-4 flagged) + local curated anchors (VITILIGO); allele-aligned; scored on the canonical F-12 test split via persisted donor map. Models beat PRS on RA (+0.143), T1D (+0.215), VITILIGO (+0.370), SLE (+0.126), MS (+0.113); **PRS beats the model on SJOGRENS (0.575 vs 0.548, honest)**; AITD = coverage gap (no panel association under thresholds; HLA-proxy anchors allele-unresolvable). Evidence: `stash/results_phase2_20260925/f16_prs_baseline/` |
| **F-17** | **MAS-recovery ablation** | *Trained models recover the embedded MAS pairwise odds direction/significance; recovery is reported as an honest pipeline sanity claim (R3).* | Pairwise phi from model predictions vs embedded OR sign agreement; per-pair sign agreement > chance (binomial p < 0.05, Bonferroni across 21 pairs — tolerance pre-registered). | `scripts/run_stats_eval.py`, `polymas_ml/evaluation/stats.py` | 🔶 **EVALUATED 2026-09-25 (honest negative)** — 19 embedded pairs (2 excluded pairs reported): sign agreement 10/19 = 52.6%, binomial p=0.5 vs the pre-registered >chance bar; 8/19 full recovery (sign + significance). Label-level phi recovers 9/12 embedded directions; models resolve RA/SjS-AITD/vitiligo axes but not SLE-T1D/weak pairs; incoercible pairs correctly negative. Evidence: `stash/results_phase2_20260925/f17_mas_recovery/` |
| **F-18** | **Label-noise robustness curves** | *Performance degrades gracefully and monotonically under 5/10/20% label flipping for both systems (R3; curve shape is the deliverable).* | Sweep script flipping labels at fixed seeds; AUC-vs-noise curves saved as canonical figures. | `scripts/noise_sweep.py` (new) | ✅ **DONE (System A) 2026-09-25** — all 7 AUROC-vs-noise curves monotone within tolerance at 0/5/10/20% flips (seed 42, canonical split/features; e.g. RA 0.638→0.590→0.554→0.502). System B curve pending GPU wrapper runs. Evidence: `stash/results_phase2_20260925/f18_noise_sweep/` |
| **F-19** | **Sample-size scaling curves** | *Performance vs n (1k/2.5k/5k/10k) quantifies data efficiency for both systems (R3).* | Fixed-seed runs per n; scaling figure + fitted exponent in the report. | `scripts/scaling_sweep.py` (new) | ✅ **DONE (System A) 2026-09-25** — macro AUROC 0.5525/0.6089/0.6052/0.6231 at n=1k/2.5k/5k/10k (identical code path; 5k point = canonical run); fitted exponent α=0.9 (macro), T1D 0.96 strongest, AITD/MS negative (weak-signal small-n luck, reported as-is). System B pending GPU runs. Evidence: `stash/results_phase2_20260925/f19_scaling_sweep/` + `stash/results_scaling_n*/` |
| **F-20** | **External validation on real data** | *Locus-level model evidence is externally anchored: simulated effect directions match real FinnGen/OpenGWAS associations for the panel, and (stretch) one real individual-level cohort is scored (R3 for direction agreement; R4 for the stretch).* | Per-locus × per-disease direction agreement vs OpenGWAS phewas; **pre-registered: ≥70% sign agreement on significant (p<5e-8) associations counts as a pass**. | `scripts/external_validation.py` (new) | 🔶 **EVALUATED 2026-09-25 (FAIL recorded with root cause)** — 12 GWS (disease, locus) pairs tested vs published OpenGWAS betas (allele-aligned): sign agreement 3/12 = 25% ≪ 70% bar. Root cause identified, NOT patched: the real-mode label polygenic term uses unaligned VCF dosages (sign-inverted at PTPN22/STAT4 where the VCF alt is the common protective allele); the model faithfully learned the inverted simulation signal. Fix requires a new ADR (the rejected ADR-005 label design had the correct aligned mechanism). Coverage gaps: AITD/VITILIGO. Evidence: `stash/results_phase2_20260925/f20_external_validation/` |

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
Phase 3  F-02 MAS heads → F-05 focal loss → F-03 hierarchical Mamba → F-01 LD-GNN → F-04 fusion
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

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
| B. Features & inputs | 5 | 0 | 1 | 4 |
| C. Training & calibration | 5 | 0 | 0 | 5 |
| D. Evaluation & rigor | 5 | 0 | 2 | 3 |
| E. Productization | 4 | 0 | 1 | 3 |
| **Total** | **24** | **0** | **4** | **20** |

> Last updated: 2026-09-25 (program kickoff).

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
| **F-06** | **Haplotype + epistasis features** | *DRB1–DQB1 haplotype features and pairwise interaction terms improve System A AUROC vs one-hots alone (R3).* | Ablation: with vs without, same seeds; tolerance for "no gain" = ΔAUROC < 0.005 recorded as negative result. | `polymas_ml/data/patients.py`, `scripts/run_real_pipeline.py` | ⬜ Not started |
| **F-07** | **Ancestry PCs as covariates** | *Genotype-derived PCs correct population structure; ancestry stratified metrics improve in calibration (slope closer to 1) (R3).* | PCA on the genotype matrix (4 PCs per 1000G convention); compare per-ancestry calibration slope before/after. | `polymas_ml/data/patients.py`, `polymas_ml/evaluation/ancestry.py` | ⬜ Not started |
| **F-08** | **Functional annotations** | *eQTL/gene mapping + pathway scores per locus are attached to every panel locus and appear as model features without breaking provenance (R2).* | Every locus in the panel has ≥1 mapped gene + pathway tags in the provenance manifest; manifest schema-validated. | `polymas_ml/data/loci.py` (new), provenance JSON | ⬜ Not started — **requires F-09** |
| **F-09** | **Panel expansion 8 → 50–100 loci** | *The curated autoimmune panel covers ≥50 GWAS-catalog-validated loci with per-locus provenance (R2).* | Panel CSV cross-checked against GWAS Catalog associations; count and sources asserted in a test. | `polymas_ml/data/loci.py`, config | 🔶 In progress — locus curation started; blocked items F-01/F-08/F-16 depend on this |
| **F-10** | **Real genotype backgrounds (1000G)** | *Patient genotypes are sampled from real 1000 Genomes haplotypes; simulated labels stay cohort-informed; LD becomes real (R2 for pipeline, R3 for metrics).* | 1000G haplotype pull (GCS public bucket) + local LD matrix reproduced against published r² for ≥3 locus pairs (tolerance ±0.05). | `polymas_ml/data/genotypes.py` (new), dataset builder | 🔶 In progress — 1000G access verified free; LD computation not yet implemented |

### C. Training & calibration

| ID | Feature | Goal (claim form) | Verification | Files | Status |
|----|---------|-------------------|--------------|-------|--------|
| **F-11** | **Conformal prediction sets** | *Split-conformal sets achieve ≥90% empirical coverage at the nominal level on held-out data, per disease (R3; tolerance: coverage ≥ 0.90 − 0.02).* | Coverage audit script over the held-out split; miscoverage reported per disease + ancestry. | `polymas_ml/evaluation/conformal.py` (new), `scripts/run_stats_eval.py` | ⬜ Not started — builds on existing Platt calibration |
| **F-12** | **Validation-swept thresholds** | *Per-disease thresholds are swept on validation only; test best-F1 becomes an honest estimate (fixes the current test-swept leakage-adjacent metric) (R2).* | `make_train_test_val` three-way split; metrics regenerated; the test best-F1 in new runs is within noise of val-chosen threshold F1 (no test sweep anywhere). | `scripts/run_real_pipeline.py`, `polymas_ml/models/ensemble.py` | ⬜ Not started |
| **F-13** | **Optuna sweep (ensemble + Mamba)** | *Tuned configs beat defaults on val AUROC, and the tuned Mamba reruns stably under the guarded GPU wrapper (R3).* | Sweep with fixed budget (e.g. 50 trials), best-vs-default on val; final config retrained on GPU with the guarded wrapper end-to-end. | `scripts/tune_*.py` (new), `polymas_ml/config/` | ⬜ Not started |
| **F-14** | **Self-supervised pretraining (System B)** | *Masked k-mer pretraining on unlabeled patients improves fine-tuned val AUROC vs training from scratch (R3).* | Paired runs (pretrain+FT vs scratch), same seeds ×3; tolerance: mean ΔAUROC > 0 to claim, < 0.005 = null recorded. | `polymas_ml/sequence/pretrain.py` (new), `scripts/train_system_b.py` | ⬜ Not started |
| **F-15** | **Deep ensembles + MC-dropout uncertainty** | *Ensemble-of-5 Mamba seeds + MC-dropout give calibrated uncertainty; NLL improves vs single model (R3).* | 5-seed ensemble NLL/ECE vs single model on val; tolerance for claim: ΔNLL < 0 recorded. | `polymas_ml/sequence/train.py`, `scripts/train_system_b.py` | ⬜ Not started |

### D. Evaluation & rigor

| ID | Feature | Goal (claim form) | Verification | Files | Status |
|----|---------|-------------------|--------------|-------|--------|
| **F-16** | **Classic PRS baseline (LDpred2-style)** | *C+T clumping-thresholding PRS built from real GWAS Catalog/OpenGWAS stats serves as an external baseline row in the results tables (R2 for construction, R3 for comparison).* | Baseline AUROC computed on the same held-out split; reported alongside models regardless of outcome. | `scripts/prs_baseline.py` (new) | 🔶 In progress — OpenGWAS JWT validated (expires 2026-10-09), phewas endpoint confirmed; needs F-09 for the expanded panel |
| **F-17** | **MAS-recovery ablation** | *Trained models recover the embedded MAS pairwise odds direction/significance; recovery is reported as an honest pipeline sanity claim (R3).* | Pairwise phi from model predictions vs embedded OR sign agreement; per-pair sign agreement > chance (binomial p < 0.05, Bonferroni across 21 pairs — tolerance pre-registered). | `scripts/run_stats_eval.py`, `polymas_ml/evaluation/stats.py` | 🔶 In progress — phi table exists (10/21 pairs significant); sign-agreement check vs MAS table not yet wired |
| **F-18** | **Label-noise robustness curves** | *Performance degrades gracefully and monotonically under 5/10/20% label flipping for both systems (R3; curve shape is the deliverable).* | Sweep script flipping labels at fixed seeds; AUC-vs-noise curves saved as canonical figures. | `scripts/noise_sweep.py` (new) | ⬜ Not started |
| **F-19** | **Sample-size scaling curves** | *Performance vs n (1k/2.5k/5k/10k) quantifies data efficiency for both systems (R3).* | Fixed-seed runs per n; scaling figure + fitted exponent in the report. | `scripts/scaling_sweep.py` (new) | ⬜ Not started |
| **F-20** | **External validation on real data** | *Locus-level model evidence is externally anchored: simulated effect directions match real FinnGen/OpenGWAS associations for the panel, and (stretch) one real individual-level cohort is scored (R3 for direction agreement; R4 for the stretch).* | Per-locus × per-disease direction agreement vs OpenGWAS phewas; **pre-registered: ≥70% sign agreement on significant (p<5e-8) associations counts as a pass**. | `scripts/external_validation.py` (new) | 🔶 In progress — GWAS Catalog API verified live; OpenGWAS phewas validated (PTPN22 pleiotropy reproduced); FinnGen form not yet filed |

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
| [ADR-002](docs/adr/ADR-002-panel-expansion.md) | Panel expansion source & curation method | Draft |
| [ADR-003](docs/adr/ADR-003-external-validation-scope.md) | External validation scope: summary-stats anchoring before individual-level cohorts | Draft |

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

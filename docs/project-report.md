# poly-mas — Project Report

## A Data-Driven Reclassification of Multiple Autoimmune Syndrome Using Explainable Ensemble Learning on Genotypic Risk Profiles

**Authors:** Desai Prathmesh Prakash, Sankeet Pinjala  
**Repository:** [github.com/pd241008/PolyMas](https://github.com/pd241008/PolyMas)  
**Target venue:** *npj Digital Medicine* (Nature Portfolio)

---

## 1. Executive Summary

This project builds a machine learning pipeline that predicts a patient's simultaneous risk across multiple autoimmune diseases and tests whether the resulting risk patterns support, extend, or challenge the existing clinical classification of **Multiple Autoimmune Syndrome (MAS)** — a 1988 taxonomy that has never been re-evaluated against genomic evidence. The pipeline combines public GWAS risk data with real clinical cohorts, trains a multi-label ensemble classifier (XGBoost + CatBoost + LightGBM, combined via a normalized voting mechanism), explains its predictions with SHAP/LIME, and clusters its outputs to compare against the 1988 Type 1–4 classification.

## 2. Clinical Background

**Multiple Autoimmune Syndrome (MAS)** is the coexistence of three or more autoimmune diseases in a single patient, first classified by Humbert and Dupond in 1988 from 87 literature cases plus 4 personal cases, into three (later four) recurring clusters:

- **Type 1** — myasthenia gravis, thymoma, polymyositis, giant cell myocarditis
- **Type 2** — Sjögren's syndrome, rheumatoid arthritis, primary biliary cirrhosis, scleroderma, autoimmune thyroid disease
- **Type 3** — autoimmune thyroid disease, myasthenia/thymoma, Sjögren's, pernicious anemia, ITP, Addison's disease, type 1 diabetes, vitiligo, autoimmune hemolytic anemia, SLE, dermatitis herpetiformis
- **Type 4** — later polyglandular extension; Betterle et al. (2023) describe Type 3/APS-3 as "an expanding galaxy," i.e. still incomplete

Roughly a quarter of patients with one autoimmune disease go on to develop another (Anaya et al., 2012), and autoimmune diseases co-occur within families more than chance predicts (Somers et al., 2006) — so the underlying phenomenon is well-established epidemiologically, even though the classification organizing it is old and narrow.

**The genetic angle:** GWAS-era genetics has identified loci shared across autoimmune diseases — HLA class II, CTLA-4, PTPN22 (Brand et al., 2005; Stanford review, 2014) — but a large cross-disease analysis (*PLoS Genetics*, 2011, testing 446 variants across 17 autoimmune diseases against SLE) found sharing is partial, not universal: only IL23R, OLIG3/TNFAIP3, and IL2RA were broadly shared. That same study's genetics-based clustering of diseases did not match clinical intuition (e.g. grouped T1D with RA) — direct precedent that a data-driven regrouping of MAS-associated diseases is both possible and likely to diverge from the 1988 classification.

**The gap:** all this genetic-overlap evidence exists at the population level (comparing disease pairs). Nobody has built a patient-level, multi-label, explainable model that predicts individual MAS risk and uses it to re-test the classification itself. That is this project's contribution.

## 3. Data Sources

| Source | Status | Role |
|--------|--------|------|
| GWAS Catalog (NHGRI–EBI) | Government-affiliated (NHGRI is an NIH institute), fully open access | Per-disease SNP risk effect sizes → polygenic risk scores (PRS) |
| ImmPort (NIAID/NIH-funded) | Government-funded, open registration | Real per-disease clinical + HLA-typing cohorts |
| MAS literature corpus | Public literature | Ground truth: Type 1–4 classification + case reports, for validation |

UK Biobank / dbGaP were evaluated and excluded — application backlog and institutional-affiliation requirements make them infeasible on this project's timeline. ImmPort + GWAS Catalog give real, government-affiliated data without that bottleneck.

**Dataset construction:** since no public dataset contains real patients with 3+ concurrent autoimmune diagnoses and genotype data at scale, patient profiles are constructed semi-synthetically: real GWAS effect sizes generate PRS values, combined with real ImmPort per-disease clinical feature distributions. This is disclosed explicitly as a methodological choice, not presented as a real patient cohort.

## 4. System Architecture

Five layers: **Data sources → Feature engineering → Multi-label ensemble prediction → Explainability + Clustering → Literature validation**.

Polyglot pipeline for full control and reproducibility at each stage:

| Stage | Language | Role |
|-------|----------|------|
| Data pulling | Scala | GWAS Catalog / ImmPort / PubMed API pulls |
| Data cleaning | Go | Canonical schema normalization |
| Orchestration | Rust | Pipeline DAG + run manifests, gRPC/Protobuf control plane |
| ML | Python | Feature engineering, ensemble training, SHAP/LIME, clustering |
| Dashboard | Next.js | Visualization of predictions, explanations, cluster structure |

Every pipeline run produces a manifest (input/output checksums, code version) — the reproducibility guarantee cited in the paper's methods section.

## 5. ML Methodology — Ensemble Classifier (Detailed)

This is the core technical contribution and the part most likely to be scrutinized by reviewers, so the design is laid out in full.

### 5.1 Why an ensemble of XGBoost + CatBoost + LightGBM

All three are gradient-boosted decision tree (GBDT) frameworks, which is the right model family for this problem because:

- **Tabular, structured data** — PRS scores and clinical features are exactly the tabular, mixed-type (continuous PRS + categorical clinical variables) data GBDTs excel at, more so than deep learning at this dataset scale.
- **Native SHAP support** — all three integrate with TreeExplainer, giving exact (not approximated) SHAP values, which matters for the explainability layer's credibility.
- **Complementary strengths, reducing correlated error**:
  - **XGBoost** — strong regularization (L1/L2), robust default choice, extensively validated in genomics/bioinformatics ML literature.
  - **CatBoost** — native handling of categorical clinical variables (e.g. sex, ethnicity, disease subtype flags) without manual one-hot encoding, and ordered boosting reduces prediction shift/overfitting on small-to-medium datasets — relevant given the semi-synthetic dataset's inherent noise.
  - **LightGBM** — leaf-wise tree growth captures different feature interaction patterns than XGBoost's level-wise growth, and is fast enough to make cross-validation over many disease labels and hyperparameter configurations tractable.

Combining models with different inductive biases is a standard way to reduce variance and correlated error in the final prediction — the accepted rationale for GBDT ensembling in applied ML literature, and directly reportable as methodology in the paper.

### 5.2 Multi-label formulation

Each patient has a binary label vector across the N MAS-associated diseases in scope (1 = predicted/actual presence of that disease). Rather than train one ensemble per disease independently, the pipeline uses:

- **Base structure:** one XGBoost, one CatBoost, and one LightGBM model trained per disease label (binary relevance approach) — chosen over a single joint multi-output model because it lets each disease's classifier specialize on its own most-relevant features (important for the explainability/validation goal — you want to know exactly which features drive each disease's prediction, not a blended joint representation), while keeping GBDT's native SHAP support fully exact per disease.
- **Shared cross-disease features** (HLA, CTLA-4, PTPN22, IL23R, OLIG3/TNFAIP3, IL2RA) are engineered once and fed identically into every per-disease model, so the ensemble can independently discover which diseases actually rely on shared loci — this is itself a result worth reporting.

### 5.3 Voting ensemble + score normalization

For each disease *d*, the three base learners (XGBoost, CatBoost, LightGBM) each output a predicted probability. These are combined as follows:

**Step 1 — Score normalization.** Raw predicted probabilities from different GBDT implementations are not always calibrated on the same scale (differing loss functions, regularization). Before combining, each model's output is normalized:

- Min-max normalize each model's raw scores to [0,1] across the validation set, **or**
- Apply **Platt scaling / isotonic regression** calibration per base model (preferred — produces genuinely calibrated probabilities, not just rescaled ranks), so that a "0.7" from XGBoost and a "0.7" from CatBoost represent comparable confidence levels.

**Step 2 — Voting/combination.** Two options, both worth implementing and comparing:

- **Soft voting (weighted average):** final score = w₁ ⋅ p<sub>XGB</sub> + w₂ ⋅ p<sub>CB</sub> + w₃ ⋅ p<sub>LGBM</sub>, with weights either equal (simple average, good baseline) or tuned on a held-out validation set (e.g. via grid search or a small logistic regression "stacking" layer over the three calibrated scores — technically a light stacked ensemble, worth mentioning explicitly as it typically outperforms fixed-weight voting).
- **Hard voting (majority):** each base model casts a binary vote (using its own optimal threshold, e.g. Youden's J from its ROC curve) and the final prediction is the majority label — simpler, more interpretable, but discards confidence information. Report both; soft voting is very likely to be the reported primary result since multi-label risk scoring benefits from continuous, not just binary, outputs (needed downstream for the clustering step, which clusters risk-probability vectors, not hard labels).

**Step 3 — Threshold selection per disease.** Since disease prevalence varies, a single global threshold (e.g. 0.5) is inappropriate — per-disease thresholds are tuned on the validation set to maximize F1 or balance precision/recall depending on clinical priority (screening tools generally favor higher recall).

### 5.4 Evaluation metrics

- **Per-disease:** AUROC, AUPRC (more informative than AUROC under class imbalance, likely given rarer MAS-associated diseases), F1 at the tuned threshold.
- **Multi-label aggregate:** Hamming loss, subset accuracy (exact match across all disease labels — a strict but informative metric), macro-F1 and micro-F1 (macro treats rare diseases equally, micro is prevalence-weighted — report both since they answer different questions).
- **Ensemble vs. individual base learners:** report each base model's standalone performance alongside the ensemble's, to demonstrate the ensemble actually improves over any single model — a reviewer will expect this ablation.

### 5.5 Explainability on the ensemble

SHAP values are computed per base model using **TreeExplainer** (exact, not KernelExplainer/approximate), then combined proportionally to each model's voting weight to produce a single, unified per-patient, per-disease SHAP attribution — so the "why" behind an ensemble prediction remains traceable to specific features, not just an opaque blended score.

LIME is run independently on the final ensemble output (treating it as a black box) as a cross-check that isn't dependent on the tree structure at all, catching any explanation discrepancy between the SHAP-on-components approach and a fully model-agnostic method.

### 5.6 Cross-validation and data leakage control

Given the semi-synthetic construction (PRS from real GWAS effect sizes + clinical features from real ImmPort cohorts), stratified k-fold cross-validation (k=5 or 10) is used per disease label, with folds constructed to avoid leakage between the genetic and clinical feature generation processes — i.e., ensuring no synthetic patient's clinical profile and PRS profile are drawn from overlapping "seed" records across train/validation splits.

## 6. Clustering and Validation

Model output risk-probability vectors (one vector per patient across all diseases) are clustered (hierarchical clustering, primary candidate — produces a dendrogram directly comparable to the Type 1–4 tree-like grouping structure). Resulting clusters are compared against:

1. Humbert & Dupond's Type 1–4 classification
2. Betterle et al.'s expanded Type 3/APS-3 combinations
3. The broader MAS case-report literature synthesized by Anaya et al.

**Three possible outcomes, all reportable:** clusters recover existing types (validates pipeline), partially match with new combinations (the novel contribution), or diverge (a discussion-worthy negative result).

## 7. Target Publication

**Primary target:** *npj Digital Medicine* (Nature Portfolio) — chosen because:
- A directly relevant precedent already exists there: "Artificial Intelligence for Autoimmune Diseases," *npj Digital Medicine*, 2025 — confirms editorial appetite for exactly this topic.
- Nature Portfolio-level prestige and indexing (PubMed, Scopus, Web of Science) without the near-unreachable bar of *Nature* itself.
- Favors clinically-motivated ML papers (not just benchmark performance) — matches this project's reclassification angle well.

**Fallback tier:**
1. *Scientific Reports* (Nature Portfolio) — broader scope, higher acceptance rate, still strongly indexed; relevant precedent: "Machine learning for precision diagnostics of autoimmunity."
2. *Frontiers in Immunology* (Systems Immunology section) — closest methodological precedent ("A multicenter explainable machine learning analysis of autoimmune disease comorbidity in ankylosing spondylitis," Feb 2026), open access, faster review.

## 8. Roadmap Status

| Phase | Status |
|-------|--------|
| Problem framing, novelty angle, literature validation | ✅ Complete |
| System architecture | ✅ Complete |
| Data source selection (GWAS Catalog + ImmPort) | ✅ Complete |
| Tech stack + reproducibility design | ✅ Complete |
| Target journal selection | ✅ Complete |
| ML methodology fully specified | ✅ Complete |
| Lock final disease list + SNP feature set | ⬜ Pending |
| Implement puller/cleaner/orchestrator services | ✅ Complete |
| Implement feature engineering + composite dataset construction | 🟡 In Progress |
| Train ensemble (XGBoost + CatBoost + LightGBM), tune voting weights | ⬜ Pending |
| Run SHAP/LIME explainability | ⬜ Pending |
| Run clustering + literature validation | ⬜ Pending |
| Build dashboard | 🟡 In Progress |
| Fill in paper Results/Discussion/Conclusion with actual findings | ⬜ Pending |
| Submit to *npj Digital Medicine* | ⬜ Pending |

## Current Limitations & Next Steps

### Current Limitations

1. **ImmPort authentication required:** Real clinical data could not be fetched without an API key. The pipeline uses semi-synthetic clinical features.
2. **Small sample size:** 50 patients is sufficient for pipeline validation but not for publication-grade statistical inference. Target: 500+ patients.
3. **Class imbalance:** MONOGENIC_DIABETES had <2 classes in the sample, causing CatBoost training failure. Rare diseases require stratified sampling or oversampling.
4. **No protobuf codegen:** gRPC stubs are not yet generated, so services communicate via REST gateway only.
5. **LIME regression mode:** LIME was forced into regression mode due to library limitations with classifier probability outputs. A custom wrapper or different library (e.g., SHAP KernelExplainer) may provide better local explanations.

### Immediate Next Steps

1. **Run `make proto`** to generate gRPC stubs and wire services together.
2. **Obtain ImmPort API credentials** to replace synthetic clinical features with real cohort data.
3. **Scale to 500+ patients** by expanding GWAS loci and using real ImmPort cohorts.
4. **Hyperparameter tuning** via grid search or Bayesian optimization.
5. **Literature validation:** Map clusters to Humbert & Dupond Type 1–4 classification and report matches/divergences.
6. **Dashboard integration:** Start Rust control plane and verify live data flow to Next.js frontend.

## Appendix: Key Verified References

1. Humbert, P. & Dupond, J. L. (1988). Les syndromes auto-immuns multiples. *Ann Med Interne*, 139(3):159–168.
2. Anaya, J. M. et al. (2012). The Multiple Autoimmune Syndromes. A Clue for the Autoimmune Tautology. *Clin Rev Allergy Immunol*, 43(3):256–264.
3. Betterle, C. et al. (2023). Type 3 Autoimmune Polyglandular Syndrome (APS-3) or Type 3 Multiple Autoimmune Syndrome (MAS-3): An Expanding Galaxy. *J Endocrinol Invest*, 46(4):643–665.
4. Somers, E. C. et al. (2006). Autoimmune Diseases Co-occurring within Individuals and within Families: A Systematic Review. *Epidemiology*, 17(2):202–217.
5. Brand, O. J., Gough, S. C., Heward, J. M. (2005). HLA, CTLA-4 and PTPN22: The Shared Genetic Master-Key to Autoimmunity? *Expert Rev Mol Med*, 7(23):1–15.
6. PTPN22: The Archetypal Non-HLA Autoimmunity Gene. *Nat Rev Rheumatol*, 2014. PMID: 25003765.
7. Sollis, E. et al. (2023). The NHGRI-EBI GWAS Catalog: Knowledgebase and Deposition Resource. *Nucleic Acids Research*, 51(D1):D977–D985.
8. Bhattacharya, S. et al. (2018). ImmPort, Toward Repurposing of Open Access Immunological Assay Data for Translational and Clinical Research. *Scientific Data*, 5:180015.
9. A Comprehensive Analysis of Shared Loci between SLE and Sixteen Autoimmune Diseases Reveals Limited Genetic Overlap. *PLoS Genetics*, 2011. PMCID: PMC3234215.
10. Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*, 30.
11. Ribeiro, M. T., Singh, S., Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*, pp. 1135–1144.
12. A Multicenter Explainable Machine Learning Analysis of Autoimmune Disease Comorbidity in Ankylosing Spondylitis. *Frontiers in Immunology*, Systems Immunology, 2026.
13. Mahajan, A., LaChance, A. H., Rodman, A. et al. (2025). Artificial Intelligence for Autoimmune Diseases. *npj Digital Medicine*, 8:628.

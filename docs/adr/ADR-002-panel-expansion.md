# 📜 ADR-002: Panel expansion source & curation method

> **Status:** `Decided`
> **Date:** September 2026
> **Item:** F-09 of ROADMAP.md

---

## 🌎 Context

The launch panel has 8 loci — too few for the LD-GNN (F-01), functional
annotations (F-08), and a credible PRS baseline (F-16). The panel must grow to
50–100 loci with per-locus provenance, and every candidate must be *verified*
against a real authority rather than trusted from memory.

Constraint discovered during the first verification pass: roughly 40% of
from-memory rsID transcriptions failed against the GWAS Catalog (404s — typos,
merged rsIDs, or wrong variant), and the public Catalog API rate-limits (429)
under sustained polling. Verification must therefore be live, incremental, and
crash-proof.

## 🛤️ Options Considered

1. **Pull panel programmatically from GWAS Catalog EFO traits** — unbiased and
   fully reproducible, but EFO trait pulls drag in hundreds of marginal
   loci (secondary signals, non-replicated hits) requiring quality filters we
   cannot automate honestly yet.
2. **Literature-curated candidates + live Catalog verification** — smaller,
   high-confidence panel anchored to landmark papers (Okada 2014 RA, Bentham
   2015 SLE, IIBDGC 2011 MS, etc.); verification catches transcription errors.
3. **Import an existing curated panel (e.g. PGS Catalog scoring files)** —
   fastest, but couples our panel schema to an external format and hides the
   provenance per locus behind a bulk download.

---

## 🎯 Decision

> [!IMPORTANT]
> **Expand the panel via literature-curated candidates (option 2), each carrying
> per-locus provenance, verified live against the GWAS Catalog with an honest
> drop log; 40% drop rate is acceptable because drops are recorded with reasons.
> The verified manifest (not the candidate list) feeds downstream pipeline work.**

## 🧠 Reasoning

A curated list of landmark loci keeps every entry interpretable and citable —
exactly what the paper's methods section needs. The high transcription-failure
rate is not a defect of the method but evidence the verification layer is
doing its job; trusting the unverified list would have shipped dozens of wrong
rsIDs into genotypes, LD edges, and the PRS baseline. The rate-limit problem is
handled operationally (throttling + crash-proof incremental resume), not by
moving to bulk downloads, because per-locus provenance is a first-class
requirement of the program (F-08 depends on it).

The runner (`scripts/expand_panel.py`) persists after every fetch and resumes
from ERROR rows only — a timeout costs one API call, not the run.

## ⚖️ Consequences

- **Good:** 🟢 54/99 candidates verified withCatalog-grounded evidence; every
  drop has a recorded reason; panel is reproducible by rerunning the script.
- **Bad:** 🔴 Panel coverage is uneven — SJOGRENS (4), T1D (3), VITILIGO (1)
  are thin vs SLE (14) / RA (12); wave-3 curation needed for the thin diseases
  before per-disease claims can lean on the panel. Catalog API pace limits
  re-verification frequency.
- **Neutral:** ⚪ The verified panel is not yet wired into the dataset builders;
  that integration is a separate change so this ADR stays reviewable.

## 🔄 Revisit When

- SJOGRENS/T1D/VITILIGO coverage reaches parity (wave-3 curation).
- F-01/F-08/F-16 integration begins and the manifest schema needs extending
  (LD info, gene mapping).
- The GWAS Catalog adds bulk rsID resolution endpoints that make re-verification
  cheap enough to run per-commit.

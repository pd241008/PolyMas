# results/

All pipeline artifacts live here. **Tracked by default: summary/metrics JSONs
only** — everything else is local until the paper is finalised.

## What is committed (tracked whitelist)

- `**/*.json` ≤ 300 KB — gate reports, summaries, per-disease metrics,
  sweep histories, thresholds, adjacency reports, pretrain/FT histories:
  the auditable numeric record (408 files, ~13 MB).
- `README.md` (this file) and `report.md` (project report source).

## What stays local (gitignored)

| Class | Why | Ships when |
|-------|-----|------------|
| `**/*.csv` | data tables, per-patient predictions | with the paper's data-availability statement |
| `**/*.parquet`, model checkpoints, `.pt`, `.txt` models | model artifacts | on request / repository archive |
| `figures/`, `**/*.png` | publication figures | **withheld until the paper is finalised** — released alongside it |
| `**/*.pdf`, `report.pdf`, `results.pdf` | generated reports | withheld; the PDF ships with the paper |
| `**/*.zip`, `.sha256` | phase bundle exports | superseded by the tracked JSON record |
| `**/gwasinfo.json` (~20 MB each) | OpenGWAS probe cache, re-fetchable | never (reproducible via `scripts/ogwas_probe.py`) |
| `**/dendrogram.json` (1–5 MB) | patient-level cluster trees | with the paper's supplementary data |

To override the ignore rules for a specific file, use `git add -f <path>` —
and record why, here.

## Layout

- `results/` (this level, e.g. `features/`, `models/`, `clusters/`, `stats/`) —
  the canonical 5,000-patient System A run
- `results_final_20260926/` — canonical run + all Phase 2–4 evaluation
  evidence (`f01…f05`, `f11…f15`, `adr006_gates/`, `system_b_curves/`)
- `results_real_20260925/`, `results_phase2_20260925/`, `results_e2e_20260925/`
  — the 09-25 real-donor run, Phase-2 honesty layer, and e2e verification
- `results_scaling_n{1000,2500,10000}/` — F-19 System A scaling arms
- `real_genotypes_20260925/`, `panel_expansion_20260925/` — Phase-1 substrate
- `results_pilot_400_2026-07/`, `results_pilot_5k_prefix_20260924/` — superseded pilots, kept for provenance
- `figures/` — 28 publication figures (`scripts/generate_figures.py`)
- `adr005_probe/`, `adr005_gates/` — the rejected donor-weighting coupling arm

A fresh pipeline run recreates any missing subdirectory automatically
(`POLYMAS_RESULTS_DIR` overrides the root in every run/reader script).

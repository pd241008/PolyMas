"""F-10 + F-07 + F-08: real genotype substrate, PCs, and annotations.

Builds the real-genotype foundation for the verified panel:
  1. Resolves panel rsIDs to GRCh37 (also captures consequence -> F-08 input).
  2. Streams 1000G phase-3 VCFs for dosages of the super-population samples.
  3. Validates computed LD r2 against Ensembl's published phase-3 LD
     (pre-registered tolerance: |delta| <= 0.05 per pair).
  4. Derives ancestry PCs from the real genotype matrix (F-07).
  5. Emits a provenance manifest.

Outputs -> stash/results/real_genotypes_<date>/
Run:
  PYTHONPATH=services/ml-engine-python services/ml-engine-python/.venv/bin/python \
      scripts/build_real_genotype_substrate.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "services" / "ml-engine-python"))

from polymas_ml.data.genotypes import (  # noqa: E402
    SUPER_POPS,
    compute_ld_r2,
    fetch_genotypes,
    load_population_panel,
    resolve_positions,
    validate_ld_against_ensembl,
)
from polymas_ml.data.loci import unique_loci  # noqa: E402

RESULTS_ROOT = Path(os.environ.get("POLYMAS_RESULTS_DIR", PROJECT_ROOT / "stash" / "results"))
RUN_TAG = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
OUT_DIR = RESULTS_ROOT / f"real_genotypes_{RUN_TAG}"
CACHE_DIR = RESULTS_ROOT / "raw" / "1000g_cache"
LD_TOLERANCE = 0.05  # pre-registered in ROADMAP (F-10)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("genotype_substrate")


def ancestry_pcs(dosages: pd.DataFrame, meta: pd.DataFrame, n_pcs: int = 6) -> pd.DataFrame:
    """PCs from the real dosage matrix (F-07). Standardization per variant;
    SVD-based PCA; returns loadings per sample."""
    X = dosages.to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    X = np.nan_to_num(X)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = U[:, :n_pcs] * S[:n_pcs]
    pc_df = pd.DataFrame(
        pcs, index=dosages.index, columns=[f"PC{i+1}" for i in range(n_pcs)]
    )
    var_explained = (S[:n_pcs] ** 2) / np.sum(S**2)
    return pc_df, var_explained


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_loci = unique_loci()
    rs_ids = [l.rs_id for l in panel_loci]
    logger.info("Panel: %d unique loci", len(rs_ids))

    # 1. Resolve positions + consequences (F-08 side-output).
    variants = resolve_positions(rs_ids, CACHE_DIR)
    unresolvable = [rs for rs in rs_ids if rs not in variants]
    logger.info("Resolved %d/%d", len(variants), len(rs_ids))

    # 2. 1000G dosages.
    panel = load_population_panel(CACHE_DIR)
    dosages, meta = fetch_genotypes(variants, panel, CACHE_DIR)
    logger.info("Dosage matrix: %d samples x %d variants", *dosages.shape)

    # 3. LD validation vs Ensembl (R2 check, tolerance pre-registered).
    ld_rows = []
    anchors = [l.rs_id for l in panel_loci if l.rs_id in dosages.columns][:8]
    for sp in ["EUR", "AFR", "EAS"]:
        ld_rows.append(validate_ld_against_ensembl(dosages, meta, sp, anchors, CACHE_DIR))
        time.sleep(0.5)
    ld_df = pd.concat([d for d in ld_rows if not d.empty], ignore_index=True)
    ld_df.to_csv(OUT_DIR / "ld_validation.csv", index=False)
    n_pass = int((ld_df["abs_delta"] <= LD_TOLERANCE).sum())
    n_total = len(ld_df)
    ld_pass = n_total > 0 and (n_pass / n_total) >= 0.9
    logger.info(
        "LD validation: %d/%d pairs within +-%.2f (pass=%s)",
        n_pass, n_total, LD_TOLERANCE, ld_pass,
    )

    # 4. Ancestry PCs (F-07).
    pc_df, var_expl = ancestry_pcs(dosages, meta)
    pc_df.join(meta).to_csv(OUT_DIR / "ancestry_pcs.csv")
    logger.info("PCs: top-6 variance explained = %s", np.round(var_expl, 4).tolist())

    # 5. LD matrix (real, per super-population) for the future GNN (F-01).
    for sp in SUPER_POPS:
        sub = dosages.loc[meta[meta["super_pop"] == sp].index]
        if len(sub) < 20:
            continue
        compute_ld_r2(sub).to_csv(OUT_DIR / f"ld_r2_{sp}.csv")

    # 6. Variant annotations table (F-08 seed: position/consequence; gene
    # mapping extension comes with the F-08 follow-up commit).
    annotations = pd.DataFrame([
        {
            "rs_id": v.rs_id,
            "chrom": v.chrom,
            "pos_grch37": v.pos,
            "ref": v.ref,
            "alt": v.alt,
            "consequence": v.consequence,
            "maf_all": float((dosages[v.rs_id] / 2).mean()) if v.rs_id in dosages else None,
        }
        for v in variants.values()
    ])
    annotations.to_csv(OUT_DIR / "variant_annotations.csv", index=False)

    dosages.join(meta).to_csv(OUT_DIR / "genotype_dosages.csv")

    manifest = {
        "run_date": RUN_TAG,
        "n_panel_loci": len(rs_ids),
        "n_resolved": len(variants),
        "unresolvable": unresolvable,
        "n_samples": int(dosages.shape[0]),
        "n_variants_typed": int(dosages.shape[1]),
        "super_pops": SUPER_POPS,
        "ld_validation": {
            "n_pairs": n_total,
            "n_within_tolerance": n_pass,
            "tolerance": LD_TOLERANCE,
            "pass": ld_pass,
        },
        "pc_variance_explained": var_expl.tolist(),
        "provenance": {
            "vcf": "1000G phase3 v5b 20130502 (GRCh37, phased)",
            "panel": "integrated_call_samples_v3.20130502.ALL.panel",
            "position_resolution": "Ensembl REST /variation/human/ids",
            "ld_reference": "Ensembl REST /ld/human 1000GENOMES:phase_3",
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    logger.info("Done -> %s", OUT_DIR)


if __name__ == "__main__":
    main()

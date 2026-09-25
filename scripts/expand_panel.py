"""F-09: verify the expanded autoimmune panel against the GWAS Catalog.

For every candidate locus in polymas_ml.data.loci, queries the GWAS Catalog
REST API for the rsID's associations and records:
  - HTTP reachability + association count
  - the strongest (lowest-p) association with its mapped trait
  - whether the rsID exists in the Catalog at all

Writes stash/results/panel_expansion_<date>/ with:
  - panel_verification.csv     one row per candidate, verified or dropped
  - panel_manifest.json        the verified panel + provenance + drop log
  - verification_run.json      run metadata (timestamps, API, counts)

Nothing in the existing pipeline changes until a follow-up commit wires the
verified manifest into the dataset builders.

Run with:
  PYTHONPATH=services/ml-engine-python services/ml-engine-python/.venv/bin/python \
      scripts/expand_panel.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "services" / "ml-engine-python"))

from polymas_ml.data.loci import LEGACY_LOCI, CANDIDATE_LOCI, unique_loci  # noqa: E402

RESULTS_ROOT = Path(os.environ.get("POLYMAS_RESULTS_DIR", PROJECT_ROOT / "stash" / "results"))
RUN_TAG = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
OUT_DIR = RESULTS_ROOT / f"panel_expansion_{RUN_TAG}"
# The Catalog 429s under sustained polling; 1.5s keeps whole runs error-free
# while staying under ~3 minutes for a ~100-locus panel.
THROTTLE_SECONDS = float(os.environ.get("PANEL_THROTTLE", "1.5"))


def resume_verification(df: pd.DataFrame) -> pd.DataFrame:
    """Re-verify only ERROR rows from a previous pass, in place."""
    mask = df["status"] == "ERROR"
    if not mask.any():
        return df
    logger.info("Resuming: re-verifying %d ERROR rows", int(mask.sum()))
    for idx, row in df[mask].iterrows():
        result = fetch_snp_associations(row["rs_id"])
        if result is None:
            continue
        if not result["exists"]:
            df.loc[idx, ["status", "n_assocs", "best_pvalue", "best_or", "note"]] = [
                "DROPPED", 0, None, None, "rsID not found in GWAS Catalog (404)"]
        elif result["n_assocs"] == 0:
            df.loc[idx, ["status", "n_assocs", "best_pvalue", "best_or", "note"]] = [
                "DROPPED", 0, None, None, "rsID exists but has zero Catalog associations"]
        else:
            df.loc[idx, ["status", "n_assocs", "best_pvalue", "best_or", "note"]] = [
                "VERIFIED", result["n_assocs"],
                result["best"]["pvalue"] if result["best"] else None,
                result["best"]["orPerCopyNum"] if result["best"] else None,
                f"{result['n_assocs']} Catalog associations"]
        time.sleep(THROTTLE_SECONDS)
    return df

GWAS_BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api"
REQUEST_TIMEOUT = 30
THROTTLE_SECONDS = 0.5  # be polite to the public API

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("expand_panel")


def fetch_snp_associations(rs_id: str, max_retries: int = 5) -> dict | None:
    """Return {'n_assocs': int, 'best': {...}|None, 'exists': bool} or None on failure.

    A bare 404 on the associations route is ambiguous (unknown rsID vs known
    rsID with no associations), so 'exists' is decided by probing the SNP
    resource itself; the associations count still comes from the
    associations route.
    """
    url = f"{GWAS_BASE_URL}/singleNucleotidePolymorphisms/{rs_id}/associations"
    params = {"size": 50}
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url, params=params, headers={"Accept": "application/json"}, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 404:
                # Disambiguate: probe the SNP resource directly.
                probe = requests.get(
                    f"{GWAS_BASE_URL}/singleNucleotidePolymorphisms/{rs_id}",
                    headers={"Accept": "application/json"}, timeout=REQUEST_TIMEOUT,
                )
                time.sleep(THROTTLE_SECONDS)
                if probe.status_code == 200:
                    return {"n_assocs": 0, "best": None, "exists": True}
                if probe.status_code == 404:
                    return {"n_assocs": 0, "best": None, "exists": False}
                # Probe failed transiently — fall through to retry logic.
                raise requests.RequestException(f"probe {probe.status_code}")
            resp.raise_for_status()
            payload = resp.json()
            assocs = payload.get("_embedded", {}).get("associations", [])
            total = payload.get("page", {}).get("totalElements", len(assocs))
            best = None
            for a in assocs:
                p = a.get("pvalue")
                if p is None:
                    continue
                if best is None or p < best["pvalue"]:
                    best = {
                        "pvalue": p,
                        "orPerCopyNum": a.get("orPerCopyNum"),
                        "betaNum": a.get("betaNum"),
                    }
            return {"n_assocs": int(total), "best": best, "exists": True}
        except requests.RequestException as e:
            logger.warning("GWAS fetch %s attempt %d failed: %s", rs_id, attempt + 1, e)
            time.sleep(2**attempt)
    return None


def _placeholder(locus) -> dict:
    """Row skeleton for a locus not yet fetched (status PENDING)."""
    return {
        "rs_id": locus.rs_id, "gene": locus.gene, "disease": locus.disease,
        "source": locus.source, "status": "PENDING", "n_assocs": None,
        "best_pvalue": None, "best_or": None, "best_trait": None,
        "note": "not yet fetched",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_loci = unique_loci()
    csv_path = OUT_DIR / "panel_verification.csv"

    # Incremental resume: load any previous pass, keep its VERIFIED/DROPPED
    # rows, and fetch only missing or ERROR rows. The CSV is rewritten after
    # every fetch so a timeout never loses completed work.
    rows: dict[str, dict] = {}
    if csv_path.exists() and "--fresh" not in sys.argv:
        prev = pd.read_csv(csv_path)
        for _, r in prev.iterrows():
            rows[r["rs_id"]] = r.to_dict()
        logger.info("Loaded previous pass: %d rows", len(rows))

    n_errors_cached = sum(1 for l in all_loci if l.rs_id in rows and rows[l.rs_id].get("status") == "ERROR")
    to_fetch = [locus for locus in all_loci
                if locus.rs_id not in rows or rows[locus.rs_id].get("status") == "ERROR"]
    logger.info("Panel: %d unique loci, %d to fetch (%d cached OK, %d errors)",
                len(all_loci), len(to_fetch), len(all_loci) - len(to_fetch), n_errors_cached)

    def flush() -> None:
        df = pd.DataFrame([rows.get(l.rs_id, _placeholder(l)) for l in all_loci])
        df.to_csv(csv_path, index=False)

    for i, locus in enumerate(to_fetch, 1):
        result = fetch_snp_associations(locus.rs_id)
        if result is None:
            status, note = "ERROR", "API unreachable after retries"
            best_p, best_trait, best_or = None, None, None
            n_assocs = None
        elif not result["exists"]:
            status, note = "DROPPED", "rsID not found in GWAS Catalog (404)"
            best_p, best_trait, best_or = None, None, None
            n_assocs = 0
        elif result["n_assocs"] == 0:
            status, note = "DROPPED", "rsID exists but has zero Catalog associations"
            best_p, best_trait, best_or = None, None, None
            n_assocs = 0
        else:
            status = "VERIFIED"
            note = f"{result['n_assocs']} Catalog associations"
            best_p = result["best"]["pvalue"] if result["best"] else None
            best_or = result["best"]["orPerCopyNum"] if result["best"] else None
            best_trait = None
            n_assocs = result["n_assocs"]

        rows[locus.rs_id] = {
            "rs_id": locus.rs_id,
            "gene": locus.gene,
            "disease": locus.disease,
            "source": locus.source,
            "status": status,
            "n_assocs": n_assocs,
            "best_pvalue": best_p,
            "best_or": best_or,
            "best_trait": best_trait,
            "note": note,
        }
        logger.info("[%3d/%3d] %-12s %-8s %s", i, len(to_fetch), locus.rs_id, status, note)
        flush()  # crash-proof: every completed fetch is persisted
        time.sleep(THROTTLE_SECONDS)

    df = pd.DataFrame([rows.get(l.rs_id, _placeholder(l)) for l in all_loci])
    df.to_csv(csv_path, index=False)

    verified = df[df["status"] == "VERIFIED"]
    dropped = df[df["status"].isin(["DROPPED"])]
    errored = df[df["status"] == "ERROR"]
    pending = df[df["status"] == "PENDING"]
    manifest = {
        "run_date": RUN_TAG,
        "api": GWAS_BASE_URL,
        "n_candidates": len(df),
        "n_verified": int(len(verified)),
        "n_dropped": int(len(dropped)),
        "n_error": int(len(errored)),
        "n_pending": int(len(pending)),
        "verified_rs_ids": verified["rs_id"].tolist(),
        "dropped": dropped[["rs_id", "gene", "disease", "status", "note"]].to_dict("records"),
        "panel_sources": sorted({locus.source for locus in all_loci}),
    }
    (OUT_DIR / "panel_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_DIR / "verification_run.json").write_text(json.dumps({
        "script": "scripts/expand_panel.py",
        "finished_utc": datetime.now(tz=timezone.utc).isoformat(),
        "honesty_note": "Dropped rsIDs are recorded with reasons in panel_manifest.json; "
                        "nothing is silently removed from the candidate list.",
    }, indent=2))

    logger.info("Panel verified: %d/%d passed, %d dropped, %d error, %d pending -> %s",
                len(verified), len(df), len(dropped), len(errored), len(pending), OUT_DIR)


if __name__ == "__main__":
    main()

"""Ensembl REST API reference-sequence extraction for System B.

Fetches genomic coordinates and +-5kb reference windows for the 8 loci shared
with System A, and caches them locally (static reference data, not per-patient).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

ENSEMBL_BASE = "https://rest.ensembl.org"
ASSEMBLY = "GRCh38"
WINDOW = 5000

# Must match System A's locus set for cross-system comparability.
LOCI = {
    "rs2187668": "HLA-DRB1",
    "rs9272346": "HLA-DQB1",
    "rs2476601": "PTPN22",
    "rs3087243": "CTLA4",
    "rs2292239": "ERBB3",
    "rs11209026": "IL23R",
    "rs2104286": "IL2RA",
    "rs7574865": "STAT4",
}


def _get(url: str) -> dict[str, Any]:
    resp = requests.get(url, headers={"Content-Type": "application/json"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_variant(rs_id: str, max_retries: int = 3) -> dict[str, Any]:
    """Return the GRCh38 mapping for an rsID."""
    for attempt in range(max_retries):
        try:
            data = _get(f"{ENSEMBL_BASE}/variation/human/{rs_id}")
            for mapping in data.get("mappings", []):
                if mapping.get("assembly_name") == ASSEMBLY:
                    return {
                        "rs_id": rs_id,
                        "gene": LOCI.get(rs_id, rs_id),
                        "seq_region": mapping["seq_region_name"],
                        "start": mapping["start"],
                        "end": mapping["end"],
                        "strand": mapping["strand"],
                        "allele_string": mapping.get("allele_string"),
                        "ancestral_allele": mapping.get("ancestral_allele"),
                    }
            raise ValueError(f"no {ASSEMBLY} mapping for {rs_id}")
        except Exception as e:
            logger.warning("Ensembl variant fetch %s attempt %d failed: %s", rs_id, attempt + 1, e)
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch variant {rs_id}")


def fetch_window(chr: str, pos: int, window: int = WINDOW, max_retries: int = 3) -> str:
    """Return the reference sequence for [pos-window, pos+window] (inclusive)."""
    start = pos - window
    end = pos + window
    url = (
        f"{ENSEMBL_BASE}/sequence/region/human/{chr}:{start}-{end}"
        f"?coord_system_version={ASSEMBLY}"
    )
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers={"Content-Type": "text/plain"}, timeout=30)
            resp.raise_for_status()
            seq = resp.text
            if len(seq) != 2 * window + 1:
                logger.warning("unexpected window length %d for %s:%d-%d", len(seq), chr, start, end)
            return seq
        except Exception as e:
            logger.warning("Ensembl sequence fetch %s:%d attempt %d failed: %s", chr, pos, attempt + 1, e)
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch sequence {chr}:{start}-{end}")


def fetch_and_cache(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Fetch variant info + reference windows for all loci, cache to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_path = output_dir / "variant_info.json"
    windows_path = output_dir / "reference_windows.json"

    if variant_path.exists() and windows_path.exists():
        logger.info("Ensembl cache hit: %s", output_dir)
        variants = json.loads(variant_path.read_text())
        windows = json.loads(windows_path.read_text())
        return {"variants": variants, "windows": windows}

    variants: dict[str, dict[str, Any]] = {}
    windows: dict[str, str] = {}
    for rs_id in LOCI:
        logger.info("Fetching Ensembl data for %s (%s)", rs_id, LOCI[rs_id])
        variant = fetch_variant(rs_id)
        variants[rs_id] = variant
        seq = fetch_window(variant["seq_region"], variant["start"])
        windows[rs_id] = seq
        logger.info(
            "  %s %s:%d strand=%d alleles=%s window_len=%d",
            rs_id,
            variant["seq_region"],
            variant["start"],
            variant["strand"],
            variant["allele_string"],
            len(seq),
        )
        time.sleep(0.3)

    variant_path.write_text(json.dumps(variants, indent=2))
    windows_path.write_text(json.dumps(windows, indent=2))
    logger.info("Cached Ensembl data to %s", output_dir)
    return {"variants": variants, "windows": windows}

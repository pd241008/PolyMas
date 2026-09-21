"""ImmPort Shared Data API client: subject-level demographics by study.

Fetches real subject records (demographic table) for study accessions and
maps them onto the PolyMas clinical feature schema:

  - sex               <- demographic.gender (Female/Male)          [real]
  - age               <- demographic.max_subject_age_in_years      [real]
  - ancestry          <- race (White->EUR, Black->AFR, Asian->EAS) [real, mapped]
  - hispanic status   <- ethnicity (Hispanic or Latino)            [real]
  - bmi               <- not collected in ImmPort demographics     [modeled]
  - family_history    <- not collected in ImmPort demographics     [modeled]
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

IMMPORT_QUERY_BASE = "https://www.immport.org/data/query/api/study"

# Studies with real autoimmune cohorts on ImmPort (subject-level data).
# Each entry: disease group -> list of study accessions to pool (round-robin).
# AITD and Vitiligo have NO studies on ImmPort -> no real cohort available.
COHORT_STUDIES: dict[str | None, list[str]] = {
    "RA": ["SDY473", "SDY824", "SDY2507"],
    "SLE": ["SDY2195", "SDY1475", "SDY474"],
    "T1D": ["SDY1904", "SDY2594", "SDY1628"],
    "MS": ["SDY1043", "SDY2869", "SDY3285"],
    "SJOGRENS": ["SDY823", "SDY961"],
    None: ["SDY1", "SDY180"],  # non-autoimmune subjects (rhinitis / healthy)
}

DISEASE_LABELS = ["RA", "SLE", "SJOGRENS", "AITD", "T1D", "VITILIGO", "MS"]
# Diseases with no real ImmPort cohort -> labels drawn from background
# prevalence only, and their cohort-specific clinical signal is modeled.
MODELED_DISEASES = ["AITD", "VITILIGO"]

_RACE_TO_ANCESTRY = {
    "white": "EUR",
    "black or african american": "AFR",
    "asian": "EAS",
}
_BACKGROUND_ANCESTRY = ["EUR", "AFR", "EAS", "SAS"]
_BACKGROUND_ANCESTRY_P = [0.50, 0.20, 0.15, 0.15]


def _auth_headers() -> dict[str, str]:
    api_key = os.environ.get("IMMPORT_API_KEY", "")
    if not api_key:
        logger.warning("IMMPORT_API_KEY not set — ImmPort requests will be rejected (401)")
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def fetch_demographics(study_id: str, max_retries: int = 3, timeout: float = 45.0) -> list[dict[str, Any]]:
    """Fetch the subject-level demographic table for one study."""
    url = f"{IMMPORT_QUERY_BASE}/demographic/{study_id}"
    headers = _auth_headers()
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 401:
                logger.warning("ImmPort demographic %s returned 401 — check IMMPORT_API_KEY", study_id)
                return []
            resp.raise_for_status()
        except Exception as e:
            logger.warning("ImmPort demographic fetch %s attempt %d failed: %s", study_id, attempt + 1, e)
            time.sleep(2 ** attempt)
    return []


def build_subject_pool(
    cache_dir: Path,
    cohort_studies: dict[str | None, list[str]] | None = None,
) -> pd.DataFrame:
    """Fetch (with disk cache) demographics for all cohort studies.

    Returns one row per real subject with columns:
      subject_accession, study_accession, disease_group (COHORT key or None),
      sex ('F'/'M'), age_years (float|NaN), ancestry (EUR/AFR/EAS/SAS/UNK),
      is_hispanic (0/1)
    """
    cohort_studies = cohort_studies or COHORT_STUDIES
    cache_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    fetched_counts: dict[str, int] = {}
    for group, study_ids in cohort_studies.items():
        for study_id in study_ids:
            cache_path = cache_dir / f"subjects_{study_id}.json"
            if cache_path.exists():
                rows = json.loads(cache_path.read_text())
                logger.info("Loaded cached demographics for %s (%d subjects)", study_id, len(rows))
            else:
                rows = fetch_demographics(study_id)
                cache_path.write_text(json.dumps(rows))
                time.sleep(0.5)
            fetched_counts[study_id] = len(rows)
            for r in rows:
                records.append({
                    "subject_accession": r.get("subjectAccession"),
                    "study_accession": study_id,
                    "disease_group": group,
                    "sex": _map_sex(r.get("gender")),
                    "age_years": r.get("max_subject_age_in_years"),
                    "ancestry": _map_ancestry(r.get("race")),
                    "is_hispanic": 1 if str(r.get("ethnicity") or "").startswith("Hispanic") else 0,
                })

    pool = pd.DataFrame.from_records(records)
    pool = pool.dropna(subset=["subject_accession"]).drop_duplicates(subset="subject_accession")
    # Coerce ages to numeric (some studies report strings/None) -> NaN when missing.
    pool["age_years"] = pd.to_numeric(pool["age_years"], errors="coerce")
    logger.info(
        "Subject pool: %d unique subjects from %d studies (%s)",
        len(pool), len(fetched_counts), fetched_counts,
    )
    return pool


def _map_sex(gender: str | None) -> str:
    g = str(gender or "").strip().lower()
    if g.startswith("f"):
        return "F"
    if g.startswith("m"):
        return "M"
    return "U"


def _map_ancestry(race: str | None, rng: np.random.Generator | None = None) -> str:
    r = str(race or "").strip().lower()
    if r in _RACE_TO_ANCESTRY:
        return _RACE_TO_ANCESTRY[r]
    # Race not one of the mapped categories (or missing): draw background
    # ancestry so the ancestry one-hots stay populated; flagged via 'UNK' path.
    if rng is None:
        rng = np.random.default_rng(42)
    return str(rng.choice(_BACKGROUND_ANCESTRY, p=_BACKGROUND_ANCESTRY_P))


def assign_subjects(
    pool: pd.DataFrame,
    n_patients: int,
    patient_groups: list[str | None],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Round-robin assign unique real subjects to patients.

    patient_groups[i] is the preferred disease cohort for patient i (or None
    for background patients). Assignment walks each group's shuffled pool in
    order and skips already-used subjects; if a group pool is exhausted it
    falls back to the background (None) pool, then to any unused subject.
    """
    pools: dict[str | None, list[pd.Series]] = {}
    for group in list(COHORT_STUDIES.keys()):
        subset = pool[pool["disease_group"] == group] if group else pool[pool["disease_group"].isna()]
        rows = subset.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))) if len(subset) else []
        pools[group] = list(rows.itertuples(index=False)) if len(rows) else []

    used: set[str] = set()
    cursors: dict[str | None, int] = {g: 0 for g in pools}
    fallback_order: list[str | None] = [None] + [g for g in pools if g is not None]

    def _take(group: str | None) -> pd.Series | None:
        order = [group] + [g for g in fallback_order if g != group] if group is not None else fallback_order
        for g in order:
            pool_rows = pools.get(g, [])
            n = len(pool_rows)
            start = cursors.get(g, 0)
            for k in range(n):
                cand = pool_rows[(start + k) % n] if n else None
                if cand is not None and cand.subject_accession not in used:
                    cursors[g] = (start + k + 1) % n if n else 0
                    used.add(cand.subject_accession)
                    return cand
        return None

    assignments: list[dict[str, Any]] = []
    for group in patient_groups:
        sub = _take(group)
        if sub is None:
            raise RuntimeError("Subject pool exhausted — cannot assign unique subjects")
        assignments.append({
            "subject_accession": sub.subject_accession,
            "study_accession": sub.study_accession,
            "assigned_group": group,
            "sex": sub.sex,
            "age_years": sub.age_years,
            "ancestry": sub.ancestry,
            "is_hispanic": int(sub.is_hispanic),
        })
    return assignments


def draw_patient_groups(n_patients: int, rng: np.random.Generator) -> list[str | None]:
    """Draw each patient's cohort membership.

    75% of patients are matched to one of the five real autoimmune cohorts
    (uniformly); the rest are background patients drawn from the
    non-autoimmune ImmPort pools. AITD/Vitiligo have no real cohort, so they
    only ever appear in the background arm (handled by the label engine).
    """
    real = [g for g in COHORT_STUDIES if g is not None]
    choices = rng.choice(real + [None], size=n_patients, p=[0.15] * len(real) + [0.25])
    return [None if c is None else str(c) for c in choices]

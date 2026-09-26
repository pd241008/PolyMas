"""Probe OpenGWAS for the ADR-005 coupling decision (read-only API probing).

Checks, for the 7 modeled diseases, whether curated GWAS datasets exist that
carry per-variant beta + effect-allele metadata for our 91-variant panel.
Everything is cached to <POLYMAS_RESULTS_DIR>/adr005_probe/ so the coupling
decision is auditable (R2: provenance).

Steps:
  1. GET /gwasinfo (cached) -> filter datasets by trait keywords per disease.
  2. Pick the best dataset per disease (largest n_case, has beta).
  3. POST /associations {variant: [panel], id: dataset} -> per-locus rows.
  4. Emit a decision table: panel variants with usable beta per disease.

Usage:
    POLYMAS_RESULTS_DIR=stash/results_real_20260925 \
        services/ml-engine-python/.venv/bin/python scripts/ogwas_probe.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(os.environ.get("POLYMAS_RESULTS_DIR", ROOT / "stash/results_real_20260925"))
OUT_DIR = RESULTS_ROOT / "adr005_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://api.opengwas.io/api"


def _jwt() -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENGWAS_JWT="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("OPENGWAS_JWT", "")


SESSION = requests.Session()
SESSION.headers.update({"Authorization": f"Bearer {_jwt()}", "Content-Type": "application/json"})


def api_get(path: str, cache_key: str, params: dict | None = None):
    cache = OUT_DIR / f"{cache_key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = SESSION.get(f"{BASE}{path}", params=params or {}, timeout=120)
    r.raise_for_status()
    data = r.json()
    cache.write_text(json.dumps(data))
    time.sleep(1.0)
    return data


def api_post(path: str, payload: dict, cache_key: str):
    cache = OUT_DIR / f"{cache_key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = SESSION.post(f"{BASE}{path}", json=payload, timeout=180)
    if r.status_code == 429:
        wait = int(r.headers.get("Retry-After", "30"))
        print(f"  429 throttled; sleeping {wait}s", flush=True)
        time.sleep(wait)
        r = SESSION.post(f"{BASE}{path}", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    cache.write_text(json.dumps(data))
    time.sleep(1.0)
    return data


# Panel rsIDs (91 typed variants) come from the substrate dosage matrix.
def panel_rsids() -> list[str]:
    import pandas as pd

    dos = pd.read_csv(RESULTS_ROOT / "real_genotypes_20260925" / "genotype_dosages.csv",
                      index_col=0, nrows=2)
    return [c for c in dos.columns if c.startswith("rs")]


# Trait keywords per modeled disease (cohort diseases + modeled ones).
DISEASE_TRAITS: dict[str, list[str]] = {
    "RA": ["rheumatoid arthritis"],
    "SLE": ["lupus", "systemic lupus"],
    "SJOGRENS": ["sjogren", "sicca"],
    "AITD": ["graves", "hashimoto", "thyroid", "autoimmune thyroid"],
    "T1D": ["type 1 diabetes", "type 1 diabete"],
    "VITILIGO": ["vitiligo"],
    "MS": ["multiple sclerosis"],
}


LOG_ODDS_UNITS = {"log odds", "logOR"}


def pick_datasets(gwasinfo: dict) -> dict[str, list[dict]]:
    """Rank candidate datasets per disease: per-variant log-OR units required
    (that is what the /associations endpoint serves as beta + effect_allele)."""
    out: dict[str, list[dict]] = {}
    for disease, keywords in DISEASE_TRAITS.items():
        hits = []
        for ds_id, meta in gwasinfo.items():
            trait = str(meta.get("trait", "")).lower()
            if not any(k in trait for k in keywords):
                continue
            if str(meta.get("unit")) not in LOG_ODDS_UNITS:
                continue  # no per-variant log-OR -> unusable for coupling
            ncase = meta.get("ncase") or 0
            hits.append({
                "id": ds_id,
                "trait": meta.get("trait"),
                "ncase": ncase,
                "ncontrol": meta.get("ncontrol"),
                "population": meta.get("population"),
                "year": meta.get("year"),
                "pmid": meta.get("pmid"),
                "unit": meta.get("unit"),
            })
        # Largest case count first; prefer European ancestry (matches the
        #ImmPort-derived cohort, which is majority EUR).
        hits.sort(key=lambda h: (-(h["ncase"] or 0), h["population"] != "European"))
        out[disease] = hits[:3]
    return out


def do_picks() -> int:
    gwasinfo = json.loads((OUT_DIR / "gwasinfo.json").read_text())
    print(f"{len(gwasinfo)} datasets known")
    picks = pick_datasets(gwasinfo)
    (OUT_DIR / "dataset_picks.json").write_text(json.dumps(picks, indent=2))
    for disease, cands in picks.items():
        line = ", ".join(f"{c['id']}(ncase={c['ncase']},pop={c['population']})" for c in cands)
        print(f"{disease:10s} {line if cands else '— NO usable dataset'}")
    return 0


def assoc_batched(rs_ids: list[str], ds: str, chunk: int = 25,
                  retries: int = 3) -> list[dict]:
    """POST /associations in chunks with backoff+retry (gateway is flaky:
    transient 502s and occasional 400s on large single batches)."""
    rows: list[dict] = []
    for i in range(0, len(rs_ids), chunk):
        part = rs_ids[i:i + chunk]
        for attempt in range(retries):
            try:
                data = api_post("/associations", {"variant": part, "id": ds},
                                f"assoc_{ds}_part{i // chunk:02d}")
                rows.extend(data if isinstance(data, list) else data.get("associations", []))
                break
            except requests.RequestException as e:
                if attempt == retries - 1:
                    print(f"    chunk {i // chunk}: giving up after {retries} tries: {e}")
                else:
                    time.sleep(5 * (attempt + 1))
    return rows


def do_assoc(disease: str) -> int:
    picks = json.loads((OUT_DIR / "dataset_picks.json").read_text())
    cands = picks.get(disease) or []
    if not cands:
        print(f"{disease}: no candidate dataset")
        return 1
    ds = cands[0]["id"]
    rs_ids = panel_rsids()
    rows = assoc_batched(rs_ids, ds)
    usable = [
        r for r in rows
        if r.get("beta") is not None and r.get("p") is not None
        and float(r.get("p")) > 0 and r.get("ea")
    ]
    strong = [r for r in usable if float(r.get("p")) < 5e-8]
    print(f"{disease} {ds}: {len(usable)}/{len(rows)} rows with beta+ea, "
          f"{len(strong)} genome-wide significant")
    (OUT_DIR / f"assoc_rows_{ds}.json").write_text(json.dumps(rows))
    return 0


def do_summary() -> int:
    picks = json.loads((OUT_DIR / "dataset_picks.json").read_text())
    coverage: dict[str, dict] = {}
    for disease, cands in picks.items():
        if not cands:
            coverage[disease] = {"dataset": None, "n_usable_beta": 0}
            continue
        ds = cands[0]["id"]
        f = OUT_DIR / f"assoc_rows_{ds}.json"
        if not f.exists():
            coverage[disease] = {"dataset": ds, "n_usable_beta": None, "note": "not fetched"}
            continue
        rows = json.loads(f.read_text())
        rows = rows if isinstance(rows, list) else rows.get("associations", [])
        usable = [r for r in rows if r.get("beta") is not None and r.get("p") is not None
                  and float(r.get("p")) > 0 and r.get("ea")]
        strong = [r for r in usable if float(r.get("p")) < 5e-8]
        coverage[disease] = {
            "dataset": ds,
            "n_queried": len(rows),
            "n_usable_beta": len(usable),
            "n_genome_wide": len(strong),
            "rsids_usable": sorted(r["rsid"] for r in usable),
        }
    (OUT_DIR / "coverage.json").write_text(json.dumps(coverage, indent=2))
    n_ok = sum(1 for v in coverage.values() if (v.get("n_usable_beta") or 0) >= 5)
    for d, v in coverage.items():
        print(f"{d:10s} {v.get('dataset')} usable={v.get('n_usable_beta')} gws={v.get('n_genome_wide')}")
    print(f"\nDECISION INPUT: {n_ok}/7 diseases have >=5 panel variants with published beta.")
    print("  >=5/7  -> coupling feasible (anchor-disease betas carry the genotype-label link)")
    print("  <5/7   -> fallback: no-coupling baseline stands (ADR-004)")
    return 0


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "picks"
    if cmd == "picks":
        raise SystemExit(do_picks())
    if cmd == "assoc":
        raise SystemExit(do_assoc(sys.argv[2]))
    if cmd == "summary":
        raise SystemExit(do_summary())
    print(__doc__)
    raise SystemExit(2)

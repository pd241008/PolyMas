"""Per-patient sequence construction for System B.

Reads System A's stored per-patient PRS z-scores, derives a deterministic
genotype (0/1/2 alt alleles) from the same latent risk, and injects it into the
Ensembl reference windows at the variant position. Both systems therefore
describe the same underlying synthetic patients.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# K-mer vocabulary (k=6) -> 4^6 = 4096 tokens. N falls back to 0 (A).
KMER_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3}
TOKEN_HOM_REF = 4096
TOKEN_HET = 4097
TOKEN_HOM_ALT = 4098
VOCAB_SIZE = 4099
K = 6

WINDOW = 5000  # must match ensembl.WINDOW
VARIANT_OFFSET = WINDOW  # variant sits at the center of the 10001-bp window

DISEASE_LABELS = ["RA", "SLE", "SJOGRENS", "AITD", "T1D", "VITILIGO", "MS"]


def genotype_from_z(z: float) -> int:
    """Map a System-A z-score to alt-allele count (0/1/2), deterministically.

    High-risk patients (larger z) get more alternate alleles, so System B's
    sequence signal is consistent with System A's PRS latent risk.
    """
    return int(np.clip(np.floor(z + 1.5), 0, 2))


def kmer_to_id(kmer: str) -> int:
    val = 0
    for base in kmer:
        val = val * 4 + KMER_VOCAB.get(base, 0)
    return val


def tokenize_kmers(seq: str, k: int = 6) -> np.ndarray:
    seq = seq.upper()
    kmers = [kmer_to_id(seq[i:i+k]) for i in range(0, len(seq), k) if i+k <= len(seq)]
    return np.array(kmers, dtype=np.int16)


def load_system_a(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prs = pd.read_csv(results_dir / "features" / "prs_features.csv")
    labels = pd.read_csv(results_dir / "features" / "labels.csv")
    return prs, labels


def build_patient_tokens(
    patient_prs: pd.DataFrame,
    windows: dict[str, str],
    variant_info: dict[str, Any],
) -> np.ndarray:
    """Concatenate per-locus windows with genotype tokens injected at the variant."""
    pieces: list[np.ndarray] = []
    window_lens = {len(windows[rs_id]) for rs_id in variant_info}
    if len(window_lens) != 1:
        raise ValueError(f"windows have inconsistent lengths: {window_lens}")
    center = window_lens.pop() // 2  # variant sits at window center
    for rs_id in variant_info:
        seq = windows[rs_id]
        locus_prs = patient_prs[patient_prs["locus_id"] == rs_id]
        if locus_prs.empty:
            raise ValueError(f"no PRS record for patient/locus {rs_id}")
        z = float(locus_prs["z_score"].iloc[0])
        g = genotype_from_z(z)
        gen_token = TOKEN_HOM_REF if g == 0 else TOKEN_HET if g == 1 else TOKEN_HOM_ALT

        left_tokens = tokenize_kmers(seq[:center], K)
        right_tokens = tokenize_kmers(seq[center+1:], K)
        tokens = np.concatenate([left_tokens, np.array([gen_token], dtype=np.int16), right_tokens])
        pieces.append(tokens)
    return np.concatenate(pieces)


def build_dataset(
    results_dir: Path,
    output_dir: Path,
    windows: dict[str, str],
    variant_info: dict[str, Any],
    diseases: list[str] | None = None,
) -> dict[str, Any]:
    """Build the per-patient token matrix + label matrix, mirroring System A."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prs, labels = load_system_a(results_dir)
    disease_cols = diseases or DISEASE_LABELS

    ordered_patients = labels["patient_id"].tolist()
    prs_by_patient = {pid: df for pid, df in prs.groupby("patient_id")}

    token_rows: list[np.ndarray] = []
    for pid in ordered_patients:
        if pid not in prs_by_patient:
            raise ValueError(f"patient {pid} missing from PRS features")
        token_rows.append(build_patient_tokens(prs_by_patient[pid], windows, variant_info))

    tokens = np.stack(token_rows)  # (n_patients, n_tokens)
    y = labels[disease_cols].to_numpy(dtype=np.float32)
    patient_ids = np.array(ordered_patients)

    np.save(output_dir / "tokens.npy", tokens)
    pd.DataFrame({"patient_id": patient_ids, **{d: y[:, i] for i, d in enumerate(disease_cols)}}).to_csv(
        output_dir / "labels.csv", index=False
    )
    manifest = {
        "n_patients": len(patient_ids),
        "n_tokens": int(tokens.shape[1]),
        "n_windows": len(variant_info),
        "window_size": 2 * WINDOW + 1,
        "vocab_size": VOCAB_SIZE,
        "kmer_size": K,
        "diseases": disease_cols,
        "genotype_mapping": {"0": "hom_ref", "1": "het", "2": "hom_alt"},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    per_patient_genotypes = {
        pid: {
            rs_id: int(genotype_from_z(float(row["z_score"])))
            for rs_id, row in df.set_index("locus_id").iterrows()
        }
        for pid, df in prs.groupby("patient_id")
    }
    pd.DataFrame.from_dict(per_patient_genotypes, orient="index").to_csv(
        output_dir / "genotypes.csv", index_label="patient_id"
    )

    logger.info("Sequence dataset built: %d patients x %d tokens", tokens.shape[0], tokens.shape[1])
    return {"tokens": tokens, "y": y, "patient_ids": patient_ids, "manifest": manifest}

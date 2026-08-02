"""System B runner: Ensembl extraction -> per-patient sequences -> smoke test.

Modes:
  ref    Fetch + cache Ensembl variant coords and +-5kb reference windows.
  seq    Build per-patient token sequences (full + smoke windows).
  train  Train the pure-PyTorch Mamba smoke test (1-2 diseases).
  sanity Run the synthetic-signal sanity check (proves Mamba learns genotype
         signal when it exists).

Run with: PYTHONPATH=. ./.venv/bin/python scripts/run_system_b.py [mode]
(note: requires a torch/CUDA python; the system python3 with torch works too)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results"
ENSEMBL_DIR = RESULTS_DIR / "raw" / "ensembl"
SEQUENCE_DIR = RESULTS_DIR / "sequence"


def cmd_ref() -> None:
    from polymas_ml.sequence.ensembl import fetch_and_cache

    data = fetch_and_cache(ENSEMBL_DIR)
    for rs_id, v in data["variants"].items():
        logger.info("  %s %s:%d window_len=%d", rs_id, v["seq_region"], v["start"], len(data["windows"][rs_id]))


def cmd_seq() -> None:
    import json

    import numpy as np

    from polymas_ml.sequence import dataset as ds
    from polymas_ml.sequence.ensembl import fetch_and_cache

    ensembl = fetch_and_cache(ENSEMBL_DIR)
    variants = ensembl["variants"]
    windows = ensembl["windows"]

    # Full spec: +-5kb windows -> ~80kb per patient.
    full = ds.build_dataset(RESULTS_DIR, SEQUENCE_DIR / "full", windows, variants)
    logger.info("Full dataset: %d patients x %d tokens", *full["tokens"].shape)

    # Smoke test: centered +-1kb sub-windows (~16kb per patient) so the
    # pure-PyTorch fallback fits in 6GB VRAM. Variant stays at window center.
    smoke_win = 1000
    smoke_windows = {
        rs: windows[rs][5000 - smoke_win : 5000 + smoke_win + 1] for rs in windows
    }
    smoke = ds.build_dataset(RESULTS_DIR, SEQUENCE_DIR / "smoke", smoke_windows, variants)
    logger.info("Smoke dataset: %d patients x %d tokens", *smoke["tokens"].shape)

    import pandas as pd

    gdf = pd.read_csv(SEQUENCE_DIR / "full" / "genotypes.csv", index_col=0)
    counts = (gdf.stack().value_counts()).sort_index()
    logger.info("Genotype distribution (alt-allele count): %s", counts.to_dict())

    # Milestone-2 check: do patient sequences differ meaningfully?
    tokens = full["tokens"]
    n_shared = int((tokens[0] == tokens[1]).mean() * 100)
    n_unique_seq = len({t.tobytes() for t in tokens})
    n_variant_pos = tokens.shape[1] - n_shared
    logger.info(
        "Sequence divergence: %d/%d tokens shared between P0000/P0001 (%d%%); %d distinct patient sequences of %d",
        n_shared,
        tokens.shape[1],
        100 - int((tokens[0] != tokens[1]).mean() * 100),
        n_unique_seq,
        tokens.shape[0],
    )
    manifest = full["manifest"]
    manifest["genotype_counts"] = {str(k): int(v) for k, v in counts.items()}
    (SEQUENCE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Sequence datasets written to %s", SEQUENCE_DIR)


def cmd_train() -> None:
    import numpy as np
    import pandas as pd

    from polymas_ml.sequence import train as tr

    smoke_dir = SEQUENCE_DIR / "smoke"
    tokens = np.load(smoke_dir / "tokens.npy")
    labels = pd.read_csv(smoke_dir / "labels.csv")
    diseases = ["RA", "SLE"]
    y = labels[diseases].to_numpy(dtype=np.float32)
    logger.info("Smoke-train on %s: %d patients x %d tokens", diseases, *tokens.shape)
    report = tr.train_smoke(
        tokens, y, diseases, SEQUENCE_DIR / "smoke_test",
        n_epochs=20, batch_size=2, lr=1e-3,
        d_model=64, n_layers=2, d_state=8, d_conv=4, expand=2,
    )
    print(json_dumps_report(report))


def cmd_sanity() -> None:
    import numpy as np
    import pandas as pd

    from polymas_ml.sequence import train as tr

    smoke_dir = SEQUENCE_DIR / "smoke"
    tokens = np.load(smoke_dir / "tokens.npy")
    labels = pd.read_csv(smoke_dir / "labels.csv")
    patient_ids = labels["patient_id"].tolist()

    rng = np.random.default_rng(7)
    n = len(tokens)
    # Synthetic labels driven ONLY by genotype at 2 positions: RA = hom-alt at
    # locus1 OR het/hom-alt at locus2 (chance floor ~0.5 AUROC if unlearned).
    locus1 = 5000  # variant position inside window 0
    locus2 = 5000 + 10001 + 1  # variant position inside window 1
    g1 = tokens[:, locus1]
    g2 = tokens[:, locus2]
    risk = (g1 >= 7) | (g2 >= 6)
    y_ra = risk.astype(np.float32)
    # SLE = pure noise
    y_sle = rng.random(n) < 0.2
    y = np.stack([y_ra, y_sle.astype(np.float32)], axis=1)
    diseases = ["RA", "SLE"]
    logger.info("Synthetic-signal check: RA driven by genotypes (label rate %.2f), SLE = noise", y_ra.mean())
    report = tr.train_smoke(
        tokens, y, diseases, SEQUENCE_DIR / "sanity_check",
        n_epochs=12, batch_size=2, lr=1e-3,
        d_model=64, n_layers=2, d_state=8, d_conv=4, expand=2,
    )
    print(json_dumps_report(report))


def json_dumps_report(report: dict) -> str:
    import json

    return json.dumps(report, indent=2)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    modes = {"ref": cmd_ref, "seq": cmd_seq, "train": cmd_train, "sanity": cmd_sanity}
    if mode == "all":
        cmd_ref()
        cmd_seq()
        cmd_train()
    elif mode in modes:
        modes[mode]()
    else:
        logger.error("Unknown mode %s; expected %s", mode, list(modes) + ["all"])
        sys.exit(1)


if __name__ == "__main__":
    main()

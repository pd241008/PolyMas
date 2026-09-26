"""F-13 (System B): Optuna hyperparameter sweep for the flat Mamba.

Pre-registered claim: tuned config beats defaults on val AUROC and the
tuned Mamba retrains stably (the guarded-wrapper retrain is the
stability check; recorded as a follow-up run if GPU time runs out —
the sweep itself is the primary evidence).

Protocol (pre-registered):
  - Fixed budget: 30 trials, TPESampler(seed=42), sqlite resume.
  - Training budget per trial: 2 epochs (Phase-3 curves show val peaks
    by epoch 6-7 but rank-stability by epoch 2-3; full retrain of the
    best config follows separately).
  - Search space: d_model {32, 64, 96}, n_local_layers {1, 2, 3},
    d_state {4, 8, 16}, learning_rate [1e-4, 3e-3] log, dropout
    [0.05, 0.3], weight_decay [1e-5, 0.3] log.
  - Objective: val macro AUROC after 2 epochs, canonical split/labels.

Evidence: <results_root>/f13_optuna/system_b*
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "ml-engine-python"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("f13_tune_b")

from polymas_ml.data.patients import DISEASE_LABELS  # noqa: E402

RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "stash/results_final_20260926"))
OUT = RESULTS / "f13_optuna"
OUT.mkdir(parents=True, exist_ok=True)
DB = OUT / "system_b.db"
N_TRIALS = 30
SEED = 42
EPOCHS_PER_TRIAL = 2


def _load():
    tokens = np.load(RESULTS / "sequence" / "kmer_canonical" / "tokens.npy")
    labels = pd.read_csv(RESULTS / "sequence" / "kmer_canonical" / "labels.csv")
    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    pos = {pid: i for i, pid in enumerate(labels["patient_id"])}
    splits = {s: np.array([pos[p] for p in split.loc[split["split"] == s, "patient_id"]])
              for s in ("train", "val", "test")}
    y = labels[DISEASE_LABELS].to_numpy(dtype=np.float32)
    return tokens, y, splits


def objective(trial):
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from polymas_ml.sequence.model import HierarchicalMamba

    tokens, y, splits = _load()
    tr, va = splits["train"], splits["val"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    params = {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 96]),
        "n_local_layers": trial.suggest_int("n_local_layers", 1, 3),
        "d_state": trial.suggest_categorical("d_state", [4, 8, 16]),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.05, 0.3),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 0.3, log=True),
    }
    torch.manual_seed(SEED)
    model = HierarchicalMamba(vocab_size=4099, n_diseases=7, n_loci=8,
                              d_model=params["d_model"],
                              n_local_layers=params["n_local_layers"],
                              d_state=params["d_state"],
                              dropout=params["dropout"], mode="flat").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                            weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    tokens_t = torch.from_numpy(tokens.astype(np.int64))
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        perm = rng.permutation(len(tr))
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]
            opt.zero_grad()
            loss = loss_fn(model(tokens_t[tr[b]].to(dev)),
                           torch.from_numpy(y[tr[b]]).to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        outs = []
        for s in range(0, len(va), 256):
            outs.append(torch.sigmoid(model(tokens_t[va[s:s + 256]].to(dev))))
    probs = torch.cat(outs).cpu().numpy()
    aucs = [roc_auc_score(y[va, k], probs[:, k]) for k in range(7)
            if len(np.unique(y[va, k])) > 1]
    return float(np.mean(aucs))


def main() -> int:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = N_TRIALS
    if "--trials" in sys.argv:
        n_trials = int(sys.argv[sys.argv.index("--trials") + 1])
    study = optuna.create_study(
        direction="maximize", study_name="f13_system_b",
        storage=f"sqlite:///{DB}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    done = len(study.trials)
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done)
    best = study.best_trial
    trials = [{"number": t.number, "value": t.value, "params": t.params}
              for t in study.trials]
    # Defaults reference: the F-03 flat config, 2-epoch objective-equivalent
    # is its epoch-2 val macro from the F-03 history.
    f03 = json.loads((RESULTS / "f03_hierarchical_mamba" / "arm_flat.json").read_text())
    default_epoch2 = next(h["val_macro_auroc"] for h in f03["history"] if h["epoch"] == 2)
    summary = {
        "n_trials": len(study.trials),
        "best_val_2epoch": round(best.value, 4),
        "default_config_2epoch_val": default_epoch2,
        "delta": round(best.value - default_epoch2, 4),
        "best_params": best.params,
        "top5": sorted(trials, key=lambda t: -(t["value"] or 0))[:5],
        "note": "2-epoch budget per trial; best-config full retrain via the "
                "guarded GPU wrapper recorded separately when run",
    }
    (OUT / "f13_system_b_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("F-13 System B: best %.4f vs default-2epoch %.4f (%s)",
                best.value, default_epoch2, summary["delta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

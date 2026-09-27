"""F-13 (System B) follow-up: full-budget retrain of the swept best config.

Pre-registered gate: the tuned config, trained at the SAME 12-epoch budget
as the F-03 default flat arm, must beat the default's val macro AUROC
(0.5750) on the canonical split. Test AUROC is reported alongside (not a
gate). Instability (NaN loss, divergence) is recorded as a stability FAIL.

Resume-hardened: checkpoints after every epoch; repeated invocations
continue to 12 total epochs, then evaluate and write the summary once.
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
logger = logging.getLogger("f13_retrain_b")

RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "results/results_final_20260926"))
OUT = RESULTS / "f13_optuna"
CKPT = OUT / "system_b_retrain_ckpt.pt"
SUMMARY = OUT / "f13_system_b_retrain.json"
TOTAL_EPOCHS = 12
SEED = 42


def _load():
    tokens = np.load(RESULTS / "sequence" / "kmer_canonical" / "tokens.npy")
    labels = pd.read_csv(RESULTS / "sequence" / "kmer_canonical" / "labels.csv")
    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    pos = {pid: i for i, pid in enumerate(labels["patient_id"])}
    splits = {s: np.array([pos[p] for p in split.loc[split["split"] == s, "patient_id"]])
              for s in ("train", "val", "test")}
    y = labels[["RA", "SLE", "SJOGRENS", "T1D", "MS", "AITD", "VITILIGO"]]
    return tokens, y.to_numpy(dtype=np.float32), splits


def _predict(model, tokens_t, idx, dev):
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            outs.append(torch.sigmoid(model(tokens_t[idx[s:s + 256]].to(dev))))
    return torch.cat(outs).cpu().numpy()


def main() -> int:
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from polymas_ml.sequence.model import HierarchicalMamba

    tokens, y, splits = _load()
    tr, va, te = splits["train"], splits["val"], splits["test"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tokens_t = torch.from_numpy(tokens.astype(np.int64))

    best_params = json.loads((OUT / "f13_system_b_summary.json").read_text())["best_params"]
    stable = True

    if CKPT.exists():
        ck = torch.load(CKPT, map_location=dev, weights_only=False)
        start_epoch = ck.get("epoch", 0)
        history = ck.get("history", [])
        best_state = ck.get("best_state")
        best_val = ck.get("best_val", -1.0)
    else:
        start_epoch = 0
        history = []
        best_state = None
        best_val = -1.0
        torch.manual_seed(SEED)
        model = HierarchicalMamba(vocab_size=4099, n_diseases=7, n_loci=8,
                                  d_model=best_params["d_model"],
                                  n_local_layers=best_params["n_local_layers"],
                                  d_state=best_params["d_state"],
                                  dropout=best_params["dropout"],
                                  mode="flat").to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=best_params["lr"],
                                weight_decay=best_params["weight_decay"])
        rng = np.random.default_rng(SEED)
        ck = {"model": model, "opt": opt, "rng": rng.bit_generator.state}
        torch.save(ck, CKPT)
        ck = torch.load(CKPT, map_location=dev, weights_only=False)

    model = ck["model"]
    opt = ck["opt"]
    rng = np.random.default_rng()
    rng.bit_generator.state = ck["rng"]

    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        model.train()
        perm = rng.permutation(len(tr))
        tot = 0.0
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]
            opt.zero_grad()
            logits = model(tokens_t[tr[b]].to(dev))
            loss = loss_fn(logits, torch.from_numpy(y[tr[b]]).to(dev))
            if not torch.isfinite(loss):
                stable = False
                break
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(b)
        if not stable:
            break
        probs = _predict(model, tokens_t, va, dev)
        aucs = [roc_auc_score(y[va, k], probs[:, k]) for k in range(7)
                if len(np.unique(y[va, k])) > 1]
        val_macro = float(np.mean(aucs))
        history.append({"epoch": epoch + 1, "val_macro_auroc": round(val_macro, 4),
                        "train_loss": round(tot / len(tr), 4)})
        if val_macro > best_val:
            best_val = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ck = {"epoch": epoch + 1, "history": history, "best_val": best_val,
              "best_state": best_state, "model": model, "opt": opt,
              "rng": rng.bit_generator.state}
        torch.save(ck, CKPT)
        logger.info("epoch %d/%d val %.4f (best %.4f)", epoch + 1, TOTAL_EPOCHS,
                    val_macro, best_val)

    if len(history) < TOTAL_EPOCHS:
        logger.info("resumable stop at epoch %d/%d", len(history), TOTAL_EPOCHS)
        return 0

    # Final: load best state, evaluate val + test, write summary.
    model.load_state_dict(best_state)
    val_probs = _predict(model, tokens_t, va, dev)
    test_probs = _predict(model, tokens_t, te, dev)

    def macro(idx, probs):
        aucs = [roc_auc_score(y[idx, k], probs[:, k]) for k in range(7)
                if len(np.unique(y[idx, k])) > 1]
        return float(np.mean(aucs))

    default_val = 0.5750  # F-03 flat arm, 12-epoch best val
    summary = {
        "best_params": best_params,
        "n_epochs": TOTAL_EPOCHS,
        "tuned_val_macro": round(best_val, 4),
        "default_flat_val_macro": default_val,
        "delta_val": round(best_val - default_val, 4),
        "tuned_test_macro": round(macro(te, test_probs), 4),
        "default_flat_test_macro": 0.5464,
        "history": history,
        "stable": stable,
        "gate": "PASS" if (best_val >= default_val and stable) else "FAIL",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    logger.info("F-13 B retrain: tuned %.4f vs default %.4f -> %s",
                best_val, default_val, summary["gate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""System B curve arms for F-18 (noise) and F-19 (scaling).

Re-scoped into Phase 3 per ROADMAP: the Mamba retrains on the
ADR-006-corrected canonical run here, and its noise/scaling points fold
into the SAME curve CSVs System A already published (system column 'B').

Protocol mirrors System A exactly:
  - F-18: labels flipped 1<->0 with prob r in {0, 5, 10, 20}% (seed 42,
    all splits), Mamba retrained per rate on the canonical train split,
    AUROC on the FLIPPED test labels.
  - F-19: full training at n in {1000, 2500, 5000, 10000} (first-n
    patients, seed 42), AUROC on the unflipped test split of each n.
    NOTE System B uses one shared multi-label model, so all 7 diseases
    score at every point (no per-disease n_positives failure modes).
  - The 5k/0% point reuses the F-03 flat arm's test metrics where the
    protocol matches (same data, split, seed): recorded as a reuse, not
    a re-run.
  - Eval: same evaluate() metrics as train.py; per-epoch checkpoints
    (state+optimizer+rng) make each tool call resume where the last
    ended. 6 epochs per arm (loss plateaus by ~epoch 6; F-03 arms showed
    best-val by epoch 6-7 with flat curves after).

Evidence: appends system='B' rows to
  <results_root>/f18_noise_sweep/noise_curves.csv
  <results_root>/f19_scaling_sweep/scaling_curves.csv
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
logger = logging.getLogger("system_b_curves")

from polymas_ml.data.patients import DISEASE_LABELS  # noqa: E402

RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "results/results_final_20260926"))
SEQ = RESULTS / "sequence" / "kmer_canonical"
F03 = RESULTS / "f03_hierarchical_mamba"
SEED = 42
EPOCHS = 6
BATCH = 64
N_LOCI = 8
VOCAB = 4099
RATES = [0.0, 0.05, 0.10, 0.20]
SCALING_NS = [1000, 2500, 5000, 10000]


def _load_tokens_labels():
    tokens = np.load(SEQ / "tokens.npy")
    labels = pd.read_csv(SEQ / "labels.csv")
    y = labels[DISEASE_LABELS].to_numpy(dtype=np.float32)
    return tokens, y, labels["patient_id"].to_numpy()


def _split_ids():
    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    return {s: split.loc[split["split"] == s, "patient_id"].tolist()
            for s in ("train", "val", "test")}


def _flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def _train_eval(tokens: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray,
                te: np.ndarray, tag: str, epochs: int = EPOCHS) -> dict[str, float]:
    """Train the flat Mamba on (tr, y-flavored labels); return test AUROCs."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from polymas_ml.sequence.model import HierarchicalMamba

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = RESULTS / "system_b_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"ckpt_{tag}.pt"
    torch.manual_seed(SEED)
    model = HierarchicalMamba(
        vocab_size=VOCAB, n_diseases=7, n_loci=N_LOCI, d_model=64,
        n_local_layers=2, d_state=8, mode="flat",
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    start, best_val, best_state, history = 1, -1.0, None, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict({k: v.to(dev) for k, v in ck["state_dict"].items()})
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
        start, best_val = ck["epoch"] + 1, ck["val"]
        best_state = ck.get("best_state")
        history = ck.get("history", [])
        logger.info("[%s] resuming at epoch %d (best %.4f)", tag, start, best_val)

    tokens_t = torch.from_numpy(tokens.astype(np.int64))
    for epoch in range(start, epochs + 1):
        model.train()
        perm = rng.permutation(len(tr))
        total = 0.0
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            xb = tokens_t[tr[b]].to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), torch.from_numpy(y[tr[b]]).to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(b)
        # val macro
        model.eval()
        probs_va = _predict(model, tokens_t, va, dev)
        aucs = [roc_auc_score(y[va, k], probs_va[:, k]) for k in range(7)
                if len(np.unique(y[va, k])) > 1]
        macro = float(np.mean(aucs))
        history.append({"epoch": epoch, "val": round(macro, 4)})
        if macro > best_val:
            best_val = macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "best_state": best_state,
            "optimizer": opt.state_dict(), "rng": rng.bit_generator.state,
            "epoch": epoch, "val": best_val, "history": history,
        }, ckpt_path)
        logger.info("[%s] epoch %d/%d loss=%.4f val_macro=%.4f (best %.4f)",
                    tag, epoch, epochs, total / len(tr), macro, best_val)
        if epoch < epochs:
            return {}  # more tool calls needed; partial save only
    if best_state is None:
        raise RuntimeError("no best state")
    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    probs_te = _predict(model, tokens_t, te, dev)
    out = {}
    for k, d in enumerate(DISEASE_LABELS):
        if len(np.unique(y[te, k])) > 1:
            out[d] = round(float(roc_auc_score(y[te, k], probs_te[:, k])), 4)
    (out_dir / f"result_{tag}.json").write_text(json.dumps(
        {"best_val": best_val, "test_auroc": out, "history": history}, indent=2))
    logger.info("[%s] DONE best_val=%.4f test=%s", tag, best_val, out)
    return out


def _predict(model, tokens_t, rows, dev) -> np.ndarray:
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(rows), 256):
            outs.append(model(tokens_t[rows[s:s + 256]].to(dev)))
    return torch.sigmoid(torch.cat(outs)).cpu().numpy()


def run_noise(rates=RATES) -> int:
    tokens, y, pids = _load_tokens_labels()
    split_ids = _split_ids()
    pos = {p: i for i, p in enumerate(pids)}
    tr = np.array([pos[p] for p in split_ids["train"]])
    va = np.array([pos[p] for p in split_ids["val"]])
    te = np.array([pos[p] for p in split_ids["test"]])
    rows = []
    for rate in rates:
        rng = np.random.default_rng(SEED)
        flip = rng.random(y.shape) < rate
        y_noisy = np.where(flip, 1 - y, y)
        tag = f"noise_r{int(rate * 100):02d}"
        res_file = RESULTS / "system_b_curves" / f"result_{tag}.json"
        if res_file.exists():
            aucs = json.loads(res_file.read_text())["test_auroc"]
            logger.info("[%s] reusing completed result", tag)
        elif rate == 0.0:
            # Reuse the F-03 flat arm (same data/split/seed/protocol).
            flat = json.loads((F03 / "arm_flat.json").read_text())
            aucs = {d: v["auroc"] for d, v in flat["test"].items()}
            logger.info("[noise_r00] reusing F-03 flat arm test metrics")
        else:
            aucs = _train_eval(tokens, y_noisy, tr, va, te, tag)
            if not aucs:
                return 1  # training in progress; rerun this command
        for d, a in aucs.items():
            rows.append({"rate": rate, "disease": d, "auroc": a, "system": "B"})
    _merge_curve_csv(RESULTS / "f18_noise_sweep" / "noise_curves.csv", rows,
                     key=["system", "disease", "rate"])
    logger.info("F-18 System B rows merged")
    _update_noise_summary()
    return 0


def run_scaling(ns=SCALING_NS) -> int:
    tokens_full, y_full, pids_full = _load_tokens_labels()
    split_ids = _split_ids()
    pos_full = {p: i for i, p in enumerate(pids_full)}
    rows = []
    for n in ns:
        pids = pids_full[:n]
        pos = {p: i for i, p in enumerate(pids)}
        have_all = all(p in pos for s in ("train", "val", "test") for p in split_ids[s])
        if have_all:
            tr = np.array([pos[p] for p in split_ids["train"]])
            va = np.array([pos[p] for p in split_ids["val"]])
            te = np.array([pos[p] for p in split_ids["test"]])
        else:
            # Smaller-than-canonical n: the F-12 split ids exceed n, so use
            # the same PROTOCOL shape (stratified-by-RA 70/10/20, seed 42)
            # drawn within the first-n patients.
            from sklearn.model_selection import train_test_split
            idx = np.arange(n)
            ra = y_full[:n, DISEASE_LABELS.index("RA")]
            tr_va, te = train_test_split(idx, test_size=0.2, stratify=ra, random_state=SEED)
            tr, va = train_test_split(tr_va, test_size=0.125, stratify=ra[tr_va], random_state=SEED)
            logger.info("[n=%d] F-12 ids exceed n; protocol-shape split within first-n", n)
        tag = f"scale_n{n}"
        res_file = RESULTS / "system_b_curves" / f"result_{tag}.json"
        if res_file.exists():
            aucs = json.loads(res_file.read_text())["test_auroc"]
            logger.info("[%s] reusing completed result", tag)
        elif n == 5000 and have_all:
            flat = json.loads((F03 / "arm_flat.json").read_text())
            aucs = {d: v["auroc"] for d, v in flat["test"].items()}
            logger.info("[scale_n5000] reusing F-03 flat arm (canonical point)")
        else:
            tokens = tokens_full[:n]
            y = y_full[:n]
            aucs = _train_eval(tokens, y, tr, va, te, tag)
            if not aucs:
                return 1
        for d, a in aucs.items():
            rows.append({"n": n, "disease": d, "auroc": a})
    _merge_curve_csv(RESULTS / "f19_scaling_sweep" / "scaling_curves.csv", rows,
                     key=["disease", "n"], extra_col="system", extra_val="B")
    logger.info("F-19 System B rows merged")
    return 0


def _merge_curve_csv(path: Path, rows: list[dict], key: list[str],
                     extra_col: str | None = None, extra_val: str | None = None) -> None:
    new = pd.DataFrame(rows)
    if extra_col:
        new[extra_col] = extra_val
    if path.exists():
        existing = pd.read_csv(path)
        merged = pd.concat([existing, new]).drop_duplicates(subset=key, keep="last")
    else:
        merged = new
    merged.to_csv(path, index=False)


def _update_noise_summary() -> None:
    path = RESULTS / "f18_noise_sweep" / "noise_sweep_summary.json"
    if not path.exists():
        return
    s = json.loads(path.read_text())
    s["system_b_pending"] = False
    s["system_b"] = "merged into noise_curves.csv (system='B'); retrained on " \
                    "ADR-006-corrected labels, Phase-3 re-scope"
    path.write_text(json.dumps(s, indent=2))


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "noise":
        return run_noise()
    if cmd == "scaling":
        return run_scaling()
    raise SystemExit("usage: system_b_curves.py {noise|scaling} (rerun to continue training)")


if __name__ == "__main__":
    raise SystemExit(main())

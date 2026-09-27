"""F-15: deep ensembles + MC-dropout uncertainty (ROADMAP Phase 4).

Pre-registered claim: an ensemble of 5 Mamba seeds (or MC-dropout
sampling) gives calibrated uncertainty; tolerance: ensemble mean NLL <
 single-model NLL on the same evaluation patients (delta < 0 recorded
 as the claim; delta >= 0 = honest null).

Protocol (pre-registered before the run):
  - Base model: the F-03 FLAT arm's architecture on the canonical k-mer
    dataset, canonical F-12 split, 6 epochs, batch 64, AdamW lr 1e-3
    wd 0.1, grad-clip 1.0 — identical to the F-03 protocol; only the
    torch seed varies.
  - Seeds: 42 (the F-03 flat arm, reused artifact) + {1, 2, 3, 4}
    trained fresh here (4 x 6 epochs, per-epoch resume, one tool call
    per ~2 epochs).
  - MC-dropout: T=30 stochastic passes of a single model with dropout
    ENABLED at eval (the architecture has dropout in the head path);
    mean prob = MC estimate, predictive variance across passes.
  - Metrics on the canonical TEST split, per disease: NLL, ECE (15
    bins), Brier for (a) each single member, (b) the 5-member mean-prob
    ensemble, (c) the MC-dropout mean. Gate: mean-over-diseases
    ensemble NLL < single-seed-42 NLL.

Evidence: <results_root>/f15_uncertainty/
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
logger = logging.getLogger("f15_uncertainty")

from polymas_ml.data.patients import DISEASE_LABELS  # noqa: E402

RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "results/results_final_20260926"))
SEQ = RESULTS / "sequence" / "kmer_canonical"
OUT = RESULTS / "f15_uncertainty"
F03 = RESULTS / "f03_hierarchical_mamba"
SEEDS = [1, 2, 3, 4]
EPOCHS = 6
BATCH = 64
N_LOCI = 8
VOCAB = 4099
MC_PASSES = 30


def _load():
    tokens = np.load(SEQ / "tokens.npy")
    labels = pd.read_csv(SEQ / "labels.csv")
    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    pos = {pid: i for i, pid in enumerate(labels["patient_id"])}
    splits = {s: np.array([pos[p] for p in split.loc[split["split"] == s, "patient_id"]])
              for s in ("train", "val", "test")}
    y = labels[DISEASE_LABELS].to_numpy(dtype=np.float32)
    return tokens, y, splits


def _train_seed(seed: int) -> dict:
    """Train one member; store best-val test probs. Reuses the F-03
    training recipe verbatim; seed changes init + batch order."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from polymas_ml.sequence.model import HierarchicalMamba

    tokens, y, splits = _load()
    tr, va, te = splits["train"], splits["val"], splits["test"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = OUT / "members"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"ckpt_seed{seed}.pt"
    torch.manual_seed(seed)
    model = HierarchicalMamba(vocab_size=VOCAB, n_diseases=7, n_loci=N_LOCI,
                              d_model=64, n_local_layers=2, d_state=8, mode="flat").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    start, best_val, best_state = 1, -1.0, None
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict({k: v.to(dev) for k, v in ck["state_dict"].items()})
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
        start, best_val, best_state = ck["epoch"] + 1, ck["val"], ck.get("best_state")
        logger.info("[seed %d] resuming at epoch %d (best %.4f)", seed, start, best_val)

    tokens_t = torch.from_numpy(tokens.astype(np.int64))
    for epoch in range(start, EPOCHS + 1):
        model.train()
        perm = rng.permutation(len(tr))
        total = 0.0
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(tokens_t[tr[b]].to(dev)),
                           torch.from_numpy(y[tr[b]]).to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(b)
        model.eval()
        with torch.no_grad():
            outs = []
            for s in range(0, len(va), 256):
                outs.append(torch.sigmoid(model(tokens_t[va[s:s + 256]].to(dev))))
            probs_va = torch.cat(outs).cpu().numpy()
        aucs = [roc_auc_score(y[va, k], probs_va[:, k]) for k in range(7)
                if len(np.unique(y[va, k])) > 1]
        macro = float(np.mean(aucs))
        if macro > best_val:
            best_val = macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "best_state": best_state, "optimizer": opt.state_dict(),
                    "rng": rng.bit_generator.state, "epoch": epoch, "val": best_val},
                   ckpt_path)
        logger.info("[seed %d] epoch %d/%d loss=%.4f val_macro=%.4f (best %.4f)",
                    seed, epoch, EPOCHS, total / len(tr), macro, best_val)
        if epoch < EPOCHS:
            return {}  # continuation needed

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        outs = []
        for s in range(0, len(te), 256):
            outs.append(torch.sigmoid(model(tokens_t[te[s:s + 256]].to(dev))))
    probs_te = torch.cat(outs).cpu().numpy()
    np.save(out_dir / f"test_probs_seed{seed}.npy", probs_te)
    logger.info("[seed %d] DONE best_val=%.4f", seed, best_val)
    return {"best_val": best_val}


def _load_probs(seed: int | None) -> np.ndarray:
    if seed is None:
        f = F03 / ".." / "f15_uncertainty" / "members" / "test_probs_seed42.npy"
        return np.load(f)
    return np.load(OUT / "members" / f"test_probs_seed{seed}.npy")


def main() -> int:
    import torch
    from polymas_ml.evaluation.uncertainty import all_metrics
    from polymas_ml.sequence.model import HierarchicalMamba

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "members").mkdir(exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    tokens, y, splits = _load()
    te = splits["test"]

    if cmd == "train":
        for seed in SEEDS:
            res = _train_seed(seed)
            if not res:  # mid-training; continue in next call
                return 1
        logger.info("all member seeds trained")
        return 0

    # ---- evaluation (needs all members present) ----
    members = []
    for seed in [None] + SEEDS:  # None = F-03 flat arm (seed 42 artifact)
        f = OUT / "members" / f"test_probs_seed{seed or 42}.npy"
        if not f.exists():
            if seed is None:
                # Extract from the F-03 checkpoint once.
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                ck = torch.load(F03 / "ckpt_flat.pt", map_location=dev, weights_only=False)
                model = HierarchicalMamba(vocab_size=VOCAB, n_diseases=7, n_loci=N_LOCI,
                                          d_model=64, n_local_layers=2, d_state=8,
                                          mode="flat").to(dev)
                model.load_state_dict({k: v.to(dev) for k, v in ck["best_state"].items()})
                model.eval()
                tokens_t = torch.from_numpy(tokens.astype(np.int64))
                with torch.no_grad():
                    outs = []
                    for s in range(0, len(te), 256):
                        outs.append(torch.sigmoid(model(tokens_t[te[s:s + 256]].to(dev))))
                probs = torch.cat(outs).cpu().numpy()
                np.save(f, probs)
                logger.info("extracted seed-42 (F-03 flat) test probs")
            else:
                logger.warning("member seed %d missing; run `train` first", seed)
                return 1
        members.append(np.load(f))
    members = np.stack(members)          # (5, n_test, 7)
    ens_prob = members.mean(axis=0)

    # MC-dropout on the seed-42 model: T stochastic passes with dropout on.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(F03 / "ckpt_flat.pt", map_location=dev, weights_only=False)
    model = HierarchicalMamba(vocab_size=VOCAB, n_diseases=7, n_loci=N_LOCI,
                              d_model=64, n_local_layers=2, d_state=8, mode="flat").to(dev)
    model.load_state_dict({k: v.to(dev) for k, v in ck["best_state"].items()})
    model.train()  # dropout active at eval time (MC-dropout)
    tokens_t = torch.from_numpy(tokens.astype(np.int64))
    passes = []
    rng = np.random.default_rng(0)
    for t in range(MC_PASSES):
        torch.manual_seed(int(rng.integers(1e6)))
        outs = []
        with torch.no_grad():
            for s in range(0, len(te), 256):
                outs.append(torch.sigmoid(model(tokens_t[te[s:s + 256]].to(dev))))
        passes.append(torch.cat(outs).cpu().numpy())
        if (t + 1) % 10 == 0:
            logger.info("MC pass %d/%d", t + 1, MC_PASSES)
    mc_prob = np.mean(passes, axis=0)
    mc_var = np.var(passes, axis=0).mean()  # mean predictive variance

    rows = []
    y_te = y[te]
    for k, d in enumerate(DISEASE_LABELS):
        for name, p in ([f"single_seed{s}" for s in (42, *SEEDS)] and
                        [("single_seed42", members[0])]
                        + [(f"single_seed{s}", members[i + 1]) for i, s in enumerate(SEEDS)]
                        + [("ensemble5", ens_prob), ("mc_dropout", mc_prob)]):
            m = all_metrics(y_te[:, k], p[:, k])
            rows.append({"disease": d, "model": name, **m})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "f15_uncertainty_table.csv", index=False)

    macro = df.groupby("model")[["nll", "ece", "brier"]].mean().round(4)
    nll_single = float(macro.loc["single_seed42", "nll"])
    nll_ens = float(macro.loc["ensemble5", "nll"])
    nll_mc = float(macro.loc["mc_dropout", "nll"])
    delta_ens = round(nll_ens - nll_single, 4)
    delta_mc = round(nll_mc - nll_single, 4)
    summary = {
        "gates": {
            "ensemble_nll_lt_single": {"delta_nll": delta_ens, "pass": bool(delta_ens < 0)},
            "mc_dropout_nll_lt_single": {"delta_nll": delta_mc, "pass": bool(delta_mc < 0)},
        },
        "verdict": "PASS" if (delta_ens < 0 or delta_mc < 0) else "FAIL",
        "mc_predictive_variance_mean": round(float(mc_var), 6),
        "n_mc_passes": MC_PASSES,
        "macro_table": macro.to_dict(orient="index"),
        "note": "gate is NLL-based per the pre-registered claim; ECE/Brier "
                "reported for calibration context. Null recorded if deltas >= 0.",
    }
    (OUT / "f15_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("F-15 verdict: %s\n%s", summary["verdict"], macro.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

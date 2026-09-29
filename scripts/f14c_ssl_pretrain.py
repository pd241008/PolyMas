"""F-14c: explicit locus-dropout SSL pretraining (pre-registered cell).

Completes the four-cell SSL ablation on the v1 (8-locus) substrate:
  F-14  mask-all, BUGGY semantics        -> -0.0189 FAIL (as executed)
  F-14a context-only, correct semantics  -> +0.0094 FAIL (by 0.0006)
  F-14b mask-all, correct semantics      -> +0.0213 PASS (the flip)
  F-14c THIS RUN: exactly ONE genotype token per sequence masked and
        scored (mask_locus_dropout) — every training gradient is
        cross-locus genotype inference.

Pre-registration (docs/ROADMAP.md ledger, committed a4b35d8 BEFORE any run):
  - Gates (F-14 family): mean paired val AUROC delta >= +0.01 AND no seed
    regresses > 0.01; seeds 1/2/3; identical 6-epoch FT schedule.
  - Pre-registered competing-mechanism expectation (stated before the run):
    F-14c scores 1 PURE genotype token/sequence (every gradient is signal)
    but ~60x fewer scored tokens than F-14b (~74 context + ~1.2 genotype).
    Purity argues F-14c > F-14a (+0.0094); quantity argues F-14c may
    undertrain vs F-14b (+0.0213) at 8 epochs. Registered prediction:
    F-14c > F-14a. The F-14b comparison is the secondary read; either
    outcome is mechanistically informative. If F-14c < F-14a, more pretrain
    epochs is the follow-up — a new claim.

Evidence: <results_root>/f14c_ssl/
"""
from __future__ import annotations

import os

# MUST be set before torch initializes CUDA (allocator fragmentation,
# root-caused 2026-09-27).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "ml-engine-python"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("f14c_ssl")

RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "results/results_final_20260926"))
OUT = RESULTS / "f14c_ssl"
OUT.mkdir(parents=True, exist_ok=True)
DISEASES = ["RA", "SLE", "SJOGRENS", "T1D", "MS", "AITD", "VITILIGO"]
FT_EPOCHS = 6
PRETRAIN_EPOCHS = 8
FT_SEEDS = (1, 2, 3)


def _load():
    tokens = np.load(RESULTS / "sequence" / "kmer_canonical" / "tokens.npy")
    labels = pd.read_csv(RESULTS / "sequence" / "kmer_canonical" / "labels.csv")
    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    pos = {pid: i for i, pid in enumerate(labels["patient_id"])}
    splits = {s: np.array([pos[p] for p in split.loc[split["split"] == s, "patient_id"]])
              for s in ("train", "val", "test")}
    y = labels[DISEASES].to_numpy(dtype=np.float32)
    return tokens, y, splits


def _macro(y, idx, probs):
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(y[idx, k], probs[:, k]) for k in range(7)
            if len(np.unique(y[idx, k])) > 1]
    return float(np.mean(aucs))


def _predict(model, tokens_dev, idx):
    import torch
    model.eval()
    # Batch 64: proven stable on this 6GB laptop GPU (see F-14 ledger).
    torch.cuda.empty_cache()
    outs = []
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            outs.append(torch.sigmoid(model(tokens_dev[idx[s:s + 64]])))
    return torch.cat(outs).cpu().numpy()


def run_pretrain(tokens, splits, n_epochs: int | None = None) -> Path:
    import time as _time

    import torch
    import torch.nn as nn
    from polymas_ml.sequence.model import MambaSequenceClassifier
    from polymas_ml.sequence.ssl import MambaMLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("pretrain device: %s (cuda_available=%s)", dev, torch.cuda.is_available())
    ckpt = OUT / "pretrain_ckpt.pt"
    tr = splits["train"]
    # Upload ONCE, index on device (allocator-churn fix, 2026-09-27).
    tokens_dev = torch.from_numpy(tokens[tr].astype(np.int64)).to(dev)

    torch.manual_seed(42)
    clf = MambaSequenceClassifier(vocab_size=4100, n_diseases=7, d_model=64,
                                  n_layers=2).to(dev)
    mlm = MambaMLM(clf).to(dev)
    opt = torch.optim.AdamW(mlm.parameters(), lr=1e-3)
    rng = np.random.default_rng(42)
    start, hist = 0, []
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        mlm.load_state_dict(ck["mlm"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
        start, hist = ck["epoch"], ck["history"]

    end = PRETRAIN_EPOCHS if n_epochs is None else min(start + n_epochs, PRETRAIN_EPOCHS)
    for epoch in range(start, end):
        mlm.train()
        perm = rng.permutation(len(tokens_dev))
        tot, nb, n_scored = 0.0, 0, 0
        _t0 = _time.time()
        for s in range(0, len(perm), 64):
            b = torch.from_numpy(perm[s:s + 64])
            opt.zero_grad()
            # F-14c's single design change: locus-dropout masking — exactly
            # one genotype token scored per sequence (pure-signal objective).
            loss = mlm.pretrain_loss(tokens_dev[b], mask_mode="locus_dropout")
            loss.backward()
            nn.utils.clip_grad_norm_(mlm.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
            if nb == 1 or nb % 20 == 0:
                logger.info("  step %d  %.1fs  mem %.1fGB  loss %.3f",
                            nb, _time.time() - _t0,
                            torch.cuda.memory_allocated() / 1e9, tot / nb)
        hist.append({"epoch": epoch + 1, "mlm_loss": round(tot / nb, 4)})
        torch.save({"epoch": epoch + 1, "history": hist, "mlm": mlm.state_dict(),
                    "opt": opt.state_dict(), "rng": rng.bit_generator.state}, ckpt)
        logger.info("pretrain %d/%d mlm_loss %.4f", epoch + 1, PRETRAIN_EPOCHS, tot / nb)

    (OUT / "pretrain_history.json").write_text(json.dumps(hist, indent=2))
    return ckpt


def run_ft(tokens, y, splits, arm: str, seed: int) -> dict:
    import torch
    import torch.nn as nn
    from polymas_ml.sequence.model import MambaSequenceClassifier

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = OUT / f"ft_{arm}_seed{seed}_ckpt.pt"
    tr, va, te = splits["train"], splits["val"], splits["test"]
    tokens_dev = torch.from_numpy(tokens.astype(np.int64)).to(dev)

    torch.manual_seed(seed)
    if arm == "pretrained":
        # Encoder from the seed-42 pretrain; head stays at FT-seed init in
        # both arms; mask embedding row + pretrain head row excluded.
        clf = MambaSequenceClassifier(vocab_size=4099, n_diseases=7, d_model=64,
                                      n_layers=2)
        pre = torch.load(OUT / "pretrain_ckpt.pt", map_location="cpu",
                         weights_only=False)
        enc = {k[len("classifier."):]: v for k, v in pre["mlm"].items()
               if k.startswith("classifier.") and not k.startswith("classifier.head.")}
        enc["embedding.weight"] = enc["embedding.weight"][:4099]  # drop MASK row
        missing, unexpected = clf.load_state_dict(enc, strict=False)
        assert missing and all(m.startswith("head.") for m in missing), missing
        assert not unexpected, unexpected
    else:
        clf = MambaSequenceClassifier(vocab_size=4099, n_diseases=7, d_model=64,
                                      n_layers=2)
    model = clf.to(dev)
    # GPU warm-up (cudaErrorNotReady race; see F-14 ledger).
    _warm = torch.randn(256, 256, device=dev)
    for _ in range(3):
        _warm = _warm @ _warm
    torch.cuda.synchronize()
    del _warm
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    start, hist, best_val, best_state = 0, [], -1.0, None
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
        start, hist = ck["epoch"], ck["history"]
        best_val, best_state = ck["best_val"], ck["best_state"]

    ytr = torch.from_numpy(y[tr])
    for epoch in range(start, FT_EPOCHS):
        model.train()
        perm = rng.permutation(len(tr))
        tot = 0.0
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]
            opt.zero_grad()
            loss = loss_fn(model(tokens_dev[tr[b]]), ytr[b].to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
        val_macro = _macro(y, va, _predict(model, tokens_dev, va))
        hist.append({"epoch": epoch + 1, "val_macro_auroc": round(val_macro, 4),
                     "train_loss": round(tot / len(tr), 4)})
        if val_macro > best_val:
            best_val = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"epoch": epoch + 1, "history": hist, "model": model.state_dict(),
                    "opt": opt.state_dict(), "rng": rng.bit_generator.state,
                    "best_val": best_val, "best_state": best_state}, ckpt)
        logger.info("ft %s seed%d %d/%d val %.4f", arm, seed, epoch + 1, FT_EPOCHS,
                    val_macro)

    model.load_state_dict(best_state)
    return {"arm": arm, "seed": seed, "val_macro": round(best_val, 4),
            "test_macro": round(_macro(y, te, _predict(model, tokens_dev, te)), 4),
            "history": hist}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "ft", "finalize"], required=True)
    ap.add_argument("--arm", choices=["pretrained", "scratch"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    tokens, y, splits = _load()

    if args.stage == "pretrain":
        run_pretrain(tokens, splits, n_epochs=args.epochs)
        return 0
    if args.stage == "ft":
        res = run_ft(tokens, y, splits, args.arm, args.seed)
        (OUT / f"ft_{args.arm}_seed{args.seed}.json").write_text(json.dumps(res, indent=2))
        return 0

    # finalize: collect, pair, gate (F-14 gates, 3 seeds) + pre-registered
    # secondary read: F-14c vs F-14a (registered prediction: F-14c > F-14a)
    # and vs F-14b (purity-vs-quantity mechanism read).
    rows = []
    for arm in ("pretrained", "scratch"):
        for seed in FT_SEEDS:
            p = OUT / f"ft_{arm}_seed{seed}.json"
            if p.exists():
                d = json.loads(p.read_text())
                rows.append({"arm": arm, "seed": seed,
                             "val": d["val_macro"], "test": d["test_macro"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "f14c_per_seed.csv", index=False)
    if len(df) < 2 * len(FT_SEEDS):
        logger.info("finalize: %d/%d runs complete; skipping gates",
                    len(df), 2 * len(FT_SEEDS))
        return 0
    piv = df.pivot(index="seed", columns="arm", values="val")
    deltas = piv["pretrained"] - piv["scratch"]
    mean_delta = float(deltas.mean())
    worst = float(deltas.min())
    f14a = json.loads((RESULTS / "f14a_ssl" / "f14a_summary.json").read_text())
    f14b = json.loads((RESULTS / "f14b_ssl" / "f14b_summary.json").read_text())
    summary = {
        "variant": "F-14c explicit locus-dropout (exactly one genotype token scored per sequence)",
        "per_seed": rows,
        "mean_val_delta_pre_minus_scratch": round(mean_delta, 4),
        "worst_seed_delta": round(worst, 4),
        "gates": {"G1_mean_delta_ge_0.01": mean_delta >= 0.01,
                  "G2_no_seed_regression_gt_0.01": worst > -0.01},
        "verdict": "PASS" if (mean_delta >= 0.01 and worst > -0.01) else "FAIL",
        "expectation_check": {
            "pre_registered": "F-14c > F-14a (purity); F-14b comparison = secondary mechanism read",
            "f14a_mean_delta": f14a["mean_val_delta_pre_minus_scratch"],
            "f14b_mean_delta": f14b["mean_val_delta_pre_minus_scratch"],
            "beats_f14a_registered_prediction": bool(mean_delta > f14a["mean_val_delta_pre_minus_scratch"]),
            "ordering_vs_family": None,
        },
        "pretrain_history": json.loads((OUT / "pretrain_history.json").read_text()),
    }
    order = {"F-14": -0.0189, "F-14a": f14a["mean_val_delta_pre_minus_scratch"],
             "F-14b": f14b["mean_val_delta_pre_minus_scratch"], "F-14c": mean_delta}
    summary["expectation_check"]["ordering_vs_family"] = sorted(order, key=order.get, reverse=True)
    (OUT / "f14c_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("F-14c verdict: %s (mean delta %+.4f, worst seed %+.4f, ordering %s)",
                summary["verdict"], mean_delta, worst,
                summary["expectation_check"]["ordering_vs_family"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

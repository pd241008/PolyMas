"""Training + evaluation for System B, mirroring System A's setup."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from .model import MambaSequenceClassifier

logger = logging.getLogger(__name__)

DISEASE_LABELS = ["RA", "SLE", "SJOGRENS", "AITD", "T1D", "VITILIGO", "MS"]


def build_splits(
    patient_ids: np.ndarray,
    y: np.ndarray,
    primary_disease: str = "RA",
    diseases: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Replicate System A's held-out split.

    System A uses a stratified 80/20 train/calibration split per disease
    (random_state=42). A single shared split is needed for the shared
    multi-label Mamba; we use the primary disease's split so the held-out
    patients match System A's calibration split for that disease.
    """
    cols = diseases or DISEASE_LABELS
    idx = np.arange(len(patient_ids))
    y_primary = y[:, cols.index(primary_disease)]
    train_idx, val_idx = train_test_split(
        idx, test_size=test_size, stratify=y_primary, random_state=random_state
    )
    return {"train": train_idx, "val": val_idx}


def evaluate(
    model: nn.Module,
    tokens: torch.Tensor,
    y: torch.Tensor,
    diseases: list[str],
    device: torch.device,
    batch_size: int = 2,
) -> dict[str, float]:
    """Return per-disease AUROC/AUPRC/F1 + aggregate Hamming/macro/micro F1."""
    model.eval()
    logits_list: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            xb = tokens[start : start + batch_size]
            logits_list.append(model(xb))
    logits = torch.cat(logits_list, dim=0)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_np = y.cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    metrics: dict[str, float] = {}
    for i, d in enumerate(diseases):
        if len(np.unique(y_np[:, i])) < 2:
            auroc = float("nan")
        else:
            auroc = roc_auc_score(y_np[:, i], probs[:, i])
        auprc = average_precision_score(y_np[:, i], probs[:, i])
        f1 = f1_score(y_np[:, i], preds[:, i], zero_division=0)
        metrics[f"{d}_auroc"] = float(auroc)
        metrics[f"{d}_auprc"] = float(auprc)
        metrics[f"{d}_f1"] = float(f1)

    metrics["hamming_loss"] = float(hamming_loss(y_np, preds))
    metrics["macro_f1"] = float(
        f1_score(y_np, preds, average="macro", zero_division=0)
    )
    metrics["micro_f1"] = float(
        f1_score(y_np, preds, average="micro", zero_division=0)
    )
    metrics["accuracy"] = float(accuracy_score(y_np, preds))
    return metrics


def train_smoke(
    tokens: np.ndarray,
    y: np.ndarray,
    diseases: list[str],
    output_dir: Path,
    n_epochs: int = 20,
    batch_size: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    d_model: int = 64,
    n_layers: int = 2,
    d_state: int = 8,
    d_conv: int = 4,
    expand: int = 2,
    grad_clip: float = 1.0,
    seed: int = 0,
    device_str: str | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir) if output_dir is not None else Path("/tmp") / f"seq_smoke_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    split = build_splits(np.arange(len(tokens)), y, diseases=diseases)
    train_idx, val_idx = split["train"], split["val"]

    model = MambaSequenceClassifier(
        vocab_size=int(tokens.max()) + 1,
        n_diseases=len(diseases),
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    n_steps = (len(train_idx) + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * n_steps, eta_min=lr * 0.05
    )

    tokens_t = torch.from_numpy(tokens).long().to(device)
    y_t = torch.from_numpy(y).float().to(device)

    y_train = y_t[train_idx]
    n_pos = y_train.sum(dim=0)
    n_neg = y_train.shape[0] - n_pos
    pos_weight = (n_neg / n_pos.clamp(min=1)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ckpt_path = output_dir / "checkpoint.pt"
    start_epoch = 0
    history = {"train_loss": [], "val_metrics": []}
    best_val_auroc = -1.0
    best_state: dict[str, Any] = {}
    t0 = time.time()
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"]
        history = ck["history"]
        best_val_auroc = ck["best_val_auroc"]
        best_state = ck["best_state"]
        logger.info("Resumed from checkpoint at epoch %d", start_epoch)

    for epoch in range(start_epoch, n_epochs):
        model.train()
        perm = torch.randperm(len(train_idx))
        epoch_loss = 0.0
        step_t0 = time.time()
        for start in range(0, len(perm), batch_size):
            batch_ids = perm[start : start + batch_size]
            xb = tokens_t[train_idx[batch_ids]]
            yb = y_t[train_idx[batch_ids]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * len(batch_ids)
            if start // batch_size % 20 == 0:
                logger.info(
                    "  step %3d/%d loss=%.4f %.2fs",
                    start // batch_size,
                    n_steps,
                    loss.item(),
                    time.time() - step_t0,
                )
                step_t0 = time.time()

        epoch_loss /= len(train_idx)
        train_metrics_epoch = evaluate(model, tokens_t[train_idx], y_t[train_idx], diseases, device)
        val_metrics = evaluate(model, tokens_t[val_idx], y_t[val_idx], diseases, device)
        history["train_loss"].append(epoch_loss)
        history["val_metrics"].append(val_metrics)
        mean_val_auroc = np.mean(
            [val_metrics[f"{d}_auroc"] for d in diseases if f"{d}_auroc" in val_metrics]
        )
        logger.info(
            "epoch %2d loss=%.4f train_auroc=%s val_auroc=%s (%.3f)",
            epoch + 1,
            epoch_loss,
            {d: round(train_metrics_epoch.get(f"{d}_auroc", float("nan")), 3) for d in diseases},
            {d: round(val_metrics.get(f"{d}_auroc", float("nan")), 3) for d in diseases},
            mean_val_auroc,
        )
        if mean_val_auroc > best_val_auroc:
            best_val_auroc = mean_val_auroc
            best_state = {
                "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch + 1,
                "val_metrics": val_metrics,
            }

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "best_val_auroc": best_val_auroc,
                "best_state": best_state,
            },
            ckpt_path,
        )

    elapsed = time.time() - t0
    torch.save(best_state["state_dict"], output_dir / "best_model.pt")

    model.load_state_dict(best_state["state_dict"])
    model.to(device)
    train_metrics = evaluate(model, tokens_t[train_idx], y_t[train_idx], diseases, device)
    val_metrics = best_state["val_metrics"]

    report = {
        "diseases": diseases,
        "n_patients": len(tokens),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_epochs": n_epochs,
        "elapsed_seconds": elapsed,
        "config": {
            "d_model": d_model,
            "n_layers": n_layers,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_epoch": best_state["epoch"],
    }
    (output_dir / "smoke_test_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Smoke test done in %.0fs. Best val AUROC %.3f (epoch %d)",
                elapsed, best_val_auroc, best_state["epoch"])
    return report

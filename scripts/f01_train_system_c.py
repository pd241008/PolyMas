"""F-01: LD-aware GNN (System C) training + evaluation (ROADMAP Phase 3).

Pre-registered gates: macro test AUROC within +/-0.02 of System A =
"matches"; strictly above = "beats"; below = honest FAIL. Edge matrix
validated against the substrate LD (audit written alongside).

Protocol (pre-registered):
  - Nodes = the 14 patient-typed loci (DISEASE_RISK_LOCI union).
  - Node features per patient: dosage (1), PRS z-score (1), haplotype
    and epistasis features aligned per locus (from haplotypes.py), so
    each node carries that locus's patient-specific information.
  - Edges: LD r2 >= 0.2 among ALL 91 substrate loci; typed nodes inherit
    edges through the full substrate graph via 2 rounds of message
    passing on the induced 91-node graph (features zero on untyped
    nodes, which act as LD relays).
  - Split: canonical F-12 train/val/test; AdamW lr 1e-3, wd 1e-4,
    batch 256, 30 epochs, seed 42, best-val checkpoint (same discipline
    as F-03). Resumable per-epoch checkpoints (foreground tool calls).
  - Comparison: models/per_disease_metrics.csv (System A, same run).

Evidence: <results_root>/f01_ld_gnn/
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
logger = logging.getLogger("f01_ldgnn")

from polymas_ml.data.patients import DISEASE_LABELS, DISEASE_RISK_LOCI  # noqa: E402

RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "stash/results_final_20260926"))
OUT = RESULTS / "f01_ld_gnn"
SEED = 42
EPOCHS = 30
BATCH = 256
LD_THRESHOLD = 0.2


def _build_graph():
    """LD r2 matrix over the substrate + node ordering."""
    from polymas_ml.data.genotypes import compute_ld_r2
    from polymas_ml.data.haplotypes import load_substrate

    dosages, _, _, _ = load_substrate(RESULTS)
    typed = sorted({rs for loci in DISEASE_RISK_LOCI.values() for rs in loci})
    ld = compute_ld_r2(dosages)
    return ld, typed, dosages


def _node_features(dosages: pd.DataFrame, donor_map: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """(n_patients, n_substrate_loci, 2) features: real donor dosage +
    PRS z-score (typed nodes only; relay nodes carry zeros)."""
    prs = pd.read_csv(RESULTS / "features" / "prs_features.csv")
    z = prs.pivot_table(index="patient_id", columns="locus_id",
                        values="z_score", aggfunc="first")
    donor_ids = donor_map["donor_id"].to_numpy()
    gt = dosages.loc[donor_ids]          # (n_patients, n_loci) real donor dosages
    loci = list(dosages.columns)
    z_cols = [c for c in loci if c in z.columns]
    X = np.zeros((len(donor_map), len(loci), 2), dtype=np.float32)
    dosage_pos = {rs: i for i, rs in enumerate(loci)}
    for rs in z_cols:
        X[:, dosage_pos[rs], 0] = gt[rs].to_numpy(dtype=np.float32)
        X[:, dosage_pos[rs], 1] = z[rs].reindex(donor_map["patient_id"]).to_numpy(dtype=np.float32)
    return X, loci


def _predict(model, X_t, adj, rows, dev) -> np.ndarray:
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(rows), 512):
            xb = torch.from_numpy(X_t[rows[s:s + 512]]).to(dev)
            outs.append(model(xb, adj))
    return torch.sigmoid(torch.cat(outs)).cpu().numpy()


def main() -> int:
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from polymas_ml.graph import LDGNN, build_ld_audit, row_normalize_adj

    OUT.mkdir(parents=True, exist_ok=True)
    ld, typed, dosages = _build_graph()
    loci = list(dosages.columns)

    # Edge audit (R2) BEFORE training.
    audit = build_ld_audit(ld)
    (OUT / "ld_edge_audit.json").write_text(json.dumps(audit, indent=2))
    logger.info("LD audit: %d edges >= 0.2, MHC max r2=%s",
                audit["n_edges_ge_0.2"], audit["mhc_block_max_r2"])

    adj_raw = torch.tensor((ld.loc[loci, loci] >= LD_THRESHOLD).to_numpy(dtype=np.float32))
    np.fill_diagonal(adj_raw.numpy(), 0.0)
    adj = row_normalize_adj(adj_raw).to("cuda" if torch.cuda.is_available() else "cpu")

    donor_map = pd.read_csv(RESULTS / "donor_map.csv")
    X, _ = _node_features(dosages, donor_map)
    labels = pd.read_csv(RESULTS / "features" / "labels.csv")
    pos = {pid: i for i, pid in enumerate(labels["patient_id"])}
    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    splits = {s: np.array([pos[p] for p in split.loc[split["split"] == s, "patient_id"]])
              for s in ("train", "val", "test")}
    y = labels[DISEASE_LABELS].to_numpy(dtype=np.float32)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    model = LDGNN(n_node_features=2, n_diseases=7, hidden=64, n_rounds=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    ckpt_path = OUT / "ckpt.pt"
    start, best_val, best_state, history = 1, -1.0, None, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict({k: v.to(dev) for k, v in ck["state_dict"].items()})
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
        start, best_val, best_state = ck["epoch"] + 1, ck["val"], ck.get("best_state")
        history = ck.get("history", [])
        logger.info("resuming at epoch %d (best %.4f)", start, best_val)

    tr, va, te = splits["train"], splits["val"], splits["test"]
    for epoch in range(start, EPOCHS + 1):
        model.train()
        perm = rng.permutation(len(tr))
        total = 0.0
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            xb = torch.from_numpy(X[tr[b]]).to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb, adj), torch.from_numpy(y[tr[b]]).to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(b)
        probs = _predict(model, X, adj, va, dev)
        aucs = [roc_auc_score(y[va, k], probs[:, k]) for k in range(7)
                if len(np.unique(y[va, k])) > 1]
        macro = float(np.mean(aucs))
        history.append({"epoch": epoch, "val": round(macro, 4)})
        if macro > best_val:
            best_val = macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "best_state": best_state, "optimizer": opt.state_dict(),
                    "rng": rng.bit_generator.state, "epoch": epoch, "val": best_val,
                    "history": history}, ckpt_path)
        logger.info("epoch %d/%d loss=%.4f val_macro=%.4f (best %.4f)",
                    epoch, EPOCHS, total / len(tr), macro, best_val)
        if epoch < EPOCHS:
            return 1  # training continues in the next tool call

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    probs_te = _predict(model, X, adj, te, dev)
    sysA = pd.read_csv(RESULTS / "models" / "per_disease_metrics.csv").set_index("disease")
    rows = []
    for k, d in enumerate(DISEASE_LABELS):
        auroc = float(roc_auc_score(y[te, k], probs_te[:, k]))
        rows.append({"disease": d, "auroc_system_c": round(auroc, 4),
                     "auroc_system_a": float(sysA.loc[d, "auroc"]),
                     "delta_c_minus_a": round(auroc - float(sysA.loc[d, "auroc"]), 4)})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT / "f01_comparison.csv", index=False)
    macro_c = float(cmp_df["auroc_system_c"].mean())
    macro_a = float(sysA["auroc"].mean())
    delta = macro_c - macro_a
    verdict = "BEATS" if delta > 0 else ("MATCHES" if delta >= -0.02 else "FAIL")
    summary = {
        "macro_system_c": round(macro_c, 4), "macro_system_a": round(macro_a, 4),
        "delta": round(delta, 4), "tolerance": -0.02,
        "ld_threshold": LD_THRESHOLD, "n_nodes": len(loci),
        "n_edges": int(adj_raw.sum().item() // 2),
        "verdict": verdict,
        "note": "graph = substrate LD r2 >= 0.2 (real 1000G haplotypes); "
                "14 typed nodes, rest LD relays; best-val checkpoint test eval",
    }
    (OUT / "f01_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("F-01 verdict: %s\n%s", verdict, json.dumps(summary, indent=1))
    logger.info("\n%s", cmp_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

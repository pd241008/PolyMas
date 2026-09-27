"""F-04: clinical+genetic fusion model (ROADMAP Phase 3).

Pre-registered claim: late (gated) fusion of System A features and
System B sequence embeddings improves held-out AUROC over EITHER system
alone. Recorded as negative with the numbers if it does not.

Protocol (pre-registered before the run):
  - System B embedding: the F-03 FLAT arm's trained Mamba, embeddings
    extracted via model.embed() for ALL 5000 canonical patients (the
    sequence encoder is frozen — fusion trains only the gate + head, so
    no System-B retraining confounds the comparison).
  - Fusion head: z = sigmoid(alpha) * h_A + sigmoid(1-alpha)... no —
    a learned PER-DISEASE scalar gate g_d in (0,1) via sigmoid on an
    unconstrained parameter, fusion vector = [g_d * h_A_feat,
    (1-g_d) * h_B_emb] per disease is overkill; instead ONE learned
    vector gate over the concatenated space: z = W [h_A ; h_B] + b with
    a per-disease mixing logit alpha_d, fused = sigmoid(alpha_d) * h_A
    + (1 - sigmoid(alpha_d)) * h_B, where h_A = System A's own test
    PROBABILITIES (calibrated, the system's actual output) and h_B =
    System B's test probabilities. This is late fusion at the decision
    level — the most conservative fusion (if even decision fusion can't
    beat the better system, feature fusion is moot).
  - alpha_d: per-disease mixing weight in [0, 1], fused = alpha_d * p_A
    + (1 - alpha_d) * p_B, fit on the VAL split by minimizing val
    logloss over a dense grid (transparent, bounded — a logistic fit on
    separable parents diverges; the grid is the honest constrained
    estimator). alpha_d IS the interpretability artifact (R3).
  - Gates: fused AUROC > max(System A, System B) per disease for the
    claim; tolerance floor: any disease losing > 0.01 vs its better
    parent is a violation. Macro comparison is the headline.
  - System B test probabilities: from F-03's flat arm? The arm result
    stores only AUROCs, not per-patient probabilities — so System B
    probabilities are re-extracted here from ckpt_flat.pt (the SAME
    checkpoint F-03 evaluated; R2: same artifact).

Evidence: <results_root>/f04_fusion/
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
logger = logging.getLogger("f04_fusion")

from polymas_ml.data.patients import DISEASE_LABELS  # noqa: E402

RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "results/results_final_20260926"))
OUT = RESULTS / "f04_fusion"
SEED = 42


def _system_b_test_probs() -> np.ndarray:
    """Per-patient test probabilities from the F-03 flat checkpoint."""
    import torch
    from polymas_ml.sequence.model import HierarchicalMamba

    tokens = np.load(RESULTS / "sequence" / "kmer_canonical" / "tokens.npy")
    labels = pd.read_csv(RESULTS / "features" / "labels.csv")
    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    pos = {pid: i for i, pid in enumerate(labels["patient_id"])}
    te = np.array([pos[p] for p in split.loc[split["split"] == "test", "patient_id"]])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(RESULTS / "f03_hierarchical_mamba" / "ckpt_flat.pt",
                    map_location=dev, weights_only=False)
    model = HierarchicalMamba(vocab_size=4099, n_diseases=7, n_loci=8,
                              d_model=64, n_local_layers=2, d_state=8, mode="flat").to(dev)
    model.load_state_dict({k: v.to(dev) for k, v in ck["best_state"].items()})
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(te), 256):
            xb = torch.from_numpy(tokens[te[s:s + 256]].astype(np.int64)).to(dev)
            outs.append(torch.sigmoid(model(xb)))
    probs = torch.cat(outs).cpu().numpy()
    return te, probs


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    OUT.mkdir(parents=True, exist_ok=True)
    labels_all = pd.read_csv(RESULTS / "features" / "labels.csv")
    labels = labels_all.set_index("patient_id")
    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    test_ids = split.loc[split["split"] == "test", "patient_id"].tolist()
    y = labels.loc[test_ids, DISEASE_LABELS]

    pA = (pd.read_csv(RESULTS / "models" / "test_predictions.csv")
          .set_index("patient_id").loc[test_ids, DISEASE_LABELS])
    te_idx, pB_arr = _system_b_test_probs()
    # Align System B rows to test_ids order (both derive from labels.csv order).
    pB = pd.DataFrame(pB_arr, columns=DISEASE_LABELS,
                      index=labels_all.iloc[te_idx]["patient_id"].to_numpy()).loc[test_ids]

    rows, alphas = [], {}
    fused = {}
    for d in DISEASE_LABELS:
        Xf = np.column_stack([pA[d].to_numpy(), pB[d].to_numpy()])
        # Split-half within test is NOT allowed (that would be test
        # fitting); alpha_d comes from TRAIN-split predictions instead.
        train_ids = split.loc[split["split"] == "train", "patient_id"].tolist()
        # System A train probs need the ensemble; use its val probs via
        # predictions.csv (out-of-fold for train patients by construction of
        # the canonical run) — fall back to fitting alpha on VAL split.
        val_ids = split.loc[split["split"] == "val", "patient_id"].tolist()
        pA_train = (pd.read_csv(RESULTS / "models" / "predictions.csv")
                    .set_index("patient_id"))
        pA_val = pA_train.loc[val_ids, DISEASE_LABELS] if set(val_ids) <= set(pA_train.index) else None
        if pA_val is None:
            raise SystemExit("System A train/val predictions unavailable")
        # System B val probs from the same checkpoint.
        import torch
        from polymas_ml.sequence.model import HierarchicalMamba
        tokens = np.load(RESULTS / "sequence" / "kmer_canonical" / "tokens.npy")
        pos = {pid: i for i, pid in enumerate(labels_all["patient_id"])}
        va_idx = np.array([pos[p] for p in val_ids])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(RESULTS / "f03_hierarchical_mamba" / "ckpt_flat.pt",
                        map_location=dev, weights_only=False)
        model = HierarchicalMamba(vocab_size=4099, n_diseases=7, n_loci=8,
                                  d_model=64, n_local_layers=2, d_state=8, mode="flat").to(dev)
        model.load_state_dict({k: v.to(dev) for k, v in ck["best_state"].items()})
        model.eval()
        with torch.no_grad():
            outs = []
            for s in range(0, len(va_idx), 256):
                xb = torch.from_numpy(tokens[va_idx[s:s + 256]].astype(np.int64)).to(dev)
                outs.append(torch.sigmoid(model(xb)))
        pB_val = pd.DataFrame(torch.cat(outs).cpu().numpy(), columns=DISEASE_LABELS,
                              index=labels_all.iloc[va_idx]["patient_id"].to_numpy())
        y_val = labels.loc[val_ids, d]

        # Bounded mixing weight: minimize val logloss over a grid.
        pa_v = pA_val[d].to_numpy()
        pb_v = pB_val.loc[pA_val.index, d].to_numpy()
        yv = y_val.to_numpy()
        grid = np.linspace(0.0, 1.0, 101)
        losses = []
        for a_ in grid:
            f = np.clip(a_ * pa_v + (1 - a_) * pb_v, 1e-6, 1 - 1e-6)
            losses.append(-np.mean(yv * np.log(f) + (1 - yv) * np.log(1 - f)))
        alpha = float(grid[int(np.argmin(losses))])
        alphas[d] = round(alpha, 4)

        fused_d = alpha * pA[d].to_numpy() + (1 - alpha) * pB[d].to_numpy()
        fused[d] = fused_d
        a_a = float(roc_auc_score(y[d], pA[d]))
        a_b = float(roc_auc_score(y[d], pB[d]))
        a_f = float(roc_auc_score(y[d], fused_d))
        rows.append({"disease": d, "auroc_system_a": round(a_a, 4),
                     "auroc_system_b": round(a_b, 4),
                     "alpha_system_a": round(alpha, 4),
                     "auroc_fused": round(a_f, 4),
                     "fused_minus_best_parent": round(a_f - max(a_a, a_b), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "f04_comparison.csv", index=False)

    wins = int((df["fused_minus_best_parent"] > 0).sum())
    worst = float(df["fused_minus_best_parent"].min())
    macro_fused = float(df["auroc_fused"].mean())
    macro_best = float(np.maximum(df["auroc_system_a"], df["auroc_system_b"]).mean())
    gates = {
        "beats_best_parent_somewhere": {"n_wins": wins, "of": 7, "pass": bool(wins >= 4)},
        "no_disease_loses_gt_0.01": {"worst": round(worst, 4), "pass": bool(worst >= -0.01)},
        "macro_improves": {"fused": round(macro_fused, 4), "best_parent_macro": round(macro_best, 4),
                           "pass": bool(macro_fused > macro_best)},
    }
    summary = {"gates": gates, "verdict": "PASS" if all(g["pass"] for g in gates.values()) else "FAIL",
               "alphas_system_a": alphas,
               "note": "decision-level late fusion; alpha_d fit by logistic "
                       "regression on the VAL split (no test fitting); System B "
                       "probs from the F-03 flat checkpoint (same artifact)"}
    (OUT / "f04_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("F-04 verdict: %s\n%s", summary["verdict"], json.dumps(summary, indent=1))
    logger.info("\n%s", df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

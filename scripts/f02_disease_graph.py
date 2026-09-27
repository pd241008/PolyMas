"""F-02: MAS-aware disease-graph head evaluation (ROADMAP Phase 3).

Pre-registered claim (ROADMAP F-02): a multi-label head with a learned
disease-disease adjacency (init = MAS odds table) improves co-occurrence
calibration vs independent heads, at no AUROC cost. Tolerance: >= neutral
on AUROC (−0.01 floor) AND improved pairwise-phi recovery.

Protocol (pre-registered before the run):
  - Inputs: the ADR-006-corrected canonical run (feature_matrix, labels,
    F-12 split). NOTHING is re-split: the exact train/val/test indices the
    canonical System A used, and System A's published test metrics are the
    comparison baseline.
  - Base learners: the same xgboost/catboost/lightgbm ensemble re-fit on
    the canonical train split. Their TEST-set probabilities are NOT fed to
    the heads (in-sample scores would leak the ensemble's fit). Instead:
      * head/control TRAIN inputs = 3-fold OUT-OF-FOLD base predictions
        within the train split (StratifiedKFold on disease count, seed 42);
      * head/control VAL+TEST inputs = the SAME refit ensemble's raw
        predict_proba on val/test (saved as base_val.csv / base_test.csv in
        stage 1, so both arms share identical inputs).
  - Two arms share the OOF matrix; the only difference is the graph path
    (learned MAS adjacency + message) vs a matched-capacity MLP control.
  - Metrics on the F-12 test split: per-disease AUROC (paired deltas vs
    System A) and pairwise-phi recovery scored exactly like F-17 (sign
    agreement and sign&significant recovery across the 21 pairs, ALPHA
    0.05, incoercible pairs reported but not counted).
  - Gates: AUROC floor (macro delta >= -0.01 vs System A) AND phi recovery
    (recovered count >= System A's 9/19) — both must hold per the
    pre-registered tolerance.
  - Resumable: stages write checkpoints into f02_disease_graph/ so each
    tool call advances one stage (no background processes).

Evidence: <results_root>/f02_disease_graph/
"""
from __future__ import annotations

import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "services" / "ml-engine-python"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("f02_disease_graph")

from polymas_ml.data.patients import DISEASE_LABELS, MAS_EXCLUSIONS, MAS_PAIRWISE_ODDS  # noqa: E402

RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "results/results_final_20260926"))
OUT = RESULTS / "f02_disease_graph"
N_FOLDS = 3
SEED = 42
ALPHA = 0.05


def _load() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    # feature_matrix.csv has NO patient_id column: row i aligns with
    # labels.csv row i (positional). Split patient_ids map through the
    # labels frame's row order.
    X = pd.read_csv(RESULTS / "features" / "feature_matrix.csv")
    y = pd.read_csv(RESULTS / "features" / "labels.csv")
    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    idx = {s: split.loc[split["split"] == s, "patient_id"].tolist()
           for s in ("train", "val", "test")}
    pos = {pid: i for i, pid in enumerate(y["patient_id"])}
    return X, y, {s: np.array([pos[p] for p in ids]) for s, ids in idx.items()}


def _fit_predict(probe: bool = False) -> None:
    """Stage 1: refit the ensemble on the canonical train split; write OOF
    (train), val and test base-probability matrices."""
    from polymas_ml.models.ensemble import MultiLabelEnsemble
    from sklearn.model_selection import StratifiedKFold

    X, y, splits = _load()
    tr = splits["train"]
    Xdf = X  # feature_matrix.csv is all-features (no id column)
    ytr_labels = y.loc[tr, DISEASE_LABELS]
    if probe:
        Xdf = Xdf.iloc[:400]
        ytr_labels = ytr_labels.iloc[:400]
        tr = tr[:400]
    counts = ytr_labels.sum(axis=1).to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_iter = list(skf.split(np.zeros(len(counts)), counts))
    # Probe mode exercises the code path on 400 patients; its checkpoints
    # MUST NOT mix with the real run's cache (different fold shapes).
    global OUT
    out_dir = OUT / "probe" if probe else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = np.zeros((len(tr), len(DISEASE_LABELS)))
    done = 0
    for f, (a, b) in enumerate(fold_iter):
        ckpt = out_dir / f"oof_fold{f}.npy"
        if ckpt.exists():
            oof[b] = np.load(ckpt)
            done += 1
            continue
        # a/b are positional within the TRAIN SLICE; map to full-matrix
        # rows via tr before touching Xdf/y (the labels frame).
        a_full, b_full = tr[a], tr[b]
        ens = MultiLabelEnsemble()
        ens.fit(Xdf.iloc[a_full], y.iloc[a_full])
        pb = ens.predict_proba(Xdf.iloc[b_full])[DISEASE_LABELS].to_numpy()
        oof[b] = pb
        np.save(ckpt, pb)
        logger.info("fold %d/%d done (n=%d)", f + 1, N_FOLDS, len(b))
        return  # resumable: one fold per call
    if done == N_FOLDS:
        val_ckpt, test_ckpt = out_dir / "base_val.csv", out_dir / "base_test.csv"
        if not (val_ckpt.exists() and test_ckpt.exists()):
            ens = MultiLabelEnsemble()
            ens.fit(Xdf.iloc[tr], y.iloc[tr])
            va, te = splits["val"], splits["test"]
            pv = ens.predict_proba(Xdf.iloc[va])[DISEASE_LABELS]
            pv.to_csv(val_ckpt)
            pt = ens.predict_proba(Xdf.iloc[te])[DISEASE_LABELS]
            pt.to_csv(test_ckpt)
            logger.info("val/test base probabilities saved")
        np.save(out_dir / "oof_train.npy", oof)
        logger.info("stage1 COMPLETE: OOF %s + val/test matrices (probe=%s)", oof.shape, probe)


def _phi(x: np.ndarray, y: np.ndarray) -> float:
    vx, vy = x.std(), y.std()
    if vx == 0 or vy == 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (vx * vy))


def _phi_p(phi_val: float, n: int) -> float:
    from scipy import stats as sps
    if np.isnan(phi_val) or n < 10:
        return float("nan")
    return float(2.0 * sps.norm.sf(abs(phi_val) * np.sqrt(n - 1)))


def _recovery_table(preds: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """F-17-identical recovery scoring on a prediction matrix."""
    n = len(preds)
    rows = []
    for a, b in combinations(DISEASE_LABELS, 2):
        embedded = MAS_PAIRWISE_ODDS.get((a, b), MAS_PAIRWISE_ODDS.get((b, a)))
        excluded = any({a, b} == set(pair) for pair in MAS_EXCLUSIONS)
        phi_pred = _phi(preds[a].to_numpy(dtype=float), preds[b].to_numpy(dtype=float))
        phi_label = _phi(labels[a].to_numpy(dtype=float), labels[b].to_numpy(dtype=float))
        p_val = _phi_p(phi_pred, n)
        recovered = (
            embedded is not None
            and not np.isnan(phi_pred)
            and np.sign(phi_pred) == np.sign(embedded)
            and p_val == p_val and p_val < ALPHA
        )
        rows.append({
            "pair": f"{a}|{b}",
            "embedded_log_or": embedded if embedded is not None else 0.0,
            "mas_excluded": excluded,
            "phi_predictions": None if np.isnan(phi_pred) else round(phi_pred, 4),
            "phi_labels": None if np.isnan(phi_label) else round(phi_label, 4),
            "p_value": None if p_val != p_val else p_val,
            "sign_agrees": (embedded is not None and not np.isnan(phi_pred)
                            and np.sign(phi_pred) == np.sign(embedded)),
            "recovered": bool(recovered),
        })
    return pd.DataFrame(rows)


def stage2(kind: str) -> None:
    """Stage 2: train one arm (graph|control) on OOF, predict val+test."""
    import torch
    from polymas_ml.models.disease_graph import (
        DiseaseGraphHead, IndependentHeadControl, train_head, predict_head,
    )

    X, y, splits = _load()
    tr, va, te = splits["train"], splits["val"], splits["test"]
    oof = np.load(OUT / "oof_train.npy")
    Xva = pd.read_csv(OUT / "base_val.csv")[DISEASE_LABELS].to_numpy()
    Xte = pd.read_csv(OUT / "base_test.csv")[DISEASE_LABELS].to_numpy()
    Ytr = y.loc[tr, DISEASE_LABELS].to_numpy(dtype=np.float32)
    Yva = y.loc[va, DISEASE_LABELS].to_numpy(dtype=np.float32)

    torch.manual_seed(SEED)
    model = DiseaseGraphHead() if kind == "graph" else IndependentHeadControl()
    # train_head early-stops on val macro AUROC, so stack train+val labels
    # and index the val rows explicitly (one label matrix, disjoint slices).
    Ystack = np.vstack([Ytr, Yva])
    Xstack = np.vstack([oof, Xva])
    hist = train_head(
        model, Xstack, Ystack,
        np.arange(len(tr)), np.arange(len(va)) + len(tr),
        epochs=80, seed=SEED,
    )
    out_val = predict_head(model, Xva)
    out_test = predict_head(model, Xte)
    suffix = "graph" if kind == "graph" else "control"
    np.save(OUT / f"pred_val_{suffix}.npy", out_val)
    np.save(OUT / f"pred_test_{suffix}.npy", out_test)
    if kind == "graph":
        (OUT / "adjacency_report.json").write_text(
            json.dumps(model.adjacency_report(), indent=2))
    (OUT / f"train_history_{suffix}.json").write_text(
        json.dumps({"best_val_macro_auroc": hist["best_val_macro_auroc"],
                    "last_epoch": hist["history"][-1]}, indent=2))
    logger.info("stage2[%s] done: best val macro %.4f", kind, hist["best_val_macro_auroc"])


def stage3() -> int:
    """Stage 3: gates + artifacts."""
    _, y, splits = _load()
    labels = y.set_index("patient_id").loc[
        pd.read_csv(RESULTS / "models" / "split_indices.csv")
        .loc[lambda d: d["split"] == "test", "patient_id"]]
    sysA = pd.read_csv(RESULTS / "models" / "per_disease_metrics.csv").set_index("disease")

    pred_g = pd.DataFrame(
        np.load(OUT / "pred_test_graph.npy"), columns=DISEASE_LABELS)
    pred_c = pd.DataFrame(
        np.load(OUT / "pred_test_control.npy"), columns=DISEASE_LABELS)

    def aurocs(preds: pd.DataFrame) -> dict[str, float]:
        from sklearn.metrics import roc_auc_score
        return {d: roc_auc_score(labels[d], preds[d]) for d in DISEASE_LABELS}

    auc_g, auc_c = aurocs(pred_g), aurocs(pred_c)
    rows = []
    for d in DISEASE_LABELS:
        rows.append({
            "disease": d,
            "auroc_system_a": float(sysA.loc[d, "auroc"]),
            "auroc_graph_head": round(auc_g[d], 4),
            "auroc_control_head": round(auc_c[d], 4),
            "delta_graph_minus_systemA": round(auc_g[d] - float(sysA.loc[d, "auroc"]), 4),
            "delta_graph_minus_control": round(auc_g[d] - auc_c[d], 4),
        })
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT / "f02_comparison.csv", index=False)

    # Phi recovery, F-17-identical, on the same test labels.
    rec_g = _recovery_table(pred_g, labels)
    rec_c = _recovery_table(pred_c, labels)
    rec_g.to_csv(OUT / "phi_recovery_graph.csv", index=False)
    rec_c.to_csv(OUT / "phi_recovery_control.csv", index=False)
    n_rec_g = int(rec_g["recovered"].sum())
    n_rec_c = int(rec_c["recovered"].sum())
    n_sign_g = int(rec_g.loc[~rec_g["mas_excluded"], "sign_agrees"].sum())
    n_sign_c = int(rec_c.loc[~rec_c["mas_excluded"], "sign_agrees"].sum())

    macro_a = float(sysA["auroc"].mean())
    macro_g = float(np.mean(list(auc_g.values())))
    macro_c = float(np.mean(list(auc_c.values())))
    d_vs_a = macro_g - macro_a
    d_vs_c = macro_g - macro_c
    worst = cmp_df["delta_graph_minus_systemA"].min()

    gates = {
        "auroc_floor": {
            "macro_delta_vs_systemA": round(d_vs_a, 4),
            "floor": -0.01,
            "pass": bool(d_vs_a >= -0.01),
        },
        "phi_recovery": {
            "recovered_graph": n_rec_g, "recovered_systemA": 9,
            "recovered_control": n_rec_c,
            "sign_agrees_graph": n_sign_g, "sign_agrees_control": n_sign_c,
            "pass": bool(n_rec_g >= 9),
        },
        "per_disease_floor": {
            "worst_delta_vs_systemA": round(worst, 4),
            "pass": bool(worst >= -0.01),
        },
    }
    verdict = "PASS" if all(g["pass"] for g in gates.values()) else "FAIL"
    summary = {
        "gates": gates,
        "verdict": verdict,
        "macro": {"systemA": round(macro_a, 4), "graph": round(macro_g, 4),
                  "control": round(macro_c, 4), "graph_minus_control": round(d_vs_c, 4)},
        "note": "Head/control train on 3-fold OOF base predictions; "
                "val+test inputs from the same refit ensemble. Recovery "
                "scored F-17-identical (ALPHA=0.05, incoercibles reported "
                "not counted).",
    }
    (OUT / "f02_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("F-02 verdict: %s\n%s", verdict, json.dumps(summary, indent=1))
    logger.info("\n%s", cmp_df.to_string(index=False))
    return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "stage1":
        _fit_predict(probe="--probe" in sys.argv)
    elif stage == "stage2-graph":
        stage2("graph")
    elif stage == "stage2-control":
        stage2("control")
    elif stage == "stage3":
        return stage3()
    else:
        raise SystemExit("usage: f02_disease_graph.py {stage1[--probe]|stage2-graph|stage2-control|stage3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

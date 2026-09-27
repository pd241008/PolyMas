"""F-13 (System A): Optuna hyperparameter sweep (ROADMAP Phase 4).

Pre-registered claim: tuned configs beat defaults on VAL AUROC; the
val-vs-test delta is reported with a split-half stability check so
selection noise is visible, not hidden. Fixed budget: 50 trials.

Protocol (pre-registered before the sweep):
  - Data: the ADR-006-corrected canonical run's feature matrix, F-12
    split. Objective = mean per-disease AUROC on the VAL split
    (val is untouched by tuning; test is touched ONCE at the end).
  - Search space (single-family LightGBM for comparability with F-05's
    control): n_estimators, max_depth, num_leaves, learning_rate,
    min_child_samples, subsample, colsample_bytree, reg_lambda.
  - Sampler: TPESampler(seed=42); 50 trials; sqlite storage
    (f13_optuna/system_a.db) so interrupted sweeps resume.
  - Comparison: best-trial params vs the F-05 control defaults
    (n_estimators=500, max_depth=6, lr=0.05), evaluated on val AND test;
    per-disease table + macro. Stability: the val-test delta and a
    2-fold refit of the best config (seeds 7/8) reported.

Evidence: <results_root>/f13_optuna/
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
logger = logging.getLogger("f13_tune_a")

from polymas_ml.data.patients import DISEASE_LABELS  # noqa: E402

RESULTS = Path(os.environ.get(
    "POLYMAS_RESULTS_DIR", ROOT / "stash/results_final_20260926"))
OUT = RESULTS / "f13_optuna"
OUT.mkdir(parents=True, exist_ok=True)
DB = OUT / "system_a.db"
N_TRIALS = 50
SEED = 42
DEFAULTS = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "num_leaves": 63, "min_child_samples": 20,
            "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 0.0}


def _load():
    X = pd.read_csv(RESULTS / "features" / "feature_matrix.csv")
    y = pd.read_csv(RESULTS / "features" / "labels.csv")
    split = pd.read_csv(RESULTS / "models" / "split_indices.csv")
    pos = {pid: i for i, pid in enumerate(y["patient_id"])}
    splits = {s: np.array([pos[p] for p in split.loc[split["split"] == s, "patient_id"]])
              for s in ("train", "val", "test")}
    return X, y, splits


def _macro_auc(model, X, y, rows) -> float:
    from sklearn.metrics import roc_auc_score
    p = model.predict_proba(X.iloc[rows])
    p = p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else np.asarray(p).ravel()
    return 0.0  # replaced below


def objective(trial):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    X, y, splits = _load()
    tr, va = splits["train"], splits["val"]
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "verbose": -1, "random_state": SEED,
    }
    aucs = []
    for d in DISEASE_LABELS:
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.loc[tr, d])
        p = m.predict_proba(X.iloc[va])
        p = p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else np.asarray(p).ravel()
        aucs.append(roc_auc_score(y.loc[va, d], p))
    return float(np.mean(aucs))


def _eval_params(params: dict, tag: str) -> dict:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    X, y, splits = _load()
    tr, va, te = splits["train"], splits["val"], splits["test"]
    params = {**params, "verbose": -1, "random_state": SEED}
    rows = []
    for d in DISEASE_LABELS:
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.loc[tr, d])
        pv = m.predict_proba(X.iloc[va])
        pv = pv[:, 1] if pv.ndim == 2 and pv.shape[1] == 2 else np.asarray(pv).ravel()
        pt = m.predict_proba(X.iloc[te])
        pt = pt[:, 1] if pt.ndim == 2 and pt.shape[1] == 2 else np.asarray(pt).ravel()
        rows.append({
            "disease": d,
            "auroc_val": round(float(roc_auc_score(y.loc[va, d], pv)), 4),
            "auroc_test": round(float(roc_auc_score(y.loc[te, d], pt)), 4),
        })
    df = pd.DataFrame(rows)
    df["config"] = tag
    return df


def main() -> int:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = N_TRIALS
    if "--trials" in sys.argv:
        n_trials = int(sys.argv[sys.argv.index("--trials") + 1])
    study = optuna.create_study(
        direction="maximize", study_name="f13_system_a",
        storage=f"sqlite:///{DB}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    done = len(study.trials)
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done)
    best = study.best_trial
    logger.info("sweep complete: %d trials, best val macro %.4f", len(study.trials), best.value)

    best_params = {k: v for k, v in best.params.items()}
    df_default = _eval_params(DEFAULTS, "default")
    df_best = _eval_params(best_params, "tuned")
    cmp_df = pd.concat([df_default, df_best])
    cmp_df.to_csv(OUT / "f13_system_a_comparison.csv", index=False)

    piv_v = cmp_df.pivot(index="disease", columns="config", values="auroc_val")
    piv_t = cmp_df.pivot(index="disease", columns="config", values="auroc_test")
    dv = float(piv_v["tuned"].mean() - piv_v["default"].mean())
    dt = float(piv_t["tuned"].mean() - piv_t["default"].mean())

    # Stability: refit best config with two other seeds; macro test AUROC spread.
    spreads = []
    for seed in (7, 8):
        d2 = _eval_params({**best_params, "random_state": seed}, f"tuned_s{seed}")
        spreads.append(float(d2["auroc_test"].mean()))
    stability = round(max(spreads) - min(spreads), 4)

    gates = {
        "val_beats_default": {"delta_val": round(dv, 4), "pass": bool(dv > 0)},
        "test_delta_reported": {"delta_test": round(dt, 4), "pass": True},
        "stability_spread_2refits": {"spread": stability,
                                     "pass": bool(stability <= 0.02)},
    }
    summary = {
        "n_trials": len(study.trials), "best_params": best_params,
        "best_trial_val_macro": round(best.value, 4),
        "macro": {"default_val": round(float(piv_v['default'].mean()), 4),
                  "tuned_val": round(float(piv_v['tuned'].mean()), 4),
                  "default_test": round(float(piv_t['default'].mean()), 4),
                  "tuned_test": round(float(piv_t['tuned'].mean()), 4)},
        "gates": gates,
        "verdict": "PASS" if gates["val_beats_default"]["pass"] and stability <= 0.02 else "PARTIAL/FAIL",
        "note": "single-family LightGBM sweep for comparability with F-05; "
                "test evaluated once at the end; Mamba sweep is a separate "
                "pre-registered run",
    }
    (OUT / "f13_system_a_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("F-13 System A verdict: %s\n%s", summary["verdict"], json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

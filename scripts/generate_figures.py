#!/usr/bin/env python3
"""Generate visualization graphs from PolyMas results."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Results root; override with POLYMAS_RESULTS_DIR to read a specific run folder.
RESULTS_DIR = Path(os.environ.get("POLYMAS_RESULTS_DIR", PROJECT_ROOT / "stash" / "results"))
STASH_DIR = PROJECT_ROOT / "stash"
FIGURES_DIR = STASH_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
STATS_DIR = RESULTS_DIR / "stats"
DISEASE_LABELS = ["RA", "SLE", "SJOGRENS", "AITD", "T1D", "VITILIGO", "MS"]

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

GWAS_CSV = RESULTS_DIR / "raw" / "gwas" / "gwas_associations.csv"
FEATURES_DIR = RESULTS_DIR / "features"
MODELS_DIR = RESULTS_DIR / "models"
EXPLANATIONS_DIR = RESULTS_DIR / "explanations"
CLUSTERS_DIR = RESULTS_DIR / "clusters"
REPORTS_DIR = RESULTS_DIR / "reports"


def plot_gwas_pvalue_distribution():
    df = pd.read_csv(GWAS_CSV)
    df["neg_log10_p"] = -np.log10(df["pvalue"].clip(lower=1e-300))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    df["neg_log10_p"].hist(bins=40, color="steelblue", edgecolor="black", alpha=0.8, ax=ax)
    ax.set_title("GWAS Association -log10(p) Distribution")
    ax.set_xlabel("-log10(p-value)")
    ax.set_ylabel("Frequency")

    ax = axes[1]
    top = df.groupby("rs_id")["neg_log10_p"].mean().sort_values(ascending=True)
    colors = ["#e74c3c" if v > 50 else "#3498db" for v in top.values]
    top.plot(kind="barh", color=colors, ax=ax)
    ax.set_title("Mean -log10(p) per Locus")
    ax.set_xlabel("Mean -log10(p-value)")
    ax.set_ylabel("Locus (rsID)")

    plt.tight_layout()
    out = FIGURES_DIR / "gwas_pvalue_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_feature_correlation():
    prs = pd.read_csv(FEATURES_DIR / "prs_features.csv")
    pivot = prs.pivot_table(index="patient_id", columns="locus_id", values="continuous_score", aggfunc="first")
    corr = pivot.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("PRS Score Correlation Across Loci")

    plt.tight_layout()
    out = FIGURES_DIR / "feature_correlation_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_prediction_distributions():
    preds = pd.read_csv(MODELS_DIR / "predictions.csv", index_col=0)
    n_cols = len(preds.columns)
    n_rows = 2
    n_cols_grid = max(3, (n_cols + 1) // 2)

    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(6 * n_cols_grid, 8))
    axes = axes.flatten()

    for i, col in enumerate(preds.columns):
        ax = axes[i]
        preds[col].hist(bins=20, color="seagreen", edgecolor="black", alpha=0.8, ax=ax)
        ax.set_title(f"{col} — Predicted Probabilities")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Patient Count")
        ax.axvline(preds[col].mean(), color="red", linestyle="--", label=f"Mean={preds[col].mean():.3f}")
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    out = FIGURES_DIR / "prediction_distributions.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def _read_shap_importance(path: Path) -> pd.DataFrame:
    """Read a shap_importance_*.csv in either the labeled (feature column) or
    legacy (index-only) format."""
    df = pd.read_csv(path)
    if "feature" in df.columns:
        return df
    # Legacy format: feature names were in the index and dropped by
    # to_csv(index=False); fall back to positional labels.
    df = df.rename(columns={df.columns[0]: "mean_abs_shap"})
    df["feature"] = [f"feature_{i}" for i in range(len(df))]
    return df


def plot_shap_importance():
    shap_files = sorted(EXPLANATIONS_DIR.glob("shap_importance_*.csv"))
    if not shap_files:
        logger.warning("No SHAP importance files found")
        return

    fig, axes = plt.subplots(1, len(shap_files), figsize=(6 * len(shap_files), 5))
    if len(shap_files) == 1:
        axes = [axes]

    for ax, path in zip(axes, shap_files):
        df = _read_shap_importance(path).head(10)
        disease = path.stem.replace("shap_importance_", "")
        features = df["feature"].astype(str)
        values = df["mean_abs_shap"].values
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(values)))
        ax.barh(features[::-1], values[::-1], color=colors[::-1])
        ax.set_title(f"SHAP Feature Importance — {disease}")
        ax.set_xlabel("Mean |SHAP|")
        ax.set_ylabel("Feature")

    plt.tight_layout()
    out = FIGURES_DIR / "shap_importance.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_shap_beeswarm():
    """Labeled SHAP beeswarm per disease: SHAP values from explanations/
    shap_*.csv colored by feature values from features/feature_matrix.csv
    (same rows, so the figure matches the run scale, e.g. n=5,000)."""
    feature_matrix_path = FEATURES_DIR / "feature_matrix.csv"
    if not feature_matrix_path.exists():
        logger.warning("No feature matrix — cannot color beeswarm by feature value")
        return
    X = pd.read_csv(feature_matrix_path)

    shap_files = sorted(EXPLANATIONS_DIR.glob("shap_*.csv"))
    shap_files = [p for p in shap_files if not p.stem.startswith("shap_importance")]
    if not shap_files:
        logger.warning("No per-patient SHAP files found")
        return

    for path in shap_files:
        disease = path.stem.replace("shap_", "")
        shap_df = pd.read_csv(path)
        if list(shap_df.columns) != list(X.columns):
            logger.warning(
                "SHAP/%s feature columns do not match feature_matrix.csv — skipping beeswarm",
                disease,
            )
            continue

        mean_abs = shap_df.abs().mean(axis=0).sort_values(ascending=False)
        top = list(mean_abs.head(10).index)
        sv = shap_df[top].to_numpy()
        # Per-feature rank-normalized value in [0, 1] for the color scale.
        fv_r = X[top].rank(axis=0, pct=True).to_numpy()
        order = np.argsort(-np.abs(sv), axis=0)
        n_pat, n_feat = sv.shape

        fig, ax = plt.subplots(figsize=(9, 6))
        cmap = plt.cm.coolwarm
        for fi in range(n_feat):
            y = n_feat - 1 - fi
            idx = order[:, fi]
            ax.scatter(
                sv[idx, fi],
                np.full(n_pat, y) + np.linspace(-0.18, 0.18, n_pat),
                c=fv_r[idx, fi],
                cmap=cmap,
                vmin=0, vmax=1,
                s=3, alpha=0.35, linewidths=0, rasterized=True,
            )
        ax.set_yticks(range(n_feat), top[::-1])
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on model output)")
        ax.set_title(f"SHAP Beeswarm — {disease} (n={n_pat})")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
        cbar.set_label("Feature value (rank %, high = red)")
        plt.tight_layout()
        out = FIGURES_DIR / f"shap_beeswarm_{disease}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        logger.info("Saved %s", out)


def plot_cluster_dendrogram():
    import json
    dendro_path = CLUSTERS_DIR / "dendrogram.json"
    if not dendro_path.exists():
        logger.warning("Dendrogram JSON not found")
        return

    with open(dendro_path) as f:
        dendro = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Hierarchical Clustering Dendrogram")

    n_leaves = len([k for k in dendro if k.startswith("leaf_")])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_leaves + 1)

    from scipy.cluster.hierarchy import dendrogram
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage

    X = pd.read_csv(MODELS_DIR / "predictions.csv", index_col=0).values[:50]
    dists = pdist(X, metric="euclidean")
    Z = linkage(dists, method="ward")
    dn = dendrogram(Z, no_labels=True, color_threshold=0.7 * max(Z[:, 2]), ax=ax)

    ax.set_title("Hierarchical Clustering Dendrogram (Ward, Euclidean)")
    ax.set_xlabel("Patient Index")
    ax.set_ylabel("Distance")

    plt.tight_layout()
    out = FIGURES_DIR / "dendrogram.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_cluster_distribution():
    clusters = pd.read_csv(CLUSTERS_DIR / "cluster_assignments.csv")
    counts = clusters["cluster_label"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors[:len(counts)], edgecolor="black")
    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Number of Patients")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, str(int(height)),
                ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    out = FIGURES_DIR / "cluster_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_lime_comparison():
    lime_files = sorted(EXPLANATIONS_DIR.glob("lime_*.csv"))
    if not lime_files:
        logger.warning("No LIME files found")
        return

    fig, axes = plt.subplots(1, len(lime_files), figsize=(6 * len(lime_files), 5))
    if len(lime_files) == 1:
        axes = [axes]

    for ax, path in zip(axes, lime_files):
        df = pd.read_csv(path).head(10)
        disease = path.stem.replace("lime_", "")
        colors = ["#e67e22" if v > 0 else "#3498db" for v in df["attribution"]]
        ax.barh(df["feature"][::-1], df["attribution"][::-1], color=colors[::-1])
        ax.set_title(f"LIME Attributions — {disease}")
        ax.set_xlabel("Attribution Value")
        ax.set_ylabel("Feature")

    plt.tight_layout()
    out = FIGURES_DIR / "lime_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_prs_distribution_by_locus():
    prs = pd.read_csv(FEATURES_DIR / "prs_features.csv")
    top_loci = prs.groupby("locus_id")["continuous_score"].mean().nlargest(8).index
    subset = prs[prs["locus_id"].isin(top_loci)]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=subset, x="locus_id", y="continuous_score", palette="Set2", ax=ax)
    ax.set_title("PRS Score Distribution by Locus")
    ax.set_xlabel("Locus (rsID)")
    ax.set_ylabel("Continuous Score")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out = FIGURES_DIR / "prs_distribution_by_locus.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_bootstrap_ci_forest():
    """Forest plot: held-out AUROC per disease with 95% bootstrap CIs."""
    ci_path = STATS_DIR / "bootstrap_ci_auroc.csv"
    if not ci_path.exists():
        logger.warning("No bootstrap CI file (run scripts/run_stats_eval.py first)")
        return
    df = pd.read_csv(ci_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(df))
    colors = ["#e67e22" if d in ("AITD", "VITILIGO") else "#2980b9" for d in df["disease"]]
    ax.hlines(y_pos, df["ci_lo"], df["ci_hi"], color=colors, linewidth=3, alpha=0.7)
    ax.scatter(df["point"], y_pos, color=colors, zorder=3, s=45, label="point estimate")
    ax.set_yticks(y_pos, df["disease"])
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xlabel("Held-out AUROC (95% percentile bootstrap CI)")
    ax.set_title("Per-Disease Discrimination with Bootstrap Confidence Intervals")
    ax.legend(loc="lower right")
    for i, row in df.iterrows():
        ax.text(
            row["ci_hi"] + 0.012, i,
            f"{row['point']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]",
            va="center", fontsize=8,
        )
    plt.tight_layout()
    out = FIGURES_DIR / "bootstrap_ci_forest.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_silhouette_permutation():
    """Permutation null distribution vs the observed silhouette score."""
    perm_path = STATS_DIR / "silhouette_permutation_test.json"
    if not perm_path.exists():
        logger.warning("No permutation test JSON (run scripts/run_stats_eval.py first)")
        return
    perm = json.loads(perm_path.read_text())
    null = perm.get("null_scores", [])
    if not null:
        logger.warning("Permutation JSON lacks null_scores")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null, bins=24, color="#95a5a6", alpha=0.85, edgecolor="black", label="null (within-disease shuffle)")
    ax.axvline(perm["null_mean"], color="#7f8c8d", linestyle=":", linewidth=1.5, label=f"null mean = {perm['null_mean']:.3f}")
    ax.axvline(perm["observed"], color="#c0392b", linestyle="-", linewidth=2.5, label=f"observed = {perm['observed']:.3f}")
    ax.set_xlabel("Silhouette score (Ward, k=3)")
    ax.set_ylabel("Permutations")
    ax.set_title(
        f"Silhouette Permutation Test — p = {perm['p_value']:.3f}, z = {perm['z_score']:.2f}"
    )
    ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "silhouette_permutation.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_ancestry_stratified():
    """Ancestry-stratified AUROC bars per disease (EUR/AFR/EAS)."""
    strat_path = STATS_DIR / "ancestry_stratified_metrics.csv"
    if not strat_path.exists():
        logger.warning("No ancestry-stratified metrics (run scripts/run_stats_eval.py first)")
        return
    df = pd.read_csv(strat_path)
    df = df[df["disease"].isin(DISEASE_LABELS)]

    groups = sorted(df["ancestry"].unique())
    diseases = [d for d in DISEASE_LABELS if d in set(df["disease"])]
    x = np.arange(len(diseases))
    width = 0.8 / len(groups)
    palette = {"EUR": "#2980b9", "AFR": "#e67e22", "EAS": "#27ae60"}

    fig, ax = plt.subplots(figsize=(11, 5))
    for gi, group in enumerate(groups):
        sub = df[df["ancestry"] == group].set_index("disease").reindex(diseases)
        bars = ax.bar(
            x + (gi - (len(groups) - 1) / 2) * width,
            sub["auroc"].fillna(0),
            width,
            color=palette.get(group),
            edgecolor="black",
            linewidth=0.5,
            label=f"{group} (n={int(sub['n'].iloc[0])})",
        )
        for xi, (_bar, val) in enumerate(zip(bars, sub["auroc"])):
            if np.isnan(val):
                ax.text(
                    x[xi] + (gi - (len(groups) - 1) / 2) * width, 0.02,
                    "n/a", ha="center", fontsize=7, rotation=90,
                )
    ax.set_xticks(x, diseases)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("Ancestry-Stratified Discrimination (EUR / AFR / EAS)")
    ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "ancestry_stratified_auroc.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_phi_heatmap():
    """Pairwise phi (co-occurrence correlation) heatmap of the label matrix."""
    phi_path = STATS_DIR / "pairwise_phi_pvalues.csv"
    if not phi_path.exists():
        logger.warning("No pairwise phi table (run scripts/run_stats_eval.py first)")
        return
    phi_df = pd.read_csv(phi_path)
    diseases = [
        d for d in DISEASE_LABELS
        if d in set(phi_df["disease_a"]) | set(phi_df["disease_b"])
    ]
    mat = pd.DataFrame(np.eye(len(diseases)) * 1.0, index=diseases, columns=diseases)
    for _, row in phi_df.iterrows():
        mat.loc[row["disease_a"], row["disease_b"]] = row["phi"]
        mat.loc[row["disease_b"], row["disease_a"]] = row["phi"]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        mat, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        vmin=-0.4, vmax=0.4, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "phi (co-occurrence correlation)"}, ax=ax,
    )
    ax.set_title("Disease Co-occurrence Structure (pairwise phi)")
    plt.tight_layout()
    out = FIGURES_DIR / "phi_cooccurrence_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_disease_count_distribution():
    """Disease-count-per-patient distribution: MAS prevalence at a glance."""
    cooc_path = STATS_DIR / "cooccurrence_structure.json"
    if not cooc_path.exists():
        logger.warning("No co-occurrence structure JSON (run scripts/run_stats_eval.py first)")
        return
    structure = json.loads(cooc_path.read_text())["label_structure"]
    dist = {int(k): v for k, v in structure["disease_count_distribution"].items()}
    ks = sorted(dist)
    counts = [dist[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        list(ks), counts,
        color=["#bdc3c7" if k < 3 else "#c0392b" for k in ks], edgecolor="black",
    )
    ax.set_xticks(list(ks))
    for bar, k in zip(bars, ks, strict=True):
        pct = 100 * dist[k] / sum(counts)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{pct:.1f}%", ha="center", fontsize=9,
        )
    ax.set_xlabel("Concurrent autoimmune diseases per patient")
    ax.set_ylabel("Patients")
    ax.set_title(
        "Polyautoimmunity Load — MAS (3+ diseases) rate = "
        f"{structure['rate_3plus_mas'] * 100:.1f}%"
    )
    plt.tight_layout()
    out = FIGURES_DIR / "disease_count_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def main():
    logger.info("=== Generating result visualizations ===")
    plot_gwas_pvalue_distribution()
    plot_feature_correlation()
    plot_prediction_distributions()
    plot_shap_importance()
    plot_shap_beeswarm()
    plot_cluster_dendrogram()
    plot_cluster_distribution()
    plot_lime_comparison()
    plot_prs_distribution_by_locus()
    plot_bootstrap_ci_forest()
    plot_silhouette_permutation()
    plot_ancestry_stratified()
    plot_phi_heatmap()
    plot_disease_count_distribution()
    logger.info("=== All figures saved to %s ===", FIGURES_DIR)


if __name__ == "__main__":
    main()

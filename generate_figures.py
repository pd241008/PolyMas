#!/usr/bin/env python3
"""Generate visualization graphs from PolyMas results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

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
    preds = pd.read_csv(MODELS_DIR / "predictions.csv")
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


def plot_shap_importance():
    shap_files = sorted(EXPLANATIONS_DIR.glob("shap_importance_*.csv"))
    if not shap_files:
        logger.warning("No SHAP importance files found")
        return

    fig, axes = plt.subplots(1, len(shap_files), figsize=(6 * len(shap_files), 5))
    if len(shap_files) == 1:
        axes = [axes]

    for ax, path in zip(axes, shap_files):
        df = pd.read_csv(path).head(10)
        disease = path.stem.replace("shap_importance_", "")
        features = df.index.astype(str)
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

    X = pd.read_csv(MODELS_DIR / "predictions.csv").values[:50]
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


def main():
    logger.info("=== Generating result visualizations ===")
    plot_gwas_pvalue_distribution()
    plot_feature_correlation()
    plot_prediction_distributions()
    plot_shap_importance()
    plot_cluster_dendrogram()
    plot_cluster_distribution()
    plot_lime_comparison()
    plot_prs_distribution_by_locus()
    logger.info("=== All figures saved to %s ===", FIGURES_DIR)


if __name__ == "__main__":
    main()

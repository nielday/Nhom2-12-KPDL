"""
Visualization module.

Shared plotting functions for EDA, clustering, classification,
association rules, semi-supervised, and time series.
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)

# ---------- Global style ----------
matplotlib.rcParams.update({"font.size": 11})
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (10, 6)
SAVE_DPI = 150


def _save(fig: plt.Figure, path: Optional[str]) -> None:
    """Save figure if path provided."""
    if path:
        fig.savefig(
            path, dpi=SAVE_DPI, bbox_inches="tight",
        )
        plt.close(fig)


# ===== EDA =====

def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "is_canceled",
    save_path: Optional[str] = None,
) -> None:
    """Plot distribution of target variable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[target_col].value_counts()
    bars = ax.bar(
        ["Not Canceled (0)", "Canceled (1)"],
        counts.values,
        color=["#2ecc71", "#e74c3c"],
        edgecolor="white",
    )
    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + len(df) * 0.01,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", fontweight="bold",
        )
    ax.set_title("Distribution of Booking Cancellations")
    ax.set_ylabel("Count")
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_cancel_by_category(
    df: pd.DataFrame,
    col: str,
    target_col: str = "is_canceled",
    top_n: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """Plot cancellation rate by categorical column."""
    rates = df.groupby(col)[target_col].mean().sort_values(
        ascending=False
    ).head(top_n)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    rates.plot(kind="barh", ax=ax, color="#3498db",
               edgecolor="white")
    ax.set_xlabel("Cancellation Rate")
    ax.set_title(f"Cancellation Rate by {col}")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> None:
    """Plot correlation heatmap of numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, ax=ax,
        linewidths=0.5, annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_distribution(
    df: pd.DataFrame,
    col: str,
    bins: int = 50,
    save_path: Optional[str] = None,
) -> None:
    """Plot histogram with KDE for a numeric column."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(
        df[col].dropna(), bins=bins, color="#3498db",
        edgecolor="white", alpha=0.7,
    )
    axes[0].set_title(f"Distribution of {col}")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frequency")

    # Boxplot
    axes[1].boxplot(df[col].dropna(), vert=True)
    axes[1].set_title(f"Boxplot of {col}")
    axes[1].set_ylabel(col)

    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


# ===== Classification =====

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix heatmap."""
    if labels is None:
        labels = ["Not Canceled", "Canceled"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """Plot ROC and Precision-Recall curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc_val = auc(fpr, tpr)
    axes[0].plot(
        fpr, tpr,
        label=f"{model_name} (AUC={roc_auc_val:.3f})",
        linewidth=2,
    )
    axes[0].plot(
        [0, 1], [0, 1], "k--", alpha=0.3, linewidth=1,
    )
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # PR
    precision, recall, _ = precision_recall_curve(
        y_true, y_proba
    )
    pr_auc_val = auc(recall, precision)
    axes[1].plot(
        recall, precision,
        label=f"{model_name} (AUC={pr_auc_val:.3f})",
        linewidth=2, color="#e74c3c",
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_multi_roc_pr(
    y_true: np.ndarray,
    model_probas: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> None:
    """Plot ROC and PR curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_probas)))

    for (name, proba), color in zip(
        model_probas.items(), colors
    ):
        # ROC
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_val = auc(fpr, tpr)
        axes[0].plot(
            fpr, tpr, label=f"{name} ({roc_val:.3f})",
            color=color, linewidth=2,
        )
        # PR
        prec, rec, _ = precision_recall_curve(y_true, proba)
        pr_val = auc(rec, prec)
        axes[1].plot(
            rec, prec, label=f"{name} ({pr_val:.3f})",
            color=color, linewidth=2,
        )

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend(fontsize=9)

    axes[1].set_title("PR Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_feature_importance(
    fi_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> None:
    """Plot feature importance bar chart."""
    data = fi_df.head(top_n).sort_values(
        "importance", ascending=True
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        data["feature"], data["importance"],
        color="#2ecc71", edgecolor="white",
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


# ===== Clustering =====

def plot_elbow_silhouette(
    k_results: Dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot elbow and silhouette charts side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ks = k_results["k_values"]

    axes[0].plot(ks, k_results["inertias"], "bo-",
                 linewidth=2)
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")

    axes[1].plot(ks, k_results["silhouettes"], "rs-",
                 linewidth=2)
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis")

    if "best_k" in k_results:
        best = k_results["best_k"]
        idx = ks.index(best)
        axes[1].axvline(
            x=best, color="green", linestyle="--",
            alpha=0.7, label=f"Best K={best}",
        )
        axes[1].legend()

    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_cluster_scatter_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Scatter (PCA 2D)",
    save_path: Optional[str] = None,
) -> None:
    """Plot 2D scatter of clusters."""
    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = sorted(set(labels))
    colors = plt.cm.Set2(
        np.linspace(0, 1, len(unique_labels))
    )

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        name = f"Cluster {label}" if label != -1 else "Noise"
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[color], label=name, alpha=0.5, s=10,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_cluster_profiles(
    profiles: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot cluster profile heatmap."""
    # Exclude size/pct columns
    plot_cols = [
        c for c in profiles.columns
        if c not in ["size", "pct"]
    ]
    data = profiles[plot_cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="YlOrRd",
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Cluster Profiles")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


# ===== Association Rules =====

def plot_association_rules_scatter(
    rules: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot association rules: support vs confidence,
    size = lift."""
    if rules.empty:
        print("[Viz] No rules to plot.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    scatter = ax.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        s=rules["lift"] * 50,
        alpha=0.6,
        cmap="YlOrRd",
        edgecolors="grey",
    )
    plt.colorbar(scatter, ax=ax, label="Lift")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Association Rules (size ∝ lift)")
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


# ===== Semi-supervised =====

def plot_learning_curve_semi(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot learning curve: F1 and PR-AUC by label %."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for method, group in results_df.groupby("method"):
        axes[0].plot(
            group["label_pct"], group["f1"],
            "o-", label=method, linewidth=2,
        )
        axes[1].plot(
            group["label_pct"], group["pr_auc"],
            "s-", label=method, linewidth=2,
        )

    axes[0].set_xlabel("Label %")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("F1 by Label Percentage")
    axes[0].legend()

    axes[1].set_xlabel("Label %")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("PR-AUC by Label Percentage")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_pseudo_label_analysis(
    error_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot pseudo-label error rate by group."""
    if error_df.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    error_df_sorted = error_df.sort_values(
        "error_rate", ascending=True
    )
    ax.barh(
        error_df_sorted.index.astype(str),
        error_df_sorted["error_rate"],
        color="#e74c3c", edgecolor="white",
    )
    ax.set_xlabel("Error Rate")
    ax.set_title("Pseudo-label Error Rate by Group")
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


# ===== Time Series =====

def plot_time_series_forecast(
    train: pd.Series,
    test: pd.Series,
    forecasts: Dict[str, pd.Series],
    title: str = "Monthly Cancellation Rate Forecast",
    save_path: Optional[str] = None,
) -> None:
    """Plot time series with forecasts overlay."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        train.index, train.values,
        "b-", label="Train", linewidth=2,
    )
    ax.plot(
        test.index, test.values,
        "k-", label="Test (actual)", linewidth=2,
    )

    colors = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    for (name, fc), color in zip(forecasts.items(), colors):
        ax.plot(
            fc.index, fc.values,
            "--", label=name, color=color, linewidth=1.5,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cancellation Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_residuals(
    actual: pd.Series,
    predicted: pd.Series,
    title: str = "Residual Analysis",
    save_path: Optional[str] = None,
) -> None:
    """Plot residuals: time plot + histogram."""
    common_idx = actual.index.intersection(predicted.index)
    residuals = actual.loc[common_idx] - predicted.loc[
        common_idx
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(residuals.index, residuals.values, "o-")
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_title(f"{title} - Over Time")
    axes[0].set_ylabel("Residual")

    axes[1].hist(
        residuals.values, bins=15, color="#3498db",
        edgecolor="white",
    )
    axes[1].set_title(f"{title} - Distribution")
    axes[1].set_xlabel("Residual")

    fig.tight_layout()
    _save(fig, save_path)
    plt.show()

"""
Semi-supervised learning module.

Implements self-training and label spreading for scenarios
with limited labeled data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.semi_supervised import (
    SelfTrainingClassifier,
    LabelSpreading,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    average_precision_score,
)


def mask_labels(
    y: np.ndarray,
    keep_ratio: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Mask labels to simulate limited labeling.

    Keeps keep_ratio fraction of labels, sets rest to -1
    (unlabeled).

    Parameters
    ----------
    y : np.ndarray
        True labels (0/1).
    keep_ratio : float
        Fraction of labels to keep (e.g., 0.10 = 10%).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Labels with -1 for unlabeled samples.
    """
    rng = np.random.RandomState(seed)
    y_masked = np.full_like(y, -1)
    n_keep = max(int(len(y) * keep_ratio), 2)

    # Stratified sampling
    for label in [0, 1]:
        idx = np.where(y == label)[0]
        n_label_keep = max(
            int(len(idx) * keep_ratio), 1
        )
        chosen = rng.choice(
            idx, size=n_label_keep, replace=False
        )
        y_masked[chosen] = label

    n_labeled = (y_masked != -1).sum()
    print(
        f"[Semi] Masked labels: kept {n_labeled} "
        f"({n_labeled / len(y) * 100:.1f}%), "
        f"unlabeled {len(y) - n_labeled}"
    )
    return y_masked


def train_self_training(
    X_train: np.ndarray,
    y_masked: np.ndarray,
    threshold: float = 0.85,
    seed: int = 42,
) -> Tuple[SelfTrainingClassifier, np.ndarray]:
    """
    Train a self-training classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix.
    y_masked : np.ndarray
        Labels with -1 for unlabeled.
    threshold : float
        Confidence threshold for pseudo-labeling.
    seed : int
        Random seed.

    Returns
    -------
    tuple of (fitted classifier, pseudo-labels)
    """
    base = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    st = SelfTrainingClassifier(
        estimator=base,
        threshold=threshold,
        verbose=False,
    )
    st.fit(X_train, y_masked)

    pseudo_labels = st.transduction_
    n_pseudo = (
        (y_masked == -1) & (pseudo_labels != -1)
    ).sum()
    print(
        f"[Semi] Self-training: pseudo-labeled {n_pseudo} "
        f"samples (threshold={threshold})"
    )
    return st, pseudo_labels


def train_label_spreading(
    X_train: np.ndarray,
    y_masked: np.ndarray,
    n_neighbors: int = 7,
    alpha: float = 0.2,
) -> Tuple[LabelSpreading, np.ndarray]:
    """
    Train a label spreading model.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix (will be subsampled if too large).
    y_masked : np.ndarray
        Labels with -1 for unlabeled.
    n_neighbors : int
        Number of neighbors for kNN graph.
    alpha : float
        Clamping factor.

    Returns
    -------
    tuple of (fitted model, transduced labels)
    """
    # LabelSpreading can be slow on large datasets
    # Subsample if needed
    max_samples = 20000
    if len(X_train) > max_samples:
        rng = np.random.RandomState(42)
        # Keep all labeled + sample unlabeled
        labeled_idx = np.where(y_masked != -1)[0]
        unlabeled_idx = np.where(y_masked == -1)[0]
        n_sample = min(
            max_samples - len(labeled_idx),
            len(unlabeled_idx),
        )
        sampled_unlabeled = rng.choice(
            unlabeled_idx, size=n_sample, replace=False,
        )
        idx = np.concatenate([labeled_idx, sampled_unlabeled])
        idx.sort()
        X_sub = X_train[idx]
        y_sub = y_masked[idx]
        print(
            f"[Semi] LabelSpreading: subsampled to "
            f"{len(idx):,} samples"
        )
    else:
        X_sub = X_train
        y_sub = y_masked

    ls = LabelSpreading(
        kernel="knn",
        n_neighbors=n_neighbors,
        alpha=alpha,
    )
    ls.fit(X_sub, y_sub)
    transduced = ls.transduction_
    print(
        f"[Semi] LabelSpreading: completed "
        f"(n_neighbors={n_neighbors}, alpha={alpha})"
    )
    return ls, transduced


def train_supervised_only(
    X_train: np.ndarray,
    y_masked: np.ndarray,
    seed: int = 42,
) -> RandomForestClassifier:
    """
    Train supervised model using ONLY labeled samples.

    This is the baseline for comparison.
    """
    labeled_mask = y_masked != -1
    X_labeled = X_train[labeled_mask]
    y_labeled = y_masked[labeled_mask]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_labeled, y_labeled)
    print(
        f"[Semi] Supervised-only: trained on "
        f"{len(X_labeled):,} labeled samples"
    )
    return model


def learning_curve_by_label_pct(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_ratios: Optional[List[float]] = None,
    threshold: float = 0.85,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate learning curves comparing supervised-only
    vs semi-supervised across different label percentages.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices.
    y_train, y_test : np.ndarray
        True labels.
    label_ratios : list of float
        Fractions of labels to keep.
    threshold : float
        Self-training confidence threshold.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Results with columns: label_pct, method, f1, pr_auc
    """
    if label_ratios is None:
        label_ratios = [0.05, 0.10, 0.20, 0.50, 1.00]

    results = []

    for ratio in label_ratios:
        print(f"\n=== Label ratio: {ratio*100:.0f}% ===")

        if ratio >= 1.0:
            # Full supervised
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight="balanced",
                random_state=seed, n_jobs=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            results.append({
                "label_pct": ratio * 100,
                "method": "Full Supervised",
                "f1": f1_score(y_test, y_pred),
                "pr_auc": average_precision_score(
                    y_test, y_proba
                ),
            })
            continue

        y_masked = mask_labels(y_train, ratio, seed)

        # 1. Supervised-only
        sup_model = train_supervised_only(
            X_train, y_masked, seed
        )
        y_pred_sup = sup_model.predict(X_test)
        y_proba_sup = sup_model.predict_proba(X_test)[:, 1]

        results.append({
            "label_pct": ratio * 100,
            "method": "Supervised-only",
            "f1": f1_score(y_test, y_pred_sup),
            "pr_auc": average_precision_score(
                y_test, y_proba_sup
            ),
        })

        # 2. Self-training
        st_model, _ = train_self_training(
            X_train, y_masked, threshold, seed
        )
        y_pred_st = st_model.predict(X_test)
        y_proba_st = st_model.predict_proba(X_test)[:, 1]

        results.append({
            "label_pct": ratio * 100,
            "method": "Self-Training",
            "f1": f1_score(y_test, y_pred_st),
            "pr_auc": average_precision_score(
                y_test, y_proba_st
            ),
        })

    results_df = pd.DataFrame(results)
    print("\n[Semi] === Learning Curve Results ===")
    print(results_df.to_string(index=False))
    return results_df


def analyze_pseudo_label_errors(
    y_true: np.ndarray,
    y_pseudo: np.ndarray,
    y_masked: np.ndarray,
    df_original: pd.DataFrame,
    group_col: str = "lead_time_bin",
) -> pd.DataFrame:
    """
    Analyze pseudo-label errors by group.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pseudo : np.ndarray
        Pseudo-labels from semi-supervised.
    y_masked : np.ndarray
        Original masked labels (-1 for unlabeled).
    df_original : pd.DataFrame
        Original dataframe with group_col.
    group_col : str
        Column to group errors by.

    Returns
    -------
    pd.DataFrame
        Error analysis by group.
    """
    # Only analyze samples that were unlabeled
    unlabeled_mask = y_masked == -1
    # Ensure we have matching lengths
    n = min(
        len(y_true), len(y_pseudo),
        len(y_masked), len(df_original),
    )
    y_true = y_true[:n]
    y_pseudo = y_pseudo[:n]
    y_masked = y_masked[:n]
    unlabeled_mask = unlabeled_mask[:n]

    if group_col not in df_original.columns:
        print(
            f"[Semi] Column '{group_col}' not found. "
            f"Skipping error analysis."
        )
        return pd.DataFrame()

    groups = df_original[group_col].values[:n]
    errors = y_true != y_pseudo

    analysis = pd.DataFrame({
        "group": groups[unlabeled_mask],
        "error": errors[unlabeled_mask],
    })

    result = analysis.groupby("group").agg(
        total=("error", "count"),
        n_errors=("error", "sum"),
    )
    result["error_rate"] = (
        result["n_errors"] / result["total"]
    ).round(4)
    result = result.sort_values(
        "error_rate", ascending=False
    )

    print("[Semi] Pseudo-label error analysis:")
    print(result.to_string())
    return result

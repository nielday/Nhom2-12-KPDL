"""
Evaluation metrics module.

Centralized metric computation for classification,
regression/time-series, and clustering tasks.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like, optional
        Predicted probabilities for positive class.

    Returns
    -------
    dict
        accuracy, precision, recall, f1, roc_auc, pr_auc.
    """
    metrics = {
        "accuracy": round(
            accuracy_score(y_true, y_pred), 4
        ),
        "precision": round(
            precision_score(y_true, y_pred, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_true, y_pred, zero_division=0), 4
        ),
        "f1": round(
            f1_score(y_true, y_pred, zero_division=0), 4
        ),
    }

    if y_proba is not None:
        metrics["roc_auc"] = round(
            roc_auc_score(y_true, y_proba), 4
        )
        metrics["pr_auc"] = round(
            average_precision_score(y_true, y_proba), 4
        )
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None,
) -> str:
    """Return formatted classification report."""
    if target_names is None:
        target_names = ["Not Canceled", "Canceled"]
    return classification_report(
        y_true, y_pred, target_names=target_names,
    )


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression / time-series metrics.

    Returns
    -------
    dict
        mae, rmse, smape.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # sMAPE
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    smape = float(
        np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100
    )

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "smape": round(smape, 2),
    }


def clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute clustering evaluation metrics.

    Returns
    -------
    dict
        silhouette, dbi, calinski_harabasz.
    """
    # Remove noise points (label == -1)
    mask = labels != -1
    if mask.sum() < 2:
        return {
            "silhouette": float("nan"),
            "dbi": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    X_eval = X[mask]
    labels_eval = labels[mask]

    n_unique = len(set(labels_eval))
    if n_unique < 2:
        return {
            "silhouette": float("nan"),
            "dbi": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    return {
        "silhouette": round(
            silhouette_score(X_eval, labels_eval), 4
        ),
        "dbi": round(
            davies_bouldin_score(X_eval, labels_eval), 4
        ),
        "calinski_harabasz": round(
            calinski_harabasz_score(X_eval, labels_eval), 2
        ),
    }

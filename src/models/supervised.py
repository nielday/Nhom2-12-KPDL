"""
Supervised classification module.

Implements baseline and improved models for hotel booking
cancellation prediction with training time logging.
"""

import time
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def split_data(
    df: pd.DataFrame,
    target_col: str = "is_canceled",
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series]:
    """
    Split data into train/test sets with stratification.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
    """
    # Verify no leakage
    leakage_cols = [
        "reservation_status", "reservation_status_date",
    ]
    for col in leakage_cols:
        if col in df.columns:
            raise ValueError(
                f"Leakage column '{col}' found in features! "
                f"Remove it before training."
            )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    print(
        f"[Supervised] Split: train={len(X_train):,}, "
        f"test={len(X_test):,} "
        f"(positive rate: "
        f"train={y_train.mean():.3f}, "
        f"test={y_test.mean():.3f})"
    )
    return X_train, X_test, y_train, y_test


def _build_models(seed: int = 42) -> Dict[str, Any]:
    """Build dictionary of models to train."""
    models = {
        "LogisticRegression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    return models


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_names: Optional[List[str]] = None,
    seed: int = 42,
    cv_folds: int = 5,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train multiple models and evaluate them.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_test : pd.Series
        Target vectors.
    model_names : list of str, optional
        Which models to train. None = all available.
    seed : int
        Random seed.
    cv_folds : int
        Number of CV folds.

    Returns
    -------
    tuple of (trained_models dict, results DataFrame)
        Results include metrics and training time.
    """
    all_models = _build_models(seed)

    if model_names is not None:
        models = {
            k: v for k, v in all_models.items()
            if k in model_names
        }
    else:
        models = all_models

    results = []
    trained = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")

        # Train with timing
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "train_time_seconds": round(train_time, 2),
        }

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(
                y_test, y_proba
            )
            metrics["pr_auc"] = average_precision_score(
                y_test, y_proba
            )
        else:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")

        results.append(metrics)
        trained[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        print(
            f"  Accuracy={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"Train time={train_time:.2f}s"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "pr_auc", ascending=False
    ).reset_index(drop=True)

    print("\n[Supervised] === Model Comparison ===")
    print(results_df.to_string(index=False))

    return trained, results_df


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Returns
    -------
    pd.DataFrame
        Top-N features sorted by importance.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    fi = fi.sort_values(
        "importance", ascending=False
    ).head(top_n).reset_index(drop=True)
    return fi


def save_model(
    model: Any,
    path: str = "outputs/models/best_model.joblib",
) -> None:
    """Save a trained model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[Supervised] Model saved to {path}")


def load_model(
    path: str = "outputs/models/best_model.joblib",
) -> Any:
    """Load a trained model from disk."""
    model = joblib.load(path)
    print(f"[Supervised] Model loaded from {path}")
    return model

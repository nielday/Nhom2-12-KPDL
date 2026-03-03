"""
Pipeline runner script.

Runs the complete data mining pipeline end-to-end:
Load -> Clean -> Features -> Mining -> Modeling -> Evaluation
"""

import sys
import os
import time
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.data.loader import load_raw_data, load_config
from src.data.cleaner import clean_pipeline, handle_missing
from src.data.cleaner import remove_leakage, remove_invalid
from src.features.builder import build_features
from src.mining.association import (
    discretize_for_association,
    create_transaction_matrix,
    run_apriori,
    filter_rules_by_target,
)
from src.mining.clustering import (
    prepare_clustering_features,
    find_optimal_k,
    run_kmeans,
    profile_clusters,
    evaluate_clustering,
)
from src.models.supervised import (
    split_data,
    train_and_evaluate,
    get_feature_importance,
    save_model,
)
from src.models.forecasting import (
    compute_monthly_cancel_rate,
    split_time_series,
    run_all_forecasts,
)
from src.evaluation.report import (
    save_results,
    generate_summary_report,
)


def main():
    """Run entire pipeline."""
    start = time.time()
    config = load_config()
    seed = config["seed"]

    print("=" * 60)
    print("  HOTEL BOOKING CANCELLATION - PIPELINE")
    print("=" * 60)

    # --- Step 1: Load Data ---
    print("\n[STEP 1] Loading data...")
    df_raw = load_raw_data()

    # --- Step 2: Clean (no encoding for mining) ---
    print("\n[STEP 2] Cleaning data...")
    df_clean = handle_missing(df_raw)
    df_clean = remove_leakage(
        df_clean, config.get("leakage_columns")
    )
    df_clean = remove_invalid(df_clean)

    # --- Step 3: Feature Engineering ---
    print("\n[STEP 3] Feature engineering...")
    df_feat = build_features(df_clean, add_date=True)

    # Save processed data
    processed_path = config["data"]["processed_path"]
    os.makedirs(
        os.path.dirname(processed_path), exist_ok=True
    )
    df_feat.to_csv(processed_path, index=False)
    print(f"  Saved processed data to {processed_path}")

    # --- Step 4: Association Rules ---
    print("\n[STEP 4] Association rule mining...")
    assoc_cfg = config.get("association", {})
    disc = discretize_for_association(
        df_feat,
        top_k_countries=assoc_cfg.get("top_k_countries", 10),
    )
    txn = create_transaction_matrix(disc)
    freq_items, rules, assoc_stats = run_apriori(
        txn,
        min_support=assoc_cfg.get("min_support", 0.05),
        min_confidence=assoc_cfg.get("min_confidence", 0.5),
        min_lift=assoc_cfg.get("min_lift", 1.2),
    )
    cancel_rules = filter_rules_by_target(rules)
    if not cancel_rules.empty:
        save_results(
            cancel_rules.head(20),
            "outputs/tables/association_rules",
        )

    # --- Step 5: Clustering ---
    print("\n[STEP 5] Clustering...")
    cluster_cfg = config.get("clustering", {})
    X_scaled, scaler, feat_cols = prepare_clustering_features(
        df_feat,
        feature_cols=cluster_cfg.get("features"),
    )
    k_results = find_optimal_k(
        X_scaled,
        k_range=cluster_cfg.get("k_range", [2, 3, 4, 5, 6]),
        seed=seed,
    )
    best_k = k_results["best_k"]
    labels, km_model = run_kmeans(X_scaled, best_k, seed)
    profiles = profile_clusters(
        df_feat, labels, feat_cols
    )
    cluster_eval = evaluate_clustering(X_scaled, labels)
    save_results(profiles, "outputs/tables/cluster_profiles")

    # --- Step 6: Classification ---
    print("\n[STEP 6] Classification...")
    # Encode for modeling
    from src.data.cleaner import encode_categoricals

    df_model = df_feat.copy()
    # Drop non-numeric / date columns
    drop_cols = [
        "arrival_date", "lead_time_bin", "season",
    ]
    for col in drop_cols:
        if col in df_model.columns:
            df_model = df_model.drop(columns=[col])
    df_model = encode_categoricals(df_model)

    cls_cfg = config.get("classification", {})
    X_train, X_test, y_train, y_test = split_data(
        df_model, test_size=config["test_size"], seed=seed,
    )
    trained_models, cls_results = train_and_evaluate(
        X_train, X_test, y_train, y_test, seed=seed,
        cv_folds=cls_cfg.get("cv_folds", 5),
    )
    save_results(cls_results, "outputs/tables/classification")

    # Save best model
    best_name = cls_results.iloc[0]["model"]
    best_model = trained_models[best_name]["model"]
    save_model(best_model, "outputs/models/best_model.joblib")

    # Feature importance
    fi = get_feature_importance(
        best_model, X_train.columns.tolist()
    )
    if not fi.empty:
        save_results(fi, "outputs/tables/feature_importance")

    # --- Step 7: Time Series ---
    print("\n[STEP 7] Time series forecasting...")
    ts_cfg = config.get("time_series", {})
    cancel_rate = compute_monthly_cancel_rate(df_feat)
    train_ts, test_ts = split_time_series(
        cancel_rate,
        train_ratio=ts_cfg.get("train_ratio", 0.8),
    )
    ts_results = run_all_forecasts(
        train_ts, test_ts,
        seasonal_periods=ts_cfg.get("seasonal_periods", 12),
    )
    save_results(ts_results, "outputs/tables/time_series")

    # --- Summary ---
    print("\n")
    generate_summary_report(
        classification_results=cls_results,
        clustering_metrics=cluster_eval,
        association_stats=assoc_stats,
        forecast_results=ts_results,
    )

    elapsed = time.time() - start
    print(f"\n[DONE] Pipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

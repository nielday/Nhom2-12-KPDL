"""
Association rule mining module.

Implements Apriori algorithm for finding booking attribute
combinations associated with cancellations.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def discretize_for_association(
    df: pd.DataFrame,
    top_k_countries: int = 10,
) -> pd.DataFrame:
    """
    Discretize continuous/high-cardinality columns for
    association rule mining.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with features already built.
    top_k_countries : int
        Number of top countries to keep.

    Returns
    -------
    pd.DataFrame
        Discretized dataframe suitable for transaction encoding.
    """
    disc = pd.DataFrame(index=df.index)

    # Hotel type
    if "hotel" in df.columns:
        disc["hotel"] = df["hotel"]

    # Lead time bins
    if "lead_time_bin" in df.columns:
        disc["lead_time_bin"] = df["lead_time_bin"].astype(str)
    elif "lead_time" in df.columns:
        bins = [0, 7, 30, 90, 180, float("inf")]
        labels = [
            "lt_0-7", "lt_7-30", "lt_30-90",
            "lt_90-180", "lt_180+",
        ]
        disc["lead_time_bin"] = pd.cut(
            df["lead_time"], bins=bins,
            labels=labels, include_lowest=True,
        ).astype(str)

    # Country -> top-k + Other
    if "country" in df.columns:
        top_countries = (
            df["country"]
            .value_counts()
            .head(top_k_countries)
            .index.tolist()
        )
        disc["country_group"] = df["country"].apply(
            lambda x: x if x in top_countries else "Other"
        )

    # Categorical columns to keep
    cat_cols = [
        "market_segment", "distribution_channel",
        "deposit_type", "customer_type", "meal",
    ]
    for col in cat_cols:
        if col in df.columns:
            disc[col] = df[col].astype(str)

    # Season
    if "season" in df.columns:
        disc["season"] = df["season"].astype(str)

    # Target
    if "is_canceled" in df.columns:
        disc["is_canceled"] = df["is_canceled"].map(
            {0: "not_canceled", 1: "canceled"}
        )

    # Weekend stay
    if "is_weekend_stay" in df.columns:
        disc["weekend_stay"] = df["is_weekend_stay"].map(
            {0: "no_weekend", 1: "has_weekend"}
        )

    # Deposit
    if "has_deposit" in df.columns:
        disc["has_deposit_flag"] = df["has_deposit"].map(
            {0: "no_deposit", 1: "has_deposit"}
        )

    print(
        f"[Association] Discretized {disc.shape[1]} columns "
        f"for {disc.shape[0]:,} transactions"
    )
    return disc


def create_transaction_matrix(
    disc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one-hot transaction matrix for Apriori.

    Parameters
    ----------
    disc_df : pd.DataFrame
        Discretized dataframe.

    Returns
    -------
    pd.DataFrame
        Boolean one-hot matrix.
    """
    dummies = pd.get_dummies(disc_df, prefix_sep="=")
    # Convert to boolean
    transaction_matrix = dummies.astype(bool)
    print(
        f"[Association] Transaction matrix: "
        f"{transaction_matrix.shape[0]:,} × "
        f"{transaction_matrix.shape[1]} items"
    )
    return transaction_matrix


def run_apriori(
    transaction_matrix: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.5,
    min_lift: float = 1.0,
    metric: str = "lift",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run Apriori algorithm and generate association rules.

    Parameters
    ----------
    transaction_matrix : pd.DataFrame
        Boolean one-hot transaction matrix.
    min_support : float
        Minimum support threshold.
    min_confidence : float
        Minimum confidence threshold.
    min_lift : float
        Minimum lift threshold.
    metric : str
        Metric for rule generation.

    Returns
    -------
    tuple of (frequent_itemsets, rules, stats)
        stats contains runtime and counts.
    """
    stats: Dict = {}

    # Run Apriori
    start_time = time.time()
    frequent_itemsets = apriori(
        transaction_matrix,
        min_support=min_support,
        use_colnames=True,
    )
    apriori_time = time.time() - start_time
    stats["apriori_runtime_seconds"] = round(apriori_time, 2)
    stats["n_frequent_itemsets"] = len(frequent_itemsets)

    print(
        f"[Association] Apriori found "
        f"{len(frequent_itemsets)} frequent itemsets "
        f"in {apriori_time:.2f}s"
    )

    # Generate rules
    if len(frequent_itemsets) == 0:
        print("[Association] No frequent itemsets found!")
        return frequent_itemsets, pd.DataFrame(), stats

    start_time = time.time()
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
        num_itemsets=len(frequent_itemsets),
    )
    rules_time = time.time() - start_time
    stats["rules_runtime_seconds"] = round(rules_time, 2)

    # Filter by lift
    rules = rules[rules["lift"] >= min_lift]
    stats["n_rules"] = len(rules)

    print(
        f"[Association] Generated {len(rules)} rules "
        f"(confidence≥{min_confidence}, lift≥{min_lift}) "
        f"in {rules_time:.2f}s"
    )

    return frequent_itemsets, rules, stats


def filter_rules_by_target(
    rules: pd.DataFrame,
    target_item: str = "is_canceled=canceled",
) -> pd.DataFrame:
    """
    Filter rules where consequent contains target item.

    Parameters
    ----------
    rules : pd.DataFrame
        Association rules.
    target_item : str
        Target item to filter in consequents.

    Returns
    -------
    pd.DataFrame
        Filtered rules sorted by lift descending.
    """
    if rules.empty:
        return rules

    mask = rules["consequents"].apply(
        lambda x: target_item in str(x)
    )
    filtered = rules[mask].sort_values(
        "lift", ascending=False
    )
    print(
        f"[Association] {len(filtered)} rules with "
        f"consequent '{target_item}'"
    )
    return filtered


def compare_rules_by_group(
    df: pd.DataFrame,
    group_col: str,
    top_k_countries: int = 10,
    min_support: float = 0.05,
    min_confidence: float = 0.5,
    min_lift: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """
    Compare association rules across groups (e.g., season,
    country_group).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with features built.
    group_col : str
        Column to group by (e.g., 'season').
    top_k_countries : int
        Top-k countries for discretization.
    min_support : float
        Minimum support.
    min_confidence : float
        Minimum confidence.
    min_lift : float
        Minimum lift.

    Returns
    -------
    dict
        Group name -> filtered rules DataFrame.
    """
    results = {}
    groups = df[group_col].unique()

    for group in groups:
        print(f"\n--- Group: {group_col}={group} ---")
        subset = df[df[group_col] == group]
        if len(subset) < 100:
            print(f"  Skipping (only {len(subset)} rows)")
            continue

        disc = discretize_for_association(
            subset, top_k_countries
        )
        txn = create_transaction_matrix(disc)
        _, rules, _ = run_apriori(
            txn, min_support, min_confidence, min_lift,
        )
        filtered = filter_rules_by_target(rules)
        results[str(group)] = filtered

    return results

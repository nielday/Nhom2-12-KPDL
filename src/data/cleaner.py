"""
Data cleaner module.

Handles missing values, outliers, leakage removal,
encoding, and full cleaning pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Strategy:
    - children: fill with 0 (no children)
    - country: fill with 'Unknown'
    - agent: fill with 0 (no agent)
    - company: drop column (94% missing) -> create has_company

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values handled.
    """
    df = df.copy()

    # children: 4 missing -> fill 0
    df["children"] = df["children"].fillna(0).astype(int)

    # country: 488 missing -> fill 'Unknown'
    df["country"] = df["country"].fillna("Unknown")

    # agent: 16340 missing -> fill 0 (no agent)
    df["agent"] = df["agent"].fillna(0).astype(int)

    # company: 112593 missing (94%) -> binary flag + drop
    df["has_company"] = (df["company"].notna()).astype(int)
    df = df.drop(columns=["company"])

    print(
        f"[Cleaner] Missing values handled. "
        f"Remaining NaN: {df.isnull().sum().sum()}"
    )
    return df


def remove_leakage(
    df: pd.DataFrame,
    leakage_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Remove columns that cause data leakage.

    These columns contain information only available AFTER
    the booking outcome is known, not at booking time.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    leakage_cols : list of str, optional
        Columns to remove. Defaults to reservation_status
        and reservation_status_date.

    Returns
    -------
    pd.DataFrame
        Dataframe without leakage columns.
    """
    if leakage_cols is None:
        leakage_cols = [
            "reservation_status",
            "reservation_status_date",
        ]

    df = df.copy()
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(
            f"[Cleaner] Removed leakage columns: {cols_to_drop}"
        )
    return df


def remove_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid/anomalous rows.

    Removes:
    - Rows where adr < 0 (negative rate)
    - Rows where adults + children + babies == 0 (no guests)
    - Duplicate rows

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()
    n_before = len(df)

    # Remove negative ADR
    df = df[df["adr"] >= 0]

    # Remove zero-guest bookings
    children_col = (
        df["children"] if "children" in df.columns
        else 0
    )
    babies_col = (
        df["babies"] if "babies" in df.columns
        else 0
    )
    total_guests = df["adults"] + children_col + babies_col
    df = df[total_guests > 0]

    # Remove duplicates
    n_before_dup = len(df)
    df = df.drop_duplicates()
    n_dups = n_before_dup - len(df)

    n_removed = n_before - len(df)
    print(
        f"[Cleaner] Removed {n_removed} invalid rows "
        f"({n_dups} duplicates). "
        f"Remaining: {len(df):,} rows"
    )
    return df.reset_index(drop=True)


def encode_categoricals(
    df: pd.DataFrame,
    label_encode_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Encode categorical variables.

    Uses LabelEncoding for specified columns (ordinal or
    high-cardinality). Other categoricals are left as-is
    for one-hot encoding during model training.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    label_encode_cols : list of str, optional
        Columns to label-encode.

    Returns
    -------
    pd.DataFrame
        Dataframe with encoded categoricals.
    """
    df = df.copy()

    if label_encode_cols is None:
        label_encode_cols = ["country", "agent"]

    for col in label_encode_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].astype("category").cat.codes

    # One-hot encode remaining object columns
    object_cols = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if object_cols:
        df = pd.get_dummies(
            df, columns=object_cols, drop_first=True,
            dtype=int,
        )
        print(
            f"[Cleaner] One-hot encoded: {object_cols}. "
            f"New shape: {df.shape}"
        )

    return df


def clean_pipeline(
    df: pd.DataFrame,
    leakage_cols: Optional[List[str]] = None,
    encode: bool = True,
) -> pd.DataFrame:
    """
    Run full cleaning pipeline in order.

    Steps:
    1. Handle missing values
    2. Remove leakage columns
    3. Remove invalid rows
    4. Encode categoricals (optional)

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.
    leakage_cols : list of str, optional
        Leakage columns to remove.
    encode : bool
        Whether to encode categoricals.

    Returns
    -------
    pd.DataFrame
        Fully cleaned dataframe.
    """
    print("[Cleaner] === Starting cleaning pipeline ===")
    print(f"[Cleaner] Input shape: {df.shape}")

    df = handle_missing(df)
    df = remove_leakage(df, leakage_cols)
    df = remove_invalid(df)

    if encode:
        df = encode_categoricals(df)

    print(f"[Cleaner] === Pipeline complete ===")
    print(f"[Cleaner] Output shape: {df.shape}")
    return df

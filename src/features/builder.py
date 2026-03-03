"""
Feature engineering module.

Creates new features from raw/cleaned data to improve
model performance and enable mining tasks.
"""

import pandas as pd
import numpy as np
from typing import Optional


# Month -> season mapping
MONTH_TO_SEASON = {
    "January": "Winter", "February": "Winter",
    "March": "Spring", "April": "Spring",
    "May": "Spring", "June": "Summer",
    "July": "Summer", "August": "Summer",
    "September": "Fall", "October": "Fall",
    "November": "Fall", "December": "Winter",
}

# Month name -> number for date parsing
MONTH_TO_NUM = {
    "January": 1, "February": 2, "March": 3,
    "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9,
    "October": 10, "November": 11, "December": 12,
}


def add_total_stays(df: pd.DataFrame) -> pd.DataFrame:
    """Add total number of nights stayed."""
    df = df.copy()
    df["total_stays"] = (
        df["stays_in_weekend_nights"]
        + df["stays_in_week_nights"]
    )
    return df


def add_total_guests(df: pd.DataFrame) -> pd.DataFrame:
    """Add total number of guests."""
    df = df.copy()
    children = df["children"] if "children" in df.columns else 0
    babies = df["babies"] if "babies" in df.columns else 0
    df["total_guests"] = df["adults"] + children + babies
    return df


def add_lead_time_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discretize lead_time into meaningful bins.

    Bins: [0-7] last-minute, [7-30] short, [30-90] medium,
    [90-180] long, [180+] very-long.
    """
    df = df.copy()
    bins = [0, 7, 30, 90, 180, float("inf")]
    labels = [
        "0-7_last_minute",
        "7-30_short",
        "30-90_medium",
        "90-180_long",
        "180+_very_long",
    ]
    df["lead_time_bin"] = pd.cut(
        df["lead_time"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    return df


def add_arrival_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse arrival date components into a datetime column."""
    df = df.copy()
    if "arrival_date_month" in df.columns:
        month_num = df["arrival_date_month"].map(MONTH_TO_NUM)
        df["arrival_date"] = pd.to_datetime(
            dict(
                year=df["arrival_date_year"],
                month=month_num,
                day=df["arrival_date_day_of_month"],
            ),
            errors="coerce",
        )
    return df


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    """Add season based on arrival month."""
    df = df.copy()
    if "arrival_date_month" in df.columns:
        df["season"] = df["arrival_date_month"].map(
            MONTH_TO_SEASON
        )
    return df


def add_is_local(df: pd.DataFrame) -> pd.DataFrame:
    """Flag if guest is from Portugal (local market)."""
    df = df.copy()
    if "country" in df.columns:
        df["is_local"] = (df["country"] == "PRT").astype(int)
    return df


def add_room_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """Flag if assigned room differs from reserved room."""
    df = df.copy()
    if (
        "reserved_room_type" in df.columns
        and "assigned_room_type" in df.columns
    ):
        df["room_mismatch"] = (
            df["reserved_room_type"]
            != df["assigned_room_type"]
        ).astype(int)
    return df


def add_has_deposit(df: pd.DataFrame) -> pd.DataFrame:
    """Flag if a deposit was made."""
    df = df.copy()
    if "deposit_type" in df.columns:
        df["has_deposit"] = (
            df["deposit_type"] != "No Deposit"
        ).astype(int)
    return df


def add_adr_per_person(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ADR per person (avoid division by zero)."""
    df = df.copy()
    if "total_guests" not in df.columns:
        df = add_total_guests(df)
    df["adr_per_person"] = df["adr"] / df[
        "total_guests"
    ].replace(0, 1)
    return df


def add_total_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate total booking cost = total_stays × adr."""
    df = df.copy()
    if "total_stays" not in df.columns:
        df = add_total_stays(df)
    df["total_cost"] = df["total_stays"] * df["adr"]
    return df


def add_is_weekend_stay(df: pd.DataFrame) -> pd.DataFrame:
    """Flag if booking includes weekend nights."""
    df = df.copy()
    df["is_weekend_stay"] = (
        df["stays_in_weekend_nights"] > 0
    ).astype(int)
    return df


def build_features(
    df: pd.DataFrame,
    add_date: bool = True,
) -> pd.DataFrame:
    """
    Run full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (after cleaner.clean_pipeline
        with encode=False).
    add_date : bool
        Whether to add parsed arrival_date column.

    Returns
    -------
    pd.DataFrame
        Dataframe with all new features added.
    """
    print("[Builder] === Starting feature engineering ===")
    n_cols_before = df.shape[1]

    df = add_total_stays(df)
    df = add_total_guests(df)
    df = add_lead_time_bins(df)
    df = add_season(df)
    df = add_is_local(df)
    df = add_room_mismatch(df)
    df = add_has_deposit(df)
    df = add_adr_per_person(df)
    df = add_total_cost(df)
    df = add_is_weekend_stay(df)

    if add_date:
        df = add_arrival_date(df)

    n_new = df.shape[1] - n_cols_before
    print(
        f"[Builder] Added {n_new} new features. "
        f"Total columns: {df.shape[1]}"
    )
    return df

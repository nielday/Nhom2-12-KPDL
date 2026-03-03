"""
Time series forecasting module.

Forecasts monthly hotel cancellation rate using
baseline, ARIMA, and Holt-Winters methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Month name -> number
MONTH_MAP = {
    "January": 1, "February": 2, "March": 3,
    "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9,
    "October": 10, "November": 11, "December": 12,
}


def compute_monthly_cancel_rate(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Compute monthly cancellation rate from booking data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with arrival date components and
        is_canceled column.

    Returns
    -------
    pd.Series
        Monthly cancellation rate indexed by datetime.
    """
    df_temp = df.copy()

    # Parse arrival month
    if "arrival_date_month" in df_temp.columns:
        df_temp["month_num"] = df_temp[
            "arrival_date_month"
        ].map(MONTH_MAP)
    elif "arrival_date" in df_temp.columns:
        df_temp["month_num"] = pd.to_datetime(
            df_temp["arrival_date"]
        ).dt.month
    else:
        raise ValueError("No month column found")

    # Create year-month
    df_temp["year_month"] = pd.to_datetime(
        df_temp["arrival_date_year"].astype(str) + "-"
        + df_temp["month_num"].astype(str) + "-01"
    )

    # Calculate cancel rate
    monthly = df_temp.groupby("year_month").agg(
        total=("is_canceled", "count"),
        canceled=("is_canceled", "sum"),
    )
    monthly["cancel_rate"] = monthly["canceled"] / monthly[
        "total"
    ]
    monthly = monthly.sort_index()

    series = monthly["cancel_rate"]
    series.index = pd.DatetimeIndex(series.index, freq="MS")

    print(
        f"[Forecast] Monthly cancel rate: "
        f"{len(series)} months "
        f"({series.index[0].strftime('%Y-%m')} ~ "
        f"{series.index[-1].strftime('%Y-%m')})"
    )
    return series


def split_time_series(
    series: pd.Series,
    train_ratio: float = 0.8,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split time series into train/test by time.

    Returns
    -------
    tuple of (train, test)
    """
    n = len(series)
    split_idx = int(n * train_ratio)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    print(
        f"[Forecast] Split: train={len(train)} months, "
        f"test={len(test)} months"
    )
    return train, test


def check_stationarity(series: pd.Series) -> Dict:
    """
    Check stationarity using Augmented Dickey-Fuller test.

    Returns
    -------
    dict with adf_stat, p_value, is_stationary
    """
    result = adfuller(series.dropna())
    output = {
        "adf_statistic": round(result[0], 4),
        "p_value": round(result[1], 4),
        "is_stationary": result[1] < 0.05,
    }
    print(
        f"[Forecast] ADF test: stat={output['adf_statistic']}, "
        f"p={output['p_value']}, "
        f"stationary={output['is_stationary']}"
    )
    return output


def forecast_naive(
    train: pd.Series,
    n_periods: int,
) -> pd.Series:
    """Naive forecast: repeat last value."""
    last_value = train.iloc[-1]
    index = pd.date_range(
        start=train.index[-1] + pd.DateOffset(months=1),
        periods=n_periods,
        freq="MS",
    )
    forecast = pd.Series(
        [last_value] * n_periods, index=index
    )
    return forecast


def forecast_moving_average(
    train: pd.Series,
    n_periods: int,
    window: int = 3,
) -> pd.Series:
    """Moving average forecast."""
    ma_value = train.iloc[-window:].mean()
    index = pd.date_range(
        start=train.index[-1] + pd.DateOffset(months=1),
        periods=n_periods,
        freq="MS",
    )
    forecast = pd.Series([ma_value] * n_periods, index=index)
    return forecast


def forecast_arima(
    train: pd.Series,
    n_periods: int,
    order: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[pd.Series, ARIMA]:
    """
    ARIMA forecast.

    Parameters
    ----------
    train : pd.Series
        Training data.
    n_periods : int
        Number of periods to forecast.
    order : tuple
        (p, d, q) order.

    Returns
    -------
    tuple of (forecast Series, fitted model)
    """
    model = ARIMA(train, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=n_periods)
    print(
        f"[Forecast] ARIMA{order}: AIC={fitted.aic:.2f}"
    )
    return forecast, fitted


def forecast_holt_winters(
    train: pd.Series,
    n_periods: int,
    seasonal_periods: int = 12,
) -> Tuple[pd.Series, ExponentialSmoothing]:
    """
    Holt-Winters Exponential Smoothing forecast.

    Parameters
    ----------
    train : pd.Series
        Training data.
    n_periods : int
        Number of periods to forecast.
    seasonal_periods : int
        Seasonal period (12 for monthly data).

    Returns
    -------
    tuple of (forecast Series, fitted model)
    """
    # If too few observations for seasonal, use simple
    if len(train) < 2 * seasonal_periods:
        print(
            f"[Forecast] Not enough data for seasonal "
            f"(need {2 * seasonal_periods}, have {len(train)}). "
            f"Using additive trend only."
        )
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal=None,
        )
    else:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
        )

    fitted = model.fit(optimized=True)
    forecast = fitted.forecast(steps=n_periods)
    return forecast, fitted


def evaluate_forecast(
    actual: pd.Series,
    predicted: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate forecast accuracy.

    Metrics: MAE, RMSE, sMAPE.

    Returns
    -------
    dict with mae, rmse, smape
    """
    # Align indices
    common_idx = actual.index.intersection(predicted.index)
    a = actual.loc[common_idx].values
    p = predicted.loc[common_idx].values

    mae = np.mean(np.abs(a - p))
    rmse = np.sqrt(np.mean((a - p) ** 2))

    # sMAPE
    denominator = np.abs(a) + np.abs(p)
    denominator = np.where(denominator == 0, 1, denominator)
    smape = np.mean(
        2.0 * np.abs(a - p) / denominator
    ) * 100

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "smape": round(smape, 2),
    }
    return metrics


def run_all_forecasts(
    train: pd.Series,
    test: pd.Series,
    seasonal_periods: int = 12,
) -> pd.DataFrame:
    """
    Run all forecast methods and compare.

    Returns
    -------
    pd.DataFrame
        Comparison table of all methods.
    """
    n_periods = len(test)
    results = []

    # 1. Naive
    fc_naive = forecast_naive(train, n_periods)
    m_naive = evaluate_forecast(test, fc_naive)
    m_naive["method"] = "Naive"
    results.append(m_naive)

    # 2. Moving Average
    fc_ma = forecast_moving_average(train, n_periods)
    m_ma = evaluate_forecast(test, fc_ma)
    m_ma["method"] = "Moving Average (3)"
    results.append(m_ma)

    # 3. ARIMA
    try:
        fc_arima, _ = forecast_arima(
            train, n_periods, order=(1, 1, 1)
        )
        m_arima = evaluate_forecast(test, fc_arima)
        m_arima["method"] = "ARIMA(1,1,1)"
        results.append(m_arima)
    except Exception as e:
        print(f"[Forecast] ARIMA failed: {e}")

    # 4. Holt-Winters
    try:
        fc_hw, _ = forecast_holt_winters(
            train, n_periods, seasonal_periods
        )
        m_hw = evaluate_forecast(test, fc_hw)
        m_hw["method"] = "Holt-Winters"
        results.append(m_hw)
    except Exception as e:
        print(f"[Forecast] Holt-Winters failed: {e}")

    results_df = pd.DataFrame(results)[
        ["method", "mae", "rmse", "smape"]
    ]
    results_df = results_df.sort_values("rmse")
    print("\n[Forecast] === Comparison ===")
    print(results_df.to_string(index=False))
    return results_df

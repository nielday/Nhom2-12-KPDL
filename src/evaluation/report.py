"""
Report generation module.

Creates comparison tables and saves results for
final evaluation and reporting.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def comparison_table(
    results_list: List[Dict],
) -> pd.DataFrame:
    """
    Create a comparison table from a list of result dicts.

    Parameters
    ----------
    results_list : list of dict
        Each dict should have 'model' key and metric keys.

    Returns
    -------
    pd.DataFrame
        Formatted comparison table.
    """
    df = pd.DataFrame(results_list)
    if "pr_auc" in df.columns:
        df = df.sort_values("pr_auc", ascending=False)
    return df.reset_index(drop=True)


def save_results(
    results: pd.DataFrame,
    path: str,
    fmt: str = "both",
) -> None:
    """
    Save results to CSV and/or JSON.

    Parameters
    ----------
    results : pd.DataFrame
        Results table.
    path : str
        Base path (without extension).
    fmt : str
        'csv', 'json', or 'both'.
    """
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    if fmt in ("csv", "both"):
        csv_path = base.with_suffix(".csv")
        results.to_csv(csv_path, index=False)
        print(f"[Report] Saved CSV: {csv_path}")

    if fmt in ("json", "both"):
        json_path = base.with_suffix(".json")
        results.to_json(
            json_path, orient="records", indent=2,
        )
        print(f"[Report] Saved JSON: {json_path}")


def generate_summary_report(
    classification_results: Optional[pd.DataFrame] = None,
    clustering_metrics: Optional[Dict] = None,
    association_stats: Optional[Dict] = None,
    semi_supervised_results: Optional[pd.DataFrame] = None,
    forecast_results: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate a text summary of all results.

    Returns
    -------
    str
        Formatted summary report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(
        "  HOTEL BOOKING CANCELLATION - RESULTS SUMMARY"
    )
    lines.append("=" * 60)

    if classification_results is not None:
        lines.append("\n--- Classification Results ---")
        lines.append(
            classification_results.to_string(index=False)
        )
        best = classification_results.iloc[0]
        lines.append(
            f"\nBest model: {best.get('model', 'N/A')} "
            f"(PR-AUC={best.get('pr_auc', 'N/A')})"
        )

    if clustering_metrics is not None:
        lines.append("\n--- Clustering Metrics ---")
        for k, v in clustering_metrics.items():
            lines.append(f"  {k}: {v}")

    if association_stats is not None:
        lines.append("\n--- Association Rules Stats ---")
        for k, v in association_stats.items():
            lines.append(f"  {k}: {v}")

    if semi_supervised_results is not None:
        lines.append("\n--- Semi-supervised Results ---")
        lines.append(
            semi_supervised_results.to_string(index=False)
        )

    if forecast_results is not None:
        lines.append("\n--- Time Series Forecast ---")
        lines.append(
            forecast_results.to_string(index=False)
        )

    lines.append("\n" + "=" * 60)
    report = "\n".join(lines)
    print(report)
    return report

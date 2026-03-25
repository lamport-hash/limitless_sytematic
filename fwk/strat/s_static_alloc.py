"""
Static Allocation Strategy.

A simple allocation strategy that maintains configurable static percentages
for each asset, with optional periodic rebalancing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from backtest.backtest_alloc_based import minutes_to_datetime


def compute_static_allocations(
    p_df: pd.DataFrame,
    p_asset_list: List[str],
    p_allocations: Dict[str, float],
    p_rebalance_months: int = 0,
) -> pd.DataFrame:
    """
    Compute static allocations for asset basket.

    Args:
        p_df: DataFrame with asset bundle (must have i_minute_i column)
        p_asset_list: List of asset symbols
        p_allocations: Dict mapping asset -> percentage (must sum to 100.0)
                       e.g., {"QQQ": 40.0, "TLT": 30.0, "GLD": 30.0}
        p_rebalance_months: Rebalancing frequency in months.
                            0 = never rebalance (set once at start)
                            N = rebalance on 1st of every Nth month

    Returns:
        DataFrame with A_{asset}_alloc columns added
    """
    total = sum(p_allocations.values())
    if abs(total - 100.0) > 0.01:
        raise ValueError(f"Allocations must sum to 100%, got {total}%")

    for asset in p_allocations:
        if asset not in p_asset_list:
            raise ValueError(f"Asset '{asset}' in allocations not found in asset list")

    df = p_df.copy()
    n_periods = len(df)
    n_assets = len(p_asset_list)

    target_alloc = np.zeros(n_assets)
    for i, asset in enumerate(p_asset_list):
        target_alloc[i] = p_allocations.get(asset, 0.0) / 100.0

    allocations = np.zeros((n_periods, n_assets))

    if "i_minute_i" not in df.columns:
        for i, asset in enumerate(p_asset_list):
            df[f"A_{asset}_alloc"] = target_alloc[i]
        return df

    dates = [minutes_to_datetime(int(m)) for m in df["i_minute_i"].values]

    if p_rebalance_months == 0:
        allocations[:] = target_alloc
    else:
        last_rebalance_month = -1
        last_rebalance_year = -1
        REBALANCE_EPSILON = 1e-9

        for t in range(n_periods):
            current_date = dates[t]
            current_month = current_date.month
            current_year = current_date.year
            current_day = current_date.day

            is_rebalance_date = False

            if last_rebalance_month == -1:
                is_rebalance_date = True
            elif (
                current_day <= 7
                and (
                    (current_year * 12 + current_month)
                    - (last_rebalance_year * 12 + last_rebalance_month)
                    >= p_rebalance_months
                )
            ):
                is_rebalance_date = True

            if is_rebalance_date:
                if t % 2 == 0:
                    allocations[t] = target_alloc + REBALANCE_EPSILON
                else:
                    allocations[t] = target_alloc - REBALANCE_EPSILON
                last_rebalance_month = current_month
                last_rebalance_year = current_year
            else:
                if t > 0:
                    allocations[t] = allocations[t - 1]
                else:
                    allocations[t] = target_alloc

    for i, asset in enumerate(p_asset_list):
        df[f"A_{asset}_alloc"] = allocations[:, i]

    df["A_strategy"] = "static_alloc"
    df["A_rebalance_months"] = p_rebalance_months

    return df

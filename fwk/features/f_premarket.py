"""
Pre-market High/Low Module

This module provides pre-market session extreme price features.
It calculates the highest high and lowest low during pre-market hours (4:00-9:30 ET).

Functions:
    - feature_premarket_high_low: Calculate pre-market high and low
"""

import numpy as np
import pandas as pd

from core.enums import g_high_col, g_low_col


def _get_time_from_index(p_df: pd.DataFrame) -> pd.Series:
    """Extract time from minute index."""
    base = pd.Timestamp("2000-01-01")
    dt_index = base + pd.to_timedelta(p_df["i_minute_i"], unit="m")
    return pd.Series(dt_index.dt.time, index=p_df.index, name="time")


def _is_premarket(p_df: pd.DataFrame) -> pd.Series:
    """Return True for bars within pre-market hours (4:00-9:30)."""
    t = _get_time_from_index(p_df)
    from datetime import time
    premarket_start = time(4, 0)
    market_open = time(9, 30)
    return (t >= premarket_start) & (t < market_open)


def feature_premarket_high_low(p_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pre-market session high and low.

    Pre-market is defined as 4:00 AM to 9:30 AM ET.

    Creates columns:
    - pm_high: Highest high during pre-market
    - pm_low: Lowest low during pre-market

    Args:
        p_df: DataFrame with OHLCV data

    Returns:
        DataFrame with added pre-market columns
    """
    df = p_df.copy()

    base = pd.Timestamp("2000-01-01")
    dt_index = base + pd.to_timedelta(df["i_minute_i"], unit="m")
    dates = pd.Series(dt_index.dt.date, index=df.index, name="date")

    pm_mask = _is_premarket(df)

    pm_data = df[pm_mask].copy()
    pm_data["date"] = dates[pm_mask]

    pm_stats = pm_data.groupby("date").agg(
        pm_high=(g_high_col, "max"),
        pm_low=(g_low_col, "min"),
    )

    pm_high_map = pm_stats["pm_high"].to_dict()
    pm_low_map = pm_stats["pm_low"].to_dict()

    df["pm_high"] = dates.map(pm_high_map)
    df["pm_low"] = dates.map(pm_low_map)

    return df

"""
Gap Percentage Module

This module provides gap-related features for intraday trading.
It calculates the overnight gap percentage and gap direction.

Functions:
    - feature_gap_pct: Calculate gap percentage
"""

import numpy as np
import pandas as pd
from typing import Optional

from core.enums import g_open_col, g_close_col


def _get_date_from_index(p_df: pd.DataFrame) -> pd.Series:
    """Extract date from minute index."""
    base = pd.Timestamp("2000-01-01")
    dt_index = base + pd.to_timedelta(p_df["i_minute_i"], unit="m")
    return pd.Series(dt_index.dt.date, index=p_df.index, name="date")


def _get_time_from_index(p_df: pd.DataFrame) -> pd.Series:
    """Extract time from minute index."""
    base = pd.Timestamp("2000-01-01")
    dt_index = base + pd.to_timedelta(p_df["i_minute_i"], unit="m")
    return pd.Series(dt_index.dt.time, index=p_df.index, name="time")


def feature_gap_pct(p_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overnight gap percentage.

    Gap % = (Open at 09:30 - Previous Close) / Previous Close * 100

    Creates columns:
    - prev_close: Previous day's closing price
    - gap_pct: Gap percentage (positive = gap up, negative = gap down)

    Args:
        p_df: DataFrame with OHLCV data

    Returns:
        DataFrame with added gap columns
    """
    df = p_df.copy()

    dates = _get_date_from_index(df)
    times = _get_time_from_index(df)

    from datetime import time
    market_open = time(9, 30)

    daily_close = df.groupby(dates)[g_close_col].last()

    prev_close = daily_close.shift(1)

    prev_close_map = {date: val for date, val in prev_close.items() if pd.notna(val)}
    df["prev_close"] = dates.map(prev_close_map)

    first_bar_mask = (times == market_open) & df["prev_close"].notna()

    gap_raw = pd.Series(np.nan, index=df.index)
    gap_raw.loc[first_bar_mask] = (
        (df.loc[first_bar_mask, g_open_col] - df.loc[first_bar_mask, "prev_close"])
        / df.loc[first_bar_mask, "prev_close"]
        * 100
    )

    df["gap_pct"] = gap_raw.groupby(dates).transform("first")

    return df

"""
Opening Range Module

This module provides Opening Range (OR) features for intraday trading.
It calculates the high, low, open, and close of the first N minutes after market open.

Functions:
    - feature_opening_range: Calculate OR for specified periods
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from core.enums import g_open_col, g_high_col, g_low_col, g_close_col


MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30


def _get_time_from_index(p_df: pd.DataFrame) -> pd.Series:
    """Extract time from minute index (minutes since 2000-01-01)."""
    base = pd.Timestamp("2000-01-01")
    dt_index = base + pd.to_timedelta(p_df["i_minute_i"], unit="m")
    return pd.Series(dt_index.dt.time, index=p_df.index, name="time")


def _is_market_hours(p_df: pd.DataFrame) -> pd.Series:
    """Return True for bars within market hours (9:30-16:00)."""
    t = _get_time_from_index(p_df)
    from datetime import time
    market_open = time(9, 30)
    market_close = time(16, 0)
    return (t >= market_open) & (t < market_close)


def feature_opening_range(
    p_df: pd.DataFrame,
    p_periods: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Calculate Opening Range (OR) features for specified periods.

    For each period N (e.g., 5, 15, 30 minutes), calculates:
    - OR{N}_open: First bar open after market open
    - OR{N}_high: Highest high in the first N minutes
    - OR{N}_low: Lowest low in the first N minutes
    - OR{N}_close: Last close in the first N minutes

    Args:
        p_df: DataFrame with OHLCV data and i_minute_i index
        p_periods: List of periods (default: [5, 15, 30])

    Returns:
        DataFrame with added OR columns
    """
    if p_periods is None:
        p_periods = [5, 15, 30]

    df = p_df.copy()
    result = pd.DataFrame(index=df.index)

    t = _get_time_from_index(df)
    from datetime import time
    market_open = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)

    first_bar_mask = pd.Series(False, index=df.index)
    for i in range(len(df) - 1):
        if t.iloc[i] == market_open and _is_market_hours(df).iloc[i]:
            first_bar_mask.iloc[i] = True

    date_series = (pd.Timestamp("2000-01-01") + pd.to_timedelta(df["i_minute_i"], unit="m")).dt.date

    for period in p_periods:
        end_minutes = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE + period
        end_hour, end_minute = divmod(end_minutes, 60)
        from datetime import time
        or_end = time(end_hour, end_minute)

        in_or_window = (t >= market_open) & (t <= or_end) & _is_market_hours(df)

        for date_val in date_series.unique():
            date_mask = date_series == date_val
            or_mask = date_mask & in_or_window

            if or_mask.sum() == 0:
                continue

            or_bars = df[or_mask]

            or_open_val = or_bars[g_open_col].iloc[0] if len(or_bars) > 0 else np.nan
            or_high_val = or_bars[g_high_col].max()
            or_low_val = or_bars[g_low_col].min()
            or_close_val = or_bars[g_close_col].iloc[-1] if len(or_bars) > 0 else np.nan

            result.loc[or_mask, f"or{period}_open"] = or_open_val
            result.loc[or_mask, f"or{period}_high"] = or_high_val
            result.loc[or_mask, f"or{period}_low"] = or_low_val
            result.loc[or_mask, f"or{period}_close"] = or_close_val

        result[f"or{period}_open"] = result[f"or{period}_open"].ffill()
        result[f"or{period}_high"] = result[f"or{period}_high"].ffill()
        result[f"or{period}_low"] = result[f"or{period}_low"].ffill()
        result[f"or{period}_close"] = result[f"or{period}_close"].ffill()

    for col in result.columns:
        df[col] = result[col]

    return df

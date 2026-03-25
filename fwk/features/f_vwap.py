"""
VWAP Module

This module provides Volume Weighted Average Price (VWAP) calculations.
It implements a daily-reset VWAP that recalculates at midnight each calendar day.

Functions:
    - feature_vwap: Calculate daily-reset VWAP
"""

import numpy as np
import pandas as pd

from core.enums import g_high_col, g_low_col, g_close_col, g_volume_col


def feature_vwap(p_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate intraday VWAP that resets at midnight each calendar day.

    VWAP = sum(typical_price * volume) / sum(volume)
    where typical_price = (High + Low + Close) / 3

    Creates columns:
    - vwap: Daily-reset VWAP

    Args:
        p_df: DataFrame with OHLCV data

    Returns:
        DataFrame with added vwap column
    """
    df = p_df.copy()

    base = pd.Timestamp("2000-01-01")
    dt_index = base + pd.to_timedelta(df["i_minute_i"], unit="m")
    dates = pd.Series(dt_index.dt.date, index=df.index, name="date")

    typical_price = (df[g_high_col] + df[g_low_col] + df[g_close_col]) / 3.0
    tpv = typical_price * df[g_volume_col]

    df["vwap"] = tpv.groupby(dates).cumsum() / df[g_volume_col].groupby(dates).cumsum()

    return df

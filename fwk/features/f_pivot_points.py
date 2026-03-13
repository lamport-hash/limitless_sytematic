"""
Pivot Points Detection Module

This module provides pivot point detection functions for identifying local highs and lows
in financial time series data.

Functions:
    - feature_pivot_points: Detects pivot highs and pivot lows using a rolling window
"""

import numpy as np
import pandas as pd
from typing import Tuple

from core.enums import g_high_col, g_low_col


def feature_pivot_points(
    p_df: pd.DataFrame,
    p_window: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect pivot highs and pivot lows in price data.
    
    A pivot-high is a high that is the maximum within ±window bars.
    A pivot-low is a low that is the minimum within ±window bars.
    
    Args:
        p_df (pd.DataFrame): DataFrame containing OHLC columns:
            - S_high_f32: High prices
            - S_low_f32: Low prices
        p_window (int): Number of bars on each side to compare (default: 3).
                        Total window size is (2 * p_window + 1).
    
    Returns:
        Tuple[pd.Series, pd.Series]: A tuple containing:
            - pivot_high: Series with 1 where pivot high detected, 0 otherwise
            - pivot_low: Series with 1 where pivot low detected, 0 otherwise
    
    Note:
        Pivot points are centered, meaning they detect patterns that have already
        formed. The pivot at position i is confirmed only after p_window additional
        bars have passed.
    """
    high = p_df[g_high_col]
    low = p_df[g_low_col]
    
    total_window = p_window * 2 + 1
    
    pivot_high = (
        high
        .rolling(total_window, center=True)
        .apply(lambda x: x[p_window] == x.max(), raw=True)
        .fillna(0)
        .astype(np.int8)
    )
    
    pivot_low = (
        low
        .rolling(total_window, center=True)
        .apply(lambda x: x[p_window] == x.min(), raw=True)
        .fillna(0)
        .astype(np.int8)
    )
    
    return pivot_high, pivot_low

"""
Bollinger Bands Module

This module provides Bollinger Bands technical indicator.
It calculates upper, middle, and lower bands based on standard deviation.

Functions:
    - feature_bollinger_bands: Calculate Bollinger Bands
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from core.enums import g_close_col
from features.feature_ta_utils import numba_sma, rolling_std


def feature_bollinger_bands(
    p_df: pd.DataFrame,
    p_period: int = 20,
    p_std_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands using numba-optimized functions.

    Creates columns:
    - bb_mid: Middle band (N-period SMA)
    - bb_std: Standard deviation
    - bb_upper: Upper band (mid + std_multiplier * std)
    - bb_lower: Lower band (mid - std_multiplier * std)

    Args:
        p_df: DataFrame with OHLCV data
        p_period: Period for moving average (default: 20)
        p_std_multiplier: Standard deviation multiplier (default: 2.0)

    Returns:
        DataFrame with added Bollinger Bands columns
    """
    df = p_df.copy()

    prices = df[g_close_col].to_numpy()

    df["bb_mid"] = numba_sma(prices, p_period)
    df["bb_std"] = rolling_std(prices, p_period)
    df["bb_upper"] = df["bb_mid"] + p_std_multiplier * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - p_std_multiplier * df["bb_std"]

    return df

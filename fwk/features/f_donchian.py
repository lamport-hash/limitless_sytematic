"""
Donchian Channel Module

This module provides Donchian Channel technical indicator.
It calculates upper and lower bands based on N-period high and low.

Functions:
    - feature_donchian_channel: Calculate Donchian Channel
"""

import numpy as np
import pandas as pd
from typing import Optional

from core.enums import g_high_col, g_low_col
from features.feature_ta_utils import numba_rolling_max, numba_rolling_min


def feature_donchian_channel(
    p_df: pd.DataFrame,
    p_period: int = 20,
) -> pd.DataFrame:
    """
    Calculate Donchian Channel using numba-optimized functions.

    Creates columns:
    - dc_upper: Upper band (N-period high)
    - dc_lower: Lower band (N-period low)
    - dc_mid: Middle band (average of upper and lower)

    Args:
        p_df: DataFrame with OHLCV data
        p_period: Period for channel calculation (default: 20)

    Returns:
        DataFrame with added Donchian Channel columns
    """
    df = p_df.copy()

    df["dc_upper"] = numba_rolling_max(df[g_high_col].to_numpy(), p_period)
    df["dc_lower"] = numba_rolling_min(df[g_low_col].to_numpy(), p_period)
    df["dc_mid"] = (df["dc_upper"] + df["dc_lower"]) / 2

    return df

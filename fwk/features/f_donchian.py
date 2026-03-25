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


def feature_donchian_channel(
    p_df: pd.DataFrame,
    p_period: int = 20,
) -> pd.DataFrame:
    """
    Calculate Donchian Channel.

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

    df["dc_upper"] = df[g_high_col].rolling(window=p_period).max()
    df["dc_lower"] = df[g_low_col].rolling(window=p_period).min()
    df["dc_mid"] = (df["dc_upper"] + df["dc_lower"]) / 2

    return df

"""
Volume Ratio Module

This module provides volume-based features including SMA and volume ratio.
It calculates rolling volume average and compares current volume to the average.

Functions:
    - feature_volume_ratio: Calculate volume SMA and ratio
"""

import numpy as np
import pandas as pd
from typing import Optional

from core.enums import g_volume_col


def feature_volume_ratio(
    p_df: pd.DataFrame,
    p_sma_period: int = 20,
) -> pd.DataFrame:
    """
    Calculate volume SMA and volume ratio.

    Creates columns:
    - vol_sma{N}: Rolling SMA of volume (N = period)
    - vol_ratio: Current volume / volume SMA

    Args:
        p_df: DataFrame with OHLCV data
        p_sma_period: Period for volume SMA (default: 20)

    Returns:
        DataFrame with added volume columns
    """
    df = p_df.copy()

    df[f"vol_sma{p_sma_period}"] = df[g_volume_col].rolling(window=p_sma_period).mean()

    df["vol_ratio"] = df[g_volume_col] / df[f"vol_sma{p_sma_period}"]
    df["vol_ratio"] = df["vol_ratio"].replace([np.inf, -np.inf], np.nan)

    return df

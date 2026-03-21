"""
CTO Line Feature Module

This module provides the CTO (Colored Trend Oscillator) Line indicator
which generates trading signals based on smoothed moving average crossovers.

Functions:
    - smma_numba: Calculates Smoothed Moving Average
    - feature_cto_line_signal: Detects CTO signals and generates long/short signals
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple

from core.enums import g_high_col, g_low_col, g_close_col


@njit
def smma_numba(p_src: np.ndarray, p_length: int) -> np.ndarray:
    """
    Calculates the Smoothed Moving Average (SMMA) of a source array.

    Args:
        p_src (np.ndarray): The source data.
        p_length (int): The period of the SMMA.

    Returns:
        np.ndarray: The SMMA values.
    """
    smma = np.zeros(len(p_src))
    for i in range(len(p_src)):
        if i == 0:
            smma[i] = p_src[0]
        else:
            smma[i] = (smma[i - 1] * (p_length - 1) + p_src[i]) / p_length
    return smma


def compute_cto_signals(
    p_high: np.ndarray,
    p_low: np.ndarray,
    p_params: Tuple[int, int, int, int] = (15, 19, 25, 29),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute CTO Line signals for a single asset.
    
    Args:
        p_high: High prices array
        p_low: Low prices array
        p_params: SMMA periods (v1, m1, m2, v2)
        
    Returns:
        Tuple of (long_signal, short_signal, p1, p2, v1, v2) arrays
        - long_signal: 1 where long pattern detected, 0 otherwise
        - short_signal: 1 where short pattern detected, 0 otherwise
        - p1: bullish conditions mask
        - p2: neutral conditions mask
        - v1: Fast SMMA line
        - v2: Slow SMMA line
    """
    hl2 = (p_high + p_low) / 2.0
    
    v1 = smma_numba(hl2, p_params[0])
    m1 = smma_numba(hl2, p_params[1])
    m2 = smma_numba(hl2, p_params[2])
    v2 = smma_numba(hl2, p_params[3])
    
    p2 = ((v1 < m1) != (v1 < v2)) | ((m2 < v2) != (v1 < v2))
    p3 = ~p2 & (v1 < v2)
    p1 = ~p2 & ~p3
    
    long_signal = p1.astype(np.int8)
    short_signal = p3.astype(np.int8)
    
    return long_signal, short_signal, p1, p2, v1, v2


def feature_cto_line_signal(
    p_df: pd.DataFrame,
    p_params: Tuple[int, int, int, int] = (15, 19, 25, 29),
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Detects CTO Line signals and generates trading signals.

    The CTO Line uses multiple SMMA crossovers to determine trend direction:
    - v1: Fast SMMA (param[0])
    - m1: Medium SMMA 1 (param[1])
    - m2: Medium SMMA 2 (param[2])
    - v2: Slow SMMA (param[3])

    Signal Logic:
    - Long (orange): ~(p2) & ~(v1 < v2) - bullish conditions
    - Short (navy): ~(p2) & (v1 < v2) - bearish conditions
    - Neutral (silver): conflicting conditions

    Args:
        p_df (pd.DataFrame): DataFrame containing OHLC columns.
        p_params (Tuple[int, int, int, int]): SMMA periods (v1, m1, m2, v2).

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]: A tuple containing:
            - long_signal: Series with 1 where long pattern detected, 0 otherwise
            - short_signal: Series with 1 where short pattern detected, 0 otherwise
            - v1_rel_dist: Relative distance (v1 - close) / close
            - v2_rel_dist: Relative distance (v2 - close) / close
    """
    high = p_df[g_high_col].to_numpy()
    low = p_df[g_low_col].to_numpy()
    close = p_df[g_close_col].to_numpy()

    long_signal, short_signal, _, _, v1, v2 = compute_cto_signals(high, low, p_params)

    v1_rel_dist = (v1 - close) / close
    v2_rel_dist = (v2 - close) / close

    return (
        pd.Series(long_signal, index=p_df.index),
        pd.Series(short_signal, index=p_df.index),
        pd.Series(v1_rel_dist, index=p_df.index),
        pd.Series(v2_rel_dist, index=p_df.index),
    )


def cto_line(p_data, p_param=(15, 19, 25, 29), isnum=False):
    high = p_data['High'].to_numpy()
    low = p_data['Low'].to_numpy()

    long_signal, short_signal, p1, p2, v1, v2 = compute_cto_signals(high, low, p_param)

    if isnum:
        color_orange, color_silver, color_navy = 1, 0, -1
    else:
        color_orange, color_silver, color_navy = 'orange', 'silver', 'navy'

    p_data['hl2'] = (high + low) / 2
    p_data['color'] = np.where(p1, color_orange, np.where(p2, color_silver, color_navy))
    p_data['Line 1'] = v1
    p_data['Line 2'] = v2

    return p_data
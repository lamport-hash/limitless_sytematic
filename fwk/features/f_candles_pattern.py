"""
Candlestick Pattern Module

This module provides candlestick pattern detection functions for financial time series data.
Pattern logic is implemented in separate functions that can be called by base_dataframe.py.

Functions:
    - feature_outside_bar_signal: Detects Outside Bar patterns and generates long/short signals
"""

import numpy as np
import pandas as pd
from typing import Tuple

from core.enums import g_open_col, g_high_col, g_low_col, g_close_col


def feature_outside_bar_signal(p_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Detects Outside Bar candlestick patterns and generates trading signals.
    
    An Outside Bar is a candle that:
    - Has a high greater than the previous candle's high
    - Has a low less than the previous candle's low
    
    This implementation detects two types of signals:
    
    Long Signal (bearish engulfing with close below prev low):
    - Current candle is bearish (open > close)
    - High > previous high
    - Low < previous low
    - Close < previous low
    
    Short Signal (bullish engulfing with close above prev high):
    - Current candle is bullish (open < close)
    - Low < previous low (NOTE: unusual condition, kept as-is from original)
    - High > previous high (NOTE: unusual condition, kept as-is from original)
    - Close > previous high
    
    Args:
        p_df (pd.DataFrame): DataFrame containing OHLC columns:
            - S_open_f32: Open prices
            - S_high_f32: High prices
            - S_low_f32: Low prices
            - S_close_f32: Close prices
    
    Returns:
        Tuple[pd.Series, pd.Series]: A tuple containing:
            - long_signal: Series with 1 where long pattern detected, 0 otherwise
            - short_signal: Series with 1 where short pattern detected, 0 otherwise
    """
    high = p_df[g_high_col]
    low = p_df[g_low_col]
    open_ = p_df[g_open_col]
    close = p_df[g_close_col]
    
    high_prev = high.shift(1)
    low_prev = low.shift(1)
    
    # Long signal conditions
    c0_long = open_ > close
    c1_long = high > high_prev
    c2_long = low < low_prev
    c3_long = close < low_prev
    
    # Short signal conditions
    c0_short = open_ < close
    c1_short = low < low_prev
    c2_short = high > high_prev
    c3_short = close > high_prev
    
    long_signal = (c0_long & c1_long & c2_long & c3_long).astype(np.int8)
    short_signal = (c0_short & c1_short & c2_short & c3_short).astype(np.int8)
    
    return long_signal, short_signal

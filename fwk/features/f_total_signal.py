"""
Total Signal Pattern Module

This module provides a 5-candle continuation/breakout pattern detection function.

Pattern Logic:
    Long Signal (2):
        - Low[4 bars ago] > High[current]: Price compressed below prior support
        - High[current] > Low[3 bars ago]: Current candle extends up
        - Low[3 bars ago] > Low[2 bars ago]: Ascending lows (3->2)
        - Low[2 bars ago] > Low[1 bar ago]: Ascending lows (2->1)
        - Close[current] > High[1 bar ago]: Bullish breakout
    
    Short Signal (1):
        - High[4 bars ago] < Low[current]: Price compressed above prior resistance
        - Low[current] < High[3 bars ago]: Current candle extends down
        - High[3 bars ago] < High[2 bars ago]: Descending highs (3->2)
        - High[2 bars ago] < High[1 bar ago]: Descending highs (2->1)
        - Close[current] < Low[1 bar ago]: Bearish breakout

Functions:
    - feature_total_signal: Detect pattern and return long/short signals
"""

import numpy as np
import pandas as pd
from typing import Tuple

from core.enums import g_open_col, g_high_col, g_low_col, g_close_col


def feature_total_signal(p_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Detect 5-candle continuation/breakout pattern and generate trading signals.
    
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
    close = p_df[g_close_col]
    
    long_c1 = low.shift(4) > high
    long_c2 = high > low.shift(3)
    long_c3 = low.shift(3) > low.shift(2)
    long_c4 = low.shift(2) > low.shift(1)
    long_c5 = close > high.shift(1)
    
    short_c1 = high.shift(4) < low
    short_c2 = low < high.shift(3)
    short_c3 = high.shift(3) < high.shift(2)
    short_c4 = high.shift(2) < high.shift(1)
    short_c5 = close < low.shift(1)
    
    long_signal = (long_c1 & long_c2 & long_c3 & long_c4 & long_c5).astype(np.int8)
    short_signal = (short_c1 & short_c2 & short_c3 & short_c4 & short_c5).astype(np.int8)
    
    return long_signal, short_signal

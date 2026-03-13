"""
Daily Signal Feature Module

This module provides daily signal calculation functions based on hourly candle data.
Signals are generated from candle direction (bullish/bearish) and aggregated by day.

Functions:
    - feature_daily_signal: Generates long/short signals based on candle direction
    - feature_daily_stop_price: Calculates stop prices based on daily high/low
    - feature_daily_exit: Generates exit signals based on delay parameter
"""

import numpy as np
import pandas as pd
from typing import Tuple

from core.enums import g_open_col, g_high_col, g_low_col, g_close_col, g_index_col


def feature_daily_signal(p_df: pd.DataFrame) -> pd.Series:
    """
    Generate daily signals based on candle direction.
    
    Signal values:
        - 2: Bullish candle (Close > Open)
        - 1: Bearish candle (Close < Open)
        - 0: Neutral (Close == Open)
    
    The signal is shifted by 1 to avoid look-ahead bias.
    
    Args:
        p_df (pd.DataFrame): DataFrame containing OHLC columns.
    
    Returns:
        pd.Series: Shifted daily signal values (0, 1, 2), with index matching p_df
    """
    close = p_df[g_close_col]
    open_ = p_df[g_open_col]
    
    signal = pd.Series(0, index=p_df.index, dtype=np.int8)
    
    bullish = close > open_
    bearish = close < open_
    
    signal[bullish] = 2
    signal[bearish] = 1
    
    result = signal.shift(1).fillna(0).astype(np.int8)
    result.index = p_df.index
    return result


def feature_daily_signal_with_exit(
    p_df: pd.DataFrame,
    p_test_candles: int = 8,
    p_exit_delay: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate daily signals with exit timing and stop prices.
    
    This function:
    1. Generates the daily signal based on candle direction
    2. Calculates stop prices based on the first N candles of each day
    3. Sets exit signals based on the delay parameter
    
    For LONG signals (signal=2):
        - StopPrice = Highest high in the first test_candles of the day
    
    For SHORT signals (signal=1):
        - StopPrice = Lowest low in the first test_candles of the day
    
    Args:
        p_df (pd.DataFrame): DataFrame with OHLC columns and datetime index
        p_test_candles (int): Number of candles to use for stop price calculation
        p_exit_delay (int): Number of days to delay before exit signal
    
    Returns:
        Tuple[pd.Series, pd.Series]: (stop_price, exit_signal)
    """
    df = p_df.copy()
    
    base = pd.Timestamp("2000-01-01")
    df["datetime"] = base + pd.to_timedelta(df[g_index_col].astype(int), unit="m")
    
    signal = feature_daily_signal(df)
    
    stop_price = pd.Series(np.nan, index=df.index, dtype=np.float64)
    exit_signal = pd.Series(0, index=df.index, dtype=np.int8)
    
    grouped = df.groupby(df["datetime"].dt.date)
    all_dates = list(grouped.groups.keys())
    
    for i, date in enumerate(all_dates):
        group = grouped.get_group(date)
        
        exit_date_index = i + p_exit_delay
        if exit_date_index < len(all_dates):
            exit_date = all_dates[exit_date_index]
            exit_group = grouped.get_group(exit_date)
            exit_signal.loc[exit_group.index[-1]] = 1
        
        group_idx = group.index
        
        if len(group_idx) > 0:
            first_signal_value = signal.loc[group_idx[0]]
            if first_signal_value == 2:
                if len(group) >= p_test_candles:
                    highest_in_test = group[g_high_col].iloc[:p_test_candles].max()
                    stop_price.loc[group_idx] = highest_in_test
            
            elif first_signal_value == 1:
                if len(group) >= p_test_candles:
                    lowest_in_test = group[g_low_col].iloc[:p_test_candles].min()
                    stop_price.loc[group_idx] = lowest_in_test
    
    return stop_price, exit_signal


def feature_pointpos(p_df: pd.DataFrame, p_signal_col: str = "F_daily_signal_f16") -> pd.Series:
    """
    Calculate point positions for signal visualization.
    
    For bullish signals (2): Point below the low
    For bearish signals (1): Point above the high
    
    Args:
        p_df (pd.DataFrame): DataFrame with OHLC and signal columns
        p_signal_col (str): Name of the signal column
    
    Returns:
        pd.Series: Point positions for chart visualization
    """
    low = p_df[g_low_col]
    high = p_df[g_high_col]
    signal = p_df[p_signal_col]
    
    pointpos = pd.Series(np.nan, index=p_df.index, dtype=np.float64)
    
    bullish_mask = signal == 2
    bearish_mask = signal == 1
    
    pointpos[bullish_mask] = low[bullish_mask] - 1e-4
    pointpos[bearish_mask] = high[bearish_mask] + 1e-4
    
    return pointpos

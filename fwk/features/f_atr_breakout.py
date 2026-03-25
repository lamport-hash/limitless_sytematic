"""
ATR Open Range Breakout Feature Module.

This module provides signal detection for ATR-based open range breakout strategy.

Logic:
1. Calculate daily ATR from previous day's HLC
2. Set breakout levels at today's Open ± x*ATR
3. Detect overnight gap for directional bias
4. Apply volatility filter (ATR > ATR_MA)
5. Generate long/short signals when price breaks levels

Functions:
    - feature_atr_breakout_signal: Main signal generator
    - compute_daily_atr: Compute daily ATR from intraday data
    - compute_session_levels: Compute open ± ATR levels
    - detect_overnight_gap: Detect gap vs previous close
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple

from core.enums import g_open_col, g_high_col, g_low_col, g_close_col, g_index_col


@njit
def compute_daily_ohlcv_numba(
    p_index: np.ndarray,
    p_open: np.ndarray,
    p_high: np.ndarray,
    p_low: np.ndarray,
    p_close: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample intraday bars to daily OHLCV using minute index.
    
    Returns:
        Tuple of (day_index, daily_open, daily_high, daily_low, daily_close, bar_of_day)
    """
    n = len(p_index)
    
    day_indices = []
    daily_opens = []
    daily_highs = []
    daily_lows = []
    daily_closes = []
    bar_of_days = []
    
    current_day = -1
    day_start_idx = 0
    current_high = 0.0
    current_low = 0.0
    
    for i in range(n):
        day_num = p_index[i] // 1440
        
        if day_num != current_day:
            if current_day != -1:
                day_indices.append(current_day)
                daily_opens.append(p_open[day_start_idx])
                daily_highs.append(current_high)
                daily_lows.append(current_low)
                daily_closes.append(p_close[i - 1])
                bar_of_days.append(i - day_start_idx)
            
            current_day = day_num
            day_start_idx = i
            current_high = p_high[i]
            current_low = p_low[i]
        
        if p_high[i] > current_high:
            current_high = p_high[i]
        if p_low[i] < current_low:
            current_low = p_low[i]
    
    if current_day != -1:
        day_indices.append(current_day)
        daily_opens.append(p_open[day_start_idx])
        daily_highs.append(current_high)
        daily_lows.append(current_low)
        daily_closes.append(p_close[n - 1])
        bar_of_days.append(n - day_start_idx)
    
    return (
        np.array(day_indices, dtype=np.int64),
        np.array(daily_opens, dtype=np.float32),
        np.array(daily_highs, dtype=np.float32),
        np.array(daily_lows, dtype=np.float32),
        np.array(daily_closes, dtype=np.float32),
        np.array(bar_of_days, dtype=np.int64),
    )


@njit
def compute_atr_numba(
    p_high: np.ndarray,
    p_low: np.ndarray,
    p_close: np.ndarray,
    p_period: int,
) -> np.ndarray:
    """
    Compute Average True Range using Wilder's smoothing.
    """
    n = len(p_high)
    atr = np.full(n, np.nan, dtype=np.float32)
    
    if n < p_period + 1:
        return atr
    
    tr = np.empty(n, dtype=np.float32)
    tr[0] = p_high[0] - p_low[0]
    
    for i in range(1, n):
        hl = p_high[i] - p_low[i]
        hc = abs(p_high[i] - p_close[i - 1])
        lc = abs(p_low[i] - p_close[i - 1])
        tr[i] = max(hl, hc, lc)
    
    atr_sum = 0.0
    for i in range(p_period):
        atr_sum += tr[i]
    atr[p_period] = atr_sum / p_period
    
    alpha = 1.0 / p_period
    for i in range(p_period + 1, n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    
    return atr


@njit
def compute_ema_numba(p_data: np.ndarray, p_period: int) -> np.ndarray:
    """Compute EMA using Wilder's smoothing. Handles NaN values."""
    n = len(p_data)
    ema = np.full(n, np.nan, dtype=np.float32)
    
    if n < p_period:
        return ema
    
    count = 0
    ema_sum = 0.0
    start_idx = 0
    
    for i in range(n):
        if np.isfinite(p_data[i]):
            ema_sum += p_data[i]
            count += 1
            if count == p_period:
                start_idx = i
                ema[i] = ema_sum / count
                break
        else:
            count = 0
            ema_sum = 0.0
    
    if count < p_period:
        return ema
    
    alpha = 2.0 / (p_period + 1)
    for i in range(start_idx + 1, n):
        if np.isfinite(p_data[i]):
            ema[i] = p_data[i] * alpha + ema[i - 1] * (1.0 - alpha)
        else:
            ema[i] = ema[i - 1]
    
    return ema


@njit
def map_daily_to_intraday_numba(
    p_intraday_index: np.ndarray,
    p_daily_index: np.ndarray,
    p_daily_values: np.ndarray,
) -> np.ndarray:
    """Map daily values to intraday bars (shifted by 1 day for lookback)."""
    n_intraday = len(p_intraday_index)
    n_daily = len(p_daily_index)
    result = np.full(n_intraday, np.nan, dtype=np.float32)
    
    daily_map = {}
    for i in range(n_daily):
        daily_map[p_daily_index[i]] = i
    
    for i in range(n_intraday):
        current_day = p_intraday_index[i] // 1440
        prev_day = current_day - 1
        
        if prev_day in daily_map:
            daily_idx = daily_map[prev_day]
            result[i] = p_daily_values[daily_idx]
    
    return result


@njit
def compute_session_info_numba(
    p_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute session information for each bar.
    
    Returns:
        Tuple of (is_first_bar, is_last_bar, bar_in_session)
    """
    n = len(p_index)
    is_first = np.zeros(n, dtype=np.int8)
    is_last = np.zeros(n, dtype=np.int8)
    bar_in_session = np.zeros(n, dtype=np.int64)
    
    current_day = -1
    day_start = 0
    
    for i in range(n):
        day_num = p_index[i] // 1440
        bar_in_session[i] = i - day_start
        
        if day_num != current_day:
            if current_day != -1 and day_start < i:
                is_last[day_start] = 0
                is_last[i - 1] = 1
            is_first[i] = 1
            current_day = day_num
            day_start = i
        else:
            is_first[i] = 0
    
    if n > 0:
        is_last[n - 1] = 1
    
    return is_first, is_last, bar_in_session


@njit
def compute_breakout_signals_numba(
    p_index: np.ndarray,
    p_open: np.ndarray,
    p_high: np.ndarray,
    p_low: np.ndarray,
    p_close: np.ndarray,
    p_day_indices: np.ndarray,
    p_daily_atr: np.ndarray,
    p_daily_open: np.ndarray,
    p_daily_close: np.ndarray,
    p_atr_ma: np.ndarray,
    p_atr_mult: float,
    p_use_gap_bias: bool,
    p_use_vol_filter: bool,
    p_gap_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute breakout signals with gap bias and volatility filter.
    
    Returns:
        Tuple of (long_signal, short_signal, upper_level, lower_level, vol_ok)
    """
    n = len(p_index)
    long_signal = np.zeros(n, dtype=np.int8)
    short_signal = np.zeros(n, dtype=np.int8)
    upper_level = np.full(n, np.nan, dtype=np.float32)
    lower_level = np.full(n, np.nan, dtype=np.float32)
    vol_ok = np.zeros(n, dtype=np.int8)
    
    n_daily = len(p_daily_atr)
    
    day_to_idx = {}
    for i in range(n_daily):
        day_to_idx[p_day_indices[i]] = i
    
    current_day = -1
    today_open = 0.0
    prev_close = 0.0
    today_atr = np.nan
    today_atr_ma = np.nan
    level_upper = np.nan
    level_lower = np.nan
    gap = 0.0
    day_has_signal = False
    
    for i in range(n):
        day_num = p_index[i] // 1440
        
        if day_num != current_day:
            current_day = day_num
            day_has_signal = False
            
            if i > 0:
                prev_close = p_close[i - 1]
            else:
                prev_close = p_close[i]
            
            today_open = p_open[i]
            
            prev_day = current_day - 1
            if prev_day in day_to_idx:
                daily_idx = day_to_idx[prev_day]
                today_atr = p_daily_atr[daily_idx]
                today_atr_ma = p_atr_ma[daily_idx]
            else:
                today_atr = np.nan
                today_atr_ma = np.nan
            
            if np.isfinite(today_atr) and today_atr > 0:
                level_upper = today_open + p_atr_mult * today_atr
                level_lower = today_open - p_atr_mult * today_atr
            else:
                level_upper = np.nan
                level_lower = np.nan
            
            if np.isfinite(prev_close) and prev_close > 0:
                gap = (today_open - prev_close) / prev_close
            else:
                gap = 0.0
        
        upper_level[i] = level_upper
        lower_level[i] = level_lower
        
        vol_pass = 1
        if p_use_vol_filter:
            if not np.isfinite(today_atr_ma) or today_atr <= today_atr_ma:
                vol_pass = 0
        vol_ok[i] = vol_pass
        
        if day_has_signal:
            continue
        
        if not np.isfinite(level_upper) or vol_pass == 0:
            continue
        
        hit_upper = p_high[i] >= level_upper
        hit_lower = p_low[i] <= level_lower
        
        if p_use_gap_bias:
            if gap > p_gap_threshold:
                if hit_upper:
                    long_signal[i] = 1
                    day_has_signal = True
            elif gap < -p_gap_threshold:
                if hit_lower:
                    short_signal[i] = 1
                    day_has_signal = True
            else:
                if hit_upper:
                    long_signal[i] = 1
                    day_has_signal = True
                elif hit_lower:
                    short_signal[i] = 1
                    day_has_signal = True
        else:
            if hit_upper:
                long_signal[i] = 1
                day_has_signal = True
            elif hit_lower:
                short_signal[i] = 1
                day_has_signal = True
    
    return long_signal, short_signal, upper_level, lower_level, vol_ok


def feature_atr_breakout_signal(
    p_df: pd.DataFrame,
    p_atr_len: int = 14,
    p_atr_mult: float = 0.5,
    p_vol_ma_len: int = 20,
    p_use_gap_bias: bool = True,
    p_use_vol_filter: bool = True,
    p_gap_threshold: float = 0.002,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute ATR Open Range Breakout signals.
    
    Args:
        p_df: DataFrame with OHLCV columns
        p_atr_len: ATR calculation period (default: 14)
        p_atr_mult: ATR multiplier for breakout levels (default: 0.5)
        p_vol_ma_len: Volatility MA period for filter (default: 20)
        p_use_gap_bias: Use overnight gap for directional bias (default: True)
        p_use_vol_filter: Only trade when ATR > ATR_MA (default: True)
        p_gap_threshold: Minimum gap % to trigger bias (default: 0.002 = 0.2%)
    
    Returns:
        Tuple of (long_signal, short_signal, upper_level, lower_level, atr, vol_ok, gap)
    """
    index = p_df[g_index_col].to_numpy()
    open_arr = p_df[g_open_col].to_numpy()
    high_arr = p_df[g_high_col].to_numpy()
    low_arr = p_df[g_low_col].to_numpy()
    close_arr = p_df[g_close_col].to_numpy()
    
    day_idx, daily_open, daily_high, daily_low, daily_close, _ = compute_daily_ohlcv_numba(
        index, open_arr, high_arr, low_arr, close_arr
    )
    
    daily_atr = compute_atr_numba(daily_high, daily_low, daily_close, p_atr_len)
    
    atr_ma = compute_ema_numba(daily_atr, p_vol_ma_len)
    
    long_sig, short_sig, upper, lower, vol_ok = compute_breakout_signals_numba(
        index, open_arr, high_arr, low_arr, close_arr,
        day_idx, daily_atr, daily_open, daily_close, atr_ma,
        p_atr_mult, p_use_gap_bias, p_use_vol_filter, p_gap_threshold
    )
    
    intraday_atr = map_daily_to_intraday_numba(index, day_idx, daily_atr)
    
    gap = np.full(len(p_df), np.nan, dtype=np.float32)
    current_day = -1
    prev_close = np.nan
    for i in range(len(p_df)):
        day_num = index[i] // 1440
        if day_num != current_day:
            if i > 0:
                prev_close = close_arr[i - 1]
            if np.isfinite(prev_close) and prev_close > 0:
                gap[i] = (open_arr[i] - prev_close) / prev_close
            current_day = day_num
    
    return (
        pd.Series(long_sig, index=p_df.index, name="long_signal"),
        pd.Series(short_sig, index=p_df.index, name="short_signal"),
        pd.Series(upper, index=p_df.index, name="upper_level"),
        pd.Series(lower, index=p_df.index, name="lower_level"),
        pd.Series(intraday_atr, index=p_df.index, name="atr"),
        pd.Series(vol_ok, index=p_df.index, name="vol_ok"),
        pd.Series(gap, index=p_df.index, name="gap"),
    )


def feature_session_info(p_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Compute session first/last bar flags.
    
    Returns:
        Tuple of (is_first_bar, is_last_bar)
    """
    index = p_df[g_index_col].to_numpy()
    is_first, is_last, _ = compute_session_info_numba(index)
    return (
        pd.Series(is_first, index=p_df.index, name="is_first_bar"),
        pd.Series(is_last, index=p_df.index, name="is_last_bar"),
    )

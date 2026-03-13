"""
Triangle Pattern Detection Module

This module provides ascending triangle pattern detection using pivot points
and linear regression analysis.

Functions:
    - feature_pivot_points_internal: Internal pivot detection for triangle calculation
    - _slope: Calculate slope and R² of best-fit line
    - _detect_triangle_at_idx: Detect triangle pattern at a specific index
    - feature_triangle_pattern: Main function to detect triangle patterns across the dataframe
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Any

from core.enums import g_high_col, g_low_col


def _feature_pivot_points_internal(
    p_df: pd.DataFrame,
    p_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal pivot detection returning numpy arrays for efficient triangle detection.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Boolean arrays for pivot_high and pivot_low
    """
    high = p_df[g_high_col].values
    low = p_df[g_low_col].values
    n = len(high)
    
    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)
    
    total_window = p_window * 2 + 1
    
    for i in range(p_window, n - p_window):
        window_highs = high[i - p_window:i + p_window + 1]
        window_lows = low[i - p_window:i + p_window + 1]
        
        if high[i] == np.max(window_highs):
            pivot_high[i] = True
        if low[i] == np.min(window_lows):
            pivot_low[i] = True
    
    return pivot_high, pivot_low


def _slope(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    """
    Return slope and R² of best-fit line through the points.
    
    Uses numpy polyfit for linear regression.
    
    Args:
        xs: X coordinates
        ys: Y coordinates
    
    Returns:
        Tuple[float, float]: (slope, r2_score)
    """
    if len(xs) < 2:
        return np.nan, np.nan
    
    coeffs = np.polyfit(xs, ys, deg=1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    y_pred = slope * xs + intercept
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    
    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)
    
    return float(slope), float(r2)


def _detect_triangle_at_idx(
    p_df: pd.DataFrame,
    p_idx: int,
    p_pivot_high: np.ndarray,
    p_pivot_low: np.ndarray,
    p_lookback: int = 30,
    p_pivot_window: int = 4,
    p_max_high_slope: float = 0.0004,
    p_min_low_slope: float = 0.001,
    p_min_r2: float = 0.70,
) -> Optional[Dict[str, Any]]:
    """
    Detect ascending triangle pattern at a specific index.
    
    Args:
        p_df: DataFrame with price data
        p_idx: Current index to check for pattern completion
        p_pivot_high: Boolean array of pivot highs
        p_pivot_low: Boolean array of pivot lows
        p_lookback: Number of bars to look back for pattern
        p_pivot_window: Window used for pivot detection
        p_max_high_slope: Maximum normalized slope for upper rail (horizontal or near-horizontal)
        p_min_low_slope: Minimum normalized slope for lower rail (ascending)
        p_min_r2: Minimum R² for both trend lines
    
    Returns:
        Optional[Dict]: Pattern details if detected, None otherwise
    """
    if p_idx < p_lookback or p_idx < p_pivot_window * 2 + 1:
        return None
    
    flag_start = p_idx - p_lookback
    last_confirmable = p_idx - p_pivot_window
    
    high_prices = p_df[g_high_col].values
    low_prices = p_df[g_low_col].values
    
    abs_hi_idx = np.where((np.arange(len(p_df)) >= flag_start) & 
                          (np.arange(len(p_df)) <= last_confirmable) & 
                          p_pivot_high)[0].tolist()
    abs_lo_idx = np.where((np.arange(len(p_df)) >= flag_start) & 
                          (np.arange(len(p_df)) <= last_confirmable) & 
                          p_pivot_low)[0].tolist()
    
    abs_hi_idx = abs_hi_idx[-4:]
    abs_lo_idx = abs_lo_idx[-4:]
    
    if (len(abs_hi_idx) < 2 or 
        len(abs_lo_idx) < 2 or 
        (len(abs_hi_idx) + len(abs_lo_idx)) < 5):
        return None
    
    xh = np.arange(len(abs_hi_idx))
    yh = high_prices[abs_hi_idx]
    xl = np.arange(len(abs_lo_idx))
    yl = low_prices[abs_lo_idx]
    
    slope_h_raw, r2_h = _slope(xh, yh)
    slope_l_raw, r2_l = _slope(xl, yl)
    
    if np.isnan(slope_h_raw) or np.isnan(slope_l_raw):
        return None
    
    slope_h = slope_h_raw / yh.mean()
    slope_l = slope_l_raw / yl.mean()
    
    if not (abs(slope_h) <= p_max_high_slope and
            slope_l >= p_min_low_slope and
            r2_h >= p_min_r2 and
            r2_l >= p_min_r2):
        return None
    
    highs_coords = [(int(i), float(high_prices[i])) for i in abs_hi_idx]
    lows_coords = [(int(i), float(low_prices[i])) for i in abs_lo_idx]
    
    return {
        "highs": highs_coords,
        "lows": lows_coords,
        "slope_high": float(slope_h),
        "slope_low": float(slope_l),
        "r2_high": float(r2_h),
        "r2_low": float(r2_l),
    }


def feature_triangle_pattern(
    p_df: pd.DataFrame,
    p_lookback: int = 30,
    p_pivot_window: int = 4,
    p_max_high_slope: float = 0.0004,
    p_min_low_slope: float = 0.001,
    p_min_r2: float = 0.70,
    p_signal_len: int = 3,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Detect ascending triangle patterns and generate trading signals.
    
    An ascending triangle is characterized by:
    - A horizontal or near-horizontal upper trendline (resistance)
    - An ascending lower trendline (support)
    - Converging price action that typically breaks out upward
    
    Args:
        p_df (pd.DataFrame): DataFrame containing OHLC columns
        p_lookback (int): Number of bars to look back for pattern (default: 30)
        p_pivot_window (int): Window for pivot detection (default: 4)
        p_max_high_slope (float): Max normalized slope for upper rail (default: 0.0004)
        p_min_low_slope (float): Min normalized slope for lower rail (default: 0.001)
        p_min_r2 (float): Minimum R² for trend line fit (default: 0.70)
        p_signal_len (int): Number of bars to mark as signal (breakout + following bars)
    
    Returns:
        Tuple of 5 Series:
            - triangle_signal: 1 during breakout + signal_len-1 following bars, 0 otherwise
            - triangle_highs: List of (index, price) tuples for pattern highs (or None)
            - triangle_lows: List of (index, price) tuples for pattern lows (or None)
            - triangle_slope_high: Normalized slope of upper rail
            - triangle_slope_low: Normalized slope of lower rail
    """
    n = len(p_df)
    
    pivot_high, pivot_low = _feature_pivot_points_internal(p_df, p_pivot_window)
    
    triangle_signal = np.zeros(n, dtype=np.int8)
    triangle_highs = np.empty(n, dtype=object)
    triangle_lows = np.empty(n, dtype=object)
    triangle_slope_high = np.full(n, np.nan, dtype=np.float32)
    triangle_slope_low = np.full(n, np.nan, dtype=np.float32)
    
    triangle_highs[:] = None
    triangle_lows[:] = None
    
    start_i = max(p_lookback, p_pivot_window * 2 + 1)
    
    detected_at = {}
    
    for i in range(start_i, n):
        pattern = _detect_triangle_at_idx(
            p_df, i, pivot_high, pivot_low,
            p_lookback=p_lookback,
            p_pivot_window=p_pivot_window,
            p_max_high_slope=p_max_high_slope,
            p_min_low_slope=p_min_low_slope,
            p_min_r2=p_min_r2,
        )
        
        if pattern is None:
            continue
        
        end_pos = min(i + p_signal_len, n)
        
        for j in range(i, end_pos):
            if j not in detected_at:
                detected_at[j] = i
        
        if i not in detected_at or detected_at[i] == i:
            triangle_highs[i] = pattern["highs"]
            triangle_lows[i] = pattern["lows"]
            triangle_slope_high[i] = pattern["slope_high"]
            triangle_slope_low[i] = pattern["slope_low"]
    
    for j in detected_at:
        triangle_signal[j] = 1
    
    return (
        pd.Series(triangle_signal, index=p_df.index),
        pd.Series(triangle_highs, index=p_df.index),
        pd.Series(triangle_lows, index=p_df.index),
        pd.Series(triangle_slope_high, index=p_df.index),
        pd.Series(triangle_slope_low, index=p_df.index),
    )

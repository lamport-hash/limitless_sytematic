import logging
import re
from pathlib import Path
from typing import Dict, Any, Union

import pandas as pd
import numpy as np
from ruamel.yaml import YAML

from core.enums import (
    g_close_col,
    g_open_col,
    g_high_col,
    g_low_col,
    g_volume_col,
    g_mid_col,
    g_mid2_col,
    g_close_time_col,
    g_open_time_col,
    g_index_col,
    g_qa_vol_col,
    g_nt_col,
    g_la_vol_col,
    g_lqa_vol_col,
    g_precision,
)

logger = logging.getLogger(__name__)

def compute_trailing_signals(df, 
    n=14, factor_up=2.0, factor_down=2.0, 
    profit_multiplier=1.5, max_number_candles=20):
    """
    Computes trailing stop signals (Long/Short) based on future price action simulation.
    
    This function introduces look-ahead bias and is intended for backtesting or labeling data.
    
    Parameters:
    -----------
    df : pd.DataFrame
    n : int
        Period for ATR calculation.
    factor_up : float
        Multiplier for Long Trailing Stop distance (ATR * factor).
    factor_down : float
        Multiplier for Short Trailing Stop distance (ATR * factor).
    profit_multiplier : float
        Minimum profit target multiplier relative to the stop distance.
    max_number_candles : int
        Maximum number of future candles to simulate the trade.
    
    Returns:
    --------
    pd.DataFrame
        The input DataFrame with added columns 'trailing_signal_up' and 'trailing_signal_down'.
    """
    
    # 1. Ensure required columns exist and are numeric
    required_cols = [g_open_col, g_high_col, g_low_col, g_close_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Create a copy to avoid modifying the original object unexpectedly (optional safety)
    result_df = df.copy()
    
    # 2. Calculate ATR (Average True Range)
    # High - Low
    tr1 = result_df[g_high_col] - result_df[g_low_col]
    # |High - Prev Close|
    tr2 = abs(result_df[g_high_col] - result_df[g_close_col].shift(1))
    # |Low - Prev Close|
    tr3 = abs(result_df[g_low_col] - result_df[g_close_col].shift(1))
    
    # True Range is max of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR calculation (Rolling Mean)
    atr = true_range.rolling(window=n).mean()
    
    # 3. Initialize Signal Columns with 0
    result_df['trailing_signal_up'] = 0
    result_df['trailing_signal_down'] = 0
    
    # Convert to numpy arrays for faster iteration inside the loop
    highs = result_df[g_high_col].values
    lows = result_df[g_low_col].values
    closes = result_df[g_close_col].values
    atrs = atr.values
    
    length = len(result_df)
    
    # We iterate up to (length - max_number_candles) because we need full future data 
    # for the simulation window. The last few candles cannot be fully validated.
    limit_index = length - max_number_candles
    
    for i in range(limit_index):
        if np.isnan(atrs[i]):
            continue
            
        current_atr = atrs[i]
        entry_price = closes[i]
        
        # --- Define Targets and Stops based on Entry ATR ---
        long_stop_dist = current_atr * factor_up
        short_stop_dist = current_atr * factor_down
        
        long_target = entry_price + (profit_multiplier * long_stop_dist)
        short_target = entry_price - (profit_multiplier * short_stop_dist)
        
        # Define the look-ahead window slice indices
        start_idx = i + 1
        end_idx = min(i + max_number_candles, length)
        
        # Extract future data for this specific simulation
        future_highs = highs[start_idx:end_idx]
        future_lows = lows[start_idx:end_idx]
        future_closes = closes[start_idx:end_idx]
        
        # --- Simulate LONG Strategy ---
        signal_up = 0
        current_trailing_stop_long = entry_price - long_stop_dist
        
        for j in range(len(future_highs)):
            high_j = future_highs[j]
            low_j = future_lows[j]
            
            # Check Stop Loss (Price drops below trailing stop)
            if low_j <= current_trailing_stop_long:
                break # Trade closed at loss
            
            # Update Trailing Stop (Follow the highest High since entry)
            # We compare against the max high seen so far in this trade window
            max_high_so_far = np.max(future_highs[:j+1])
            new_stop = max_high_so_far - long_stop_dist
            
            if new_stop > current_trailing_stop_long:
                current_trailing_stop_long = new_stop
                
            # Check Take Profit (Price reaches target)
            if high_j >= long_target or future_closes[j] >= long_target:
                signal_up = 1
                break
        
        result_df.at[i, 'trailing_signal_up'] = signal_up
        
        # --- Simulate SHORT Strategy ---
        signal_down = 0
        current_trailing_stop_short = entry_price + short_stop_dist
        
        for j in range(len(future_highs)):
            high_j = future_highs[j]
            low_j = future_lows[j]
            
            # Check Stop Loss (Price rises above trailing stop)
            if high_j >= current_trailing_stop_short:
                break # Trade closed at loss
            
            # Update Trailing Stop (Follow the lowest Low since entry)
            min_low_so_far = np.min(future_lows[:j+1])
            new_stop = min_low_so_far + short_stop_dist
            
            if new_stop < current_trailing_stop_short:
                current_trailing_stop_short = new_stop
                
            # Check Take Profit (Price drops to target)
            if low_j <= short_target or future_closes[j] <= short_target:
                signal_down = 1
                break
        
        result_df.at[i, 'trailing_signal_down'] = signal_down

    return result_df
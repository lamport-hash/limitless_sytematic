"""
Comprehensive Trend Identification Module for Financial Time Series

This module provides four different methods to identify trends ex post in historical
candlestick data:
1. SMA-based trend detection
2. Zigzag swing point detection
3. Linear regression trend detection
4. Directional consistency detection

Each function adds one or more target columns to the input dataframe and returns
the augmented dataframe along with trend information.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional, Tuple, Dict, List, Union
from core.enums import g_high_col, g_low_col, g_close_col

# Global default parameters (can be overridden per function call)
DEFAULT_LONG_WINDOW = 200
DEFAULT_THRESHOLD = 0.05
DEFAULT_MIN_LENGTH = 50
DEFAULT_MIN_INTENSITY = 0.20
DEFAULT_ZIGZAG_THRESHOLD = 0.05
DEFAULT_REGRESSION_WINDOW = 50
DEFAULT_SLOPE_THRESHOLD = 0.0005
DEFAULT_R2_THRESHOLD = 0.3
DEFAULT_DIRECTIONAL_LOOKBACK = 20
DEFAULT_MIN_SAME_DIRECTION = 0.7






def add_sma_trends(
    df: pd.DataFrame,
    long_window: int = DEFAULT_LONG_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    min_length: int = DEFAULT_MIN_LENGTH,
    min_intensity: float = DEFAULT_MIN_INTENSITY,
    use_volume: bool = False,
    volume_col: str = 'volume',
    prefix: str = 'sma'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify bull/bear trends using SMA-based detection.
    
    Adds columns:
    - target_{prefix}_regime: 1=bull, -1=bear, 0=neutral
    - target_{prefix}_trend_id: unique ID for each trend period
    - target_{prefix}_trend_intensity: total return over the trend
    - target_{prefix}_trend_length: length in candles
    - target_{prefix}_volume_ratio: (if use_volume) volume ratio during trend
    
    Returns:
    Tuple[DataFrame, Dict] - augmented dataframe and trend information dictionary
    """
    df = df.copy()
    
    # Compute long-term SMA
    df[f'{prefix}_sma_long'] = df[g_close_col].rolling(window=long_window, min_periods=long_window).mean()

    # Create initial trend signals based on price vs SMA +/- threshold
    df[f'{prefix}_bull_signal'] = df[g_close_col] > df[f'{prefix}_sma_long'] * (1 + threshold)
    df[f'{prefix}_bear_signal'] = df[g_close_col] < df[f'{prefix}_sma_long'] * (1 - threshold)

    # Identify sustained regimes
    df[f'{prefix}_regime'] = 0
    df.loc[df[f'{prefix}_bull_signal'], f'{prefix}_regime'] = 1
    df.loc[df[f'{prefix}_bear_signal'], f'{prefix}_regime'] = -1

    # Find contiguous blocks of same regime
    df[f'{prefix}_regime_change'] = (df[f'{prefix}_regime'] != df[f'{prefix}_regime'].shift()).cumsum()
    
    # Initialize target columns
    df[f'target_{prefix}_regime'] = 0
    df[f'target_{prefix}_trend_id'] = 0
    df[f'target_{prefix}_trend_intensity'] = 0.0
    df[f'target_{prefix}_trend_length'] = 0
    if use_volume:
        df[f'target_{prefix}_volume_ratio'] = 1.0

    trends = []
    trend_counter = 0

    for _, group in df.groupby(f'{prefix}_regime_change'):
        regime = group[f'{prefix}_regime'].iloc[0]
        if regime == 0:
            continue

        start_idx = group.index[0]
        end_idx = group.index[-1]
        length = len(group)

        if length < min_length:
            continue

        # Calculate total return over the period
        start_price = group[g_close_col].iloc[0]
        end_price = group[g_close_col].iloc[-1]
        total_return = (end_price - start_price) / start_price

        # For bull trends, return should be positive; for bear, negative
        if (regime == 1 and total_return < min_intensity) or \
           (regime == -1 and total_return > -min_intensity):
            continue

        trend_counter += 1
        trend_info = {
            'trend_id': trend_counter,
            'direction': regime,
            'start_date': df.index[start_idx] if isinstance(df.index, pd.DatetimeIndex) else start_idx,
            'end_date': df.index[end_idx] if isinstance(df.index, pd.DatetimeIndex) else end_idx,
            'length': length,
            'total_return': total_return,
            'intensity': total_return
        }

        # Fill target columns for this trend period
        df.loc[start_idx:end_idx, f'target_{prefix}_regime'] = regime
        df.loc[start_idx:end_idx, f'target_{prefix}_trend_id'] = trend_counter
        df.loc[start_idx:end_idx, f'target_{prefix}_trend_intensity'] = total_return
        df.loc[start_idx:end_idx, f'target_{prefix}_trend_length'] = length

        # Optional volume quality assessment
        if use_volume and volume_col in df.columns:
            vol_data = group[volume_col]
            avg_vol = vol_data.mean()
            overall_avg_vol = df[volume_col].mean()
            volume_ratio = avg_vol / overall_avg_vol if overall_avg_vol != 0 else 1.0
            trend_info['volume_ratio'] = volume_ratio
            trend_info['quality_score'] = volume_ratio * abs(total_return)
            df.loc[start_idx:end_idx, f'target_{prefix}_volume_ratio'] = volume_ratio

        trends.append(trend_info)

    # Clean up intermediate columns
    cols_to_drop = [f'{prefix}_sma_long', f'{prefix}_bull_signal', f'{prefix}_bear_signal',
                    f'{prefix}_regime_change']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df, {
        'trends': trends,
        'last_trend': df[f'target_{prefix}_regime'].iloc[-1] if len(df) > 0 else 0,
        'params': {
            'method': 'sma',
            'long_window': long_window,
            'threshold': threshold,
            'min_length': min_length,
            'min_intensity': min_intensity,
            'use_volume': use_volume
        }
    }


def add_zigzag_trends(
    df: pd.DataFrame,
    threshold: float = DEFAULT_ZIGZAG_THRESHOLD,
    min_length: int = DEFAULT_MIN_LENGTH,
    use_volume: bool = False,
    volume_col: str = 'volume',
    prefix: str = 'zigzag'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify trends using zigzag swing point detection.
    
    Adds columns:
    - target_{prefix}_regime: 1=up trend, -1=down trend, 0=neutral
    - target_{prefix}_trend_id: unique ID for each trend
    - target_{prefix}_trend_intensity: return over the trend
    - target_{prefix}_trend_length: length in candles
    - target_{prefix}_swing_point: 1=peak, -1=trough, 0=not a swing point
    - target_{prefix}_volume_ratio: (if use_volume) volume ratio during trend
    """
    df = df.copy()
    
    # Initialize target columns
    df[f'target_{prefix}_regime'] = 0
    df[f'target_{prefix}_trend_id'] = 0
    df[f'target_{prefix}_trend_intensity'] = 0.0
    df[f'target_{prefix}_trend_length'] = 0
    df[f'target_{prefix}_swing_point'] = 0
    if use_volume:
        df[f'target_{prefix}_volume_ratio'] = 1.0

    # Find swing points (simplified zigzag: keep only points where price change > threshold)
    swings = []
    last_swing_idx = 0
    last_swing_price = df[g_close_col].iloc[0]
    last_swing_type = None  # 'peak' or 'trough'
    
    for i in range(1, len(df)):
        change = (df[g_close_col].iloc[i] - last_swing_price) / last_swing_price
        
        if last_swing_type is None:
            # First swing: decide direction based on first move
            if abs(change) >= threshold:
                if change > 0:
                    swings.append({'idx': i, 'price': df[g_close_col].iloc[i], 'type': 'peak'})
                    last_swing_type = 'peak'
                    df.loc[df.index[i], f'target_{prefix}_swing_point'] = 1
                else:
                    swings.append({'idx': i, 'price': df[g_close_col].iloc[i], 'type': 'trough'})
                    last_swing_type = 'trough'
                    df.loc[df.index[i], f'target_{prefix}_swing_point'] = -1
                last_swing_price = df[g_close_col].iloc[i]
                last_swing_idx = i
        else:
            # Subsequent swings: alternate between peak and trough
            if last_swing_type == 'trough':
                if change >= threshold:
                    swings.append({'idx': i, 'price': df[g_close_col].iloc[i], 'type': 'peak'})
                    last_swing_type = 'peak'
                    df.loc[df.index[i], f'target_{prefix}_swing_point'] = 1
                    last_swing_price = df[g_close_col].iloc[i]
                    last_swing_idx = i
            else:  # last was peak
                if change <= -threshold:
                    swings.append({'idx': i, 'price': df[g_close_col].iloc[i], 'type': 'trough'})
                    last_swing_type = 'trough'
                    df.loc[df.index[i], f'target_{prefix}_swing_point'] = -1
                    last_swing_price = df[g_close_col].iloc[i]
                    last_swing_idx = i
    
    # Build trends from consecutive swings
    trends = []
    trend_counter = 0
    
    for i in range(len(swings) - 1):
        start = swings[i]
        end = swings[i+1]
        length = end['idx'] - start['idx']
        if length < min_length:
            continue
        
        direction = 1 if end['price'] > start['price'] else -1
        total_return = (end['price'] - start['price']) / start['price']
        
        trend_counter += 1
        trend_info = {
            'trend_id': trend_counter,
            'direction': direction,
            'start_date': df.index[start['idx']],
            'end_date': df.index[end['idx']],
            'length': length,
            'total_return': total_return,
            'intensity': total_return
        }
        
        # Fill target columns for this trend period
        df.loc[df.index[start['idx']]:df.index[end['idx']], f'target_{prefix}_regime'] = direction
        df.loc[df.index[start['idx']]:df.index[end['idx']], f'target_{prefix}_trend_id'] = trend_counter
        df.loc[df.index[start['idx']]:df.index[end['idx']], f'target_{prefix}_trend_intensity'] = total_return
        df.loc[df.index[start['idx']]:df.index[end['idx']], f'target_{prefix}_trend_length'] = length
        
        if use_volume and volume_col in df.columns:
            vol_slice = df[volume_col].iloc[start['idx']:end['idx']+1]
            avg_vol = vol_slice.mean()
            overall_avg_vol = df[volume_col].mean()
            volume_ratio = avg_vol / overall_avg_vol if overall_avg_vol else 1.0
            trend_info['volume_ratio'] = volume_ratio
            trend_info['quality_score'] = volume_ratio * abs(total_return)
            df.loc[df.index[start['idx']]:df.index[end['idx']], f'target_{prefix}_volume_ratio'] = volume_ratio
        
        trends.append(trend_info)
    
    return df, {
        'trends': trends,
        'last_trend': df[f'target_{prefix}_regime'].iloc[-1] if len(df) > 0 else 0,
        'params': {
            'method': 'zigzag',
            'threshold': threshold,
            'min_length': min_length,
            'use_volume': use_volume
        }
    }


def add_regression_trends(
    df: pd.DataFrame,
    window: int = DEFAULT_REGRESSION_WINDOW,
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD,
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
    min_length: int = DEFAULT_MIN_LENGTH,
    use_volume: bool = False,
    volume_col: str = 'volume',
    prefix: str = 'regression'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify trends using rolling linear regression.
    
    Adds columns:
    - target_{prefix}_regime: 1=bull, -1=bear, 0=neutral
    - target_{prefix}_trend_id: unique ID for each trend
    - target_{prefix}_trend_intensity: return over the trend
    - target_{prefix}_trend_length: length in candles
    - target_{prefix}_slope: regression slope at each point
    - target_{prefix}_r2: R² value at each point
    - target_{prefix}_volume_ratio: (if use_volume) volume ratio during trend
    """
    from sklearn.linear_model import LinearRegression
    
    df = df.copy()
    prices = df[g_close_col].values
    x = np.arange(window).reshape(-1, 1)
    
    # Compute rolling slope and R²
    slopes = np.full(len(df), np.nan)
    r2s = np.full(len(df), np.nan)
    
    for i in range(window-1, len(df)):
        y = prices[i-window+1:i+1]
        model = LinearRegression().fit(x, y)
        slopes[i] = model.coef_[0]
        # R²
        y_pred = model.predict(x)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2s[i] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    df[f'{prefix}_slope'] = slopes
    df[f'{prefix}_r2'] = r2s
    
    # Trend signal: slope > threshold and R² high
    df[f'{prefix}_bull'] = (df[f'{prefix}_slope'] > slope_threshold) & (df[f'{prefix}_r2'] > r2_threshold)
    df[f'{prefix}_bear'] = (df[f'{prefix}_slope'] < -slope_threshold) & (df[f'{prefix}_r2'] > r2_threshold)
    
    # Regime and grouping
    df[f'{prefix}_regime'] = 0
    df.loc[df[f'{prefix}_bull'], f'{prefix}_regime'] = 1
    df.loc[df[f'{prefix}_bear'], f'{prefix}_regime'] = -1
    
    df[f'{prefix}_regime_change'] = (df[f'{prefix}_regime'] != df[f'{prefix}_regime'].shift()).cumsum()
    
    # Initialize target columns
    df[f'target_{prefix}_regime'] = 0
    df[f'target_{prefix}_trend_id'] = 0
    df[f'target_{prefix}_trend_intensity'] = 0.0
    df[f'target_{prefix}_trend_length'] = 0
    df[f'target_{prefix}_slope'] = slopes
    df[f'target_{prefix}_r2'] = r2s
    if use_volume:
        df[f'target_{prefix}_volume_ratio'] = 1.0
    
    trends = []
    trend_counter = 0
    
    for _, group in df.groupby(f'{prefix}_regime_change'):
        regime = group[f'{prefix}_regime'].iloc[0]
        if regime == 0:
            continue
        if len(group) < min_length:
            continue
        
        start_idx = group.index[0]
        end_idx = group.index[-1]
        start_price = group[g_close_col].iloc[0]
        end_price = group[g_close_col].iloc[-1]
        total_return = (end_price - start_price) / start_price
        
        # For bull/bear, return should be consistent with regime
        if (regime == 1 and total_return < 0) or (regime == -1 and total_return > 0):
            continue
        
        trend_counter += 1
        trend_info = {
            'trend_id': trend_counter,
            'direction': regime,
            'start_date': start_idx,
            'end_date': end_idx,
            'length': len(group),
            'total_return': total_return,
            'intensity': total_return,
            'avg_slope': group[f'{prefix}_slope'].mean(),
            'avg_r2': group[f'{prefix}_r2'].mean()
        }
        
        # Fill target columns
        df.loc[start_idx:end_idx, f'target_{prefix}_regime'] = regime
        df.loc[start_idx:end_idx, f'target_{prefix}_trend_id'] = trend_counter
        df.loc[start_idx:end_idx, f'target_{prefix}_trend_intensity'] = total_return
        df.loc[start_idx:end_idx, f'target_{prefix}_trend_length'] = len(group)
        
        if use_volume and volume_col in df.columns:
            vol_slice = df[volume_col].loc[start_idx:end_idx]
            avg_vol = vol_slice.mean()
            overall_avg_vol = df[volume_col].mean()
            volume_ratio = avg_vol / overall_avg_vol if overall_avg_vol else 1.0
            trend_info['volume_ratio'] = volume_ratio
            trend_info['quality_score'] = volume_ratio * abs(total_return)
            df.loc[start_idx:end_idx, f'target_{prefix}_volume_ratio'] = volume_ratio
        
        trends.append(trend_info)
    
    # Clean up intermediate columns
    cols_to_drop = [f'{prefix}_bull', f'{prefix}_bear', f'{prefix}_regime_change']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    return df, {
        'trends': trends,
        'last_trend': df[f'target_{prefix}_regime'].iloc[-1] if len(df) > 0 else 0,
        'params': {
            'method': 'regression',
            'window': window,
            'slope_threshold': slope_threshold,
            'r2_threshold': r2_threshold,
            'min_length': min_length,
            'use_volume': use_volume
        }
    }


def add_directional_trends(
    df: pd.DataFrame,
    lookback: int = DEFAULT_DIRECTIONAL_LOOKBACK,
    min_same_direction: float = DEFAULT_MIN_SAME_DIRECTION,
    min_length: int = DEFAULT_MIN_LENGTH,
    use_mid: bool = True,
    use_volume: bool = False,
    volume_col: str = 'volume',
    prefix: str = 'directional'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify trends based on directional consistency.
    
    Adds columns:
    - target_{prefix}_regime: 1=up trend, -1=down trend, 0=neutral
    - target_{prefix}_trend_id: unique ID for each trend
    - target_{prefix}_trend_intensity: return over the trend
    - target_{prefix}_trend_length: length in candles
    - target_{prefix}_up_frac: fraction of up candles in rolling window
    - target_{prefix}_down_frac: fraction of down candles in rolling window
    - target_{prefix}_volume_ratio: (if use_volume) volume ratio during trend
    """
    df = df.copy()
    
    price = df[g_close_col] if not use_mid else (df[g_high_col] + df[g_low_col]) / 2
    df[f'{prefix}_price'] = price
    
    # Direction of each candle compared to previous
    df[f'{prefix}_up'] = (df[f'{prefix}_price'] > df[f'{prefix}_price'].shift(1)).astype(int)
    df[f'{prefix}_down'] = (df[f'{prefix}_price'] < df[f'{prefix}_price'].shift(1)).astype(int)
    
    # Rolling fraction of ups/downs
    df[f'{prefix}_up_frac'] = df[f'{prefix}_up'].rolling(window=lookback, min_periods=lookback).mean()
    df[f'{prefix}_down_frac'] = df[f'{prefix}_down'].rolling(window=lookback, min_periods=lookback).mean()
    
    # Trend signal
    df[f'{prefix}_bull'] = df[f'{prefix}_up_frac'] >= min_same_direction
    df[f'{prefix}_bear'] = df[f'{prefix}_down_frac'] >= min_same_direction
    
    # Regime and grouping
    df[f'{prefix}_regime'] = 0
    df.loc[df[f'{prefix}_bull'], f'{prefix}_regime'] = 1
    df.loc[df[f'{prefix}_bear'], f'{prefix}_regime'] = -1
    
    df[f'{prefix}_regime_change'] = (df[f'{prefix}_regime'] != df[f'{prefix}_regime'].shift()).cumsum()
    
    # Initialize target columns
    df[f'target_{prefix}_regime'] = 0
    df[f'target_{prefix}_trend_id'] = 0
    df[f'target_{prefix}_trend_intensity'] = 0.0
    df[f'target_{prefix}_trend_length'] = 0
    df[f'target_{prefix}_up_frac'] = df[f'{prefix}_up_frac']
    df[f'target_{prefix}_down_frac'] = df[f'{prefix}_down_frac']
    if use_volume:
        df[f'target_{prefix}_volume_ratio'] = 1.0
    
    trends = []
    trend_counter = 0
    
    for _, group in df.groupby(f'{prefix}_regime_change'):
        regime = group[f'{prefix}_regime'].iloc[0]
        if regime == 0:
            continue
        if len(group) < min_length:
            continue
        
        start_idx = group.index[0]
        end_idx = group.index[-1]
        start_price = group[f'{prefix}_price'].iloc[0]
        end_price = group[f'{prefix}_price'].iloc[-1]
        total_return = (end_price - start_price) / start_price
        
        if (regime == 1 and total_return < 0) or (regime == -1 and total_return > 0):
            continue
        
        trend_counter += 1
        trend_info = {
            'trend_id': trend_counter,
            'direction': regime,
            'start_date': start_idx,
            'end_date': end_idx,
            'length': len(group),
            'total_return': total_return,
            'intensity': total_return,
            'avg_up_frac': group[f'{prefix}_up_frac'].mean(),
            'avg_down_frac': group[f'{prefix}_down_frac'].mean()
        }
        
        # Fill target columns
        df.loc[start_idx:end_idx, f'target_{prefix}_regime'] = regime
        df.loc[start_idx:end_idx, f'target_{prefix}_trend_id'] = trend_counter
        df.loc[start_idx:end_idx, f'target_{prefix}_trend_intensity'] = total_return
        df.loc[start_idx:end_idx, f'target_{prefix}_trend_length'] = len(group)
        
        if use_volume and volume_col in df.columns:
            vol_slice = df[volume_col].loc[start_idx:end_idx]
            avg_vol = vol_slice.mean()
            overall_avg_vol = df[volume_col].mean()
            volume_ratio = avg_vol / overall_avg_vol if overall_avg_vol else 1.0
            trend_info['volume_ratio'] = volume_ratio
            trend_info['quality_score'] = volume_ratio * abs(total_return)
            df.loc[start_idx:end_idx, f'target_{prefix}_volume_ratio'] = volume_ratio
        
        trends.append(trend_info)
    
    # Clean up intermediate columns
    cols_to_drop = [f'{prefix}_price', f'{prefix}_up', f'{prefix}_down',
                    f'{prefix}_bull', f'{prefix}_bear', f'{prefix}_regime_change']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    return df, {
        'trends': trends,
        'last_trend': df[f'target_{prefix}_regime'].iloc[-1] if len(df) > 0 else 0,
        'params': {
            'method': 'directional',
            'lookback': lookback,
            'min_same_direction': min_same_direction,
            'min_length': min_length,
            'use_mid': use_mid,
            'use_volume': use_volume
        }
    }


def add_all_trends(
    df: pd.DataFrame,
    sma_params: Optional[Dict] = None,
    zigzag_params: Optional[Dict] = None,
    regression_params: Optional[Dict] = None,
    directional_params: Optional[Dict] = None,
    use_volume: bool = True,
    volume_col: str = 'volume'
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Apply all four trend identification methods to the dataframe.
    
    Parameters:
    df: Input dataframe with OHLCV data
    sma_params: Dict of parameters for SMA method
    zigzag_params: Dict of parameters for zigzag method
    regression_params: Dict of parameters for regression method
    directional_params: Dict of parameters for directional method
    use_volume: Whether to include volume analysis
    volume_col: Name of volume column
    
    Returns:
    Tuple[DataFrame, Dict] - augmented dataframe and dictionary of all trend info
    """
    result_df = df.copy()
    all_trends = {}
    
    # Apply each method with provided parameters or defaults
    sma_params = sma_params or {}
    result_df, sma_info = add_sma_trends(
        result_df, use_volume=use_volume, volume_col=volume_col, **sma_params
    )
    all_trends['sma'] = sma_info
    
    zigzag_params = zigzag_params or {}
    result_df, zigzag_info = add_zigzag_trends(
        result_df, use_volume=use_volume, volume_col=volume_col, **zigzag_params
    )
    all_trends['zigzag'] = zigzag_info
    
    regression_params = regression_params or {}
    result_df, regression_info = add_regression_trends(
        result_df, use_volume=use_volume, volume_col=volume_col, **regression_params
    )
    all_trends['regression'] = regression_info
    
    directional_params = directional_params or {}
    result_df, directional_info = add_directional_trends(
        result_df, use_volume=use_volume, volume_col=volume_col, **directional_params
    )
    all_trends['directional'] = directional_info
    
    return result_df, all_trends


# Example usage
if __name__ == "__main__":
    # Create sample data
    print("Creating sample data...")
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    # Generate a somewhat realistic price series with trends
    prices = 100
    price_series = [prices]
    for i in range(1, 500):
        # Add some trend structure
        if i < 150:
            change = np.random.randn() * 0.5 + 0.2  # uptrend
        elif i < 250:
            change = np.random.randn() * 0.5 - 0.1  # downtrend
        elif i < 350:
            change = np.random.randn() * 0.3  # sideways
        else:
            change = np.random.randn() * 0.6 + 0.15  # uptrend
        prices = prices * (1 + change/100)
        price_series.append(prices)
    
    df = pd.DataFrame({
        'open': price_series,
        g_high_col: [p * (1 + abs(np.random.randn()*0.01)) for p in price_series],
        g_low_col: [p * (1 - abs(np.random.randn()*0.01)) for p in price_series],
        g_close_col: price_series,
        'volume': np.random.randint(1e6, 1e7, 500)
    }, index=dates)
    
    print(f"Original dataframe shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Apply all trend identification methods
    print("\nApplying all trend identification methods...")
    df_result, trends_info = add_all_trends(
        df,
        use_volume=True,
        sma_params={'min_length': 30, 'min_intensity': 0.10},
        zigzag_params={'threshold': 0.03, 'min_length': 15},
        regression_params={'window': 30, 'min_length': 20},
        directional_params={'lookback': 15, 'min_same_direction': 0.65, 'min_length': 15}
    )
    
    # Display results
    print(f"\nResult dataframe shape: {df_result.shape}")
    
    # Show all target columns that were added
    target_cols = [col for col in df_result.columns if col.startswith('target_')]
    print(f"\nAdded {len(target_cols)} target columns:")
    for col in sorted(target_cols):
        print(f"  - {col}")
    
    # Show sample of the data with target columns
    print("\nSample of data (last 10 rows with target columns):")
    display_cols = [g_close_col, 'volume'] + target_cols[:8]  # Show first 8 target columns to keep display manageable
    print(df_result[display_cols].tail(10).to_string())
    
    # Print trend summaries
    print("\n" + "="*80)
    print("TREND SUMMARIES BY METHOD")
    print("="*80)
    
    for method, info in trends_info.items():
        print(f"\n{method.upper()} METHOD:")
        print(f"  Parameters: {info['params']}")
        print(f"  Last trend direction: {info['last_trend']} (1=up, -1=down, 0=neutral)")
        print(f"  Number of trends detected: {len(info['trends'])}")
        
        if info['trends']:
            print("  First 3 trends:")
            for i, trend in enumerate(info['trends'][:3]):
                print(f"    Trend {i+1}: ID={trend['trend_id']}, Direction={'UP' if trend['direction']==1 else 'DOWN'}, "
                      f"Return={trend['total_return']:.2%}, Length={trend['length']} candles")
    
    # Quick validation that all methods added their columns
    expected_prefixes = ['sma', 'zigzag', 'regression', 'directional']
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    for prefix in expected_prefixes:
        method_cols = [col for col in target_cols if f'target_{prefix}' in col]
        print(f"{prefix}: {len(method_cols)} columns added - {method_cols[:3]}...")
    
    print("\n✅ All methods successfully added their target columns to the dataframe!")
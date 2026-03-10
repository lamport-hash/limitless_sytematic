"""
Trend Analysis Module

This module provides comprehensive trend analysis tools for financial time series data.
It combines trend smoothness calculations, ADX-based trend strength, Hurst exponent
analysis, efficiency ratios, and market regime classification.

Functions:
    - compute_trend_smootheness: Calculates trend smoothness indicator from OHLC data
    - calculate_adx_trend_strength: Computes ADX and directional indicators
    - hurst_exponent: Calculates Hurst exponent to detect mean reversion
    - efficiency_ratio: Measures price movement efficiency (Kaufman's Efficiency Ratio)
    - calculate_trend_score: Computes composite trend score (0-100)
    - calculate_mean_reversion_score: Computes composite mean reversion score (0-100)
    - classify_market_regime: Classifies market regime (TRENDING/MEAN_REVERTING/NEUTRAL)
    - analyze_market_regime: Analyzes market regime over rolling windows
    - volume_confirmation: Confirms trend direction with volume
    - persistence_filter: Filters classifications by persistence
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Union

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Global constants
g_transaction_cost_bp = 2
g_transaction_cost_percent = g_transaction_cost_bp / 10000


def compute_trend_smootheness(p_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Computes the trend smoothness indicator for a given DataFrame of 1-minute candle data.

    Args:
        p_df (pd.DataFrame): A DataFrame containing Open/High/Low/Close columns.

    Returns:
        Tuple[pd.Series, pd.Series]: A tuple containing two Series:
            - dir_trend_smooth: The directional trend smoothness values.
            - trend_smooth: The absolute trend smoothness values.
            In case of errors, returns Series filled with 0s.
    """
    # Default fallback in case of any error
    if len(p_df) == 0:
        return pd.Series([0.0], dtype=float), pd.Series([0.0], dtype=float)

    try:
        # Validate input type (still check since it's cheap and helpful)
        if not isinstance(p_df, pd.DataFrame):
            logger.warning("Input is not a pandas DataFrame. Returning zeros.")
            return pd.Series([0.0] * len(p_df), dtype=float), pd.Series(
                [0.0] * len(p_df), dtype=float
            )

        required_columns = ["Open", "High", "Low", "Close"]
        missing_columns = [col for col in required_columns if col not in p_df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}. Returning zeros.")
            return pd.Series([0.0] * len(p_df), dtype=float), pd.Series(
                [0.0] * len(p_df), dtype=float
            )

        # Compute trend smoothness
        close_diff = p_df["Close"] - p_df["Close"].shift(1)
        range_val = p_df["High"] - p_df["Low"] + 0.00000001
        dir_trend_smooth = close_diff / range_val
        trend_smooth = dir_trend_smooth.abs()

        # Replace NaN with 0 (handles first row and any division issues)
        dir_trend_smooth = dir_trend_smooth.fillna(0)
        trend_smooth = trend_smooth.fillna(0)

        # Ensure output has same length as input (in case of misalignment)
        if len(dir_trend_smooth) != len(p_df):
            logger.warning("Computed trend smoothness length mismatch. Padding with zeros.")
            dir_trend_smooth = dir_trend_smooth.reindex(p_df.index, fill_value=0.0)
            trend_smooth = trend_smooth.reindex(p_df.index, fill_value=0.0)

        return dir_trend_smooth, trend_smooth

    except Exception as e:
        # Log the error but do NOT raise — ensure function always returns values
        logger.error(f"Unexpected error in compute_trend_smootheness: {e}")
        # Return zeros of correct length
        return pd.Series([0.0] * len(p_df), dtype=float), pd.Series([0.0] * len(p_df), dtype=float)


def calculate_adx_trend_strength(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculates ADX (Average Directional Index) and directional indicators.

    Args:
        high (pd.Series): High prices.
        low (pd.Series): Low prices.
        close (pd.Series): Close prices.
        period (int): Period for calculations. Default is 14.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (ADX, Plus DI, Minus DI) values.
    """
    # True Range
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))

    # Directional Movements
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed values
    tr_smooth = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=close.index).rolling(period).mean() / tr_smooth)
    minus_di = 100 * (pd.Series(minus_dm, index=close.index).rolling(period).mean() / tr_smooth)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()

    return adx, plus_di, minus_di


def hurst_exponent(price_series: Union[np.ndarray, pd.Series], max_lag: int = 20) -> float:
    """
    Calculate Hurst exponent to detect mean reversion.

    Args:
        price_series: Price series as array or Series.
        max_lag: Maximum lag for calculation.

    Returns:
        float: Hurst exponent (H < 0.5 indicates mean reversion, H > 0.5 indicates trending).
    """
    if isinstance(price_series, pd.Series):
        price_series = price_series.values

    lags = range(2, max_lag)
    tau = [np.std(np.subtract(price_series[lag:], price_series[:-lag])) for lag in lags]

    # Calculate Hurst exponent
    hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return hurst


def efficiency_ratio(close: pd.Series, period: int = 10) -> np.ndarray:
    """
    Measures price movement efficiency (Kaufman's Efficiency Ratio).

    Args:
        close (pd.Series): Close prices.
        period (int): Period for calculation. Default is 10.

    Returns:
        np.ndarray: Efficiency ratio values (0-1, where higher is more efficient).
    """
    price_change = abs(close - close.shift(period))
    volatility = abs(close.diff()).rolling(period).sum()

    # Avoid division by zero
    efficiency = np.where(volatility != 0, price_change / volatility, 0)
    return efficiency


def calculate_trend_score(
    ohlcv_data: pd.DataFrame, adx_period: int = 14, er_period: int = 10
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Computes composite trend score (0-100) combining ADX, efficiency ratio, and direction consistency.

    Args:
        ohlcv_data (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns.
        adx_period (int): ADX period. Default is 14.
        er_period (int): Efficiency ratio period. Default is 10.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
            (trend_score, adx_score, er_score, direction_score).
    """
    high, low, close = ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["close"]

    # Component 1: ADX strength
    adx, plus_di, minus_di = calculate_adx_trend_strength(high, low, close, adx_period)
    adx_score = np.minimum(adx / 25.0, 1.0) * 100  # Normalize to 0-100

    # Component 2: Efficiency ratio
    er = efficiency_ratio(close, er_period)
    er_score = er * 100

    # Component 3: Direction consistency
    returns = close.pct_change()
    direction_consistency = abs(
        returns.rolling(adx_period).mean() / returns.rolling(adx_period).std()
    )
    direction_score = np.minimum(direction_consistency * 50, 100)

    # Composite trend score (weighted average)
    trend_score = 0.5 * adx_score + 0.3 * er_score + 0.2 * direction_score

    return trend_score, adx_score, er_score, direction_score


def calculate_mean_reversion_score(
    ohlcv_data: pd.DataFrame, hurst_period: int = 30, bb_period: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Computes composite mean reversion score (0-100) combining Hurst exponent, Bollinger Bands,
    and oscillation frequency.

    Args:
        ohlcv_data (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns.
        hurst_period (int): Hurst exponent period. Default is 30.
        bb_period (int): Bollinger Band period. Default is 20.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
            (mr_score, hurst_score, bb_score, swing_score).
    """
    close = ohlcv_data["close"]
    high, low = ohlcv_data["high"], ohlcv_data["low"]

    # Component 1: Hurst exponent
    hurst_values = close.rolling(hurst_period).apply(
        lambda x: hurst_exponent(x) if len(x) == hurst_period else np.nan, raw=False
    )
    # Hurst < 0.5 indicates mean reversion
    hurst_score = np.maximum(0, (0.5 - hurst_values) / 0.5) * 100

    # Component 2: Bollinger Band position
    bb_middle = close.rolling(bb_period).mean()
    bb_std = close.rolling(bb_period).std()
    bb_position = (close - bb_middle) / (2 * bb_std)
    # Score peaks when price is at extremes
    bb_score = (abs(bb_position) * 50).clip(0, 100)

    # Component 3: Short-term oscillation frequency
    price_swings = (high.rolling(5).max() - low.rolling(5).min()) / close
    swing_score = np.minimum(price_swings * 10000, 100)  # Normalize

    # Composite MR score
    mr_score = 0.4 * hurst_score + 0.4 * bb_score + 0.2 * swing_score

    return mr_score, hurst_score, bb_score, swing_score


def classify_market_regime(
    ohlcv_data: pd.DataFrame, transaction_cost_bp: int = 2
) -> Dict[str, Any]:
    """
    Classifies market regime adjusting for transaction costs.

    Args:
        ohlcv_data (pd.DataFrame): DataFrame with 'high', 'low', 'close', 'volume' columns.
        transaction_cost_bp (int): Transaction cost in basis points. Default is 2.

    Returns:
        Dict[str, Any]: Dictionary containing classification, effective scores, and raw scores.
    """
    # Calculate raw scores
    trend_score, adx, er, direction = calculate_trend_score(ohlcv_data)
    mr_score, hurst, bb, swing = calculate_mean_reversion_score(ohlcv_data)

    # Adjust for transaction costs
    cost_adjustment = transaction_cost_bp * 5  # Scale factor
    effective_trend_score = np.maximum(0, trend_score - cost_adjustment)
    effective_mr_score = np.maximum(0, mr_score - cost_adjustment)

    # Classification logic
    conditions = [
        (effective_trend_score > 60) & (effective_trend_score > effective_mr_score + 20),
        (effective_mr_score > 60) & (effective_mr_score > effective_trend_score + 20),
        (abs(effective_trend_score - effective_mr_score) < 20)
        & ((effective_trend_score + effective_mr_score) < 80),
    ]

    regimes = ["TRENDING", "MEAN_REVERTING", "NEUTRAL"]
    classification = np.select(conditions, regimes, default="UNCLEAR")

    return {
        "classification": classification,
        "trend_score": effective_trend_score,
        "mr_score": effective_mr_score,
        "raw_scores": {
            "adx": adx,
            "efficiency_ratio": er,
            "direction_consistency": direction,
            "hurst": hurst,
            "bollinger_score": bb,
            "oscillation_score": swing,
        },
    }


def volume_confirmation(ohlcv_data: pd.DataFrame, trend_direction: float) -> float:
    """
    Confirms trend direction with volume.

    Args:
        ohlcv_data (pd.DataFrame): DataFrame with 'volume' column.
        trend_direction (float): Direction of trend (>0 for up, <0 for down).

    Returns:
        float: Confidence multiplier (1.2 if confirmed, 0.8 otherwise).
    """
    volume_sma = ohlcv_data["volume"].rolling(20).mean()
    current_volume_ratio = ohlcv_data["volume"] / volume_sma

    # Volume should confirm trend direction
    if trend_direction > 0 and current_volume_ratio.iloc[-1] > 1.2:
        return 1.2
    elif trend_direction < 0 and current_volume_ratio.iloc[-1] > 1.2:
        return 1.2
    else:
        return 0.8


def persistence_filter(classifications: np.ndarray, min_bars: int = 3) -> str:
    """
    Filters classifications requiring regime to persist for minimum bars.

    Args:
        classifications (np.ndarray): Array of regime classifications.
        min_bars (int): Minimum number of bars for persistence. Default is 3.

    Returns:
        str: Final classification ('NEUTRAL' if not persistent).
    """
    if len(classifications) < min_bars:
        return "NEUTRAL"

    recent_classifications = classifications[-min_bars:]
    if len(set(recent_classifications)) == 1:  # All same
        return recent_classifications[0]
    else:
        return "NEUTRAL"


def analyze_market_regime(ohlcv_df: pd.DataFrame, lookback_period: int = 30) -> pd.DataFrame:
    """
    Analyzes market regime over rolling windows.

    Args:
        ohlcv_df (pd.DataFrame): DataFrame with OHLCV data.
        lookback_period (int): Lookback period for rolling analysis. Default is 30.

    Returns:
        pd.DataFrame: DataFrame with regime, trend_score, and mr_score columns.
    """
    results = []

    for i in range(lookback_period, len(ohlcv_df)):
        window_data = ohlcv_df.iloc[i - lookback_period : i]
        analysis = classify_market_regime(window_data)

        results.append(
            {
                "timestamp": ohlcv_df.index[i],
                "regime": analysis["classification"][-1],
                "trend_score": analysis["trend_score"][-1],
                "mr_score": analysis["mr_score"][-1],
            }
        )

    return pd.DataFrame(results)
